import os
import sys
from pathlib import Path

# Project root (NeedleGNN/)
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Add TVM path (must be before any TVM imports)
_DEFAULT_TVM_PATHS = [
    "/home/nas/polin/cmu-berlin/tvm-0.10.0/python",
    str((REPO_ROOT.parent / "tvm-0.10.0" / "python").resolve()),
]
for tvm_python_path in _DEFAULT_TVM_PATHS:
    if os.path.exists(tvm_python_path) and tvm_python_path not in sys.path:
        sys.path.insert(0, tvm_python_path)
        break

# Remove user site-packages to avoid using pip-installed TVM
user_site = os.path.expanduser("~/.local/lib/python3.9/site-packages")
if user_site in sys.path:
    sys.path.remove(user_site)

import numpy as np
import networkx as nx
from collections import namedtuple
import tvm
from tvm import relay
from tvm.contrib import graph_executor

# TVM GNN utilities
from apps.gnn_tvm_utils.gconv import GraphConv


class GraphConvNetwork:
    """
    Graph Convolutional Network using TVM Relay
    
    This class encapsulates the entire GCN pipeline including:
    - Graph to sparse format conversion (CSR)
    - Normalization computation
    - TVM Relay model construction
    - Parameter preparation
    - Model compilation and execution
    """
    def __init__(self, graph, data, dataset_info, num_hidden=512, num_layers=2, 
                 device="cpu", opt_level=0, target=None, 
                 cpu_instruction_set=None, use_gpu_mma=False, dtype=None, 
                 activation=None, norm=None, bias=None, verbose=False):
                 # activation can be a string (e.g., "relu") or a relay function (e.g., relay.nn.relu)
        
        self.graph = graph
        self.data = data
        self.dataset_info = dataset_info
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.device = device
        self.cpu_instruction_set = cpu_instruction_set
        self.use_gpu_mma = use_gpu_mma
        self.verbose = verbose

        # Convert activation string to relay function if needed
        if activation is not None and isinstance(activation, str):
            # Convert string to relay function
            activation_lower = activation.lower()
            if activation_lower == "relu":
                self.activation = relay.nn.relu
            elif activation_lower == "sigmoid":
                self.activation = relay.nn.sigmoid
            elif activation_lower == "tanh":
                self.activation = relay.nn.tanh
            elif activation_lower == "leaky_relu":
                self.activation = relay.nn.leaky_relu
            else:
                raise ValueError(f"Unknown activation string: {activation}. Supported: 'relu', 'sigmoid', 'tanh', 'leaky_relu'")
        else:
            self.activation = activation
        self.bias = bias
        
        # Set dtype: if not specified, use float16 for Tensor Core, float32 otherwise
        if dtype is None:
            if use_gpu_mma and device == "cuda":
                self.dtype = "float16"
            else:
                self.dtype = "float32"
        else:
            if dtype not in ["float32", "float16"]:
                raise ValueError(f"dtype must be 'float32' or 'float16', got '{dtype}'")
            self.dtype = dtype
        
        # Set target and device based on device type and instruction set
        if target is None:
            if device == "cpu":
                # Use safe CPU target detection to avoid illegal instructions
                target, actual_inst_set = get_safe_cpu_target(cpu_instruction_set)
                if actual_inst_set != cpu_instruction_set and cpu_instruction_set is not None:
                    # Update the instruction set to reflect what was actually used
                    self.cpu_instruction_set = actual_inst_set
            elif device == "cuda":
                target = "cuda"
            else:
                raise ValueError(f"Unsupported device: {device}")
        
        self.target = target
        self.opt_level = opt_level
        
        # Create TVM device
        if device == "cpu":
            self.dev = tvm.device(target.split()[0], 0)  # Extract "llvm" from target
        elif device == "cuda":
            self.dev = tvm.cuda(0)
        else:
            raise ValueError(f"Unsupported device: {device}")
        
        # Set global verbose flag for tensorcore_strategy
        from apps.gnn_tvm_utils.tensorcore_strategy import set_verbose
        set_verbose(verbose)
        
        # Prepare parameters and build model
        self.params = self._prepare_params()
        self.norm = relay.Constant(tvm.nd.array(self.params["norm"])) if norm else None

        self.mod = None
        self.lib = None
        self.executor = None
        
    def _prepare_params(self):
        """Prepare parameters including features, adjacency matrix, and normalization"""
        params = {}
        
        # Use self.dtype for all parameters
        dtype = self.dtype
        
        # Handle features
        if hasattr(self.data.features, 'numpy'):
            features_np = self.data.features.numpy()
        else:
            features_np = self.data.features
        
        # CRITICAL: Ensure features match dataset_info dimensions (should be padded to 64 multiples)
        # Get expected shape from dataset_info
        expected_num_nodes = self.dataset_info.get('num_nodes', features_np.shape[0])
        expected_infeat = self.dataset_info.get('infeat_dim', features_np.shape[1])
        expected_shape = (expected_num_nodes, expected_infeat)
        
        # Pad if needed
        if features_np.shape != expected_shape:
            pad_rows = expected_shape[0] - features_np.shape[0]
            pad_cols = expected_shape[1] - features_np.shape[1]
            if pad_rows > 0 or pad_cols > 0:
                features_np = np.pad(
                    features_np,
                    ((0, pad_rows), (0, pad_cols)),
                    mode='constant',
                    constant_values=0
                )
                if self.verbose:
                    print(f"  [Init Params Padding] features: {self.data.features.shape if hasattr(self.data.features, 'shape') else 'unknown'} -> {features_np.shape}")
        
        params["infeats"] = features_np.astype(dtype)
        
        # Choose format based on device and MMA flag
        if self.device == "cuda" and self.use_gpu_mma:
            # Convert directly from graph to DTC format for Tensor Core (following spmm_mma.py)
            # This avoids intermediate CSR conversion and ensures consistency with spmm_mma.py
            try:
                from apps.gnn_tvm_utils.csr_to_dtc import graph_to_dtc
                # Use self.dtype for DTC conversion
                dtc_dtype = np.float16 if dtype == "float16" else np.float32
                dtc_data = graph_to_dtc(self.graph, dtype=dtc_dtype)
                
                # Store DTC format data
                params["A_compressed"] = dtc_data['A_compressed']
                params["block_col_mapping"] = dtc_data['block_col_mapping']
                params["M"] = dtc_data['M']
                params["K"] = dtc_data['K']
                
                # CRITICAL: Ensure M_padded and K_padded are multiples of 64 for Tensor Core
                M_padded_orig = dtc_data['M_padded']
                K_padded_orig = dtc_data['K_padded']
                if self.verbose:
                    print(f"  [DTC Padding] Original: M_padded={M_padded_orig} (multiple of 64: {M_padded_orig % 64 == 0}), K_padded={K_padded_orig} (multiple of 64: {K_padded_orig % 64 == 0})")
                
                M_padded = ((M_padded_orig + 63) // 64) * 64
                K_padded = ((K_padded_orig + 63) // 64) * 64
                
                # If padding changed, need to pad A_compressed
                if M_padded != M_padded_orig or K_padded != K_padded_orig:
                    A_compressed_orig = params["A_compressed"]
                    A_compressed_padded = np.zeros((M_padded, K_padded), dtype=A_compressed_orig.dtype)
                    A_compressed_padded[:M_padded_orig, :K_padded_orig] = A_compressed_orig
                    params["A_compressed"] = A_compressed_padded
                    if self.verbose:
                        print(f"  [Padding Fix] M_padded: {M_padded_orig} -> {M_padded}, K_padded: {K_padded_orig} -> {K_padded}")
                        print(f"  [Padding Fix] A_compressed padded from {A_compressed_orig.shape} to {A_compressed_padded.shape}")
                else:
                    if self.verbose:
                        print(f"  [DTC Padding] No padding needed: M_padded={M_padded}, K_padded={K_padded} (both are multiples of 64)")
                
                params["M_padded"] = M_padded
                params["K_padded"] = K_padded
                params["num_row_blocks"] = dtc_data['num_row_blocks']
                
                # Note: We don't need CSR data anymore since we're using DTC directly
                # The DTC format is used with dense operation in gconv.py
                
                params["format"] = "dtc"  # Mark as DTC format
                
                if self.verbose:
                    print(f"  Converted graph directly to DTC format: {dtc_data['M']}x{dtc_data['K']} -> {M_padded}x{K_padded} (dtype={dtype})")
                    print(f"  Using global mapping (matching spmm_mma.py)")
            except ImportError as e:
                print(f"  Warning: Could not import DTC conversion: {e}")
                print("  Falling back to CSR format")
                # Fall back to CSR
                try:
                    adjacency = nx.to_scipy_sparse_array(self.graph, format='csr')
                except AttributeError:
                    adjacency = nx.to_scipy_sparse_matrix(self.graph, format='csr')
                params["g_data"] = adjacency.data.astype(dtype)
                params["indices"] = adjacency.indices.astype("int32")
                params["indptr"] = adjacency.indptr.astype("int32")
                params["format"] = "csr"
        else:
            # Use CSR format (standard)
            try:
                adjacency = nx.to_scipy_sparse_array(self.graph, format='csr')
            except AttributeError:
                adjacency = nx.to_scipy_sparse_matrix(self.graph, format='csr')
            params["g_data"] = adjacency.data.astype(dtype)
            params["indices"] = adjacency.indices.astype("int32")
            params["indptr"] = adjacency.indptr.astype("int32")
            params["format"] = "csr"
        
        # Compute normalization w.r.t. node degrees
        degs = [self.graph.degree(i) for i in range(self.graph.number_of_nodes())]
        degs = np.array(degs, dtype=dtype)
        degs[degs == 0] = 1.0  # Avoid division by zero
        # Compute power in float32 for precision, then convert back
        norm_power = np.power(degs.astype("float32"), -0.5)
        params["norm"] = norm_power.astype(dtype)
        params["norm"] = params["norm"].reshape((params["norm"].shape[0], 1))
        
        return params
    
    def _build_relay_model(self):
        """Build TVM Relay model"""
        # Use self.dtype
        dtype = self.dtype
        
        # Create input variable
        # CRITICAL: Use actual features shape from params
        # The shape should match dataset_info['infeat_dim'] (which may be padded if pad_infeat_dim_to_32=True)
        infeats_shape = tuple(self.params["infeats"].shape)
        # Verify that infeats_shape[1] matches dataset_info['infeat_dim']
        actual_infeat_dim = infeats_shape[1]
        expected_infeat_dim = self.dataset_info['infeat_dim']
        if actual_infeat_dim != expected_infeat_dim:
            print(f"  [Warning] infeats_shape[1]={actual_infeat_dim} != dataset_info['infeat_dim']={expected_infeat_dim}")
            print(f"  [Warning] Using actual shape from params: {infeats_shape}")
            # CRITICAL FIX: Update dataset_info to match actual shape
            self.dataset_info['infeat_dim'] = actual_infeat_dim
            print(f"  [Fix] Updated dataset_info['infeat_dim'] to {actual_infeat_dim}")
        # Explicitly set dtype to match self.dtype
        infeats = relay.var("infeats", shape=infeats_shape, dtype=dtype)
        
        
        # Check format and create appropriate adjacency representation
        matrix_format = self.params.get("format", "csr")
        
        if matrix_format == "dtc":
            # DTC format: use compressed matrix directly (no CSR data needed)
            # Following spmm_mma.py, we use global mapping and dense operation
            A_compressed_np = self.params["A_compressed"]
            M_padded_param = self.params['M_padded']
            K_padded_param = self.params['K_padded']
            print(f"  [Relay Build] A_compressed numpy shape: {A_compressed_np.shape}")
            print(f"  [Relay Build] M_padded from params: {M_padded_param}, K_padded from params: {K_padded_param}")
            # CRITICAL: Ensure A_compressed has the correct shape
            # If numpy array shape doesn't match M_padded/K_padded, reshape it
            if A_compressed_np.shape != (M_padded_param, K_padded_param):
                print(f"  [Relay Build] WARNING: A_compressed shape mismatch! Reshaping from {A_compressed_np.shape} to ({M_padded_param}, {K_padded_param})")
                # Reshape to ensure correct shape
                A_compressed_np = A_compressed_np.reshape(M_padded_param, K_padded_param)
            A_compressed = relay.Constant(tvm.nd.array(A_compressed_np))
            # Store metadata for DTC format (no CSR data needed)
            Adjacency = namedtuple("Adjacency", ["A_compressed", "format", 
                                                  "block_col_mapping", "M", "K", "M_padded", "K_padded", "num_row_blocks"])
            adj = Adjacency(
                A_compressed=A_compressed,
                format="dtc",
                block_col_mapping=self.params["block_col_mapping"],
                M=self.params["M"],
                K=self.params["K"],
                M_padded=self.params["M_padded"],
                K_padded=self.params["K_padded"],
                num_row_blocks=self.params["num_row_blocks"]
            )
        else:
            # CSR format: use standard sparse representation
            g_data = relay.Constant(tvm.nd.array(self.params["g_data"]))
            indices = relay.Constant(tvm.nd.array(self.params["indices"]))
            indptr = relay.Constant(tvm.nd.array(self.params["indptr"]))
            
            Adjacency = namedtuple("Adjacency", ["data", "indices", "indptr", "format"])
            adj = Adjacency(g_data, indices, indptr, "csr")
        
        # Build layers
        layers = []
        current_input = infeats
        
        # Determine if we should use Tensor Core for sparse_dense
        use_tc = (self.device == "cuda" and self.use_gpu_mma)
        
        # CRITICAL FIX: Force all dimensions to be multiples of 32 ONLY for Tensor Core
        # For CPU and other cases, use original dimensions to avoid type mismatch
        # Get infeat_dim from dataset_info (always needed)
        infeat_dim = self.dataset_info['infeat_dim']
        
        if use_tc:
            # Tensor Core requires dimensions to be multiples of 32
            # Following SparseTIR approach - dimensions must be fixed at definition time
            block_factor_n = 64
            num_hidden_padded = ((self.num_hidden + block_factor_n - 1) // block_factor_n) * block_factor_n
            # Force infeat_dim to be multiple of 32
            infeat_dim_padded = ((infeat_dim + block_factor_n - 1) // block_factor_n) * block_factor_n
        else:
            # For CPU and other cases, use original dimensions (no padding)
            num_hidden_padded = self.num_hidden
            infeat_dim_padded = infeat_dim
        
        # First layer: input_dim -> output_dim
        layers.append(
            GraphConv(
                layer_name="layers.0",
                input_dim=infeat_dim_padded,  # Padded only for Tensor Core
                output_dim=num_hidden_padded,  # Padded only for Tensor Core
                adj=adj,
                input=current_input,
                norm=self.norm,
                activation=self.activation,
                use_tensorcore=use_tc,
                dtype=self.dtype,
            )
        )
        current_input = layers[-1]
        
        # Hidden layers: num_hidden -> num_hidden
        for i in range(1, self.num_layers - 1):
            layers.append(
                GraphConv(
                    layer_name=f"layers.{i}",
                    input_dim=num_hidden_padded,  # Padded only for Tensor Core
                    output_dim=num_hidden_padded,  # Padded only for Tensor Core
                    adj=adj,
                    input=current_input,
                    norm=self.norm,
                    activation=self.activation,
                    use_tensorcore=use_tc,
                    dtype=self.dtype,
                )
            )
            current_input = layers[-1]
        
        # Output layer: num_hidden -> num_hidden
        layers.append(
            GraphConv(
                layer_name=f"layers.{self.num_layers - 1}",
                input_dim=num_hidden_padded,  # Padded only for Tensor Core
                output_dim=num_hidden_padded,  # Padded only for Tensor Core
                adj=adj,
                input=current_input,
                norm=self.norm,
                activation=self.activation,
                use_tensorcore=use_tc,
                dtype=self.dtype,
            )
        )
        
        output = layers[-1]
        
        # Create random weights and biases
        self._init_weights()
        
        # Prepare parameters for binding (exclude infeats as it's an input)
        # Also exclude non-array metadata (format, block_col_mapping, M, K, etc.)
        exclude_keys = {"infeats", "format", "block_col_mapping", "M", "K", "M_padded", "K_padded", "num_row_blocks"}
        params_for_binding = {k: v for k, v in self.params.items() if k not in exclude_keys}
        
        # Fix bias shapes (ensure 1D)
        for key in list(params_for_binding.keys()):
            if 'bias' in key and isinstance(params_for_binding[key], np.ndarray):
                bias = params_for_binding[key]
                if len(bias.shape) == 2 and bias.shape[1] == 1:
                    params_for_binding[key] = bias.reshape(-1)
        
        # Create Relay function
        func = relay.Function(relay.analysis.free_vars(output), output)
        func = relay.build_module.bind_params_by_name(func, params_for_binding)
        mod = tvm.IRModule()
        mod["main"] = func
        
        return mod, params_for_binding
    
    def _init_weights(self):
        """Initialize random weights and biases"""
        # Set seed for reproducible weights across all tests
        np.random.seed(42)
        
        # Use self.dtype
        dtype = self.dtype
        
        # Determine if we should use Tensor Core (only pad for Tensor Core)
        use_tc = (self.device == "cuda" and self.use_gpu_mma)
        
        # Get infeat_dim from dataset_info (always needed, regardless of use_tc)
        # CRITICAL: Verify it matches actual infeats shape
        infeat_dim = self.dataset_info['infeat_dim']
        actual_infeat_dim = self.params["infeats"].shape[1]
        if infeat_dim != actual_infeat_dim:
            print(f"  [Init Weights Warning] dataset_info['infeat_dim']={infeat_dim} != actual infeats.shape[1]={actual_infeat_dim}")
            print(f"  [Init Weights Fix] Using actual infeats.shape[1]={actual_infeat_dim}")
            infeat_dim = actual_infeat_dim
            self.dataset_info['infeat_dim'] = actual_infeat_dim
        
        if use_tc:
            # CRITICAL FIX: Force all dimensions to be multiples of 32 ONLY for Tensor Core
            # Following SparseTIR approach - dimensions must be fixed at definition time
            block_factor_n = 64
            # Force num_hidden to be multiple of 32
            num_hidden_padded = ((self.num_hidden + block_factor_n - 1) // block_factor_n) * block_factor_n
            # Force infeat_dim to be multiple of 32
            # Note: infeat_dim from dataset_info may already be padded if pad_infeat_dim_to_32=True
            # But we still need to ensure it's a multiple of 32 for Tensor Core
            infeat_dim_padded = ((infeat_dim + block_factor_n - 1) // block_factor_n) * block_factor_n
        else:
            # For CPU and other cases, use original dimensions (no padding)
            num_hidden_padded = self.num_hidden
            infeat_dim_padded = infeat_dim
        
        # First layer: create weight with appropriate dimensions
        weight_0 = np.random.randn(infeat_dim, self.num_hidden).astype(dtype)
        if use_tc and (infeat_dim_padded > infeat_dim or num_hidden_padded > self.num_hidden):
            # Pad to (infeat_dim_padded, num_hidden_padded) for Tensor Core
            weight_0_padded = np.zeros((infeat_dim_padded, num_hidden_padded), dtype=dtype)
            weight_0_padded[:infeat_dim, :self.num_hidden] = weight_0
            self.params["layers.0.weight"] = weight_0_padded
            if self.verbose:
                print(f"  [Init Weights] layers.0.weight: original=({infeat_dim}, {self.num_hidden}) -> padded=({infeat_dim_padded}, {num_hidden_padded})")
        else:
            # No padding needed
            self.params["layers.0.weight"] = weight_0
            if self.verbose:
                print(f"  [Init Weights] layers.0.weight: ({infeat_dim}, {self.num_hidden}) (no padding)")
        
        # Bias
        bias_0 = np.random.randn(self.num_hidden).astype(dtype)
        if use_tc and num_hidden_padded > self.num_hidden:
            # Pad to num_hidden_padded for Tensor Core
            bias_0_padded = np.zeros(num_hidden_padded, dtype=dtype)
            bias_0_padded[:self.num_hidden] = bias_0
            self.params["layers.0.bias"] = bias_0_padded
            if self.verbose:
                print(f"  [Init Weights] layers.0.bias: original=({self.num_hidden},) -> padded=({num_hidden_padded},)")
        else:
            # No padding needed
            self.params["layers.0.bias"] = bias_0
            if self.verbose:
                print(f"  [Init Weights] layers.0.bias: ({self.num_hidden},) (no padding)")
        
        # Hidden layers
        for i in range(1, self.num_layers - 1):
            weight_i = np.random.randn(self.num_hidden, self.num_hidden).astype(dtype)
            if use_tc and num_hidden_padded > self.num_hidden:
                # Pad to (num_hidden_padded, num_hidden_padded) for Tensor Core
                weight_i_padded = np.zeros((num_hidden_padded, num_hidden_padded), dtype=dtype)
                weight_i_padded[:self.num_hidden, :self.num_hidden] = weight_i
                self.params[f"layers.{i}.weight"] = weight_i_padded
                if self.verbose:
                    print(f"  [Init Weights] layers.{i}.weight: original=({self.num_hidden}, {self.num_hidden}) -> padded=({num_hidden_padded}, {num_hidden_padded})")
            else:
                # No padding needed
                self.params[f"layers.{i}.weight"] = weight_i
                if self.verbose:
                    print(f"  [Init Weights] layers.{i}.weight: ({self.num_hidden}, {self.num_hidden}) (no padding)")
            
            # Bias
            bias_i = np.random.randn(self.num_hidden).astype(dtype)
            if use_tc and num_hidden_padded > self.num_hidden:
                # Pad to num_hidden_padded for Tensor Core
                bias_i_padded = np.zeros(num_hidden_padded, dtype=dtype)
                bias_i_padded[:self.num_hidden] = bias_i
                self.params[f"layers.{i}.bias"] = bias_i_padded
                if self.verbose:
                    print(f"  [Init Weights] layers.{i}.bias: original=({self.num_hidden},) -> padded=({num_hidden_padded},)")
            else:
                # No padding needed
                self.params[f"layers.{i}.bias"] = bias_i
                if self.verbose:
                    print(f"  [Init Weights] layers.{i}.bias: ({self.num_hidden},) (no padding)")
        
        # Output layer
        weight_out = np.random.randn(self.num_hidden, self.num_hidden).astype(dtype)
        if use_tc and num_hidden_padded > self.num_hidden:
            # Pad to (num_hidden_padded, num_hidden_padded) for Tensor Core
            weight_out_padded = np.zeros((num_hidden_padded, num_hidden_padded), dtype=dtype)
            weight_out_padded[:self.num_hidden, :self.num_hidden] = weight_out
            self.params[f"layers.{self.num_layers - 1}.weight"] = weight_out_padded
            if self.verbose:
                print(f"  [Init Weights] layers.{self.num_layers - 1}.weight: original=({self.num_hidden}, {self.num_hidden}) -> padded=({num_hidden_padded}, {num_hidden_padded})")
        else:
            # No padding needed
            self.params[f"layers.{self.num_layers - 1}.weight"] = weight_out
            if self.verbose:
                print(f"  [Init Weights] layers.{self.num_layers - 1}.weight: ({self.num_hidden}, {self.num_hidden}) (no padding)")
        
        # Bias
        bias_out = np.random.randn(self.num_hidden).astype(dtype)
        if use_tc and num_hidden_padded > self.num_hidden:
            # Pad to num_hidden_padded for Tensor Core
            bias_out_padded = np.zeros(num_hidden_padded, dtype=dtype)
            bias_out_padded[:self.num_hidden] = bias_out
            self.params[f"layers.{self.num_layers - 1}.bias"] = bias_out_padded
            if self.verbose:
                print(f"  [Init Weights] layers.{self.num_layers - 1}.bias: original=({self.num_hidden},) -> padded=({num_hidden_padded},)")
        else:
            # No padding needed
            self.params[f"layers.{self.num_layers - 1}.bias"] = bias_out
            if self.verbose:
                print(f"  [Init Weights] layers.{self.num_layers - 1}.bias: ({self.num_hidden},) (no padding)")
    
    
    def compile(self):
        """Compile the TVM Relay model"""
        # CRITICAL: Register float16 fix for CUDA sparse_dense when using float16
        # This fixes TVM 0.10.0 bug where sparse_dense uses hardcoded 0.0 (float32)
        # instead of dtype-aware constant, causing ValueError for float16
        if self.device == "cuda" and self.dtype == "float16":
            try:
                from apps.gnn_tvm_utils.spmm_cuda_alter_layout import register_fixed_sparse_dense_tir
                register_fixed_sparse_dense_tir()
                # print("✓ Registered fixed sparse_dense_tir (supports float16)")
            except ImportError as e:
                print(f"⚠️  Warning: Could not import float16 fix: {e}")
        
        # Register Tensor Core strategies if using GPU MMA
        if self.device == "cuda" and self.use_gpu_mma:
            try:
                from apps.gnn_tvm_utils.tensorcore_strategy import register_tensorcore_strategies, set_force_tensorcore
                # Force Tensor Core usage - no fallback to CUDA Core
                set_force_tensorcore(force=True)
                register_tensorcore_strategies(force_tensorcore=True)
                print("✓ Registered Tensor Core strategies for CUDA (FORCED - no fallback)")
            except ImportError as e:
                print(f"⚠️  Warning: Could not import Tensor Core strategies: {e}")
                print("   Falling back to standard CUDA implementation")
        
        # Note: Activation is applied AFTER Tensor Core computation (in Relay layer)
        # We do NOT set activation context to avoid fusion issues
        # This ensures Tensor Core schedule is clean without activation fusion
        
        self.mod, params_for_binding = self._build_relay_model()
        
        # Build with PassContext (simple approach like notebook)
        # alter_op_layout is automatically applied during build
        # For Tensor Core, disable fusion to avoid schedule issues with activation
        config = {}
        if self.device == "cuda" and self.use_gpu_mma:
            # Disable fusion to ensure clean Tensor Core schedule
            # Activation will be applied separately after Tensor Core computation
            config["relay.FuseOps.max_depth"] = 1  # Limit fusion depth
        with tvm.transform.PassContext(opt_level=self.opt_level, config=config):
            self.lib = relay.build(self.mod, self.target, params=params_for_binding)
        
        # Create graph executor
        self.executor = graph_executor.GraphModule(self.lib["default"](self.dev))
        
        # Set input
        input_data = self.params["infeats"]
        self.executor.set_input("infeats", input_data)
        
        return self.executor
    
    def forward(self, features=None):
        """
        Run forward pass (PyTorch-style)
        
        Parameters
        ----------
        features: np.ndarray or torch.Tensor, optional
            Input features. If None, uses features from data.
            If provided, updates the internal input value.
        
        Returns
        -------
        output: np.ndarray
            Output logits
        """
        if self.executor is None:
            self.compile()
        
        # Use self.dtype
        dtype = self.dtype
        
        # Get expected shape from executor (model's expected input shape)
        expected_shape = None
        if self.executor is not None:
            # Get expected shape from the first input
            input_name = "infeats"
            if hasattr(self.executor, 'get_input'):
                try:
                    expected_ndarray = self.executor.get_input(0)
                    expected_shape = expected_ndarray.shape
                except:
                    pass
        
        # If we can't get from executor, try to infer from params or dataset_info
        if expected_shape is None:
            # Try to get from dataset_info (should be padded)
            if hasattr(self, 'dataset_info') and 'infeat_dim' in self.dataset_info:
                expected_num_nodes = self.dataset_info.get('num_nodes', None)
                expected_infeat = self.dataset_info['infeat_dim']
                if expected_num_nodes:
                    expected_shape = (expected_num_nodes, expected_infeat)
        
        # Handle input features
        if features is not None:
            # Convert torch.Tensor to numpy if needed
            if hasattr(features, 'numpy'):
                features_np = features.numpy()
            elif hasattr(features, 'detach'):
                # PyTorch tensor
                features_np = features.detach().cpu().numpy()
            else:
                features_np = np.asarray(features)
            
            # Use appropriate dtype
            features_np = features_np.astype(dtype)
            
            # CRITICAL: Pad features to match expected shape if needed
            if expected_shape is not None and features_np.shape != expected_shape:
                pad_rows = expected_shape[0] - features_np.shape[0]
                pad_cols = expected_shape[1] - features_np.shape[1]
                if pad_rows > 0 or pad_cols > 0:
                    features_np = np.pad(
                        features_np,
                        ((0, pad_rows), (0, pad_cols)),
                        mode='constant',
                        constant_values=0
                    )
                    if self.verbose:
                        print(f"  [Forward Padding] features: {features.shape if hasattr(features, 'shape') else 'unknown'} -> {features_np.shape}")
            
            # Update internal params
            self.params["infeats"] = features_np
            
            # Update executor input
            self.executor.set_input("infeats", features_np)
        else:
            # Use default features from data
            input_data = self.params["infeats"]
            
            # CRITICAL: Pad input_data to match expected shape if needed
            if expected_shape is not None and input_data.shape != expected_shape:
                pad_rows = expected_shape[0] - input_data.shape[0]
                pad_cols = expected_shape[1] - input_data.shape[1]
                if pad_rows > 0 or pad_cols > 0:
                    input_data = np.pad(
                        input_data,
                        ((0, pad_rows), (0, pad_cols)),
                        mode='constant',
                        constant_values=0
                    )
                    if self.verbose:
                        print(f"  [Forward Padding] self.params['infeats']: {self.params['infeats'].shape} -> {input_data.shape}")
                    # Update params with padded version
                    self.params["infeats"] = input_data
            
            self.executor.set_input("infeats", input_data)
        
        # Run forward pass
        self.executor.run()
        output = self.executor.get_output(0).numpy()
        return output
    
    def __call__(self, features=None):
        """
        Make the model callable like PyTorch: model(inputs)
        
        Parameters
        ----------
        features: np.ndarray or torch.Tensor, optional
            Input features. If None, uses features from data.
            If provided, updates the internal input value.
        
        Returns
        -------
        output: np.ndarray
            Output logits
        """
        return self.forward(features)
    
    def show_ir(self, show_relay=True, show_tir=True, show_source=True, 
                check_tensorcore=True, verbose=True, save_to_file=None):
        """
        Display IR (Intermediate Representation) at different stages
        
        This function helps inspect the low-level IR to verify Tensor Core usage
        and understand the compilation process.
        
        Parameters
        ----------
        show_relay: bool
            If True, display Relay IR (high-level, before compilation)
        show_tir: bool
            If True, display TIR (Tensor IR, low-level after compilation)
        show_source: bool
            If True, display generated source code (CUDA/LLVM)
        check_tensorcore: bool
            If True, check for Tensor Core intrinsics in generated code
        verbose: bool
            If True, print detailed information (if False, truncates output)
        save_to_file: str or None
            If provided, save the complete (untruncated) IR output to this file path.
            The file will contain all IR information regardless of verbose setting.
        
        Returns
        -------
        dict: Dictionary containing IR information
            - 'relay_ir': str or None
            - 'tir_ir': str or None
            - 'source_code': str or None
            - 'has_tensorcore': bool
            - 'saved_file': str or None (path to saved file if save_to_file was provided)
        """
        result = {
            'relay_ir': None,
            'tir_ir': None,
            'source_code': None,
            'has_tensorcore': False,
            'saved_file': None
        }
        
        # Prepare file content if saving to file
        file_content_lines = []
        if save_to_file:
            file_content_lines.append("=" * 80)
            file_content_lines.append("IR Inspection for GraphConvNetwork")
            file_content_lines.append("=" * 80)
            file_content_lines.append("")
            file_content_lines.append(f"Device: {self.device}")
            file_content_lines.append(f"Target: {self.target}")
            file_content_lines.append(f"Use GPU MMA: {self.use_gpu_mma}")
            file_content_lines.append(f"Dtype: {self.dtype}")
            file_content_lines.append("")
        
        print("=" * 80)
        print("IR Inspection for GraphConvNetwork")
        print("=" * 80)
        print()
        
        # Ensure model is compiled
        if self.mod is None:
            # print("⚠️  Model not compiled yet. Compiling now...")
            self.compile()
            print()
        
        # 1. Show Relay IR (high-level)
        if show_relay:
            section_header = "-" * 80 + "\n1. Relay IR (High-level, before compilation)\n" + "-" * 80
            # print(section_header)
            if save_to_file:
                file_content_lines.append(section_header)
            try:
                relay_ir = str(self.mod)
                result['relay_ir'] = relay_ir
                if verbose:
                    # print(relay_ir)
                    if save_to_file:
                        file_content_lines.append(relay_ir)
                else:
                    # Show first 1000 characters
                    # print(relay_ir[:1000])
                    # if len(relay_ir) > 1000:
                        # print(f"\n... (truncated, total length: {len(relay_ir)} characters)")
                    # Always save full content to file
                    if save_to_file:
                        file_content_lines.append(relay_ir)
                        file_content_lines.append(f"\n(Total length: {len(relay_ir)} characters)")
                # print()
                if save_to_file:
                    file_content_lines.append("")
            except Exception as e:
                error_msg = f"⚠️  Could not display Relay IR: {e}"
                # print(error_msg)
                # print()
                if save_to_file:
                    file_content_lines.append(error_msg)
                    file_content_lines.append("")
        
        # 2. Show TIR (Tensor IR, low-level)
        if show_tir and self.lib is not None:
            section_header = "-" * 80 + "\n2. TIR (Tensor IR, low-level after compilation)\n" + "-" * 80
            # print(section_header)
            if save_to_file:
                file_content_lines.append(section_header)
            try:
                # Get TIR from compiled library
                tir_modules = self.lib.get_lib().imported_modules
                if len(tir_modules) > 0:
                    # Get TIR from first module (usually the main one)
                    tir_ir = str(tir_modules[0].get_source("ir"))
                    result['tir_ir'] = tir_ir
                    if verbose:
                        # print(tir_ir)
                        if save_to_file:
                            file_content_lines.append(tir_ir)
                    else:
                        # Show first 2000 characters
                        # print(tir_ir[:2000])
                        # if len(tir_ir) > 2000:
                            # print(f"\n... (truncated, total length: {len(tir_ir)} characters)")
                        # Always save full content to file
                        if save_to_file:
                            file_content_lines.append(tir_ir)
                            file_content_lines.append(f"\n(Total length: {len(tir_ir)} characters)")
                else:
                    msg = "⚠️  No TIR modules found"
                    # print(msg)
                    if save_to_file:
                        file_content_lines.append(msg)
                # print()
                if save_to_file:
                    file_content_lines.append("")
            except Exception as e:
                error_msg = f"⚠️  Could not display TIR: {e}"
                # print(error_msg)
                # print()
                if save_to_file:
                    file_content_lines.append(error_msg)
                    file_content_lines.append("")
        
        # 3. Show generated source code
        if show_source and self.lib is not None:
            section_header = "-" * 80 + "\n3. Generated Source Code (CUDA/LLVM)\n" + "-" * 80
            # print(section_header)
            if save_to_file:
                file_content_lines.append(section_header)
            try:
                # For CUDA, source code might be in imported_modules
                # Try to get CUDA code first, then fall back to main lib
                source_code = None
                lib_obj = self.lib.get_lib()
                
                # Check if this is CUDA target
                is_cuda = self.device == "cuda" or "cuda" in str(self.target).lower()
                
                if is_cuda:
                    # For CUDA, try to get source from imported modules
                    try:
                        imported_modules = lib_obj.imported_modules
                        for mod in imported_modules:
                            try:
                                mod_source = mod.get_source()
                                if mod_source and ("__global__" in mod_source or "wmma" in mod_source.lower() or "cuda" in mod_source.lower()):
                                    source_code = mod_source
                                    msg = f"  Found CUDA code in imported module"
                                    # print(msg)
                                    if save_to_file:
                                        file_content_lines.append(msg)
                                    break
                            except:
                                continue
                    except:
                        pass
                
                # Fall back to main lib source
                if source_code is None:
                    try:
                        source_code = lib_obj.get_source()
                    except:
                        pass
                
                if source_code:
                    result['source_code'] = source_code
                    if verbose:
                        # print(source_code)
                        if save_to_file:
                            file_content_lines.append(source_code)
                    else:
                        # Show first 3000 characters
                        # print(source_code[:3000])
                        # if len(source_code) > 3000:
                            # print(f"\n... (truncated, total length: {len(source_code)} characters)")
                        # Always save full content to file
                        if save_to_file:
                            file_content_lines.append(source_code)
                            file_content_lines.append(f"\n(Total length: {len(source_code)} characters)")
                else:
                    msg = "⚠️  Source code not available\n  (This might be a pre-compiled binary or the code is in a different format)"
                    # print(msg)
                    if save_to_file:
                        file_content_lines.append(msg)
                # print()
                if save_to_file:
                    file_content_lines.append("")
            except Exception as e:
                error_msg = f"⚠️  Could not display source code: {e}"
                # print(error_msg)
                # print()
                if save_to_file:
                    file_content_lines.append(error_msg)
                    file_content_lines.append("")
        
        # 4. Check for Tensor Core intrinsics
        if check_tensorcore and self.lib is not None:
            section_header = "-" * 80 + "\n4. Tensor Core Usage Check\n" + "-" * 80
            # print(section_header)
            if save_to_file:
                file_content_lines.append(section_header)
            try:
                source_code = result['source_code']
                
                # If source_code is None, try to get it again (from CUDA modules if available)
                if source_code is None:
                    lib_obj = self.lib.get_lib()
                    is_cuda = self.device == "cuda" or "cuda" in str(self.target).lower()
                    
                    if is_cuda:
                        # Try imported modules for CUDA code
                        try:
                            imported_modules = lib_obj.imported_modules
                            for mod in imported_modules:
                                try:
                                    mod_source = mod.get_source()
                                    if mod_source and ("__global__" in mod_source or "wmma" in mod_source.lower()):
                                        source_code = mod_source
                                        break
                                except:
                                    continue
                        except:
                            pass
                    
                    # Fall back to main lib
                    if source_code is None:
                        try:
                            source_code = lib_obj.get_source()
                        except:
                            pass
                
                if source_code:
                    source_lower = source_code.lower()
                    
                    # Check for Tensor Core intrinsics
                    has_wmma = "wmma" in source_lower
                    has_mma_sync = "mma.sync" in source_lower
                    has_mma = "mma" in source_lower and "sync" in source_lower
                    has_tensorcore_tag = "dense_tensorcore" in source_lower or "tensorcore" in source_lower
                    has_cuda_kernel = "__global__" in source_code or "__device__" in source_code
                    
                    result['has_tensorcore'] = has_wmma or has_mma_sync or has_mma
                    
                    check_info = [
                        f"  Device:                    {self.device}",
                        f"  Target:                    {self.target}",
                        f"  CUDA kernel found:         {has_cuda_kernel}",
                        f"  wmma intrinsics found:     {has_wmma}",
                        f"  mma.sync intrinsics found: {has_mma_sync}",
                        f"  mma intrinsics found:      {has_mma}",
                        f"  tensorcore tag found:      {has_tensorcore_tag}",
                        ""
                    ]
                    # for line in check_info:
                        # print(line)
                    if save_to_file:
                        file_content_lines.extend(check_info)
                    
                    if result['has_tensorcore']:
                        msg = [
                            "  ✓ CONFIRMED: Tensor Core intrinsics are present in generated code!",
                            "  ✓ Tensor Core is being used (no fallback to CUDA Core)"
                        ]
                        # for line in msg:
                            # print(line)
                        if save_to_file:
                            file_content_lines.extend(msg)
                    else:
                        if not has_cuda_kernel:
                            msg = [
                                "  ⚠️  WARNING: No CUDA kernel found in source code",
                                "  ⚠️  This might be LLVM IR (CPU code) instead of CUDA code",
                                "  ⚠️  Check if device='cuda' and target includes 'cuda'"
                            ]
                        else:
                            msg = [
                                "  ⚠️  WARNING: Tensor Core intrinsics NOT found in CUDA code",
                                "  ⚠️  This may indicate fallback to CUDA Core or standard implementation"
                            ]
                        # for line in msg:
                            # print(line)
                        if save_to_file:
                            file_content_lines.extend(msg)
                        
                        # Show relevant code snippets
                        if verbose and source_code:
                            lines = source_code.split('\n')
                            snippet_lines = ["\n  Relevant code snippets:"]
                            count = 0
                            for i, line in enumerate(lines):
                                if any(keyword in line.lower() for keyword in ['dense', 'gemm', 'matmul', 'mma', 'wmma', '__global__', 'cuda']):
                                    snippet_lines.append(f"    Line {i+1}: {line[:100]}")
                                    count += 1
                                    if count >= 10:  # Limit to 10 lines
                                        break
                            # for line in snippet_lines:
                                # print(line)
                            if save_to_file:
                                file_content_lines.extend(snippet_lines)
                else:
                    msg = [
                        "  ⚠️  Could not retrieve source code for Tensor Core check",
                        "  ⚠️  Source code might not be available (pre-compiled binary)"
                    ]
                    #  for line in msg:
                        # print(line)
                    if save_to_file:
                        file_content_lines.extend(msg)
                # print()
                if save_to_file:
                    file_content_lines.append("")
            except Exception as e:
                error_msg = f"⚠️  Could not check Tensor Core usage: {e}"
                # print(error_msg)
                import traceback
                if verbose:
                    traceback_str = traceback.format_exc()
                    # print(traceback_str)
                    if save_to_file:
                        file_content_lines.append(error_msg)
                        file_content_lines.append(traceback_str)
                # print()
                if save_to_file:
                    file_content_lines.append("")
        
        # Save to file if requested
        if save_to_file:
            try:
                file_content = "\n".join(file_content_lines)
                with open(save_to_file, 'w', encoding='utf-8') as f:
                    f.write(file_content)
                result['saved_file'] = save_to_file
                print(f"✓ Complete IR output saved to: {save_to_file}")
                print()
            except Exception as e:
                print(f"⚠️  Could not save to file {save_to_file}: {e}")
                print()
        
        print("=" * 80)
        return result


def detect_cpu_features():
    """
    Detect CPU features and available instruction sets
    Uses both compile-time and runtime detection for reliability
    
    Returns
    -------
    dict: Dictionary with available CPU features
        - 'avx2': bool
        - 'avx512': bool
    """
    features = {
        'avx2': False,
        'avx512': False
    }
    
    try:
        import subprocess
        import platform
        import tempfile
        
        # Method 1: Check /proc/cpuinfo (Linux only)
        if platform.system() == "Linux":
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    cpuinfo = f.read()
                    # Check flags line for instruction sets
                    if 'avx2' in cpuinfo.lower():
                        features['avx2'] = True
                    if 'avx512' in cpuinfo.lower() or 'avx512f' in cpuinfo.lower():
                        features['avx512'] = True
            except:
                pass
        
        # Method 2: Runtime test - compile and actually run a test program
        # This is more reliable as it tests actual execution capability
        try:
            # Test AVX2 with runtime execution
            test_code_avx2 = """
#include <immintrin.h>
#include <stdio.h>
int main() {
    __m256i a = _mm256_setzero_si256();
    __m256i b = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
    __m256i c = _mm256_add_epi32(a, b);
    return 0;
}
"""
            with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
                f.write(test_code_avx2)
                test_file = f.name
            
            try:
                # Compile
                compile_result = subprocess.run(
                    ['gcc', '-mavx2', '-O0', test_file, '-o', test_file + '.out'],
                    capture_output=True,
                    timeout=5
                )
                if compile_result.returncode == 0:
                    # Try to run it
                    run_result = subprocess.run(
                        [test_file + '.out'],
                        capture_output=True,
                        timeout=2
                    )
                    if run_result.returncode == 0:
                        features['avx2'] = True
            finally:
                # Cleanup
                try:
                    os.unlink(test_file)
                    os.unlink(test_file + '.out')
                except:
                    pass
        except Exception:
            pass
        
        try:
            # Test AVX512 with runtime execution
            test_code_avx512 = """
#include <immintrin.h>
#include <stdio.h>
int main() {
    __m512i a = _mm512_setzero_si512();
    __m512i b = _mm512_set_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
    __m512i c = _mm512_add_epi32(a, b);
    return 0;
}
"""
            with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
                f.write(test_code_avx512)
                test_file = f.name
            
            try:
                # Compile
                compile_result = subprocess.run(
                    ['gcc', '-mavx512f', '-O0', test_file, '-o', test_file + '.out'],
                    capture_output=True,
                    timeout=5
                )
                if compile_result.returncode == 0:
                    # Try to run it
                    run_result = subprocess.run(
                        [test_file + '.out'],
                        capture_output=True,
                        timeout=2
                    )
                    if run_result.returncode == 0:
                        features['avx512'] = True
            finally:
                # Cleanup
                try:
                    os.unlink(test_file)
                    os.unlink(test_file + '.out')
                except:
                    pass
        except Exception:
            pass
            
    except Exception:
        # If detection fails, assume no special features
        pass
    
    return features


def get_safe_cpu_target(cpu_instruction_set=None):
    """
    Get a safe CPU target based on requested instruction set and CPU capabilities
    
    Parameters
    ----------
    cpu_instruction_set: str, optional
        Requested instruction set ("avx2", "avx512", or None)
    
    Returns
    -------
    tuple: (target_str, actual_instruction_set)
        target_str: TVM target string
        actual_instruction_set: Actually used instruction set (may differ from requested)
    """
    if cpu_instruction_set is None:
        return "llvm", None
    
    # Detect CPU features
    features = detect_cpu_features()
    
    if cpu_instruction_set == "avx512":
        if features.get('avx512', False):
            return "llvm -mcpu=skylake-avx512", "avx512"
        else:
            # Fall back to AVX2 if available
            if features.get('avx2', False):
                print("Warning: AVX512 not available, falling back to AVX2")
                return "llvm -mcpu=core-avx2", "avx2"
            else:
                print("Warning: AVX512 not available, using default LLVM")
                return "llvm", None
    
    elif cpu_instruction_set == "avx2":
        if features.get('avx2', False):
            return "llvm -mcpu=core-avx2", "avx2"
        else:
            print("Warning: AVX2 not available, using default LLVM")
            return "llvm", None
    
    return "llvm", None
