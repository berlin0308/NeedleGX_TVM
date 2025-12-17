import sys
import os
import time
import numpy as np
import networkx as nx

# Add paths
sys.path.append('./python')
sys.path.append('./NeedleGNN/python')

# Import both Needle and TVM implementations
import needle as ndl
import needle.nn as nn
from apps.models import GCNModel as NeedleGCNModel
from apps.tvm.models import GraphConvNetwork as TVMGCNModel

class GCNModel:
    """
    Unified interface for both Needle and TVM GCN implementations
    
    This class provides a consistent API for both backends, allowing easy
    switching between implementations for comparison and benchmarking.
    """
    
    def __init__(
        self,
        backend="needle",  # "needle" or "tvm"
        in_features=None,
        hidden_features=16,
        num_classes=None,
        num_layers=2,
        dropout=0.5,
        device=None,
        dtype="float32",
        # Needle-specific parameters
        use_sparse=False,
        # TVM-specific parameters
        dataset=None,
        graph=None,
        data=None,
        dataset_info=None,
        opt_level=0,
        cpu_instruction_set=None,
        use_gpu_mma=False,
        activation=None,
        verbose=False,
    ):
        """
        Initialize unified GCN model
        
        Parameters
        ----------
        backend: str
            "needle" for Needle native implementation
            "tvm" for TVM optimized implementation
        dataset: CoraDataset, optional
            If provided and backend is "tvm", will automatically convert to TVM format
        """
        self.backend = backend
        self.device = device
        self.dtype = dtype
        self.num_layers = num_layers
        self.hidden_features = hidden_features
        self.verbose = verbose
        
        if backend == "needle":
            # Initialize Needle GCN model
            if in_features is None or num_classes is None:
                raise ValueError("Needle backend requires in_features and num_classes")
            
            self.model = NeedleGCNModel(
                in_features=in_features,
                hidden_features=hidden_features,
                num_classes=num_classes,
                num_layers=num_layers,
                dropout=dropout,
                device=device,
                dtype=dtype,
            )
            self.model.eval()
            self.use_sparse = use_sparse
            self._needle_adjacency = None
            
        elif backend == "tvm":
            # If dataset is provided, convert it to TVM format
            if dataset is not None:
                graph, data, dataset_info = self._convert_dataset_to_tvm(dataset)
            
            # Initialize TVM GCN model
            if graph is None or data is None or dataset_info is None:
                raise ValueError("TVM backend requires either dataset or (graph, data, dataset_info)")

            # Accept either a Needle device or a simple string like "cpu"/"cuda"
            if isinstance(device, str):
                tvm_device = device
            else:
                tvm_device = "cpu" if device is None or device.name == "cpu" else "cuda"
            
            self.model = TVMGCNModel(
                graph=graph,
                data=data,
                dataset_info=dataset_info,
                num_hidden=hidden_features,
                num_layers=num_layers,
                device=tvm_device,
                opt_level=opt_level,
                cpu_instruction_set=cpu_instruction_set,
                use_gpu_mma=use_gpu_mma,
                dtype=dtype,
                activation=activation,
                verbose=verbose,
            )
            # TVM model needs compilation
            self._compiled = False
            
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'needle' or 'tvm'")
    
    @property
    def target(self):
        """Get TVM target (only for TVM backend)"""
        if self.backend == "tvm" and hasattr(self.model, 'target'):
            return self.model.target
        return None
    
    def _convert_dataset_to_tvm(self, dataset):
        """
        Convert CoraDataset to TVM format (NetworkX graph, DataWrapper, dataset_info)
        
        This method is called automatically when backend="tvm" and dataset is provided.
        Pads num_nodes and infeat to multiples of 64 for Tensor Core optimization.
        """
        import torch
        import numpy as np
        
        # CRITICAL: Pad num_nodes and infeat to multiples of 64 for Tensor Core
        original_num_nodes = dataset.num_nodes
        original_infeat = dataset.features.shape[1]
        
        # Pad num_nodes to multiple of 64
        padded_num_nodes = ((original_num_nodes + 63) // 64) * 64
        pad_nodes = padded_num_nodes - original_num_nodes
        
        # Pad infeat to multiple of 64
        padded_infeat = ((original_infeat + 63) // 64) * 64
        pad_features = padded_infeat - original_infeat
        
        # Pad features: pad rows (num_nodes) and columns (infeat)
        features_np = dataset.features.astype("float32")
        if pad_nodes > 0 or pad_features > 0:
            features_padded = np.pad(
                features_np, 
                ((0, pad_nodes), (0, pad_features)), 
                mode='constant', 
                constant_values=0
            )
            if self.verbose:
                print(f"  [TVM Padding] num_nodes: {original_num_nodes} -> {padded_num_nodes} (pad {pad_nodes} nodes)")
                print(f"  [TVM Padding] infeat: {original_infeat} -> {padded_infeat} (pad {pad_features} features)")
        else:
            features_padded = features_np
        
        # Pad labels
        labels_padded = np.pad(
            dataset.labels,
            (0, pad_nodes),
            mode='constant',
            constant_values=0
        ) if pad_nodes > 0 else dataset.labels
        
        # Pad masks (padded nodes are not used in train/val/test)
        train_mask_padded = np.pad(
            dataset.train_mask,
            (0, pad_nodes),
            mode='constant',
            constant_values=False
        ) if pad_nodes > 0 else dataset.train_mask
        
        val_mask_padded = np.pad(
            dataset.val_mask,
            (0, pad_nodes),
            mode='constant',
            constant_values=False
        ) if pad_nodes > 0 else dataset.val_mask
        
        test_mask_padded = np.pad(
            dataset.test_mask,
            (0, pad_nodes),
            mode='constant',
            constant_values=False
        ) if pad_nodes > 0 else dataset.test_mask
        
        # Convert to NetworkX graph with padded nodes
        graph_nx = nx.Graph()
        graph_nx.add_nodes_from(range(padded_num_nodes))
        
        # Add original edges (undirected, with self-loops for original nodes only)
        edge_src = dataset.edge_src
        edge_dst = dataset.edge_dst
        edges = {(min(int(u), int(v)), max(int(u), int(v))) for u, v in zip(edge_src, edge_dst)}
        graph_nx.add_edges_from(edges)
        # Add self-loops only for original nodes
        graph_nx.add_edges_from((n, n) for n in range(original_num_nodes))
        
        # Create DataWrapper
        class DataWrapper:
            def __init__(self, features, labels, train_mask, val_mask, test_mask, num_classes, dtype="float32"):
                tensor_type = torch.HalfTensor if dtype == "float16" else torch.FloatTensor
                self.features = tensor_type(features)
                self.labels = labels
                self.train_mask = train_mask
                self.val_mask = val_mask
                self.test_mask = test_mask
                self.num_labels = num_classes
        
        # Prepare data with padded dimensions
        data = DataWrapper(
            features=features_padded,
            labels=labels_padded,
            train_mask=train_mask_padded,
            val_mask=val_mask_padded,
            test_mask=test_mask_padded,
            num_classes=dataset.num_classes,
            dtype=self.dtype
        )
        
        # Calculate dataset info with padded dimensions
        num_edges_total = len(graph_nx.edges())
        num_self_loops = sum(1 for edge in graph_nx.edges() if edge[0] == edge[1])
        num_non_self_edges = num_edges_total - num_self_loops
        nnz_in_matrix = num_self_loops + num_non_self_edges * 2
        total_positions = padded_num_nodes * padded_num_nodes
        sparsity = 1.0 - (nnz_in_matrix / total_positions) if total_positions > 0 else 0.0
        
        dataset_info = {
            "num_graphs": 1,
            "num_nodes": padded_num_nodes,  # Use padded num_nodes
            "num_edges": num_edges_total,
            "infeat_dim": padded_infeat,  # Use padded infeat_dim
            "num_classes": dataset.num_classes,
            "sparsity": sparsity,
        }
        
        return graph_nx, data, dataset_info
    
    def tune(self, num_measure_trials=40, log_file=None, verbose=True, 
             runner_timeout=10, runner_number=3, runner_repeat=1, 
             runner_min_repeat_ms=100, enable_cpu_cache_flush=True,
             early_stopping=None, num_measures_per_round=64):
        """
        Tune sparse_dense operations using auto-scheduler (TVM CPU backend only)
        
        IMPORTANT: This method must be called from the directory where you want
        task_input files to be saved, as subprocesses will look for them in the
        current working directory.
        """
        """
        Tune sparse_dense operations using auto-scheduler (TVM CPU backend only)
        
        This method manually creates sparse_dense SearchTask (like spmm_llvm_tune.py)
        and tunes it using auto-scheduler with custom sketch rules.
        Only sparse_dense layer is tuned, other operations use default schedules.
        
        Parameters
        ----------
        num_measure_trials: int
            Number of measurement trials (more = better but slower)
            Default: 40 (good balance between quality and time)
            Recommended: 100-1000 for production use
        log_file: str, optional
            Path to save tuning records. If None, uses default name based on model parameters.
        verbose: bool
            Whether to print tuning progress
        runner_timeout: int
            Timeout in seconds for each measurement (default: 10)
            Increase if measurements timeout (try 30-60 for complex schedules)
        runner_number: int
            Number of runs per measurement for averaging (default: 3)
            More runs = more accurate but slower
        runner_repeat: int
            Number of repeat measurements (default: 1)
            Increase for more stable measurements
        runner_min_repeat_ms: int
            Minimum time per measurement in milliseconds (default: 100)
            Increase if measurements are too short to be accurate
        enable_cpu_cache_flush: bool
            Whether to flush CPU cache between measurements (default: True)
            Disable if it causes issues or slows down too much
        early_stopping: int, optional
            Stop tuning if no improvement after N measurements (default: None = disabled)
            Set to 100-200 to save time if no improvement
        num_measures_per_round: int
            Number of schedules to measure per search round (default: 64)
            More = better exploration but slower per round
        
        Returns
        -------
        str
            Path to the tuning log file
        """
        if self.backend != "tvm":
            raise ValueError("Tuning is only available for TVM backend")
        
        if self.model.device != "cpu":
            raise ValueError("Tuning is only available for CPU device (not CUDA)")
        
        import tvm
        from tvm import te, auto_scheduler, runtime, topi
        from tvm.auto_scheduler import workload_registry
        import networkx as nx
        from scipy import sparse as sp
        import numpy as np  # Required for np.random.seed and array operations
        import sys  # For sys.stdout.flush()
        
        # Import custom sketch rule functions
        try:
            from apps.gnn_tvm_utils.sparse_dense_tune import meet_condition_func, apply_func
        except ImportError:
            # Try alternative path
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            sys.path.insert(0, project_root)
            from apps.gnn_tvm_utils.sparse_dense_tune import meet_condition_func, apply_func
        
        # Get model parameters
        graph = self.model.graph
        num_nodes = self.model.dataset_info['num_nodes']
        infeat_dim = self.model.dataset_info['infeat_dim']
        num_hidden = self.model.num_hidden
        dtype = self.model.dtype
        
        # Get target
        target = tvm.target.Target(self.model.target)
        
        # Generate log file name if not provided
        if log_file is None:
            log_file = f"sparse_dense_gcn_tune_{num_nodes}_{infeat_dim}_{num_hidden}.json"
        
        sys.stdout.flush()
        
        # Convert graph to CSR format to get sparse matrix structure
        if verbose:
            print("\n[Step 1/4] Converting graph to CSR format...")
            sys.stdout.flush()
        try:
            adjacency = nx.to_scipy_sparse_array(graph, format='csr')
        except AttributeError:
            adjacency = nx.to_scipy_sparse_matrix(graph, format='csr')
        
        if verbose:
            print(f"  ✓ CSR conversion complete: {adjacency.shape}, nnz={adjacency.nnz}")
            sys.stdout.flush()
        
        # Get sparse matrix dimensions
        M = num_nodes  # Number of rows (nodes)
        K = num_nodes  # Number of columns (nodes)
        N = M  # Always use M for N to match measurement workload_key
        
        # Get BSR block size (default from TVM)
        BS_R = 16
        BS_C = 1
        
        # Convert CSR to BSR format manually (following spmm_llvm_tune.py exactly)
        # This is needed because we need to provide task_inputs for sparse operations
        if verbose:
            print("\n[Step 2/4] Converting CSR to BSR format...")
            sys.stdout.flush()
        
        np.random.seed(42)
        
        # Convert CSR to BSR format (following spmm_llvm_tune.py logic exactly)
        # First, create a dense matrix from CSR, then convert to BSR like spmm_llvm_tune.py
        num_row_blocks = (M + BS_R - 1) // BS_R
        num_col_blocks = (K + BS_C - 1) // BS_C
        
        if verbose:
            print(f"  Converting to BSR: {num_row_blocks} row blocks, {num_col_blocks} col blocks")
            sys.stdout.flush()
        
        # Method 1: Use scipy's tobsr() if available (more reliable)
        try:
            # Convert CSR to BSR using scipy
            bsr_matrix = adjacency.tobsr(blocksize=(BS_R, BS_C))
            
            # Extract BSR data
            W_data_np = bsr_matrix.data.astype(dtype)  # Shape: (num_blocks, BS_R, BS_C)
            W_indices_np = bsr_matrix.indices.astype("int32")  # Column indices
            W_indptr_np = bsr_matrix.indptr.astype("int32")  # Row pointers
            
            if verbose:
                print(f"  ✓ Used scipy.tobsr() for conversion")
                print(f"  ✓ BSR conversion complete: {W_data_np.shape[0]} non-zero blocks")
                sys.stdout.flush()
        except:
            # Fallback: Manual conversion
            if verbose:
                print(f"  Using manual BSR conversion")
                sys.stdout.flush()
            
            # Get CSR data
            if hasattr(adjacency, 'data'):
                csr_data = adjacency.data.astype(dtype)
                csr_indices = adjacency.indices.astype("int32")
                csr_indptr = adjacency.indptr.astype("int32")
            else:
                raise ValueError("Cannot extract CSR data from adjacency matrix")
            
            # Create dense matrix from CSR (like spmm_llvm_tune.py creates W_dense)
            W_dense = np.zeros((M, K), dtype=dtype)
            for i in range(M):
                for j_idx in range(csr_indptr[i], csr_indptr[i + 1]):
                    j = csr_indices[j_idx]
                    W_dense[i, j] = csr_data[j_idx]
            
            # Collect non-zero positions
            non_zero_positions = set()
            for i in range(M):
                for j_idx in range(csr_indptr[i], csr_indptr[i + 1]):
                    j = csr_indices[j_idx]
                    non_zero_positions.add((i, j))
            
            # Convert to BSR format (EXACTLY like spmm_llvm_tune.py)
            non_zero_blocks = []
            block_to_index = {}
            block_list = []
            
            for i, j in non_zero_positions:
                block_row = i // BS_R
                block_col = j // BS_C
                block_key = (block_row, block_col)
                if block_key not in block_to_index:
                    block_to_index[block_key] = len(non_zero_blocks)
                    block_list.append(block_key)
                    # Extract block data (EXACTLY like spmm_llvm_tune.py)
                    row_start = block_row * BS_R
                    col_start = block_col * BS_C
                    row_end = min(row_start + BS_R, M)
                    col_end = min(col_start + BS_C, K)
                    block_data = W_dense[row_start:row_end, col_start:col_end].copy()
                    # Pad if necessary (EXACTLY like spmm_llvm_tune.py)
                    if block_data.shape[0] < BS_R or block_data.shape[1] < BS_C:
                        padded_block = np.zeros((BS_R, BS_C), dtype=dtype)
                        padded_block[:block_data.shape[0], :block_data.shape[1]] = block_data
                        block_data = padded_block
                    non_zero_blocks.append(block_data.reshape(BS_R, BS_C))
            
            # Sort blocks by row, then by column for BSR format
            block_list_sorted = sorted(block_list)
            
            # Reorder W_data_np to match sorted block order
            W_data_ordered = []
            for block_key in block_list_sorted:
                block_idx = block_to_index[block_key]
                W_data_ordered.append(non_zero_blocks[block_idx])
            
            # Create BSR data array (ordered by row, then column)
            W_data_np = np.array(W_data_ordered, dtype=dtype)  # Shape: (num_blocks, BS_R, BS_C)
            
            # Create indices array (column index for each block, sorted by row)
            W_indices_np = np.array([block_col for block_row, block_col in block_list_sorted], dtype="int32")
            
            # Create indptr array (row pointer) - EXACTLY like spmm_llvm_tune.py
            W_indptr_np = np.zeros(num_row_blocks + 1, dtype="int32")
            current_block_idx = 0
            for block_row in range(num_row_blocks):
                W_indptr_np[block_row] = current_block_idx
                # Count blocks in this row
                while current_block_idx < len(block_list_sorted) and block_list_sorted[current_block_idx][0] == block_row:
                    current_block_idx += 1
            W_indptr_np[num_row_blocks] = current_block_idx
            
            if verbose:
                print(f"  ✓ BSR conversion complete: {len(block_list_sorted)} non-zero blocks")
                sys.stdout.flush()
        
        prefix = "sparse_dense_bsr_%d_%d_%d_%d_%d_%d_" % (
            M,
            M,
            BS_R,
            BS_C,
            W_indices_np.shape[0],
            W_indptr_np.shape[0],
        )
        
        def sparse_dense_workload(M, N, K, w_data_shape, w_indices_shape, w_indptr_shape, dtype):
            
            X = te.placeholder(shape=(M, K), dtype=dtype, name="X")  # Dense input matrix
            W_data = te.placeholder(shape=w_data_shape, dtype=dtype, name=prefix + "W_data")  # Sparse data blocks
            W_indices = te.placeholder(shape=w_indices_shape, dtype="int32", name=prefix + "W_indices")  # Column indices
            W_indptr = te.placeholder(shape=w_indptr_shape, dtype="int32", name=prefix + "W_indptr")  # Row pointers
            
            # Define the computation: Y = X @ W^T (sparse)
            # This creates a compute graph automatically
            out = topi.nn.sparse_dense(X, W_data, W_indices, W_indptr)
            
            return [X, W_data, W_indices, W_indptr, out]
        
        workload_registry.register_workload(sparse_dense_workload, override=True)
        
        if len(W_data_np) == 0:
            raise ValueError("No non-zero blocks found in sparse matrix! Cannot tune empty matrix.")
        
        # Verify BSR structure is valid
        if W_indptr_np[num_row_blocks] == 0:
            raise ValueError("Invalid BSR indptr: no blocks found!")
        
        # Verify shapes match
        assert W_data_np.shape[0] == W_indices_np.shape[0], \
            f"BSR data and indices mismatch: {W_data_np.shape[0]} != {W_indices_np.shape[0]}"
        assert W_indptr_np.shape[0] == num_row_blocks + 1, \
            f"BSR indptr shape mismatch: {W_indptr_np.shape[0]} != {num_row_blocks + 1}"
        
        # Create task_inputs dict (exactly like spmm_llvm_tune.py)
        # IMPORTANT: Ensure we use the same prefix format as spmm_llvm_tune.py
        task_inputs_dict = {
            prefix + "W_data": runtime.ndarray.array(W_data_np),
            prefix + "W_indices": runtime.ndarray.array(W_indices_np),
            prefix + "W_indptr": runtime.ndarray.array(W_indptr_np),
        }
        
        if verbose:
            print(f"  Task inputs keys: {list(task_inputs_dict.keys())}")
            print(f"  Prefix: {prefix}")
            sys.stdout.flush()
        
        # Pre-register task_inputs to ensure they're available in subprocess
        # This is critical because subprocesses may not have access to the global table
        from tvm.auto_scheduler.search_task import register_task_input_buffer, get_task_input_buffer
        workload_key_preview = None  # Will be set after SearchTask creation
        for input_name, input_data in task_inputs_dict.items():
            # Pre-register with a temporary workload_key, will update after SearchTask creation
            # But first, let's just ensure the data is ready
            if verbose:
                print(f"    Preparing {input_name} (shape: {input_data.shape})")
        sys.stdout.flush()
        
        # Create SearchTask (SIMPLIFIED: Use N=M to match measurement)
        # Use task_inputs_overwrite=True to allow re-registration when tune() is called multiple times
        task = auto_scheduler.SearchTask(
            func=sparse_dense_workload,
            args=(M, M, K, W_data_np.shape, W_indices_np.shape, W_indptr_np.shape, dtype),  # Use M for N to match measurement
            target=target,
            task_inputs=task_inputs_dict,
            task_inputs_overwrite=True,  # Allow overwrite when tune() is called multiple times
            task_inputs_save_to_file=True,  # Save to file so subprocess can load
        )
        
        workload_key = task.workload_key
        
        if verbose:
            print(f"  Verifying task_inputs registration...")
            print(f"    Workload key: {workload_key[:80]}...")
            
            print(f"\n{'='*80}")
            print("SEARCH SPACE INFORMATION")
            print(f"{'='*80}")
            
            # 1. Print compute DAG
            print("\n1. COMPUTATIONAL DAG (Computation Graph):")
            print("-" * 80)
            print("This shows WHAT operations need to be performed:")
            print(task.compute_dag)
            print("-" * 80)
            
            # 2. Print workload key
            print(f"\n2. WORKLOAD KEY:")
            print("-" * 80)
            print(f"   {task.workload_key}")
            print("-" * 80)
            
            # 3. Print initial state info
            print(f"\n3. INITIAL STATE (Default Schedule):")
            print("-" * 80)
            try:
                init_state = task.compute_dag.init_state
                if hasattr(task.compute_dag, 'ops'):
                    print(f"   Number of operations: {len(task.compute_dag.ops)}")
                if hasattr(init_state, 'transform_steps'):
                    print(f"   Number of transform steps: {len(init_state.transform_steps)}")
                else:
                    print(f"   Number of transform steps: 0 (naive schedule)")
            except Exception as e:
                print(f"   Could not extract initial state info: {e}")
            print("-" * 80)
            
            # 4. Print search space definition
            print(f"\n4. SEARCH SPACE DEFINITION:")
            print("-" * 80)
            print("   Custom Sketch Rule: 'SparseDense'")
            print("   - Provides initial loop structure:")
            print("     * Loop splits: i -> [i0, i1, i2], nb_j -> [j0, j1]")
            print("     * Loop reorder: [i0, j0, i1, j1, row_offset, i2, j, c]")
            print("     * Compute-at: sparse_dense_block at consumer")
            print("   - Auto-Scheduler will then tune:")
            print("     * Tile sizes (how big each split should be)")
            print("     * Parallelization (which loops to parallelize)")
            print("     * Vectorization (which loops to vectorize)")
            print("     * Loop unrolling (how much to unroll)")
            print("     * Compute placement (where to compute intermediate results)")
            print("-" * 80)
            
            # 5. Print task dimensions
            print(f"\n5. TASK DIMENSIONS:")
            print("-" * 80)
            print(f"   M (rows): {M}")
            print(f"   N (cols): {N}")
            print(f"   K (features): {K}")
            print(f"   BSR block size: {BS_R}x{BS_C}")
            print(f"   Number of row blocks: {num_row_blocks}")
            print(f"   Number of non-zero blocks: {len(W_data_np)}")
            print(f"   Sparsity: {1.0 - len(W_data_np) / (num_row_blocks * (N // BS_C)):.2%}")
            print("-" * 80)
            
            sys.stdout.flush()
        
        # Create search policy with custom sketch rule
        if verbose:
            print("\n  Creating search policy with custom sketch rule...")
            sys.stdout.flush()
        
        search_policy = auto_scheduler.SketchPolicy(
            task,
            program_cost_model=auto_scheduler.XGBModel(),
            init_search_callbacks=[
                auto_scheduler.PreloadCustomSketchRule(meet_condition_func, apply_func, "SparseDense")
            ],
        )
        
        if verbose:
            print(f"\n6. SEARCH STRATEGY:")
            print("-" * 80)
            print("   Policy: SketchPolicy")
            print("   Cost Model: XGBModel (XGBoost-based predictor)")
            print("   Custom Sketch Rule: 'SparseDense' (preloaded)")
            print("\n   Search Process:")
            print("     1. Generate Sketches: Custom rule creates initial loop structure")
            print("     2. Sample Initial Population: Random tile sizes, parallelization, etc.")
            print("     3. Evolutionary Search (GA): Mutate and optimize schedules")
            print("     4. Measure: Actually run and measure performance")
            print("     5. Learn: Cost model learns which schedules are better")
            print("-" * 80)
            sys.stdout.flush()
        
        # Create tuning options
        # Use LocalRunner for CPU (more reliable than RPC)
        from tvm.auto_scheduler import LocalRunner, LocalBuilder
        
        runner = LocalRunner(
            timeout=runner_timeout,  # Timeout per measurement
            number=runner_number,    # Number of runs per measurement
            repeat=runner_repeat,    # Number of repeat measurements
            min_repeat_ms=runner_min_repeat_ms,  # Minimum time per measurement
            enable_cpu_cache_flush=enable_cpu_cache_flush,  # Flush CPU cache
        )
        
        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=num_measure_trials,
            early_stopping=early_stopping,
            num_measures_per_round=num_measures_per_round,
            runner=runner,  # Use LocalRunner for better reliability
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
            verbose=2 if verbose else 1,  # Use verbose=1 at minimum to show progress
        )
        
        sys.stdout.flush()
                
        workload_key = task.workload_key
        cwd = os.getcwd()
        
        # Re-register and save to file
        saved_files = []
        for input_name, input_data in task_inputs_dict.items():
            # Register in global table
            register_task_input_buffer(
                workload_key, input_name, input_data,
                overwrite=True, save_to_file=True
            )
            
            try:
                np_data = input_data.numpy()
                # Create filename matching TVM's format: {name}.{shape}_{dtype}.npy
                filename = input_name + "."
                for i in np_data.shape:
                    filename += "%d_" % (i)
                filename += "%s" % (np_data.dtype)
                filename += ".npy"
                
                # Save to current directory (absolute path)
                filepath = os.path.join(cwd, filename)
                np_data.tofile(filepath, " ")
                saved_files.append(filepath)
                
                if verbose:
                    print(f"    ✓ Saved {input_name} to {filename} (shape: {np_data.shape}, size: {np_data.nbytes} bytes)")
            except Exception as e:
                print(f"    ✗ Failed to save {input_name}: {e}")
                import traceback
                traceback.print_exc()
        
        if verbose:
            print(f"  ✓ Task_inputs re-registered and saved to {len(saved_files)} files")
            print(f"    Files location: {cwd}")
            print(f"    ⚠️  If tuning fails with 'task_inputs not found', ensure subprocess")
            print(f"       runs from this directory or files are accessible")
            sys.stdout.flush()
        
        task.tune(tune_option, search_policy)
        
        # Store log file path for use in compile()
        self._tune_log_file = log_file
        
        # Check if we have valid records
        try:
            from tvm.auto_scheduler.measure_record import load_records
            records = load_records(log_file)
            valid_records = []
            for inp, res in records:
                if res.error_no == 0:
                    # Check if costs are valid (not 1e+10 which indicates failure)
                    costs = [c.value for c in res.costs if hasattr(c, 'value')]
                    if costs and min(costs) < 1e9:
                        valid_records.append((inp, res))
            
            if len(valid_records) == 0:
                pass
            else:
                best_cost = min([min([c.value for c in res.costs if hasattr(c, 'value')]) 
                                 for _, res in valid_records])
                print(f"\n{'='*80}")
                print("✓ Tuning completed!")
                print(f"Results saved to: {log_file}")
                print(f"  - Found {len(valid_records)} valid record(s) out of {len(records)} total")
                print(f"  - Best cost: {best_cost:.6f}")
                print(f"\n📝 Note:")
                print(f"  - Only sparse_dense operation was tuned")
                print(f"  - Other operations (dense, etc.) will use default schedules")
                print(f"  - Tuning results will be applied during compilation IF workload keys match")
                print(f"  - If workload keys don't match, default schedules will be used")
                print(f"{'='*80}\n")
        except Exception as e:
            pass
        
        sys.stdout.flush()
        
        return log_file
    
    def compile(self, use_tuning=True):
        """
        Compile the model (only needed for TVM backend)
        
        Parameters
        ----------
        use_tuning: bool
            If True and tuning log file exists, apply tuning results during compilation.
            Default: True
        """
        if self.backend == "tvm" and not self._compiled:
            # Apply tuning results if available
            if use_tuning and hasattr(self, '_tune_log_file') and os.path.exists(self._tune_log_file):
                import tvm
                from tvm import auto_scheduler
                
                # Check if log file has valid records
                try:
                    from tvm.auto_scheduler.measure_record import load_records
                    records = load_records(self._tune_log_file)
                    valid_records = [r for r in records if r[1].error_no == 0 and len([c for c in r[1].costs if hasattr(c, 'value') and c.value < 1e9]) > 0]
                    
                    if len(valid_records) == 0:
                        # print(f"   WARNING: No valid tuning records found in {self._tune_log_file}")
                        # print(f"   All measurements failed (cost = 1e+10).")
                        # print(f"   Compiling WITHOUT auto_scheduler (using default schedules).")
                        # Compile without auto_scheduler since we have no valid tuning results
                        self.model.compile()
                        return
                    
                    if self.verbose:
                        print(f"\nApplying tuning results from: {self._tune_log_file}")
                        print(f"   Found {len(valid_records)} valid record(s) out of {len(records)} total")
                
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️  Could not check tuning log: {e}")
                        print(f"   Attempting to apply anyway...")
                
                # Use ApplyHistoryBest to apply tuning results
                try:
                    with auto_scheduler.ApplyHistoryBest(self._tune_log_file, include_compatible=True):
                        with tvm.transform.PassContext(
                            opt_level=self.model.opt_level,
                            config={"relay.backend.use_auto_scheduler": True}
                        ):
                            self.model.compile()
                    if self.verbose:
                        print(f"✓ Applied tuning context (auto_scheduler enabled)")
                        print(f"  Note: Even without valid tuning records, auto_scheduler may use default optimizations")
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️  Could not apply tuning results: {e}")
                        print(f"   Compiling without auto_scheduler...")
                    self.model.compile()
            else:
                # Compile without tuning (no auto_scheduler, use default schedules)
                self.model.compile()
            
            self._compiled = True
        elif self.backend == "needle":
            print("Note: Needle backend does not require compilation")
    
    def forward(self, features, adjacency=None):
        """
        Forward pass with unified interface
        
        Parameters
        ----------
        features: ndl.Tensor or np.ndarray
            Node features
        adjacency: ndl.Tensor, tuple, or None
            Adjacency matrix (for Needle) or None (for TVM, uses internal graph)
        
        Returns
        -------
        output: np.ndarray
            Model output (always returns numpy array for consistency)
        """
        if self.backend == "needle":
            # Needle backend requires adjacency
            if adjacency is None:
                raise ValueError("Needle backend requires adjacency parameter")
            
            # Convert features to needle Tensor if needed
            if isinstance(features, np.ndarray):
                features = ndl.Tensor(
                    features,
                    device=self.device,
                    dtype=self.dtype,
                    requires_grad=False
                )
            
            # Run forward pass
            output = self.model(features, adjacency)
            
            # Convert to numpy
            if isinstance(output, ndl.Tensor):
                return output.numpy()
            return output
            
        elif self.backend == "tvm":
            # TVM backend uses internal graph, features can be updated
            # Ensure model is compiled
            if not self._compiled:
                self.compile()
            
            # Convert features to numpy if needed
            if isinstance(features, ndl.Tensor):
                features = features.numpy()
            
            # CRITICAL: Pad features to match model's expected shape (64 multiples)
            # Get expected shape from model's internal params
            expected_shape = self.model.params["infeats"].shape
            actual_shape = features.shape
            
            if actual_shape != expected_shape:
                # Need to pad features to match expected shape
                pad_rows = expected_shape[0] - actual_shape[0]
                pad_cols = expected_shape[1] - actual_shape[1]
                
                if pad_rows > 0 or pad_cols > 0:
                    features = np.pad(
                        features,
                        ((0, pad_rows), (0, pad_cols)),
                        mode='constant',
                        constant_values=0
                    )
                    if self.verbose:
                        print(f"  [Forward Padding] features: {actual_shape} -> {features.shape}")
            
            # TVM forward pass (uses internal graph)
            output = self.model.forward(features)
            return output
    
    def __call__(self, features, adjacency=None):
        """Make model callable"""
        return self.forward(features, adjacency)
    
    def load_weights(self, weight_path):
        """Load pretrained weights (Needle backend only)"""
        if self.backend == "needle":
            # from NeedleGNN.apps.simple_ml import load_npz_weights
            # print("Loading weights for Needle backend...")
            from apps.simple_ml import load_npz_weights
            load_npz_weights(self.model, weight_path, device=self.device, dtype=self.dtype)
        else:
            print("Warning: Weight loading for TVM backend not implemented in this example")


def build_needle_adjacency(edge_src, edge_dst, inv_sqrt_deg, sparse=False, device=None, dtype="float32"):
    """
    Build normalized adjacency matrix for Needle backend
    
    This is a helper function to convert graph data to Needle adjacency format
    """
    from needle.backend_ndarray import SparseNDArray
    
    src = edge_src.numpy().astype("int64") if isinstance(edge_src, ndl.Tensor) else edge_src
    dst = edge_dst.numpy().astype("int64") if isinstance(edge_dst, ndl.Tensor) else edge_dst
    inv = inv_sqrt_deg.numpy().reshape(-1) if isinstance(inv_sqrt_deg, ndl.Tensor) else inv_sqrt_deg.reshape(-1)
    
    values = inv[src] * inv[dst]
    num_nodes = inv.shape[0]
    shape = (num_nodes, num_nodes)
    
    if sparse:
        adj = SparseNDArray(dst, src, values, shape, device=device)
    else:
        dense = np.zeros(shape, dtype=np.float32)
        np.add.at(dense, (dst, src), values)
        adj = dense
    
    return ndl.Tensor(adj, device=device, dtype=dtype, requires_grad=False)


def example_needle_backend():
    """Example: Using Needle native GCN"""
    print("=" * 60)
    print("Example 1: Needle Native Backend")
    print("=" * 60)
    
    # Load Cora dataset (Needle format)
    try:
        # from NeedleGNN.apps.simple_ml import load_cora_graph
        from apps.simple_ml import load_cora_graph
        cora_dir = os.path.join("data", "cora")
        if os.path.exists(os.path.join(cora_dir, "cora.content")):
            weight_path = os.path.join(cora_dir, "gcn_pytorch.npz")
            # Default hidden dim matches the trained PyTorch export if present.
            hidden_dim = 16
            if os.path.exists(weight_path):
                with np.load(weight_path) as weights:
                    w0 = weights.get("layers.0.weight")
                    if w0 is not None and w0.shape[1] > 0:
                        hidden_dim = w0.shape[1]

            dataset, graph = load_cora_graph(cora_dir, device=ndl.cpu(), dtype="float32")
            
            # Create unified model with Needle backend
            model = GCNModel(
                backend="needle",
                in_features=graph["features"].shape[1],
                hidden_features=hidden_dim,
                num_classes=dataset.num_classes,
                num_layers=2,
                dropout=0.5,
                device=ndl.cpu(),
                dtype="float32",
            )
            
            # Build adjacency
            adjacency = build_needle_adjacency(
                graph["edge_index"][0],
                graph["edge_index"][1],
                graph["inv_sqrt_deg"],
                sparse=False,
                device=ndl.cpu(),
                dtype="float32",
            )

            # Load pretrained weights exported from PyTorch for fair comparison
            weights_loaded = False
            if os.path.exists(weight_path):
                print(f"Loading pretrained weights from {weight_path}")
                model.load_weights(weight_path)
                weights_loaded = True
            else:
                print("Pretrained weights not found, using random initialization.")
            
            # Run inference
            print("Running inference...")
            start_time = time.time()
            output = model(graph["features"], adjacency)
            elapsed_time = time.time() - start_time
            
            print(f"Inference time: {elapsed_time:.4f} seconds")
            print(f"Output shape: {output.shape}")
            print(f"Output dtype: {output.dtype}")
            
            # Calculate accuracy
            predictions = output.argmax(axis=1)
            labels = dataset.labels
            test_mask = dataset.test_mask
            test_accuracy = float((predictions[test_mask] == labels[test_mask]).mean())
            print(f"Test accuracy: {test_accuracy:.4f}")

            # Compare with PyTorch implementation using the same exported weights
            if weights_loaded:
                try:
                    import torch
                    from train_cora_gcn import TorchGCN

                    def load_torch_weights_from_npz(model, path, device):
                        with np.load(path) as arrays:
                            missing = [name for name in model.state_dict() if name not in arrays]
                            if missing:
                                raise KeyError(f"Weights missing for parameters: {missing}")
                            with torch.no_grad():
                                for name, param in model.named_parameters():
                                    param.copy_(torch.from_numpy(arrays[name]).to(device))

                    torch_device = torch.device("cpu")
                    torch_model = TorchGCN(
                        in_dim=graph["features"].shape[1],
                        hidden_dim=hidden_dim,
                        out_dim=dataset.num_classes,
                        num_layers=2,
                        dropout=0.5,
                    ).to(torch_device)
                    load_torch_weights_from_npz(torch_model, weight_path, torch_device)
                    torch_model.eval()

                    torch_features = torch.tensor(dataset.features, dtype=torch.float32, device=torch_device)
                    torch_edge_src = torch.tensor(dataset.edge_src, dtype=torch.long, device=torch_device)
                    torch_edge_dst = torch.tensor(dataset.edge_dst, dtype=torch.long, device=torch_device)
                    torch_inv = torch.tensor(dataset.inv_sqrt_deg, dtype=torch.float32, device=torch_device)

                    with torch.no_grad():
                        torch_logits = torch_model(torch_features, torch_edge_src, torch_edge_dst, torch_inv)
                    torch_preds = torch_logits.argmax(dim=1).cpu().numpy()
                    torch_test_acc = float((torch_preds[test_mask] == labels[test_mask]).mean())

                    print("\nPyTorch reference (exported weights)")
                    print(f"Test accuracy: {torch_test_acc:.4f}")

                    torch_logits_np = torch_logits.cpu().numpy()
                    if torch_logits_np.shape == output.shape:
                        diff = np.abs(output - torch_logits_np)
                        print(f"Max logit diff (Needle vs PyTorch): {diff.max():.6e}")
                        print(f"Mean logit diff (Needle vs PyTorch): {diff.mean():.6e}")
                    else:
                        print(f"Shape mismatch between Needle ({output.shape}) and PyTorch ({torch_logits_np.shape}) logits.")
                except Exception as compare_err:
                    print(f"Skipping PyTorch comparison due to: {compare_err}")
            
        else:
            print("Cora dataset not found, skipping Needle example")
    except Exception as e:
        print(f"Error in Needle example: {e}")
        import traceback
        traceback.print_exc()


def example_tvm_backend():
    """Example: Using TVM optimized GCN"""
    print("\n" + "=" * 60)
    print("Example 2: TVM Optimized Backend")
    print("=" * 60)
    
    try:
        # Load dataset (TVM format)
        from dataset.graph_dataset import random_graph_dataset
        
        graph, data, dataset_info = random_graph_dataset(
            num_nodes=128,
            infeat_dim=32,
            num_classes=16,
            sparsity=0.9,
            dtype="float32"
        )
        
        # Create unified model with TVM backend
        model = GCNModel(
            backend="tvm",
            hidden_features=32,
            num_layers=2,
            graph=graph,
            data=data,
            dataset_info=dataset_info,
            device="cpu",
            dtype="float32",
            opt_level=0,
        )
        
        # Compile model
        print("Compiling TVM model...")
        model.compile()
        print("Compilation complete")
        
        # Run inference
        print("Running inference...")
        start_time = time.time()
        output = model.forward(data.features.numpy() if hasattr(data.features, 'numpy') else data.features)
        elapsed_time = time.time() - start_time
        
        print(f"Inference time: {elapsed_time:.4f} seconds")
        print(f"Output shape: {output.shape}")
        print(f"Output dtype: {output.dtype}")
        
    except Exception as e:
        print(f"Error in TVM example: {e}")
        import traceback
        traceback.print_exc()


def example_tvm_backend_with_tuning():
    """Example: Using TVM optimized GCN with auto-scheduler tuning"""
    print("\n" + "=" * 60)
    print("Example 2b: TVM Optimized Backend with Auto-Scheduler Tuning")
    print("=" * 60)
    
    try:
        # Load dataset (TVM format)
        from dataset.graph_dataset import random_graph_dataset
        
        graph, data, dataset_info = random_graph_dataset(
            num_nodes=128,
            infeat_dim=32,
            num_classes=16,
            sparsity=0.9,
            dtype="float32"
        )
        
        # Create unified model with TVM backend
        model = GCNModel(
            backend="tvm",
            hidden_features=32,
            num_layers=2,
            graph=graph,
            data=data,
            dataset_info=dataset_info,
            device="cpu",
            dtype="float32",
            opt_level=0,
            verbose=True,
        )
        
        # Tune sparse_dense operations (only for CPU)
        print("\n" + "-" * 60)
        print("Step 1: Tuning sparse_dense operations...")
        print("-" * 60)
        tune_log_file = model.tune(
            num_measure_trials=20,  # Use fewer trials for faster demo (use 40+ for production)
            verbose=True
        )
        print(f"Tuning log saved to: {tune_log_file}")
        
        # Compile model with tuning results
        print("\n" + "-" * 60)
        print("Step 2: Compiling TVM model with tuning results...")
        print("-" * 60)
        model.compile(use_tuning=True)
        print("Compilation complete")
        
        # Run inference
        print("\n" + "-" * 60)
        print("Step 3: Running inference...")
        print("-" * 60)
        start_time = time.time()
        output = model.forward(data.features.numpy() if hasattr(data.features, 'numpy') else data.features)
        elapsed_time = time.time() - start_time
        
        print(f"Inference time: {elapsed_time:.4f} seconds")
        print(f"Output shape: {output.shape}")
        print(f"Output dtype: {output.dtype}")
        
        print("\n" + "=" * 60)
        print("Note: Tuning improves performance by optimizing sparse_dense operations")
        print("      The tuning results are saved and can be reused for future compilations")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error in TVM tuning example: {e}")
        import traceback
        traceback.print_exc()


def example_comparison():
    """Example: Comparing both backends on the same data"""
    print("\n" + "=" * 60)
    print("Example 3: Backend Comparison")
    print("=" * 60)
    
    try:
        # Create synthetic data
        num_nodes = 64
        infeat_dim = 16
        num_classes = 8
        hidden_dim = 16
        
        # Generate random graph
        graph = nx.erdos_renyi_graph(num_nodes, 0.3, seed=42)
        features = np.random.randn(num_nodes, infeat_dim).astype(np.float32)
        
        # For Needle: build adjacency
        edge_src = np.array([e[0] for e in graph.edges()])
        edge_dst = np.array([e[1] for e in graph.edges()])
        
        # Add self-loops
        all_nodes = np.arange(num_nodes)
        edge_src = np.concatenate([edge_src, all_nodes])
        edge_dst = np.concatenate([edge_dst, all_nodes])
        
        # Compute degrees
        degrees = np.zeros(num_nodes)
        for i in range(num_nodes):
            degrees[i] = np.sum(edge_src == i) + np.sum(edge_dst == i)
        degrees[degrees == 0] = 1.0
        inv_sqrt_deg = np.power(degrees, -0.5)
        
        # Build Needle adjacency
        needle_adj = build_needle_adjacency(
            edge_src, edge_dst, inv_sqrt_deg,
            sparse=False, device=ndl.cpu(), dtype="float32"
        )
        
        # Create Needle model
        print("Creating Needle model...")
        needle_model = GCNModel(
            backend="needle",
            in_features=infeat_dim,
            hidden_features=hidden_dim,
            num_classes=num_classes,
            num_layers=2,
            device=ndl.cpu(),
            dtype="float32",
        )
        
        # Run Needle inference
        needle_features = ndl.Tensor(features, device=ndl.cpu(), dtype="float32")
        start_time = time.time()
        needle_output = needle_model(needle_features, needle_adj)
        needle_time = time.time() - start_time
        
        print(f"Needle inference time: {needle_time:.4f} seconds")
        print(f"Needle output shape: {needle_output.shape}")
        
        # For TVM: prepare data
        try:
            from dataset.graph_dataset import DataWrapper  # Some versions only define this inside loaders
        except ImportError:
            class DataWrapper:
                def __init__(self, features):
                    self.features = features

        data_tvm = DataWrapper(features=features)
        dataset_info_tvm = {
            'num_nodes': num_nodes,
            'infeat_dim': infeat_dim,
            'num_classes': num_classes,
            'num_edges': graph.number_of_edges(),
        }
        
        # Create TVM model
        print("\nCreating TVM model...")
        tvm_model = GCNModel(
            backend="tvm",
            hidden_features=hidden_dim,
            num_layers=2,
            graph=graph,
            data=data_tvm,
            dataset_info=dataset_info_tvm,
            device="cpu",
            dtype="float32",
        )
        
        # Compile and run TVM inference
        tvm_model.compile()
        start_time = time.time()
        tvm_output = tvm_model.forward(features)
        tvm_time = time.time() - start_time
        
        print(f"TVM inference time: {tvm_time:.4f} seconds")
        print(f"TVM output shape: {tvm_output.shape}")
        
        # Compare outputs
        print("\n" + "-" * 60)
        print("Output Comparison")
        print("-" * 60)
        
        # Note: Outputs may differ due to different implementations
        # We compare shapes and show statistics
        print(f"Shape match: {needle_output.shape == tvm_output.shape}")
        if needle_output.shape == tvm_output.shape:
            diff = np.abs(needle_output - tvm_output)
            print(f"Max difference: {np.max(diff):.6e}")
            print(f"Mean difference: {np.mean(diff):.6e}")
            print(f"Relative difference: {np.mean(diff) / (np.abs(needle_output).mean() + 1e-8):.6e}")
        
        print(f"\nSpeed comparison:")
        print(f"  Needle: {needle_time:.4f}s")
        print(f"  TVM:    {tvm_time:.4f}s")
        if needle_time > 0:
            print(f"  Speedup: {needle_time / tvm_time:.2f}x")
        
    except Exception as e:
        print(f"Error in comparison example: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Unified GCN Model Examples")
    print("=" * 60)
    print()
    
    # Run examples
    example_needle_backend()
    example_tvm_backend()
    
    # Uncomment to run tuning example (takes longer)
    # example_tvm_backend_with_tuning()
    
    example_comparison()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
    print("\nNote: To use auto-scheduler tuning for TVM CPU backend, call:")
    print("      model.tune(num_measure_trials=40)")
    print("      model.compile(use_tuning=True)")
    print("=" * 60)
