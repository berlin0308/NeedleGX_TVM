from tvm import relay
from tvm import te
import numpy as np

# Global context to pass activation information from gconv.py to schedule
# This allows us to pass activation (e.g., ReLU) information to schedule_dtc_spmm_tensorcore
_tensorcore_activation_context = {
    'activation': None
}

def set_tensorcore_activation_context(activation=None):
    """
    Set global context for Tensor Core activation operations
    
    This allows gconv.py to explicitly pass activation information
    to the schedule function. All layers will use the same activation.
    
    Parameters
    ----------
    activation: function, optional
        Activation function (e.g., relay.nn.relu)
    """
    global _tensorcore_activation_context
    _tensorcore_activation_context = {
        'activation': activation
    }

def get_tensorcore_activation_context():
    """Get current Tensor Core activation context"""
    return _tensorcore_activation_context.copy()

def clear_tensorcore_activation_context():
    """Clear Tensor Core activation context"""
    global _tensorcore_activation_context
    _tensorcore_activation_context = {
        'activation': None
    }

######################################################################
# Define Graph Convolution Layer in Relay
# ---------------------------------------
# To run GCN on TVM, we first need to implement Graph Convolution Layer.
# You may refer to https://github.com/dmlc/dgl/blob/master/python/dgl/nn/mxnet/conv/graphconv.py for a GraphConv Layer implemented in DGL with MXNet Backend
#
# The layer is defined with below operations, note that we apply two transposes to keep adjacency matrix on right hand side of sparse_dense operator,
# this method is temporary and will be updated in next few weeks when we have sparse matrix transpose and support for left sparse operator.
#
#  .. math::
#
#            \mbox{GraphConv}(A, H, W)   = A * H * W
#                                        = ((H * W)^t * A^t)^t
#                                        = ((W^t * H^t) * A^t)^t
######################################################################


def GraphConv(layer_name, input_dim, output_dim, adj, input, norm=None, bias=False, activation=None, use_tensorcore=False, dtype="float32", verbose=False):
    """
    Parameters
    ----------
    layer_name: str
    Name of layer

    input_dim: int
    Input dimension per node feature

    output_dim: int,
    Output dimension per node feature

    adj: namedtuple,
    Graph representation (Adjacency Matrix) in Sparse Format.
    For CSR: (`data`, `indices`, `indptr`, `format="csr"`)
    For DTC: (`A_compressed`, `format="dtc"`, `block_col_mapping`, `M`, `K`, `M_padded`, `K_padded`, `num_row_blocks`)

    input: relay.Expr,
    Input feature to current layer with shape [num_nodes, input_dim]

    norm: relay.Expr,
    Norm passed to this layer to normalize features before and after Convolution.

    bias: bool
    Set bias to True to add bias when doing GCN layer

    activation: <function relay.op.nn>,
    Activation function applies to the output. e.g. relay.nn.{relu, sigmoid, log_softmax, softmax, leaky_relu}

    use_tensorcore: bool
    Whether to use Tensor Core (MMA) intrinsics for sparse_dense operation.
    If True, uses Tensor Core implementation (requires GPU and proper op registration).
    If False, uses standard relay.nn.sparse_dense (works on both CPU and GPU).

    dtype: str
    Data type to use for weights and operations. Options: "float32" or "float16".
    Default: "float32"

    Returns
    ----------
    output: tvm.relay.Expr
    The Output Tensor for this layer [num_nodes, output_dim]
    """
    if norm is not None:
        input = relay.multiply(input, norm)

    # CRITICAL FIX: Following SparseTIR approach - dimensions should ALREADY be padded at model creation
    # If use_tensorcore, input_dim and output_dim should already be multiples of 32 from models.py
    # We only pad here if they're not already padded (for backward compatibility)
    if use_tensorcore:
        block_factor_n = 64
        # Check if dimensions are already multiples of 32 (padded at model creation)
        if input_dim % block_factor_n == 0 and output_dim % block_factor_n == 0:
            # Dimensions are already padded, use them directly
            input_dim_padded = input_dim
            output_dim_padded = output_dim
        else:
            # Dimensions are not padded, pad them here (should not happen if model is created correctly)
            input_dim_padded = ((input_dim + block_factor_n - 1) // block_factor_n) * block_factor_n
            output_dim_padded = ((output_dim + block_factor_n - 1) // block_factor_n) * block_factor_n
    else:
        input_dim_padded = input_dim
        output_dim_padded = output_dim
    
    # Create weight variable - use padded dimensions (should already be 32-aligned from model creation)
    weight = relay.var(layer_name + ".weight", shape=(input_dim_padded, output_dim_padded), dtype=dtype)
    weight_t = relay.transpose(weight)
    
    # CRITICAL: Pad input to input_dim_padded if needed (for Tensor Core)
    # This should only happen if input_dim is not already padded
    if use_tensorcore and input_dim_padded > input_dim:
        # Pad input from (num_nodes, input_dim) to (num_nodes, input_dim_padded)
        pad_width_input = ((0, 0), (0, input_dim_padded - input_dim))
        pad_value_input = relay.const(0.0, dtype=dtype)
        input_padded = relay.nn.pad(input, pad_width_input, pad_value_input, pad_mode="constant")
    else:
        input_padded = input
    
    # First dense: weight_t @ input_padded^T (original implementation for CUDA Core compatibility)
    # Use specified dtype for output
    # CRITICAL: Original implementation uses dense(weight_t, input_padded) which computes weight_t @ input_padded^T
    #   - weight_t (output_dim_padded, input_dim_padded)
    #   - input_padded (num_nodes, input_dim_padded)
    #   - input_padded^T (input_dim_padded, num_nodes)
    #   - output (output_dim_padded, num_nodes)
    # This matches the original TVM GCN example implementation
    dense = relay.nn.dense(weight_t, input_padded, units=None, out_dtype=dtype)
    
    # Check adjacency matrix format
    matrix_format = getattr(adj, 'format', 'csr')
    
    if matrix_format == "dtc" and use_tensorcore:
        # DTC format: use A_compressed with dense operation (spmm_mma.py implementation)
        # STRICTLY following spmm_mma.py logic for column mapping
        # 
        # dense is (num_nodes, output_dim_padded) from dense(input_padded, weight_t)
        # where input_padded is (num_nodes, input_dim_padded) and weight_t is (output_dim_padded, input_dim_padded)
        # So dense = input_padded @ weight_t^T = (num_nodes, input_dim_padded) @ (input_dim_padded, output_dim_padded) = (num_nodes, output_dim_padded)
        # No need to transpose - dense is already in the correct format
        
        # Get dimensions
        K_padded = adj.K_padded
        M_padded = adj.M_padded
        M_original = adj.M
        K_original = adj.K
        block_col_mapping = adj.block_col_mapping  # List of lists: one per row block
        
        # Debug: Check A_compressed shape in Relay
        # Note: We can't directly print Relay expressions, but we can check adj.A_compressed
        if verbose:
            print(f"  [GraphConv DTC] M_original={M_original}, M_padded={M_padded}, K_original={K_original}, K_padded={K_padded}")
        
        # CRITICAL FIX: dense(weight_t, input_padded) computes weight_t @ input_padded^T
        #   - weight_t (output_dim_padded, input_dim_padded) = (32, 32)
        #   - input_padded (num_nodes, input_dim_padded) = (128, 32)
        #   - input_padded^T (input_dim_padded, num_nodes) = (32, 128)
        #   - dense = weight_t @ input_padded^T = (32, 32) @ (32, 128) = (32, 128)
        # So dense is (output_dim_padded, num_nodes) = (32, 128), NOT (num_nodes, output_dim_padded)
        # We need to transpose to get (num_nodes, output_dim_padded) = (128, 32)
        dense_T = relay.transpose(dense)  # (num_nodes, output_dim_padded) = (128, 32)
        
        # Build global column mapping following spmm_mma.py logic (lines 422-532)
        # Collect all unique columns used across all blocks
        all_non_zero_cols = set()
        for block_cols in block_col_mapping:
            all_non_zero_cols.update(block_cols)
        A_compressed_used_cols = sorted(list(all_non_zero_cols))
        
        # Build mapping_cols: map compressed column index to original column index
        # Following spmm_mma.py lines 514-525
        mapping_cols = list(A_compressed_used_cols)
        
        # Pad to K_padded (same logic as spmm_mma.py)
        while len(mapping_cols) < K_padded:
            added = False
            for j in range(K_original):
                if j not in mapping_cols:
                    mapping_cols.append(j)
                    added = True
                    break
            if not added:
                # If all columns are already in mapping_cols, pad with last column
                if len(mapping_cols) > 0:
                    mapping_cols.append(mapping_cols[-1])
                else:
                    mapping_cols.append(0)
            if len(mapping_cols) >= K_padded:
                break
        # Ensure exactly K_padded elements
        mapping_cols = mapping_cols[:K_padded]
        if len(mapping_cols) < K_padded:
            # Final padding if still needed
            while len(mapping_cols) < K_padded:
                mapping_cols.append(mapping_cols[-1] if mapping_cols else 0)
        
        mapping_cols_array = np.array(mapping_cols, dtype="int32")
        mapping_cols_const = relay.const(mapping_cols_array)
        
        # Debug: Print mapping_cols to verify it's correct
        if verbose:
            print(f"  [DTC Debug] K_padded={K_padded}, len(mapping_cols)={len(mapping_cols)}")
            print(f"  [DTC Debug] mapping_cols[:10]={mapping_cols[:10] if len(mapping_cols) > 10 else mapping_cols}")
            print(f"  [DTC Debug] dense_T shape should be (num_nodes, output_dim_padded) = ({M_original}, {output_dim_padded})")
        
        # CRITICAL FIX: output_dim_padded is ALREADY a multiple of 32 (from model creation)
        # Following SparseTIR approach - dimensions are fixed at definition time
        N_padded = output_dim_padded  # Use the padded dimension (already 32-aligned from model creation)
        N_original = output_dim_padded  # Since output_dim is already padded, N_original == N_padded
        
        # Take rows from dense_T: (num_nodes, output_dim_padded) -> (K_padded, output_dim_padded)
        # Use relay.take with axis=0 to select rows based on node indices
        # mapping_cols are original column indices (node indices) used in A_compressed
        # We need to select the corresponding rows from dense_T
        # 
        # IMPORTANT: relay.take with axis=0 selects rows
        # If data is (num_nodes, output_dim_padded) and indices is (K_padded,), result is (K_padded, output_dim_padded)
        # However, Relay's type inference may fail, so we use relay.take and then explicitly reshape
        X_selected_take = relay.take(dense_T, mapping_cols_const, axis=0)  # May be (K_padded, output_dim_padded) or incorrect shape
        
        # CRITICAL FIX: Explicitly reshape to ensure correct shape inference
        # Relay's type inference for relay.take may fail, so we explicitly reshape to (K_padded, N_padded)
        # This ensures Relay correctly infers the shape for subsequent operations
        X_selected = relay.reshape(X_selected_take, [K_padded, N_padded])
        X_selected_padded = X_selected
        
        # Now X_selected_padded is (K_padded, N_padded)
        # Transpose to (N_padded, K_padded) for Relay's dense operation
        # Relay's dense computes: data @ weight^T
        # We want: A_compressed @ X_selected
        # So: dense(A_compressed, X_selected_T) = A_compressed @ X_selected
        # 
        # IMPORTANT: transpose(axes=None) reverses all dimensions
        # For 2D tensor (K_padded, N_padded), transpose() -> (N_padded, K_padded) ✓
        # Explicitly specify axes=[1, 0] to transpose: (K_padded, N_padded) -> (N_padded, K_padded)
        X_selected_T = relay.transpose(X_selected_padded, axes=[1, 0])  # (N_padded, K_padded)
        
        # Ensure K_padded meets Tensor Core requirements (multiple of WMMA_K * chunk_factor)
        # Determine chunk_factor based on K_padded (same logic as in spmm_mma.py)
        WMMA_K = 16
        if K_padded <= 32:
            chunk_factor = 2  # Requires K to be multiple of 16 * 2 = 32
        else:
            chunk_factor = 4  # Requires K to be multiple of 16 * 4 = 64
        
        K_padded_required = ((K_padded + WMMA_K * chunk_factor - 1) // (WMMA_K * chunk_factor)) * (WMMA_K * chunk_factor)
        
        # CRITICAL: Ensure A_compressed has correct shape in Relay
        # Relay's shape inference may not correctly infer the shape of relay.Constant
        # So we explicitly reshape A_compressed to ensure it has the correct shape
        # This is a workaround for Relay's shape inference issue
        # Use relay.reshape with explicit shape tuple to force Relay to recognize (M_padded, K_padded)
        # This ensures Relay's shape inference correctly identifies the padded dimensions
        A_compressed_reshaped = relay.reshape(adj.A_compressed, [M_padded, K_padded])
        print(f"  [GraphConv DTC] Reshaped A_compressed to ({M_padded}, {K_padded}) for Relay shape inference")
        
        # Pad K dimension if needed
        if K_padded_required > K_padded:
            # Pad A_compressed: (M_padded, K_padded) -> (M_padded, K_padded_required)
            # print(f"  [GraphConv DTC] Padding K: {K_padded} -> {K_padded_required}")
            pad_width_A = ((0, 0), (0, K_padded_required - K_padded))
            pad_value = relay.const(0.0, dtype=dtype)
            A_compressed_padded = relay.nn.pad(A_compressed_reshaped, pad_width_A, pad_value, pad_mode="constant")
            
            # Pad X_selected_T: (N_padded, K_padded) -> (N_padded, K_padded_required)
            pad_width_X = ((0, 0), (0, K_padded_required - K_padded))
            X_selected_T_padded = relay.nn.pad(X_selected_T, pad_width_X, pad_value, pad_mode="constant")
            
            K_padded = K_padded_required
        else:
            A_compressed_padded = A_compressed_reshaped
            X_selected_T_padded = X_selected_T
            # print(f"  [GraphConv DTC] No K padding needed: K_padded={K_padded}, K_padded_required={K_padded_required}")
        
        # Convert to float16 for Tensor Core (required by spmm_mma.py)
        if dtype == "float16":
            A_compressed_fp16 = relay.cast(A_compressed_padded, "float16")
            X_selected_T_fp16 = relay.cast(X_selected_T_padded, "float16")
        else:
            A_compressed_fp16 = A_compressed_padded
            X_selected_T_fp16 = X_selected_T_padded
        
        # Dense operation: A_compressed @ X_selected_T^T = A_compressed @ X_selected
        # Note: Activation is applied AFTER Tensor Core computation (in Relay layer)
        # This ensures Tensor Core schedule is clean without activation fusion
        # A_compressed: (M_padded, K_padded) = (32, 64) after reshape
        # X_selected_T: (N_padded, K_padded) = (32, 64)
        # Result: (M_padded, N_padded) = (32, 32)
        # 
        # dense_tensorcore_compute will detect DTC format and use dtc_spmm_tensorcore_cuda from spmm_mma.py
        # CRITICAL: Explicitly specify units=N_padded to ensure Relay shape inference uses padded dimension
        # This ensures Relay correctly infers output shape as (M_padded, N_padded) = (32, 32)
        output = relay.nn.dense(A_compressed_fp16, X_selected_T_fp16, units=N_padded, out_dtype=dtype)  # (M_padded, N_padded) = (32, 32)
        
        # Note: We don't clear context here - keep it for all layers
        # The activation is the same for all layers in the model
        
        # Output is (M_padded, N_padded) = (num_nodes_padded, output_dim_padded) = (32, 32)
        # But we need to slice to (M_original, N_padded) = (27, 32) to match CPU/CUDA Core output
        # This ensures consistent output shape across all backends
        # 
        # Note: dense(A_compressed, X_selected_T) produces (M_padded, N_padded)
        # where M_padded = num_nodes (padded to 32), N_padded = output_dim (padded to 32)
        # 
        # CRITICAL: Slice output to (M_original, N_padded) to match actual number of nodes
        # This ensures output shape is (num_nodes, output_dim) = (27, 32) instead of (32, 32)
        # Use relay.strided_slice to slice the first dimension from 0 to M_original
        output_t = relay.strided_slice(
            output,
            begin=[0, 0],
            end=[M_original, N_padded],
            strides=[1, 1],
            axes=[0, 1]
        )  # Shape: (M_original, N_padded) = (27, 32)
        
        # output_t is now (num_nodes, output_dim) = (M_original, N_padded) = (27, 32)
        # This matches CPU/CUDA Core output shape
    else:
        # Standard sparse_dense (works on both CPU and GPU)
        # 
        # Note about relay.nn.sparse_dense:
        # - CPU: Full support with optimizations (AVX2/AVX512)
        # - GPU: Basic support, but does NOT use Tensor Core by default
        # - For GPU with Tensor Core, set use_tensorcore=True (when implemented)
        # 
        # Original implementation (restored for CUDA Core compatibility):
        # Following TVM GCN example: dense(weight_t, input) -> sparse_dense(dense, adj) -> transpose
        # - dense = weight_t @ input_padded^T = (output_dim_padded, num_nodes) = (32, 128)
        # - sparse_dense(dense, adj) computes: dense @ sparse^T
        #   - dense: (output_dim_padded, num_nodes) = (32, 128)
        #   - sparse (adj): (num_nodes, num_nodes) = (128, 128) CSR matrix
        #   - sparse^T: (num_nodes, num_nodes) = (128, 128)
        #   - output: (output_dim_padded, num_nodes) @ (num_nodes, num_nodes) = (32, 128) @ (128, 128) = (32, 128)
        # - transpose: (num_nodes, output_dim_padded) = (128, 32) ✓
        output = relay.nn.sparse_dense(dense, adj)
        # For standard sparse_dense, output is (output_dim_padded, num_nodes), need to transpose
        output_t = relay.transpose(output)
        
    if norm is not None:
        output_t = relay.multiply(output_t, norm)
    if bias is True:
        # Use specified dtype for bias to match output_t dtype
        # Use output_dim_padded to match the actual bias parameter shape (which may be padded for Tensor Core)
        _bias = relay.var(layer_name + ".bias", shape=(output_dim_padded,), dtype=dtype)
        output_t = relay.nn.bias_add(output_t, _bias, axis=-1)
    if activation is not None:
        output_t = activation(output_t)
    # Convert to float32 at the end if needed (for final output precision)
    # For now, keep as float16 for consistency
    return output_t


