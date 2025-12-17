"""
Tensor Core Strategy Registration for Relay

This module registers Tensor Core implementations for dense and sparse_dense operations
by directly using the compute and schedule functions from gemm_mma.py and spmm_mma.py.

Key functions used:
- From gemm_mma.py: dense_tensorcore_cuda, schedule_dense_tensorcore
- From spmm_mma.py: dtc_spmm_tensorcore_cuda, schedule_dtc_spmm_tensorcore

The strategy registration allows TVM to automatically select Tensor Core implementations
when compiling Relay models for CUDA targets. For DTC format, we use relay.nn.dense which
will automatically use the Tensor Core schedule via strategy registration.
"""

import sys
import os

# Add TVM path
tvm_python_path = "/home/nas/polin/cmu-berlin/sparse-gnn-tvm-tensorcore/tvm-0.10.0/python"
if tvm_python_path not in sys.path:
    sys.path.insert(0, tvm_python_path)

import tvm
from tvm import te
from tvm import relay
from tvm import topi
from tvm.relay.op import op as _op
from tvm.relay.op.strategy import wrap_topi_schedule
from tvm.relay.op.strategy.generic import wrap_compute_dense, wrap_compute_sparse_dense
from tvm.te import SpecializedCondition
from tvm.topi.utils import get_const_tuple

# Import Tensor Core implementations
try:
    from .gemm_mma import dense_tensorcore_cuda, schedule_dense_tensorcore
    from .spmm_mma import dtc_spmm_tensorcore_cuda, schedule_dtc_spmm_tensorcore
except ImportError:
    from apps.gnn_tvm_utils.gemm_mma import dense_tensorcore_cuda, schedule_dense_tensorcore
    from apps.gnn_tvm_utils.spmm_mma import dtc_spmm_tensorcore_cuda, schedule_dtc_spmm_tensorcore

# Tensor Core configuration
WMMA_M = 16
WMMA_N = 16
WMMA_K = 16

# Global verbose flag for debug messages
_verbose = False

def set_verbose(verbose):
    """Set global verbose flag for debug messages"""
    global _verbose
    _verbose = verbose

def get_verbose():
    """Get global verbose flag"""
    return _verbose


def _check_tensorcore_requirements(data_shape, weight_shape, out_dtype="float32"):
    """
    Check if dimensions meet Tensor Core requirements from gemm_mma.py
    
    For float16 Tensor Core, gemm_mma.py requires one of:
    - (batch % 8 == 0 and in_dim % 16 == 0 and out_dim % 32 == 0) OR
    - (batch % 16 == 0 and in_dim % 16 == 0 and out_dim % 16 == 0) OR
    - (batch % 32 == 0 and in_dim % 16 == 0 and out_dim % 8 == 0)
    
    For dense: data is (batch, in_dim), weight is (out_dim, in_dim)
    """
    batch, in_dim = get_const_tuple(data_shape)
    out_dim, _ = get_const_tuple(weight_shape)
    
    # Check if dimensions meet gemm_mma.py's strict requirements for float16
    # This matches the assertion in dense_tensorcore_cuda
    meets_requirements = (
        (batch % 8 == 0 and in_dim % 16 == 0 and out_dim % 32 == 0)
        or (batch % 16 == 0 and in_dim % 16 == 0 and out_dim % 16 == 0)
        or (batch % 32 == 0 and in_dim % 16 == 0 and out_dim % 8 == 0)
    )
    
    return meets_requirements


def _dense_tensorcore_cuda_force(data, weight, bias=None, out_dtype=None):
    """
    Wrapper for dense_tensorcore_cuda that skips dimension assertions in force mode
    
    This allows Tensor Core to be used even when dimensions don't strictly meet requirements.
    The schedule will handle dimension padding internally.
    """
    # Create compute without dimension checks (similar to dense_tensorcore_cuda but without asserts)
    if out_dtype is None:
        out_dtype = data.dtype
    batch, in_dim = get_const_tuple(data.shape)
    out_dim, _ = get_const_tuple(weight.shape)
    
    k = te.reduce_axis((0, in_dim), name="k")
    matmul = te.compute(
        (batch, out_dim),
        lambda i, j: te.sum(data[i, k].astype(out_dtype) * weight[j, k].astype(out_dtype), axis=k),
        name="T_dense",
        tag="dense_tensorcore",
    )
    if bias is not None:
        matmul = te.compute(
            (batch, out_dim),
            lambda i, j: matmul[i, j] + bias[j].astype(out_dtype),
            tag="broadcast",
        )
    return matmul


def dense_tensorcore_compute(data, weight, bias, out_dtype):
    """
    Compute function for dense operation using Tensor Core
    
    ALWAYS uses DTC format (dtc_spmm_tensorcore_cuda from spmm_mma.py)
    No fallback to CSR or normal dense - Tensor Core is fixed to DTC format.
    
    This function signature matches what wrap_compute_dense expects:
    wrap_compute_dense calls topi_compute(data, weight, bias, out_dtype)
    
    IMPORTANT: Tensor Core requires float16 input. 
    The caller must ensure inputs are float16 before calling this function.
    """
    # ALWAYS use DTC format for Tensor Core
    if _verbose:
        print(f"  [DTC Format] Using spmm_mma.py (fixed DTC): data={data.shape}, weight={weight.shape}")
    try:
        from .spmm_mma import dtc_spmm_tensorcore_cuda
        from tvm import te
        
        # dtc_spmm_tensorcore_cuda expects:
        #   - A_compressed (M_padded, K_padded) = data
        #   - X_selected (K_padded, N_padded)
        # Relay's dense computes: data @ weight^T
        #   - data (M_padded, K_padded) = A_compressed ✓
        #   - weight (N_padded, K_padded)
        #   - weight^T (K_padded, N_padded) = X_selected ✓
        # So we need to transpose weight to get X_selected
        # 
        # NOTE: Padding should be done in gconv.py at Relay level, not here in TE compute.
        # If dimensions don't match Tensor Core requirements, we should fail with a clear error.
        
        M_from_shape, K_from_shape = get_const_tuple(data.shape)
        weight_dim0, weight_dim1 = get_const_tuple(weight.shape)
        
        # Debug: print actual shapes to understand the issue
        if _verbose:
            print(f"  [DTC Debug] data.shape = {data.shape} (expected: (M_padded, K_padded))")
            print(f"  [DTC Debug] weight.shape = {weight.shape} (expected: (N_padded, K_padded))")
            print(f"  [DTC Debug] M_from_shape={M_from_shape}, K_from_shape={K_from_shape}, weight_dim0={weight_dim0}, weight_dim1={weight_dim1}")
        
        # CRITICAL FIX: Force M_padded to be multiple of 32, and K_padded based on actual K size
        # Relay's shape inference may incorrectly infer shapes for relay.Constant
        # For M: always force to multiple of 32 (handles case where M_from_shape=24 instead of 32)
        # For K: use the actual K_from_shape, but ensure it meets Tensor Core requirements
        block_factor_m = WMMA_M * 1 * 2  # warp_row_tiles=1, block_row_warps=2 (from schedule_dtc_spmm_tensorcore)
        
        # Force M_padded to be multiple of 32 (always pad, even if M_from_shape is already 32)
        # This handles the case where Relay shape inference incorrectly infers M_from_shape=24 instead of 32
        M_padded = ((M_from_shape + block_factor_m - 1) // block_factor_m) * block_factor_m
        if M_padded < M_from_shape:
            # This shouldn't happen, but handle it just in case
            M_padded = M_from_shape
            M_padded = ((M_padded + block_factor_m - 1) // block_factor_m) * block_factor_m
        
        # For K_padded: use K_from_shape, but ensure it meets Tensor Core requirements
        # According to spmm_mma.py:
        #   - If K <= 32: chunk_factor = 2, so need multiple of 32
        #   - If K > 32: chunk_factor = 4, so need multiple of 64
        # Use weight.shape[1] as the source of truth for K_padded (it should match K_from_shape)
        K_from_weight = weight_dim1
        if K_from_weight != K_from_shape:
            if _verbose:
                print(f"  [DTC Warning] K dimension mismatch: data.K={K_from_shape} != weight.K={K_from_weight}, using weight.K")
            K_from_shape = K_from_weight
        
        # Calculate K_padded based on actual K size
        if K_from_shape <= 32:
            K_factor = WMMA_K * 2  # chunk_factor = 2, need multiple of 32
        else:
            K_factor = WMMA_K * 4  # chunk_factor = 4, need multiple of 64
        K_padded = ((K_from_shape + K_factor - 1) // K_factor) * K_factor
        
        # Store original dimensions for padding in compute
        M_original = M_from_shape  # Use M_from_shape as M_original (actual data size)
        K_original = K_from_shape  # Use K_from_shape as K_original (actual data size)
        
        if _verbose:
            print(f"  [DTC Debug] Forced dimensions: M_from_shape={M_from_shape} -> M_padded={M_padded}, K_from_shape={K_from_shape} -> K_padded={K_padded}")
        
        # Store original dimensions for padding in compute
        # CRITICAL: We need to pad data to (M_padded, K_padded) BEFORE passing to dtc_spmm_tensorcore_cuda
        # This ensures A_compressed has the correct shape (M_padded, K_padded) to avoid out-of-bounds access
        # First pad M dimension if needed, then pad K dimension
        if M_padded > M_original:
            # Pad data from (M_original, K_from_shape) to (M_padded, K_from_shape)
            data_M_padded = te.compute(
                (M_padded, K_from_shape),
                lambda i, k: te.if_then_else(i < M_original, data[i, k], te.const(0, data.dtype)),
                name="data_M_padded"
            )
            data_after_M_padding = data_M_padded
        else:
            data_after_M_padding = data
        
        # Store data_after_M_padding for later use in K padding
        # Don't create final padding compute ops yet - wait until K_padded is calculated
        weight_padded = weight  # Will be updated after K_padded is calculated
        M_original_for_padding = M_original  # Store actual M_original for reference
        if _verbose:
            print(f"  [DTC Debug] M_original={M_original}, M_padded={M_padded}, M_original_for_padding={M_original_for_padding}")
        
        # Check if weight.shape is actually (M_padded, K_padded) instead of (N_padded, K_padded)
        # This could happen in two cases:
        # 1. GraphConv case: X_selected_T was not correctly transposed (error)
        # 2. Simple dense case: weight is (M, K) and needs to be transposed to (K, M) for X_selected
        # 
        # For simple dense: Relay's dense computes data @ weight^T
        #   - data (M, K)
        #   - weight (N, K) -> weight^T (K, N)
        #   - output (M, N)
        # But if weight is (M, K), we need to transpose it to get (K, M) for X_selected
        if weight_dim0 == M_original and weight_dim1 == K_padded:
            # weight is (M, K) instead of (N, K)
            # This could be:
            # 1. GraphConv error: X_selected_T not correctly transposed
            # 2. Simple dense: weight needs to be transposed
            # 
            # For simple dense, we can handle this by treating weight as (K, M) and transposing
            # For GraphConv, this is an error
            # 
            # Check: if weight_dim0 == M_original and M_original is large (like num_nodes),
            # this is likely a GraphConv error. But if it's a simple dense test, we should handle it.
            # 
            # Simple solution: transpose weight to get (K, M) and use M as N
            if _verbose:
                print(f"  [DTC Debug] weight.shape is ({weight_dim0}, {weight_dim1}) = (M, K), transposing to (K, M) for X_selected")
            # Transpose weight: (M, K) -> (K, M)
            # Then use M as N_original
            N_original = weight_dim0  # M becomes N after transpose
            K_padded_check = weight_dim1
            # weight is already (M, K), we need (K, M) for X_selected
            # So we'll transpose it in the X_selected creation
        else:
            # Normal case: weight is (N, K)
            N_original = weight_dim0
            K_padded_check = weight_dim1
        
        # CRITICAL FIX: Following SparseTIR approach - dimensions should ALREADY be padded at model creation
        # weight.shape[0] should already be a multiple of 32 (from model creation)
        # If it's not, it means model creation didn't work correctly
        block_factor_n = WMMA_N * 1 * 2  # warp_col_tiles=1, block_col_warps=2 (from schedule_dtc_spmm_tensorcore)
        
        # Check if weight.shape[0] is already a multiple of 32 (should be from model creation)
        if weight_dim0 % block_factor_n == 0:
            # weight.shape[0] is already a multiple of 32 (from model creation), use it directly
            N_padded = weight_dim0
            N_original = weight_dim0  # In this case, N_original == N_padded (already padded)
            if _verbose:
                print(f"  [DTC Debug] weight.shape[0]={weight_dim0} is multiple of {block_factor_n} (from model creation), using as N_padded")
        else:
            # weight.shape[0] is not a multiple of 32 - this should not happen if model is created correctly
            # But if it does, pad it here (this indicates a bug in model creation)
            if _verbose:
                print(f"  [DTC Warning] weight.shape[0]={weight_dim0} is NOT multiple of {block_factor_n} - model creation may have failed!")
                print(f"  [DTC Warning] Padding weight in TE layer (this should be avoided by ensuring model creation uses padded dimensions)")
            N_original = weight_dim0
            N_padded = ((N_original + block_factor_n - 1) // block_factor_n) * block_factor_n
            if _verbose:
                print(f"  [DTC Debug] Recalculating N_padded={N_padded}")
        
        K_padded_check = weight_dim1
        
        # K dimensions should match
        assert K_padded == K_padded_check, f"K dimension mismatch: data.K={K_padded} != weight.K={K_padded_check}"
        
        # Pad K_padded to multiple of WMMA_K * chunk_factor if needed
        # According to spmm_mma.py:
        #   - If K_from_shape <= 32: chunk_factor = 2, so need multiple of 32
        #   - If K_from_shape > 32: chunk_factor = 4, so need multiple of 64
        K_original = K_from_shape  # Use K_from_shape as K_original (actual data size)
        if K_from_shape <= 32:
            K_factor = WMMA_K * 2  # chunk_factor = 2
        else:
            K_factor = WMMA_K * 4  # chunk_factor = 4
        K_padded = ((K_original + K_factor - 1) // K_factor) * K_factor
        if _verbose:
            print(f"  [DTC Debug] K_original={K_original}, K_padded={K_padded} (will pad in compute)")
        
        # Create padding compute ops for K dimension if needed
        if K_padded > K_original:
            # Need to pad data and weight in K dimension
            # Pad data from (M_padded, K_original) to (M_padded, K_padded)
            # Note: data_after_M_padding is already (M_padded, K_from_shape) after M padding
            # Use data_after_M_padding as input (not data_padded, which might be reassigned)
            data_padded = te.compute(
                (M_padded, K_padded),
                lambda i, k: te.if_then_else(k < K_original, data_after_M_padding[i, k], te.const(0, data.dtype)),
                name="data_K_padded"
            )
            # Pad weight from (N_padded, K_original) to (N_padded, K_padded)
            # Note: Use N_padded (not N_original) to ensure correct shape
            weight_padded = te.compute(
                (N_padded, K_padded),
                lambda n, k: te.if_then_else(k < K_original, weight[n, k], te.const(0, weight.dtype)),
                name="weight_K_padded"
            )
        else:
            # K_padded == K_original, no padding needed for K dimension
            # data_after_M_padding is already (M_padded, K_original) after M padding (if needed)
            # Use it directly as data_padded
            data_padded = data_after_M_padding
            weight_padded = weight
        
        # Store original dimensions for padding in compute
        N_original_for_padding = N_original
        
        # Pad weight in N dimension if needed
        # IMPORTANT: We need to pad weight_padded to (N_padded, K_padded) to avoid boundary checks
        # But we can't use if_then_else in the lambda because it causes boundary checks in tensorize
        # Solution: Create X_selected directly with shape (K_padded, N_padded) and use weight_padded with bounds checking
        # But this still uses if_then_else...
        # 
        # Actually, the real issue is that weight_padded shape is (N_original, K_padded)
        # When we create X_selected with shape (K_padded, N_padded), we need to handle n >= N_original
        # The solution is to ensure weight_padded is already (N_padded, K_padded) before creating X_selected
        
        # CRITICAL FIX: Following SparseTIR approach - ensure dimensions are fixed at definition time
        # Get the actual N dimension from weight_padded
        weight_N_dim, weight_K_dim = get_const_tuple(weight_padded.shape)
        
        # If weight_N_dim is not N_padded, we need to pad it
        # But this will cause boundary checks if N_original is not a multiple of 32
        # The key is to ensure N_original (used in if_then_else) is also a multiple of 32
        if weight_N_dim != N_padded:
            # Debug: Print warning about dimension mismatch
            if _verbose:
                print(f"  [DTC Warning] weight.shape[0]={weight_N_dim} != N_padded={N_padded}, padding in TE layer")
            # Pad weight_padded to (N_padded, K_padded) - this will use if_then_else
            # CRITICAL: Use N_padded (not weight_N_dim) as the boundary check to avoid tensorize errors
            # But we still need to check n < weight_N_dim to avoid out-of-bounds access
            weight_padded_N = te.compute(
                (N_padded, K_padded),
                lambda n, k: te.if_then_else(n < weight_N_dim, weight_padded[n, k], te.const(0, weight_padded.dtype)),
                name="weight_N_padded"
            )
            weight_padded = weight_padded_N
            # Keep N_original = weight_N_dim for output slicing, but N_padded is used for computation
        
        # Verify weight_padded shape
        weight_padded_shape = get_const_tuple(weight_padded.shape)
        if _verbose:
            print(f"  [DTC Debug] Final weight_padded.shape={weight_padded_shape}, N_original={N_original}, N_padded={N_padded}")
        
        # Create X_selected by transposing weight_padded (which should be (N_padded, K_padded) now)
        # IMPORTANT: weight_padded should already be (N_padded, K_padded) after padding above
        # So we can directly transpose it without if_then_else
        X_selected = te.compute(
            (K_padded, N_padded),  # Use N_padded to match weight_padded shape
            lambda k, n: weight_padded[n, k],
            name="X_selected"
        )
        # Call dtc_spmm_tensorcore_cuda which returns (M_padded, N_padded)
        # CRITICAL: We need to pass the actual M_original (from data.shape) to dtc_spmm_tensorcore_cuda
        # because A_compressed_padded uses if_then_else(i < M_original, A_compressed[i, k], ...)
        # If we pass M_original=M_padded, it will try to access A_compressed[i, k] for i >= M_original (actual),
        # causing out-of-bounds memory access and CUDA kernel execution failure
        # 
        # However, we still want to avoid boundary checks in tensorize, so we need a different approach:
        # Option 1: Pass actual M_original, but ensure A_compressed is already padded to M_padded
        # Option 2: Don't use if_then_else in A_compressed_padded if M_original == M_padded
        # 
        # For now, pass actual M_original to avoid out-of-bounds access
        # The schedule should handle this correctly if dimensions are properly padded
        Y_padded = dtc_spmm_tensorcore_cuda(
            data_padded, X_selected, 
            M_original=M_original, N_original=N_padded,  # Use actual M_original to avoid out-of-bounds access
            M_padded=M_padded, N_padded=N_padded,
            out_dtype=out_dtype
        )
        
        # CRITICAL FIX: Following SparseTIR approach - dimensions should ALREADY be padded at model creation
        # N_original_for_padding should already be a multiple of 32 (from model creation)
        # So we can use it directly without padding
        block_factor_n = 32
        if N_original_for_padding % block_factor_n != 0:
            # This should not happen if model is created correctly with padded dimensions
            # But if it does, use N_padded to avoid boundary checks
            if _verbose:
                print(f"  [DTC Warning] N_original_for_padding={N_original_for_padding} is not multiple of {block_factor_n}, using N_padded={N_padded}")
            N_output = N_padded
        else:
            # N_original_for_padding is already a multiple of 32 (from model creation), use it directly
            N_output = N_original_for_padding
        
        # CRITICAL: Output shape should be (M_padded, N_padded) = (32, 32) instead of (M_original, N_original) = (23, 32)
        # This is acceptable because Relay expects padded dimensions throughout the model
        # No need to slice - return full padded output
        # Create Y_dtc_sliced with padded shape (M_padded, N_padded)
        # M_original_for_padding is already M_padded, and N_output is already N_padded
        Y = te.compute(
            (M_original_for_padding, N_output),  # This is (M_padded, N_padded) = (32, 32)
            lambda i, j: Y_padded[i, j],
            name="Y_dtc_sliced"
        )
        
        if _verbose:
            print(f"  [DTC Debug] Output shape: Y_dtc_sliced.shape = ({M_original_for_padding}, {N_output}) = (M_padded, N_padded)")
        
        return Y
    except Exception as e:
        print(f"  [ERROR] Failed to use spmm_mma.py for DTC format: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"DTC format requires spmm_mma.py implementation. Fallback disabled. Error: {e}")
    # If not in force mode and inputs are not float16, fall back to standard dense
    if not _force_tensorcore and (data.dtype != "float16" or weight.dtype != "float16"):
        # Fall back to standard dense for non-float16 inputs (only if not forcing)
        return topi.nn.dense(data, weight, bias, out_dtype)
    
    # Check if dimensions meet requirements
    meets_requirements = _check_tensorcore_requirements(data.shape, weight.shape, out_dtype)
    
    if not meets_requirements:
        if _force_tensorcore:
            # Force mode: use Tensor Core anyway (skip dimension checks)
            # Use wrapper that skips assertions - schedule will handle dimension requirements
            return _dense_tensorcore_cuda_force(data, weight, bias=bias, out_dtype=out_dtype)
        else:
            # Normal mode: fall back to standard dense
            return topi.nn.dense(data, weight, bias, out_dtype)
    
    # Use Tensor Core implementation (dimensions already meet requirements)
    # Note: dense_tensorcore_cuda expects (batch, in_dim) x (out_dim, in_dim)
    # Relay's dense expects (batch, in_dim) x (out_dim, in_dim) as well
    return dense_tensorcore_cuda(data, weight, bias=bias, out_dtype=out_dtype)


def schedule_dense_tensorcore_wrapper(outs):
    """
    Schedule wrapper for dense Tensor Core
    
    ALWAYS uses DTC format schedule (schedule_dtc_spmm_tensorcore from spmm_mma.py)
    No fallback to CSR or normal dense - Tensor Core is fixed to DTC format.
    
    Note: Only applies DTC schedule to actual matrix multiplication operations
    (those with reduce_axis). Other operations (broadcast, element-wise) should
    use standard schedules.
    
    Note: wrap_topi_schedule will wrap this to accept (attrs, outs, target),
    but we only need outs here.
    """
    # Get the output tensor
    C = outs[0]
    
    # Also check tag - broadcast and elemwise operations should not use DTC schedule
    compute_tag = getattr(C.op, 'tag', '')
    if compute_tag == 'broadcast' or compute_tag == 'elemwise':
        is_matrix_mult = False
    
    # ALWAYS use DTC format schedule for Tensor Core matrix multiplications
    if _verbose:
        print(f"  [DTC Schedule] Using schedule_dtc_spmm_tensorcore from spmm_mma.py (fixed DTC)")
    try:
        from .spmm_mma import schedule_dtc_spmm_tensorcore
        s = te.create_schedule(C.op)
        
        # Debug: print structure before applying schedule
        if _verbose:
            try:
                compute_name = getattr(C.op, 'name', 'N/A')
                compute_tag = getattr(C.op, 'tag', 'N/A')
                print(f"    Y shape: {C.shape}, name: {compute_name}, tag: {compute_tag}")
                if len(C.op.input_tensors) >= 2:
                    A = C.op.input_tensors[0]
                    X_T = C.op.input_tensors[1]
                    print(f"    A shape: {A.shape}, name: {getattr(A.op, 'name', 'N/A')}")
                    print(f"    X_T shape: {X_T.shape}, name: {getattr(X_T.op, 'name', 'N/A')}")
                    if hasattr(X_T.op, 'input_tensors') and len(X_T.op.input_tensors) > 0:
                        X = X_T.op.input_tensors[0]
                        print(f"    X shape: {X.shape}, name: {getattr(X.op, 'name', 'N/A')}")
            except:
                pass
        
        # Use Tensor Core schedule WITHOUT activation fusion
        # Activation will be applied separately in Relay layer after Tensor Core computation
        use_relu = False
        
        schedule_dtc_spmm_tensorcore(s, C, use_relu=use_relu)
        if _verbose:
            print(f"  [DTC Schedule] ✓ Successfully applied schedule_dtc_spmm_tensorcore (use_relu={use_relu})")
        return s
    except Exception as e:
        # If schedule fails, raise error - no fallback
        import traceback
        error_msg = str(e)
        error_tb = traceback.format_exc()
        print(f"  [ERROR] schedule_dtc_spmm_tensorcore failed: {error_msg}")
        print(f"  Error details: {error_tb[:800]}")
        raise RuntimeError(f"DTC format requires schedule_dtc_spmm_tensorcore. No fallback allowed. Error: {e}")


def sparse_dense_tensorcore_compute(attrs, inputs, out_type):
    """
    Compute function for sparse_dense operation using Tensor Core
    
    IMPORTANT: For DTC format, this should use dtc_spmm_tensorcore_cuda from spmm_mma.py
    However, in Relay, we only have access to CSR format (data, indices, indptr).
    The DTC format's A_compressed is not directly accessible here.
    
    For now, we use standard sparse_dense compute, but the schedule should use
    schedule_dtc_spmm_tensorcore from spmm_mma.py when DTC format is detected.
    
    TODO: Implement a way to pass A_compressed from DTC format to this compute function,
    or use alter_op_layout to convert sparse_dense to use A_compressed directly.
    """
    data, weight_data, weight_indices, weight_indptr = inputs
    out_dtype = out_type.dtype
    
    # NOTE: In DTC format, we have A_compressed but sparse_dense only receives CSR format
    # The ideal solution would be to:
    # 1. Use alter_op_layout to detect DTC format and convert to use A_compressed
    # 2. Or modify this compute to detect DTC format and use dtc_spmm_tensorcore_cuda
    # 
    # For now, we use standard sparse_dense compute
    # The schedule (schedule_sparse_dense_tensorcore_wrapper) should use 
    # schedule_dtc_spmm_tensorcore from spmm_mma.py when DTC format is detected
    
    # TODO: Implement DTC format detection and use dtc_spmm_tensorcore_cuda
    # This requires access to A_compressed, which is not in the inputs
    # Possible solutions:
    # 1. Use alter_op_layout to replace sparse_dense with a custom op that uses A_compressed
    # 2. Store A_compressed in a global registry and access it here
    # 3. Modify Relay to pass A_compressed as an additional input
    
    return topi.nn.sparse_dense(data, weight_data, weight_indices, weight_indptr, out_dtype=out_dtype)


def schedule_sparse_dense_tensorcore_wrapper(attrs, outs, target):
    """
    Schedule wrapper for sparse_dense Tensor Core
    
    For DTC format: uses schedule_dtc_spmm_tensorcore from spmm_mma.py
    For CSR format: uses standard schedule
    """
    # Try to use DTC-SpMM Tensor Core schedule from spmm_mma.py
    # This requires the compute to be in DTC format
    # For now, we use standard schedule
    # TODO: Integrate schedule_dtc_spmm_tensorcore when DTC format is detected
    
    # Import dtc_spmm schedule
    try:
        from .spmm_mma import schedule_dtc_spmm_tensorcore
        # Check if we can use DTC schedule (requires DTC format detection)
        # For now, fall back to standard schedule
        return topi.generic.schedule_sparse_dense(attrs, outs, target)
    except ImportError:
        # Fall back to standard schedule if spmm_mma is not available
        return topi.generic.schedule_sparse_dense(attrs, outs, target)


# Global flag to track if strategies are already registered
_tensorcore_strategies_registered = False

# Global flag to force Tensor Core usage (even if requirements are not fully met)
_force_tensorcore = False


def set_force_tensorcore(force=True):
    """
    Set whether to force Tensor Core usage even if requirements are not fully met
    
    Parameters
    ----------
    force: bool
        If True, force Tensor Core usage even when dimensions don't meet strict requirements.
        The implementation will pad dimensions as needed.
        If False, only use Tensor Core when all requirements are met.
    """
    global _force_tensorcore
    _force_tensorcore = force
    if force:
        print("✓ Force Tensor Core mode enabled (will pad dimensions if needed)")
    else:
        print("✓ Force Tensor Core mode disabled (strict requirement checking)")


def get_force_tensorcore():
    """Get current force Tensor Core setting"""
    return _force_tensorcore


def check_tensorcore_usage(lib, verbose=True):
    """
    Check if Tensor Core intrinsics are used in the compiled code
    
    Parameters
    ----------
    lib: tvm.runtime.Module
        Compiled TVM module
    verbose: bool
        If True, print detailed information
    
    Returns
    -------
    bool: True if Tensor Core intrinsics are found, False otherwise
    """
    try:
        source_code = lib.get_lib().get_source()
        source_lower = source_code.lower()
        
        # Check for Tensor Core intrinsics
        has_wmma = "wmma" in source_lower
        has_mma_sync = "mma.sync" in source_lower
        has_mma = "mma" in source_lower and "sync" in source_lower
        
        tensorcore_used = has_wmma or has_mma_sync or has_mma
        
        if verbose:
            if tensorcore_used:
                print("✓ Tensor Core intrinsics detected in generated code!")
                if has_wmma:
                    print("  - Found 'wmma' instructions")
                if has_mma_sync:
                    print("  - Found 'mma.sync' instructions")
            else:
                print("⚠ Tensor Core intrinsics NOT found in generated code")
                print("  - This may indicate fallback to standard CUDA cores")
                print("  - Check if dimensions meet requirements or enable force mode")
        
        return tensorcore_used
        
    except Exception as e:
        if verbose:
            print(f"⚠ Could not inspect source code: {e}")
        return False


def register_tensorcore_strategies(force_tensorcore=None):
    """
    Register Tensor Core strategies for dense and sparse_dense operations
    
    This function should be called before compiling Relay models to enable
    Tensor Core implementations for CUDA targets.
    
    Parameters
    ----------
    force_tensorcore: bool, optional
        If True, force Tensor Core usage even if requirements are not fully met.
        If None, uses the global setting from set_force_tensorcore().
    
    Note: This function can be called multiple times safely (idempotent).
    """
    global _tensorcore_strategies_registered, _force_tensorcore
    
    # Update force flag if provided
    if force_tensorcore is not None:
        _force_tensorcore = force_tensorcore
    
    # Check if already registered
    if _tensorcore_strategies_registered:
        return
    
    from tvm.relay.op.strategy import dense_strategy, sparse_dense_strategy
    
    try:
        # Register dense Tensor Core strategy
        @dense_strategy.register(["cuda", "gpu"], override=True)
        def dense_strategy_tensorcore(attrs, inputs, out_type, target):
            """Dense Tensor Core strategy for CUDA"""
            strategy = _op.OpStrategy()
            
            data, weight = inputs
            M, K = get_const_tuple(data.shape)
            N, _ = get_const_tuple(weight.shape)
            
            # Check if Tensor Core can be used
            meets_requirements = _check_tensorcore_requirements(data.shape, weight.shape, out_type.dtype)
            is_float16 = (data.dtype == "float16" and weight.dtype == "float16")
            
            # Use Tensor Core if:
            # 1. Requirements are met AND inputs are float16, OR
            # 2. Force mode is enabled AND inputs are float16 (force mode only works with float16)
            # 
            # IMPORTANT: Tensor Core requires float16. Even in force mode, if inputs are not float16,
            # we should fallback to standard CUDA strategy. This ensures CUDA Core tests (float32)
            # don't try to use Tensor Core.
            use_tensorcore = (meets_requirements and is_float16) or (_force_tensorcore and is_float16)
            
            if use_tensorcore:
                # Add Tensor Core implementation with high priority
                if _force_tensorcore and not (meets_requirements and is_float16):
                    # Force mode: use very high priority to ensure it's selected
                    # In force mode, we ONLY add Tensor Core implementation (no fallback)
                    strategy.add_implementation(
                        wrap_compute_dense(dense_tensorcore_compute),
                        wrap_topi_schedule(schedule_dense_tensorcore_wrapper),
                        name="dense_tensorcore.cuda.forced",
                        plevel=30,  # Very high priority for forced mode (higher than any fallback)
                    )
                    # In force mode, do NOT add fallback implementations
                    # This ensures Tensor Core is always used
                    return strategy
                else:
                    # Normal mode: high priority, but allow fallback
                    strategy.add_implementation(
                        wrap_compute_dense(dense_tensorcore_compute),
                        wrap_topi_schedule(schedule_dense_tensorcore_wrapper),
                        name="dense_tensorcore.cuda",
                        plevel=20,  # Higher priority than default (10)
                    )
            
            # Add default implementation as fallback (only if not in force mode)
            # In force mode, we want to use Tensor Core exclusively
            if not _force_tensorcore:
                # Use the same approach as TVM's default CUDA dense strategy
                # Check if we can use large batch schedule
                with SpecializedCondition(M >= 32):
                    strategy.add_implementation(
                        wrap_compute_dense(topi.gpu.dense_large_batch),
                        wrap_topi_schedule(topi.gpu.schedule_dense_large_batch),
                        name="dense_large_batch.gpu",
                        plevel=5,
                    )
                
                # Default: use small batch schedule
                strategy.add_implementation(
                    wrap_compute_dense(topi.gpu.dense_small_batch),
                    wrap_topi_schedule(topi.gpu.schedule_dense_small_batch),
                    name="dense_small_batch.gpu",
                    plevel=10,
                )
            
            return strategy
        
        # Register sparse_dense Tensor Core strategy
        @sparse_dense_strategy.register(["cuda", "gpu"], override=True)
        def sparse_dense_strategy_tensorcore(attrs, inputs, out_type, target):
            """Sparse_dense Tensor Core strategy for CUDA"""
            strategy = _op.OpStrategy()
            
            # For now, use standard implementation
            # TODO: Add full DTC-SpMM implementation when compute function is ready
            if _force_tensorcore:
                # Force mode: only add Tensor Core implementation (no fallback)
                strategy.add_implementation(
                    wrap_compute_sparse_dense(sparse_dense_tensorcore_compute),
                    wrap_topi_schedule(schedule_sparse_dense_tensorcore_wrapper),
                    name="sparse_dense_tensorcore.cuda.forced",
                    plevel=30,  # Very high priority for forced mode
                )
                return strategy
            else:
                # Normal mode: add Tensor Core with fallback
                strategy.add_implementation(
                    wrap_compute_sparse_dense(sparse_dense_tensorcore_compute),
                    wrap_topi_schedule(schedule_sparse_dense_tensorcore_wrapper),
                    name="sparse_dense_tensorcore.cuda",
                    plevel=15,  # Higher priority than default (10)
                )
                
                # Add default implementation as fallback
                strategy.add_implementation(
                    wrap_compute_sparse_dense(topi.nn.sparse_dense),
                    wrap_topi_schedule(topi.generic.schedule_sparse_dense),
                    name="sparse_dense.generic",
                    plevel=10,
                )
            
            return strategy
        
        _tensorcore_strategies_registered = True
        force_status = " (FORCED)" if _force_tensorcore else ""
        print(f"✓ Registered Tensor Core strategies for dense and sparse_dense operations{force_status}")
        
    except Exception as e:
        print(f"⚠️  Warning: Failed to register Tensor Core strategies: {e}")
        print("   Falling back to standard implementations")
        raise


if __name__ == "__main__":
    register_tensorcore_strategies()
    print("Tensor Core strategies registered successfully!")
