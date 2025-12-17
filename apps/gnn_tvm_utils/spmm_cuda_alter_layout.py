"""
Custom alter_op_layout for sparse_dense to fix block size issue and float16 support

This module registers a custom alter_op_layout function that automatically
replaces sparse_dense with sparse_dense_padded when using CUDA, which has
better block size handling and avoids the block size bug.

Additionally, this module provides a fixed version of sparse_dense_tir that
supports float16 by fixing the dtype mismatch bug in TVM 0.10.0.

This uses TVM's built-in alter_op_layout mechanism, which is automatically
applied during relay.build().

IMPORTANT: This registration ONLY affects CUDA/GPU targets, not CPU.

Bug Fix:
- TVM 0.10.0 has a bug in sparse_dense_tir where it uses hardcoded 0.0 (float32)
  instead of dtype-aware constant, causing ValueError for float16.
- Fixed by using tvm.tir.const(0, data.dtype) instead of 0.0
"""

import sys
import os
import numpy as np  

# Add TVM path
tvm_python_path = "/home/nas/polin/cmu-berlin/tvm-0.10.0/python"
if tvm_python_path not in sys.path:
    sys.path.insert(0, tvm_python_path)

# Lazy import to avoid affecting CPU tests
# Only import when actually needed (for CUDA)


def register_spmm_cuda_alter_layout():
    """
    Register custom alter_op_layout for sparse_dense to use sparse_dense_padded
    when possible, which avoids the block size bug.
    
    This function should be called ONLY when building CUDA models.
    It does NOT affect CPU targets.
    
    Note: sparse_dense_alter_layout is a generic function in topi.nn,
    and we register a CUDA-specific implementation here.
    
    IMPORTANT: All imports are lazy (inside the function) to avoid
    affecting CPU tests when this module is imported.
    """
    # Lazy import to avoid affecting CPU tests
    # Only import when actually registering (for CUDA)
    import tvm
    from tvm import relay
    from tvm import topi
    from tvm.topi.utils import get_const_tuple
    import scipy.sparse as sp
    import numpy as np
    from tvm.relay import transform as relay_transform
    
    # Only import CUDA-specific modules when actually needed
    # This ensures CPU tests are not affected
    try:
        from tvm.topi.cuda.sparse import (
            is_valid_for_sparse_dense_padded,
            pad_sparse_matrix
        )
    except ImportError:
        # If CUDA modules can't be imported, registration will fail
        # This is expected if CUDA is not available
        print("Warning: Could not import CUDA sparse modules")
        return
    
    # Check if already registered (optional, override=True will replace anyway)
    # We use override=True to ensure our version is used
    
    @topi.nn.sparse_dense_alter_layout.register(["cuda", "gpu"], override=True)
    def _alter_sparse_dense_layout_fixed(_attrs, inputs, _tinfos, _out_type):
        """
        Custom alter_op_layout that tries to use sparse_dense_padded when possible.
        
        This function ONLY affects CUDA/GPU targets, not CPU.
        TVM's generic_func mechanism ensures this is only called for CUDA/GPU targets.
        
        This function:
        1. Checks if sparse matrix components are Constants
        2. Checks if sparse_dense_padded is valid for the input
        3. If valid, converts to BSR and pads, then uses sparse_dense_padded
        4. Otherwise, returns None to use default implementation
        
        IMPORTANT: We force 1x1 BSR blocks to ensure consistent performance regardless
        of graph structure (clustered vs random). This prevents scipy.tobsr() from
        automatically choosing larger blocks for clustered graphs, which would cause:
        - More zero-value computation (BSR stores entire blocks including zeros)
        - Increased padding overhead
        - Lower parallel efficiency
        """
        # Double-check we're on CUDA/GPU target (shouldn't be called for CPU, but be safe)
        try:
            current_target = tvm.target.Target.current(allow_none=True)
            if current_target is None or current_target.kind.name not in ["cuda", "gpu"]:
                # Not CUDA/GPU, return None to use default (shouldn't happen due to register)
                return None
        except Exception as e:
            # If we can't check target, be safe and return None
            return None
        
        # Check if sparse matrix components are Constants (required for padding)
        if not (
            isinstance(inputs[1], relay.Constant)
            and isinstance(inputs[2], relay.Constant)
            and isinstance(inputs[3], relay.Constant)
        ):
            # Cannot pad non-constant sparse matrices, use default
            return None
        
        try:
            # Check if sparse_dense_padded is valid (same as TVM original)
            weight_data_np = inputs[1].data.numpy()
            
            # Try to check validity, catch type checking errors
            # For clustered graphs, we need more robust type checking
            is_valid = False
            try:
                # First try: use the standard check
                is_valid = is_valid_for_sparse_dense_padded(inputs[0], weight_data_np)
            except (ValueError, AttributeError, TypeError) as e:
                # Type checking failed - try manual inference
                try:
                    # Manual type inference for clustered graphs
                    # Create a temporary module to infer types
                    temp_mod = tvm.IRModule.from_expr(inputs[0])
                    inferred = relay_transform.InferType()(temp_mod)
                    if inferred["main"].checked_type.defined():
                        # Try to get shape from inferred type
                        try:
                            shape = get_const_tuple(inferred["main"].checked_type.shape)
                            if len(shape) >= 2:
                                m = shape[1]
                                warp_size = int(tvm.target.Target.current(allow_none=False).thread_warp_size)
                                bs_m = 1 if len(weight_data_np.shape) == 1 else weight_data_np.shape[1]
                                mb = m // bs_m
                                is_valid = mb >= warp_size
                        except:
                            pass
                except:
                    # If all type inference fails, skip sparse_dense_padded
                    is_valid = False
            
            if is_valid:
                # Get input dtype from multiple sources (for dtype matching)
                input_dtype_str = None
                
                # Try from tinfos first (most reliable during alter_op_layout)
                if _tinfos and len(_tinfos) > 0:
                    try:
                        input_dtype_str = str(_tinfos[0].dtype)
                    except:
                        pass
                
                # Try from checked_type
                if not input_dtype_str:
                    try:
                        if inputs[0].checked_type.defined():
                            input_dtype_str = str(inputs[0].checked_type.dtype)
                    except:
                        pass
                
                # Fallback: infer from weight_data dtype (should match input in most cases)
                if not input_dtype_str:
                    input_dtype_str = str(inputs[1].data.dtype)
                
                # Convert to numpy dtype
                if "float32" in input_dtype_str:
                    np_dtype = np.float32
                elif "float16" in input_dtype_str:
                    np_dtype = np.float16
                else:
                    np_dtype = np.float32  # Default
                
                # Convert to BSR and pad (same as TVM original, but with dtype matching)
                if len(weight_data_np.shape) == 1:
                    # CSR format - convert to BSR
                    # Ensure dtype matches input
                    weight_data_typed = weight_data_np.astype(np_dtype)
                    sparse_matrix = sp.csr_matrix(
                        (
                            weight_data_typed,
                            inputs[2].data.numpy(),
                            inputs[3].data.numpy()
                        )
                    ).tobsr(blocksize=(1, 1))  # Force 1x1 blocks to ensure consistent performance
                    # regardless of graph structure (clustered vs random)
                else:
                    # Already BSR
                    weight_data_typed = weight_data_np.astype(np_dtype)
                    sparse_matrix = sp.bsr_matrix(
                        (
                            weight_data_typed,
                            inputs[2].data.numpy(),
                            inputs[3].data.numpy()
                        )
                    )
                
                # Pad sparse matrix to warp size
                warp_size = int(tvm.target.Target.current(allow_none=False).thread_warp_size)
                sparse_matrix = pad_sparse_matrix(sparse_matrix, warp_size)
                
                # Ensure sparse matrix data dtype matches input dtype
                if sparse_matrix.data.dtype != np_dtype:
                    sparse_matrix.data = sparse_matrix.data.astype(np_dtype)
                
                # Create sparse_dense_padded (same as TVM original)
                return relay.nn._make.sparse_dense_padded(
                    inputs[0],
                    relay.Constant(tvm.nd.array(sparse_matrix.data)),
                    relay.Constant(tvm.nd.array(sparse_matrix.indices)),
                    relay.Constant(tvm.nd.array(sparse_matrix.indptr)),
                )
        except Exception as e:
            # If anything fails, fall back to default (suppress expected type checking warnings)
            error_msg = str(e)
            if "checked_type" not in error_msg and "type checker" not in error_msg.lower():
                print(f"Warning: Failed to use sparse_dense_padded: {error_msg}")
                print("Falling back to default sparse_dense implementation")
            return None
        
        # Default: use original implementation
        return None
    
    print("✓ Registered custom sparse_dense_alter_layout for CUDA")


def sparse_dense_tir_fixed(data, w_data, w_indices, w_indptr):
    """
    Fixed version of sparse_dense_tir that supports float16.
    
    This is a copy of TVM's sparse_dense_tir function with a fix for float16 support.
    The original bug is in tvm-0.10.0/python/tvm/topi/cuda/sparse.py:213 where
    it uses hardcoded 0.0 (float32) instead of dtype-aware constant.
    
    Compute data * w^T.
    
    Actually computes (w * data^T) ^ T as data needs to be in column-major
    format for performance reasons.
    
    Good resources:
    Yang, Carl, Aydın Buluç, and John D. Owens. "Design principles for sparse
    matrix multiplication on the GPU." European Conference on Parallel
    Processing. Springer, Cham, 2018. <- This code is basically row-split from here.
    Gale, Trevor, et al. "Sparse GPU Kernels for Deep Learning." arXiv preprint
    arXiv:2006.10901 (2020).
    """
    import tvm
    from tvm import te
    from tvm.topi.utils import ceil_div
    
    def gen_ir(data, w_data, w_indices, w_indptr, out):
        # pylint: disable=invalid-name, simplifiable-if-statement
        # TODO(tkonolige): use tensorcores for block multiply
        # TODO(tkonolige): use vectorize on loads
        # TODO(tkonolige): separate implementation if M is small
        # TODO(tkonolige): separate implementation for large block sizes
        ib = tvm.tir.ir_builder.create()

        if tvm.target.Target.current(allow_none=False).kind.name == "cuda":
            use_warp_storage = True
        else:
            # TVMs warp shuffle intrinsics are slow on ROCM because they use
            # LDS (shared memory) to do the shuffling. Instead, we could use
            # ROCM's support for accessing neighboring threads memory, but we
            # those intrinsics aren't accessible from TVM. For now, we just use
            # shared memory. We also default to shared memory on platforms
            # where we do not know how warp storage performs.
            use_warp_storage = False

        warp_size = int(tvm.target.Target.current(allow_none=False).thread_warp_size)
        m = data.shape[1]
        nb = w_indptr.shape[0] - 1
        # treat csr like block size 1 bsr
        if len(w_data.shape) == 1:
            bs_n = 1
            bs_k = 1
        else:
            bs_n = w_data.shape[1]
            bs_k = w_data.shape[2]
        bs_m = bs_n
        mb = m // bs_m
        mi = warp_size
        assert (
            mb >= mi
        ), "Number of block rows in dense matrix must be larger than warp size: {} vs {}.".format(
            warp_size, mb
        )
        mo = ceil_div(mb, mi)
        ni = 1  # TODO(tkonolige): how do I compute the number of warps per block?
        no = ceil_div(nb, ni)
        rowlength_bi = warp_size

        bx = te.thread_axis("blockIdx.x")
        ib.scope_attr(bx, "thread_extent", mo)
        by = te.thread_axis("blockIdx.y")
        ib.scope_attr(by, "thread_extent", no)
        tx = te.thread_axis("threadIdx.x")
        ib.scope_attr(tx, "thread_extent", warp_size)
        warp = te.thread_axis("threadIdx.y")
        ib.scope_attr(warp, "thread_extent", ni)

        out_ptr = ib.buffer_ptr(out)
        data_ptr = ib.buffer_ptr(data)
        w_data_ptr = ib.buffer_ptr(w_data)
        w_indices_ptr = ib.buffer_ptr(w_indices)
        w_indptr_ptr = ib.buffer_ptr(w_indptr)

        n_index = by * ni + warp
        m_index = bx * mi + tx
        row_start = w_indptr_ptr[n_index]

        # Guaranteed to be evenly divisible
        rowlength_bo = ceil_div(w_indptr_ptr[n_index + 1] - row_start, rowlength_bi)

        # thread local storage for bs_m x bs_n block
        block = ib.allocate(data.dtype, (bs_m, bs_n), name="block", scope="local")
        data_cache = ib.allocate(data.dtype, (mi, bs_m, bs_k), name="data_cache", scope="local")
        if use_warp_storage:
            indices = ib.allocate(w_indices.dtype, (rowlength_bi,), name="indices", scope="warp")
            w_data_cache = ib.allocate(
                w_data.dtype, (rowlength_bi, bs_n, bs_k), name="w_data_cache", scope="warp"
            )
        else:
            indices = ib.allocate(
                w_indices.dtype, (ni, rowlength_bi), name="indices", scope="shared"
            )
            w_data_cache = ib.allocate(
                w_data.dtype, (ni, rowlength_bi, bs_n, bs_k), name="w_data_cache", scope="shared"
            )

        # zero block - FIXED: use dtype-aware constant instead of hardcoded 0.0
        zero = tvm.tir.const(0, data.dtype)
        with ib.for_range(0, bs_m, name="x", kind="unroll") as x:
            with ib.for_range(0, bs_n, name="y", kind="unroll") as y:
                block[x, y] = zero  # Fixed: was 0.0 (float32), now uses correct dtype
        
        # compute into thread local storage using warp_size chunks
        with ib.for_range(0, rowlength_bo, name="bb") as bb:
            elem_idx = bb * rowlength_bi + tx
            # Cache indices. Guaranteed to be multiple of warp_size.
            if use_warp_storage:
                indices[tx] = w_indices_ptr[row_start + elem_idx]
            else:
                indices[warp, tx] = w_indices_ptr[row_start + elem_idx]
            # cache dense matrix
            # each thread has a row
            # TODO: ideally we could vectorize this
            with ib.for_range(0, rowlength_bi, name="bi") as bi:
                with ib.for_range(0, bs_m, name="x", kind="unroll") as x:
                    with ib.for_range(0, bs_k, name="z", kind="unroll") as z:
                        # This memory acces should be out of bounds when
                        # m_index >= mb (which occurs when the dense matrix
                        # rows % 32 != 0), but it seems to work just fine...
                        if use_warp_storage:
                            ind = indices[bi]
                        else:
                            ind = indices[warp, bi]
                        data_cache[bi, x, z] = data_ptr[ind * bs_k + z, m_index * bs_m + x]
            # cache w_data
            elem_idx = bb * rowlength_bi + tx
            with ib.for_range(0, bs_n, name="y", kind="unroll") as y:
                with ib.for_range(0, bs_k, name="z", kind="unroll") as z:
                    data_indices = [row_start + elem_idx] + (
                        [y, z] if len(w_data.shape) > 1 else []
                    )
                    cache_indices = [tx, y, z] if use_warp_storage else [warp, tx, y, z]
                    w_data_cache[cache_indices] = w_data_ptr[data_indices]
            with ib.for_range(0, mi, name="i") as i:
                # thread local block matmul
                with ib.for_range(0, bs_m, name="x", kind="unroll") as x:
                    with ib.for_range(0, bs_n, name="y", kind="unroll") as y:
                        with ib.for_range(0, bs_k, name="z", kind="unroll") as z:
                            if use_warp_storage:
                                w = w_data_cache[i, y, z]
                            else:
                                w = w_data_cache[warp, i, y, z]
                            block[x, y] += data_cache[i, x, z] * w
        # store results
        with ib.for_range(0, bs_m, name="x", kind="unroll") as x:
            with ib.for_range(0, bs_n, name="y", kind="unroll") as y:
                with ib.if_scope(m_index < mb):
                    with ib.if_scope(n_index < nb):
                        # It doesn't seem like we would be getting coelesced
                        # writes here, but it doesn't seem to matter
                        out_ptr[m_index * bs_m + x, n_index * bs_n + y] = block[x, y]

        return ib.get()

    data_t = tvm.topi.transpose(data)
    # handle csr
    if len(w_data.shape) == 1:
        blocksize = 1
    else:
        blocksize = w_data.shape[1]
    out_shape = (data_t.shape[1], (w_indptr.shape[0] - 1) * blocksize)
    out_buf = tvm.tir.decl_buffer(out_shape, data.dtype, "out_buf")
    out = te.extern(
        [out_shape],
        [data_t, w_data, w_indices, w_indptr, data],
        lambda ins, outs: gen_ir(ins[0], ins[1], ins[2], ins[3], outs[0]),
        dtype=data.dtype,
        out_buffers=[out_buf],
        name="sparse_dense_gpu",
        tag="sparse_dense_gpu",
    )
    return out


def register_fixed_sparse_dense_tir():
    """
    Register the fixed sparse_dense_tir function to override TVM's buggy version.
    
    This patches TVM's sparse_dense_tir to use our fixed version that supports float16.
    The original bug is in tvm-0.10.0/python/tvm/topi/cuda/sparse.py:213 where
    it uses hardcoded 0.0 (float32) instead of dtype-aware constant.
    """
    import tvm
    from tvm import topi
    
    # Patch the sparse_dense_tir function directly
    # sparse_dense_padded calls sparse_dense_tir, so replacing sparse_dense_tir
    # will fix both sparse_dense_padded and any other code that uses sparse_dense_tir
    topi.cuda.sparse.sparse_dense_tir = sparse_dense_tir_fixed
    
    # print("✓ Registered fixed sparse_dense_tir (supports float16)")


if __name__ == "__main__":
    register_spmm_cuda_alter_layout()
    register_fixed_sparse_dense_tir()
    print("Custom sparse_dense_alter_layout and fixed sparse_dense_tir registered successfully!")

