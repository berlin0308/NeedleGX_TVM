import os
import numpy as np
import tvm
import tvm.testing
from tvm import te, auto_scheduler, runtime, topi
from tvm.auto_scheduler import _ffi_api
from tvm.topi.utils import get_const_tuple
from tvm.topi.sparse.utils import random_bsr_matrix
from tvm.topi.cuda import sparse as sparse_cuda
from tvm.topi.cuda.tensor_intrin import (
    intrin_wmma_load_matrix_A,
    intrin_wmma_load_matrix_W,
    intrin_wmma_store_matrix,
    intrin_wmma_gemm,
)

M = 512
K = 512
N = 512
BS_R = 16
BS_C = 1
density = 0.2

# Generate the test data with numpy
X_np = np.random.randn(M, K).astype("float32")
# scipy.sparse doesn't support float16, generate as float32 then convert
W_sp_np = random_bsr_matrix(N, K, BS_R, BS_C, density=density, dtype="float32")
W_np = W_sp_np.todense()
Y_np = (X_np @ W_np.T).astype("float32")

# Convert to float16 for TensorCore (will be done in the computation definition)
X_np_fp16 = X_np.astype("float16")
W_data_np_fp16 = W_sp_np.data.astype("float16")

dev = tvm.cuda(1)
target = tvm.target.Target("cuda")


######################################################################
# Dense MatMul using TensorCore
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Implement dense matrix multiplication using TensorCore with 16x16x16 configuration


def dense_tensorcore_cuda(data, weight, bias=None, out_dtype=None):
    """Dense tensorcore operator on CUDA"""
    assert len(data.shape) == 2 and len(weight.shape) == 2, "only support 2-dim dense"
    if bias is not None:
        assert len(bias.shape) == 1
    if out_dtype is None:
        out_dtype = data.dtype
    batch, in_dim = get_const_tuple(data.shape)
    out_dim, _ = get_const_tuple(weight.shape)

    assert data.dtype == weight.dtype
    assert data.dtype in ["float16", "int8", "uint8", "int4", "uint4"]
    if data.dtype in ["float16", "int8", "uint8"]:
        assert (
            (batch % 8 == 0 and in_dim % 16 == 0 and out_dim % 32 == 0)
            or (batch % 16 == 0 and in_dim % 16 == 0 and out_dim % 16 == 0)
            or (batch % 32 == 0 and in_dim % 16 == 0 and out_dim % 8 == 0)
        ), (
            "The shape of (batch, in_dim, out_dim) "
            "must be multiple of (16, 16, 16) or (32, 16, 8) or (8, 16, 32) for now"
        )
    else:
        assert (
            batch % 8 == 0 and in_dim % 32 == 0 and out_dim % 8 == 0
        ), "The shape of (batch, in_dim, out_dim) must be multiple of (8, 32, 8)"

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


def schedule_dense_tensorcore(s, C):
    """Schedule dense operator using TensorCore with fixed 16x16x16 configuration"""
    A, B = s[C].op.input_tensors
    if len(B.op.input_tensors) == 1 and B.op.input_tensors[0] == A:
        s[B].compute_inline()
    batch, out_dim = get_const_tuple(C.shape)
    data_dtype = A.dtype
    out_dtype = C.dtype

    # Explicit memory access
    AS = s.cache_read(A, "shared", [C])
    BS = s.cache_read(B, "shared", [C])
    AF = s.cache_read(AS, "wmma.matrix_a", [C])
    BF = s.cache_read(BS, "wmma.matrix_b", [C])
    CF = s.cache_write(C, "wmma.accumulator")
    CS = s.cache_read(CF, "shared", [C])

    # Deal with op fusion, such as bias and relu
    if C.op not in s.outputs:
        s[C].compute_inline()
        C = s.outputs[0].output(0)

    # Fixed configuration for 16x16x16 TensorCore
    wmma_m = 16
    wmma_n = 16
    wmma_k = 16
    
    # Fixed tuning parameters
    block_row_warps = 2
    block_col_warps = 2
    warp_row_tiles = 2
    warp_col_tiles = 2
    chunk = 4
    offset = 8
    offsetCS = 8
    vec = 4

    warp_size = 32

    # Define the stride of intrin functions
    AS_align = chunk * wmma_k + offset
    BS_align = chunk * wmma_k + offset
    CS_align = warp_col_tiles * block_col_warps * wmma_n + offsetCS
    AS_stride = [AS_align, 1]
    BS_stride = [BS_align, 1]
    AF_stride = [wmma_k, 1]
    BF_stride = [wmma_k, 1]
    CF_stride = [warp_col_tiles * wmma_n, 1]
    CS_stride = [CS_align, 1]

    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    thread_x = te.thread_axis("threadIdx.x")
    thread_y = te.thread_axis("threadIdx.y")
    thread_z = te.thread_axis("threadIdx.z")

    # Schedule for dense computation
    block_factor_b = wmma_m * warp_row_tiles * block_row_warps
    block_factor_o = wmma_n * warp_col_tiles * block_col_warps
    b, o = C.op.axis
    block_i, bc = s[C].split(b, factor=block_factor_b)
    block_j, oc = s[C].split(o, factor=block_factor_o)
    s[C].reorder(block_i, block_j, bc, oc)
    t = s[C].fuse(bc, oc)
    t, vi = s[C].split(t, factor=vec)
    t, tx = s[C].split(t, factor=warp_size)
    t, ty = s[C].split(t, factor=block_row_warps)
    t, tz = s[C].split(t, factor=block_col_warps)
    s[C].bind(block_i, block_x)
    s[C].bind(block_j, block_y)
    s[C].bind(tz, thread_z)
    s[C].bind(ty, thread_y)
    s[C].bind(tx, thread_x)
    s[C].vectorize(vi)

    # Schedule for wmma store
    s[CS].compute_at(s[C], block_j)
    bb, oo = CS.op.axis
    s[CS].storage_align(bb, CS_align - 1, CS_align)
    bb, bbi = s[CS].split(bb, factor=wmma_m)
    oo, ooi = s[CS].split(oo, factor=wmma_n)
    bb, bbii = s[CS].split(bb, factor=warp_row_tiles)
    oo, ooii = s[CS].split(oo, factor=warp_col_tiles)
    s[CS].reorder(bb, oo, bbii, ooii, bbi, ooi)
    s[CS].bind(bb, thread_y)
    s[CS].bind(oo, thread_z)

    # Schedule for wmma computation
    s[CF].compute_at(s[CS], oo)
    warp_i, warp_j = CF.op.axis
    warp_i, _ii = s[CF].split(warp_i, factor=wmma_m)
    warp_j, _jj = s[CF].split(warp_j, factor=wmma_n)
    (k,) = CF.op.reduce_axis
    k, _k = s[CF].split(k, factor=wmma_k)
    ko, ki = s[CF].split(k, factor=chunk)
    s[CF].reorder(ko, ki, warp_i, warp_j, _ii, _jj, _k)

    # Schedule for wmma_matrix_a load
    s[AF].compute_at(s[CF], ki)
    b, i = AF.op.axis
    b, b_ii = s[AF].split(b, factor=wmma_m)
    i, i_jj = s[AF].split(i, factor=wmma_k)
    s[AF].reorder(b, i, b_ii, i_jj)

    # Schedule for wmma_matrix_b load
    s[BF].compute_at(s[CF], ki)
    o, i = BF.op.axis
    o, o_ii = s[BF].split(o, factor=wmma_n)
    i, i_ii = s[BF].split(i, factor=wmma_k)
    s[BF].reorder(o, i, o_ii, i_ii)

    # Schedule for A's(B's) shared memory load
    def shared_schedule(stage, strides):
        s[stage].compute_at(s[CF], ko)
        xo, yo = stage.op.axis
        s[stage].storage_align(xo, strides - 1, strides)
        t = s[stage].fuse(xo, yo)
        t, vi = s[stage].split(t, factor=vec)
        t, tx = s[stage].split(t, factor=warp_size)
        t, ty = s[stage].split(t, factor=block_row_warps)
        _, tz = s[stage].split(t, factor=block_col_warps)
        s[stage].bind(ty, thread_y)
        s[stage].bind(tz, thread_z)
        s[stage].bind(tx, thread_x)
        s[stage].vectorize(vi)

    shared_schedule(AS, AS_align)
    shared_schedule(BS, BS_align)

    shape = (wmma_m, wmma_n, wmma_k)
    AL_gemm = te.placeholder((wmma_m, wmma_k), name="AL_gemm", dtype=data_dtype)
    BL_gemm = te.placeholder((wmma_n, wmma_k), name="BL_gemm", dtype=data_dtype)
    k_gemm = te.reduce_axis((0, wmma_k), name="k_gemm")
    CL_compute = te.compute(
        (wmma_m, wmma_n),
        lambda ii, jj: te.sum(
            AL_gemm[ii, k_gemm].astype(out_dtype) * BL_gemm[jj, k_gemm].astype(out_dtype),
            axis=k_gemm,
        ),
        name="CL_compute",
    )

    # lower the computation loops down to TensorCore hardware intrinsics
    # by mapping the dense tensorcore to tensor intrinsics
    s[AF].tensorize(
        b_ii,
        intrin_wmma_load_matrix_A(
            AF_stride, AS_stride, shape, "row_major", (wmma_m, wmma_k), (wmma_m, wmma_k), data_dtype
        ),
    )
    s[BF].tensorize(
        o_ii,
        intrin_wmma_load_matrix_W(
            BF_stride, BS_stride, shape, "col_major", (wmma_n, wmma_k), (wmma_n, wmma_k), data_dtype
        ),
    )
    s[CF].tensorize(
        _ii, intrin_wmma_gemm(AL_gemm, BL_gemm, CL_compute, AF_stride, BF_stride, CF_stride, shape)
    )
    s[CS].tensorize(
        bbi,
        intrin_wmma_store_matrix(
            CS_stride, CF_stride, shape, out_dtype, (wmma_m, wmma_n), (wmma_m, wmma_n)
        ),
    )


######################################################################
# Benchmark Dense MatMul
# ^^^^^^^^^^^^^^^^^^^^^^
if __name__ == "__main__":
    
    dtype = "float16"
    
    # Generate test data
    X_np = np.random.randn(M, K).astype(dtype)
    W_np = np.random.randn(N, K).astype(dtype)
    Y_np = (X_np @ W_np.T).astype(dtype)
    
    # Define computation
    X = te.placeholder(shape=(M, K), dtype=dtype, name="X")
    W = te.placeholder(shape=(N, K), dtype=dtype, name="W")
    Y = dense_tensorcore_cuda(X, W, bias=None, out_dtype=dtype)
    
    # Create schedule
    s = te.create_schedule(Y.op)
    schedule_dense_tensorcore(s, Y)
    
    # Build the function
    func = tvm.build(s, [X, W, Y], target=target, name="dense_tensorcore")
    
    # Allocate tensors on GPU
    X_tvm = tvm.nd.array(X_np, device=dev)
    W_tvm = tvm.nd.array(W_np, device=dev)
    Y_tvm = tvm.nd.array(np.zeros((M, N), dtype=dtype), device=dev)
    
    # Warm up
    func(X_tvm, W_tvm, Y_tvm)
    dev.sync()
    
    # Benchmark
    evaluator = func.time_evaluator(func.entry_name, dev, number=100, repeat=10)
    mean_time = evaluator(X_tvm, W_tvm, Y_tvm).mean
    
    # Verify correctness
    # For float16, TensorCore uses float32 accumulator but outputs float16
    # So we need larger tolerance
    Y_np_tvm = Y_tvm.numpy()
    
    # Calculate error statistics
    diff = np.abs(Y_np.astype("float32") - Y_np_tvm.astype("float32"))
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    print(f"  Max absolute difference: {max_diff:.6f}")
    print(f"  Mean absolute difference: {mean_diff:.6f}")
    
    # Convert to float32 for more accurate comparison
    # Use larger tolerance for float16
    Y_np_f32 = Y_np.astype("float32")
    Y_np_tvm_f32 = Y_np_tvm.astype("float32")
    try:
        tvm.testing.assert_allclose(Y_np_f32, Y_np_tvm_f32, rtol=0.15, atol=0.15)
        print("  Correctness check: PASSED")
    except AssertionError as e:
        print(f"  Correctness check: FAILED")
        print(f"  Error: {str(e)[:200]}...")
    
    # Calculate GFLOPS for TVM TensorCore
    # Why 2 * M * N * K?
    # Matrix multiplication C = A @ B^T where A is (M, K) and B is (N, K)
    # - Each output element C[i,j] requires K multiply-accumulate (MAC) operations
    # - Each MAC = 1 multiplication + 1 addition = 2 floating point operations
    # - Total output elements: M * N
    # - Total FLOPS = M * N * K * 2 = 2 * M * N * K
    gflops_tvm = (2.0 * M * N * K) / (mean_time * 1e9)
    
    # Benchmark NumPy (CPU)
    import time
    # Warm up
    _ = X_np @ W_np.T
    # Benchmark - only 10 iterations
    times_np = []
    for _ in range(10):
        start = time.perf_counter()
        Y_np_ref = X_np @ W_np.T
        end = time.perf_counter()
        times_np.append(end - start)
    mean_time_np = np.mean(times_np)
    gflops_np = (2.0 * M * N * K) / (mean_time_np * 1e9)
    
    # Print results
    print("\n" + "="*60)
    print("Dense MatMul Performance Comparison")
    print("="*60)
    print(f"Shape: ({M}, {K}) x ({N}, {K})^T = ({M}, {N})")
    print(f"DataType: {dtype}")
    print("-"*60)
    print(f"TVM TensorCore (GPU):")
    print(f"  TensorCore: 16x16x16")
    print(f"  Mean time: {mean_time*1000:.4f} ms")
    print(f"  Performance: {gflops_tvm:.2f} GFLOPS")
    print("-"*60)
    print(f"NumPy (CPU):")
    print(f"  Mean time: {mean_time_np*1000:.4f} ms")
    print(f"  Performance: {gflops_np:.2f} GFLOPS")
    print("-"*60)
    speedup = mean_time_np / mean_time
    print(f"Speedup: {speedup:.2f}x")
    print("="*60)

