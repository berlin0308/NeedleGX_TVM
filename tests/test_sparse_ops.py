import numpy as np
import pytest

import needle as ndl
from needle.backend_ndarray import SparseNDArray


devices = [ndl.cpu_numpy()]
if ndl.cpu().enabled():
    devices.append(ndl.cpu())
if ndl.cuda().enabled():
    devices.append(ndl.cuda())


@pytest.mark.parametrize("device", devices)
def test_sparse_ewise_mul_matches_dense(device):
    if not device.enabled():
        pytest.skip(f"Device {device} not available.")
    shape = (2, 2)
    a = SparseNDArray([0, 1], [1, 0], [2.0, 3.0], shape, device=device)
    b = SparseNDArray([0, 0, 1], [1, 0, 1], [4.0, 5.0, 6.0], shape, device=device)
    ta = ndl.Tensor(a, device=device, requires_grad=False)
    tb = ndl.Tensor(b, device=device, requires_grad=False)

    out = ndl.ops.multiply(ta, tb)

    assert getattr(out, "is_sparse", False)
    dense_out = out.to_dense().numpy()
    expected = ta.to_dense().numpy() * tb.to_dense().numpy()
    np.testing.assert_allclose(dense_out, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("device", devices)
def test_spmm_matches_dense(device):
    if not device.enabled():
        pytest.skip(f"Device {device} not available.")
    # Sparse A (3x4) times dense B (4x2)
    A_dense = np.array(
        [[1, 0, 2, 0],
         [0, 3, 0, 0],
         [4, 0, 5, 6]], dtype=np.float32
    )
    rows, cols = np.nonzero(A_dense)
    vals = A_dense[rows, cols]
    A = SparseNDArray(rows, cols, vals, A_dense.shape, device=device)
    B = np.arange(4 * 2, dtype=np.float32).reshape(4, 2)
    tA = ndl.Tensor(A, device=device, requires_grad=False)
    tB = ndl.Tensor(B, device=device, requires_grad=False)

    out = ndl.ops.matmul(tA, tB)
    out_dense = out.to_dense().numpy() if getattr(out, "is_sparse", False) else out.numpy()
    np.testing.assert_allclose(out_dense, A_dense @ B, atol=1e-6, rtol=1e-6)


def test_spmm_cusparse_matches_dense():
    dev = ndl.cuda()
    if not dev.enabled():
        pytest.skip("CUDA not available.")
    if not getattr(dev, "__cusparse_enabled__", False):
        pytest.skip("cuSPARSE SpMM not built.")

    # Sparse A (3x4) times dense B (4x2)
    A_dense = np.array(
        [[1, 0, 2, 0],
         [0, 3, 0, 0],
         [4, 0, 5, 6]], dtype=np.float32
    )
    rows, cols = np.nonzero(A_dense)
    vals = A_dense[rows, cols]
    A = SparseNDArray(rows, cols, vals, A_dense.shape, device=dev)
    B = np.arange(4 * 2, dtype=np.float32).reshape(4, 2)
    tA = ndl.Tensor(A, device=dev, requires_grad=False)
    tB = ndl.Tensor(B, device=dev, requires_grad=False)

    # cuSPARSE path
    dev.set_use_cusparse_spmm(True)
    out = ndl.ops.matmul(tA, tB)
    out_dense = out.to_dense().numpy() if getattr(out, "is_sparse", False) else out.numpy()
    np.testing.assert_allclose(out_dense, A_dense @ B, atol=1e-6, rtol=1e-6)

    # fallback path
    dev.set_use_cusparse_spmm(False)
    out_fallback = ndl.ops.matmul(tA, tB)
    out_fallback_dense = out_fallback.to_dense().numpy() if getattr(out_fallback, "is_sparse", False) else out_fallback.numpy()
    np.testing.assert_allclose(out_fallback_dense, A_dense @ B, atol=1e-6, rtol=1e-6)

    # restore default
    dev.set_use_cusparse_spmm(True)


def test_spmm_tensor_core_matches_dense():
    dev = ndl.cuda()
    if not dev.enabled():
        pytest.skip("CUDA not available.")
    if not getattr(dev, "__cusparse_enabled__", False):
        pytest.skip("cuSPARSE SpMM not built.")
    if not getattr(dev, "__tensor_core_spmm_available__", False):
        pytest.skip("Tensor Core SpMM not available on this GPU.")

    # Sparse A (3x4) times dense B (4x2)
    A_dense = np.array(
        [[1, 0, 2, 0],
         [0, 3, 0, 0],
         [4, 0, 5, 6]], dtype=np.float32
    )
    rows, cols = np.nonzero(A_dense)
    vals = A_dense[rows, cols]
    A = SparseNDArray(rows, cols, vals, A_dense.shape, device=dev)
    B = np.arange(4 * 2, dtype=np.float32).reshape(4, 2)
    tA = ndl.Tensor(A, device=dev, requires_grad=False)
    tB = ndl.Tensor(B, device=dev, requires_grad=False)

    dev.set_use_cusparse_spmm(True)
    dev.set_use_tensor_core_spmm(True)
    out = ndl.ops.matmul(tA, tB)
    out_dense = out.to_dense().numpy() if getattr(out, "is_sparse", False) else out.numpy()
    np.testing.assert_allclose(out_dense, A_dense @ B, atol=1e-6, rtol=1e-6)

    # restore default
    dev.set_use_tensor_core_spmm(False)
