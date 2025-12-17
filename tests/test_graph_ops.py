import sys
sys.path.append("./python")

import numpy as np
import pytest

import needle as ndl


_DEVICES = [ndl.cpu(), pytest.param(ndl.cuda(),
    marks=pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU"))]


def _reference_aggregate(features, src, dst):
    out = np.zeros_like(features)
    for s, d in zip(src, dst):
        out[d] += features[s]
    return out


@pytest.mark.parametrize("num_nodes", [5, 12])
@pytest.mark.parametrize("feature_dim", [3, 7])
@pytest.mark.parametrize("num_edges", [20])
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_graph_neighbor_sum_forward(num_nodes, feature_dim, num_edges, device):
    rng = np.random.default_rng(42)
    features = rng.standard_normal((num_nodes, feature_dim), dtype=np.float32)
    src = rng.integers(0, num_nodes, size=(num_edges,), dtype=np.int32)
    dst = rng.integers(0, num_nodes, size=(num_edges,), dtype=np.int32)

    ndl_features = ndl.Tensor(features, device=device)
    ndl_src = ndl.Tensor(src.astype(np.float32), device=device)
    ndl_dst = ndl.Tensor(dst.astype(np.float32), device=device)
    out = ndl.ops.graph_neighbor_sum(ndl_features, ndl_src, ndl_dst).numpy()
    expected = _reference_aggregate(features, src, dst)
    np.testing.assert_allclose(out, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("num_nodes", [6])
@pytest.mark.parametrize("feature_dim", [4])
@pytest.mark.parametrize("num_edges", [15])
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_graph_neighbor_sum_backward(num_nodes, feature_dim, num_edges, device):
    rng = np.random.default_rng(7)
    features = rng.standard_normal((num_nodes, feature_dim), dtype=np.float32)
    src = rng.integers(0, num_nodes, size=(num_edges,), dtype=np.int32)
    dst = rng.integers(0, num_nodes, size=(num_edges,), dtype=np.int32)
    upstream = rng.standard_normal((num_nodes, feature_dim), dtype=np.float32)

    ndl_features = ndl.Tensor(features, device=device, requires_grad=True)
    ndl_src = ndl.Tensor(src.astype(np.float32), device=device)
    ndl_dst = ndl.Tensor(dst.astype(np.float32), device=device)
    ndl_upstream = ndl.Tensor(upstream, device=device)

    out = ndl.ops.graph_neighbor_sum(ndl_features, ndl_src, ndl_dst)
    loss = ndl.ops.summation(out * ndl_upstream)
    loss.backward()

    grad = ndl_features.grad.numpy()
    expected_grad = np.zeros_like(features)
    for edge_src, edge_dst in zip(src, dst):
        expected_grad[edge_src] += upstream[edge_dst]

    np.testing.assert_allclose(grad, expected_grad, atol=1e-6, rtol=1e-6)
