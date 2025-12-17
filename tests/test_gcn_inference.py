import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
PYTHON_DIR = os.path.join(ROOT, "python")
APPS_DIR = os.path.join(ROOT, "apps")
for path in (PYTHON_DIR, APPS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np
import torch
import pytest

import needle as ndl
from needle.backend_ndarray import SparseNDArray
from apps.models import GCNModel


devices = [ndl.cpu_numpy()]
if ndl.cpu().enabled():
    devices.append(ndl.cpu())
if ndl.cuda().enabled():
    devices.append(ndl.cuda())


def _generate_synthetic_graph(num_nodes=8, feature_dim=6, num_classes=4, device=None):
    rng = np.random.default_rng(3)
    features = rng.standard_normal((num_nodes, feature_dim)).astype(np.float32)
    labels = rng.integers(0, num_classes, size=(num_nodes,), dtype=np.int64)
    edges = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and rng.random() < 0.25:
                edges.append((i, j))
    for i in range(num_nodes):
        edges.append((i, i))
    edge_src = np.array([e[0] for e in edges], dtype=np.float32)
    edge_dst = np.array([e[1] for e in edges], dtype=np.float32)
    deg = np.zeros(num_nodes, dtype=np.float32)
    for d in edge_dst.astype(np.int64):
        deg[d] += 1
    inv_sqrt_deg = np.zeros_like(deg)
    mask = deg > 0
    inv_sqrt_deg[mask] = deg[mask] ** -0.5
    masks = {
        "train": np.ones(num_nodes, dtype=bool),
        "val": np.ones(num_nodes, dtype=bool),
        "test": np.ones(num_nodes, dtype=bool),
    }
    device = device if device is not None else ndl.cpu()
    return {
        "features": ndl.Tensor(features, dtype="float32", device=device),
        "labels": ndl.Tensor(labels.astype(np.float32), dtype="float32", device=device),
        "edge_index": (
            ndl.Tensor(edge_src, dtype="float32", device=device),
            ndl.Tensor(edge_dst, dtype="float32", device=device),
        ),
        "inv_sqrt_deg": ndl.Tensor(inv_sqrt_deg.reshape(num_nodes, 1), dtype="float32", device=device),
        "masks": masks,
    }, features, labels, edge_src, edge_dst, inv_sqrt_deg, masks


def _normalized_adjacency(edge_src, edge_dst, inv, sparse, device):
    """Build D^{-1/2} A D^{-1/2}."""
    src = edge_src.astype(np.int64)
    dst = edge_dst.astype(np.int64)
    values = inv[src] * inv[dst]
    num_nodes = inv.shape[0]
    shape = (num_nodes, num_nodes)
    if sparse:
        adj = SparseNDArray(dst, src, values, shape, device=device)
    else:
        dense = np.zeros(shape, dtype=np.float32)
        np.add.at(dense, (dst, src), values)
        adj = dense
    return adj


class TorchGCNLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(in_dim, out_dim))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adjacency):
        support = features @ self.weight
        return adjacency @ support


class TorchGCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.layers = torch.nn.ModuleList(
            TorchGCNLayer(dims[i], dims[i + 1]) for i in range(num_layers)
        )
        self.activation = torch.nn.ReLU()

    def forward(self, features, adjacency):
        h = features
        for i, layer in enumerate(self.layers):
            h = layer(h, adjacency)
            if i < len(self.layers) - 1:
                h = self.activation(h)
        return h


@pytest.mark.parametrize("hidden_dim", [8])
@pytest.mark.parametrize("num_layers", [2, 3])
@pytest.mark.parametrize("use_sparse", [False, True])
@pytest.mark.parametrize("device", devices)
def test_gcn_matches_pytorch(hidden_dim, num_layers, use_sparse, device):
    graph, feat_np, labels_np, edge_src, edge_dst, inv, masks = _generate_synthetic_graph(device=device)
    needle_model = GCNModel(
        in_features=feat_np.shape[1],
        hidden_features=hidden_dim,
        num_classes=labels_np.max() + 1,
        num_layers=num_layers,
        dropout=0.0,
        device=device,
    )
    torch_model = TorchGCN(
        feat_np.shape[1],
        hidden_dim,
        labels_np.max() + 1,
        num_layers=num_layers,
    )

    # Build normalized adjacency in the chosen format.
    adj_np_or_sparse = _normalized_adjacency(edge_src, edge_dst, inv, use_sparse, graph["features"].device)
    needle_adj = ndl.Tensor(adj_np_or_sparse, device=graph["features"].device, dtype="float32", requires_grad=False)
    torch_adj = torch.tensor(
        _normalized_adjacency(edge_src, edge_dst, inv, sparse=False, device=None),
        dtype=torch.float32,
    )

    # Copy weights layer by layer
    for n_layer, t_layer in zip(needle_model.layers, torch_model.layers):
        with torch.no_grad():
            t_layer.weight.copy_(torch.tensor(n_layer.weight.numpy()))
        if n_layer.bias is not None:
            # Torch reference has no bias term; enforce equivalence by zeroing bias.
            n_layer.bias.data = ndl.init.zeros_like(n_layer.bias.data)

    needle_out = needle_model(graph["features"], needle_adj).numpy()
    torch_out = torch_model(
        torch.tensor(feat_np, dtype=torch.float32),
        torch_adj,
    ).detach().numpy()

    print("Needle output:", needle_out)
    print("Torch output:", torch_out)
    np.testing.assert_allclose(needle_out, torch_out, atol=1e-5, rtol=1e-5)
