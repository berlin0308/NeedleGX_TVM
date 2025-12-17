import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
from needle.backend_ndarray import SparseNDArray
import math
import numpy as np
np.random.seed(0)


class GraphConvolution(nn.Module):
    """Single graph convolution (Kipf & Welling) supporting dense or edge-based adjacencies."""

    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            ndl.init.xavier_uniform(in_features, out_features, device=device, dtype=dtype)
        )
        self.bias = (
            nn.Parameter(ndl.init.zeros(out_features, device=device, dtype=dtype))
            if bias
            else None
        )

    def forward(self, features, adjacency):
        # CUDA sparse-dense matmul currently diverges; densify adjacency on CUDA to match reference.
        adj_data = adjacency.realize_cached_data() if isinstance(adjacency, ndl.Tensor) else adjacency
        if isinstance(adj_data, SparseNDArray) and getattr(adj_data.device, "name", None) == "cuda":
            adjacency = ndl.Tensor(
                adj_data.to_dense_ndarray(device=adj_data.device),
                device=adj_data.device,
                dtype="float32",
                requires_grad=False,
            )
        support = ndl.ops.matmul(features, self.weight)
        if isinstance(adjacency, (tuple, list)):
            edge_src, edge_dst, inv_sqrt_deg = adjacency
            inv = inv_sqrt_deg
            if len(inv.shape) == 1:
                inv = inv.reshape((inv.shape[0], 1))
            if inv.shape != support.shape:
                inv_broadcast = inv.broadcast_to(support.shape)
            else:
                inv_broadcast = inv
            support = support * inv_broadcast
            out = ndl.ops.graph_neighbor_sum(support, edge_src, edge_dst)
            if inv.shape != out.shape:
                inv_out = inv.broadcast_to(out.shape)
            else:
                inv_out = inv
            out = out * inv_out
        else:
            out = ndl.ops.matmul(adjacency, support)
        if self.bias is not None:
            bias = self.bias.reshape((1, self.out_features)).broadcast_to(out.shape)
            out = out + bias
        return out






class GCNModel(nn.Module):
    """
    Simple multi-layer GCN for inference-time usage.

    Args:
        in_features: Size of the input node features.
        hidden_features: Either an int or list controlling hidden layer widths.
        num_classes: Output dimension per node.
        num_layers: Number of graph conv layers (>=2 recommended).
        activation: Module applied after each hidden layer (default ReLU).
        dropout: Dropout probability applied after activation in hidden layers.
        edge_index: Optional tuple of (edge_src, edge_dst) tensors for building adjacency.
        inv_sqrt_deg: Optional tensor of inverse square root degrees for normalization.
        use_sparse: Whether to use sparse adjacency matrix (requires edge_index and inv_sqrt_deg).
        weight_path: Optional path to .npz file containing pretrained weights.
    """

    def __init__(
        self,
        in_features,
        hidden_features,
        num_classes,
        num_layers=2,
        activation=None,
        dropout=0.0,
        bias=True,
        device=None,
        dtype="float32",
        edge_index=None,
        inv_sqrt_deg=None,
        use_sparse=False,
        weight_path=None,
    ):
        super().__init__()
        if isinstance(hidden_features, int):
            hidden_dims = [hidden_features] * max(0, num_layers - 1)
        else:
            hidden_dims = list(hidden_features)
        if num_layers < 1:
            raise ValueError("GCNModel requires at least one layer.")
        expected_hidden_count = max(0, num_layers - 1)
        if hidden_dims and len(hidden_dims) != expected_hidden_count:
            raise ValueError(
                f"hidden_features expects {expected_hidden_count} entries but got {len(hidden_dims)}"
            )
        dims = [in_features] + hidden_dims + [num_classes]
        self.layers = [
            GraphConvolution(dims[i], dims[i + 1], bias=bias, device=device, dtype=dtype)
            for i in range(len(dims) - 1)
        ]
        self.activation = activation if activation is not None else nn.ReLU()
        self.use_activation = len(self.layers) > 1
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else None
        
        # Build adjacency matrix if edge_index and inv_sqrt_deg are provided
        self.adjacency = None
        if edge_index is not None and inv_sqrt_deg is not None:
            self.adjacency = self._build_normalized_adjacency(
                edge_index[0], edge_index[1], inv_sqrt_deg, 
                sparse=use_sparse, device=device, dtype=dtype
            )
        
        # Load weights if weight_path is provided
        if weight_path is not None:
            self.load_weights(weight_path, device=device, dtype=dtype)

    def _build_normalized_adjacency(self, edge_src, edge_dst, inv_sqrt_deg, sparse=False, device=None, dtype="float32"):
        """
        Construct normalized adjacency A_hat = D^{-1/2} A D^{-1/2} either as dense NDArray
        or SparseNDArray, using edges encoded in edge_src/edge_dst (assumed symmetric, with self loops).
        """
        from needle.backend_ndarray import SparseNDArray

        src = edge_src.numpy().astype("int64")
        dst = edge_dst.numpy().astype("int64")
        inv = inv_sqrt_deg.numpy().reshape(-1)
        values = inv[src] * inv[dst]
        num_nodes = inv.shape[0]
        shape = (num_nodes, num_nodes)
        if sparse:
            adj = SparseNDArray(dst, src, values, shape, device=device)  # rows=dst, cols=src
        else:
            import numpy as np

            dense = np.zeros(shape, dtype=np.float32)
            np.add.at(dense, (dst, src), values)
            adj = dense
        return ndl.Tensor(adj, device=device, dtype=dtype, requires_grad=False)

    def _named_parameters(self, prefix=""):
        """Recursively collect named parameters for this Module."""
        for attr, value in self.__dict__.items():
            name = f"{prefix}{attr}" if prefix else attr
            if isinstance(value, nn.Parameter):
                yield name, value
            elif isinstance(value, nn.Module):
                # Check if the module has _named_parameters method, otherwise recurse manually
                if hasattr(value, '_named_parameters'):
                    yield from value._named_parameters(name + ".")
                else:
                    # Manually recurse into the module
                    yield from self._named_parameters_recursive(value, name + ".")
            elif isinstance(value, (list, tuple)):
                for idx, item in enumerate(value):
                    list_name = f"{name}.{idx}"
                    if isinstance(item, nn.Module):
                        if hasattr(item, '_named_parameters'):
                            yield from item._named_parameters(list_name + ".")
                        else:
                            yield from self._named_parameters_recursive(item, list_name + ".")
                    elif isinstance(item, nn.Parameter):
                        yield list_name, item
    
    @staticmethod
    def _named_parameters_recursive(module, prefix=""):
        """Helper method to recursively collect parameters from modules without _named_parameters."""
        for attr, value in module.__dict__.items():
            name = f"{prefix}{attr}" if prefix else attr
            if isinstance(value, nn.Parameter):
                yield name, value
            elif isinstance(value, nn.Module):
                if hasattr(value, '_named_parameters'):
                    yield from value._named_parameters(name + ".")
                else:
                    yield from GCNModel._named_parameters_recursive(value, name + ".")
            elif isinstance(value, (list, tuple)):
                for idx, item in enumerate(value):
                    list_name = f"{name}.{idx}"
                    if isinstance(item, nn.Module):
                        if hasattr(item, '_named_parameters'):
                            yield from item._named_parameters(list_name + ".")
                        else:
                            yield from GCNModel._named_parameters_recursive(item, list_name + ".")
                    elif isinstance(item, nn.Parameter):
                        yield list_name, item

    def load_weights(self, weight_path, device=None, dtype="float32"):
        """
        Load weights from .npz file.
        """
        import numpy as np
        
        weights = np.load(weight_path)
        params = dict(self._named_parameters())
        missing = [name for name in params if name not in weights]
        if missing:
            raise KeyError(f"Weights missing for parameters: {missing}")
        for name, param in params.items():
            array = weights[name]
            param.data = ndl.Tensor(array, device=device or param.device, dtype=dtype)

    def forward(self, node_features, adjacency=None):
        """
        Forward pass through the GCN model.
        
        Args:
            node_features: Node feature matrix.
            adjacency: Optional adjacency matrix. If None, uses the adjacency built in __init__.
        """
        # Use provided adjacency or fall back to internal adjacency
        if adjacency is None:
            if self.adjacency is None:
                raise ValueError("adjacency must be provided either in __init__ or forward()")
            adjacency = self.adjacency
        
        h = node_features
        for i, layer in enumerate(self.layers):
            h = layer(h, adjacency)
            if i < len(self.layers) - 1:
                if self.dropout is not None:
                    h = self.dropout(h)
                h = self.activation(h)
        return h


class ConvBN(ndl.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 bias=True, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.conv = nn.Conv(in_channels, out_channels, kernel_size, stride=stride,
                           bias=bias, device=device, dtype=dtype)
        self.bn = nn.BatchNorm2d(out_channels, device=device, dtype=dtype)
        self.relu = nn.ReLU()
        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
        ### END YOUR SOLUTION

class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
        bias = True
        self.conv1 = ConvBN(3, 16,7, 4, bias=bias, device=device, dtype=dtype)
        self.conv2 = ConvBN(16, 32, 3, 2, bias=bias, device=device, dtype=dtype)
        self.res1 = nn.Residual(
            nn.Sequential(
                ConvBN(32, 32, 3, 1, bias=bias, device=device, dtype=dtype),
                ConvBN(32, 32, 3, 1, bias=bias, device=device, dtype=dtype),
            )
        )
        self.conv3 = ConvBN(32, 64, 3, 2, bias=bias, device=device, dtype=dtype)
        self.conv4 = ConvBN(64, 128, 3, 2, bias=bias, device=device, dtype=dtype)
        self.res2 = nn.Residual(
            nn.Sequential(
                ConvBN(128, 128, 3, 1, bias=bias, device=device, dtype=dtype),
                ConvBN(128, 128, 3, 1, bias=bias, device=device, dtype=dtype),
            )
        )
        self.linear1 = nn.Linear(128, 128, bias=bias, device=device, dtype=dtype)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, 10, bias=bias, device=device, dtype=dtype)

        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.res2(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x
        ### END YOUR SOLUTION


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', seq_len=40, device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        self.embedding = nn.Embedding(output_size, embedding_size,
                                      device=device, dtype=dtype)
        if seq_model == 'rnn':
            self.seq_model = nn.RNN(embedding_size, hidden_size,
                                    num_layers=num_layers, device=device, dtype=dtype)
        elif seq_model == 'lstm':
            self.seq_model = nn.LSTM(embedding_size, hidden_size,
                                     num_layers=num_layers, device=device, dtype=dtype)
        self.linear = nn.Linear(hidden_size, output_size, device=device, dtype=dtype)  
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        x = self.embedding(x)
        x, h = self.seq_model(x, h)
        seq_len, bs, hidden_size = x.shape
        x = x.reshape((seq_len * bs, hidden_size)) # Note: Cannot simply write -1 here!!!
        out = self.linear(x)
        return out, h
        ### END YOUR SOLUTION



if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(cifar10_train_dataset[1][0].shape)
