"""hw1/apps/simple_ml.py"""

import gzip
import os
import struct
import time
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl

import needle.nn as nn
from apps.models import *
device = ndl.cpu()

def parse_mnist(image_filesname, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION

### CIFAR-10 training ###
def epoch_general_cifar10(dataloader, model, loss_fn=nn.SoftmaxLoss(), opt=None):
    """
    Iterates over the dataloader. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


def train_cifar10(model, dataloader, n_epochs=1, optimizer=ndl.optim.Adam,
          lr=0.001, weight_decay=0.001, loss_fn=nn.SoftmaxLoss):
    """
    Performs {n_epochs} epochs of training.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


def evaluate_cifar10(model, dataloader, loss_fn=nn.SoftmaxLoss):
    """
    Computes the test accuracy and loss of the model.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


## PTB training ###
def epoch_general_ptb(data, model, seq_len=40, loss_fn=nn.SoftmaxLoss(), opt=None,
        clip=None, device=None, dtype="float32"):
    """
    Iterates over the data. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        data: data of shape (nbatch, batch_size) given from batchify function
        model: LanguageModel instance
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt is not None:
        model.train()
    else:
        model.eval()

    total_loss = 0
    total_acc = 0

    # Each row is of length row_len
    row_len, batch_size = data.shape

    iterate_c = row_len - seq_len

    for i in range(iterate_c):
        X_batch, y_batch = ndl.data.get_batch(data, i, seq_len, device=device, dtype=dtype)
        # X_batch: (seq_len, bs)
        # y_batch: (seq_len*bs,)

        # Forward pass
        if opt:
            opt.reset_grad()
        logits, _ = model(X_batch)  # logits: (seq_len*bs, output_size)

        loss = loss_fn(logits, y_batch)
        if opt:
            opt.reset_grad()
            loss.backward()
            if clip is not None:
                opt.clip_grad_norm(clip)
            opt.step()

        # loss_value, err = loss_err(logits, y_batch)

        total_loss += loss.numpy()
        total_acc += np.sum(logits.numpy().argmax(axis=1) != y_batch.numpy())

    avg_loss = total_loss / iterate_c
    avg_acc = 1 - (total_acc / (iterate_c *  seq_len))
    return avg_acc, avg_loss
    ### END YOUR SOLUTION


def train_ptb(model, data, seq_len=40, n_epochs=1, optimizer=ndl.optim.SGD,
          lr=4.0, weight_decay=0.0, loss_fn=nn.SoftmaxLoss, clip=None,
          device=None, dtype="float32"):
    """
    Performs {n_epochs} epochs of training.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    for epoch in range(n_epochs):
        opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
        avg_acc, avg_loss = epoch_general_ptb(data, model, seq_len=seq_len,
                                             loss_fn=loss_fn(), 
                                             opt=opt,
                                             clip=clip, device=device, dtype=dtype)
    return avg_acc, avg_loss
    ### END YOUR SOLUTION

def evaluate_ptb(model, data, seq_len=40, loss_fn=nn.SoftmaxLoss,
        device=None, dtype="float32"):
    """
    Computes the test accuracy and loss of the model.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    avg_acc, avg_loss = epoch_general_ptb(data, model, seq_len=seq_len,
                                         loss_fn=loss_fn(),
                                         device=device, dtype=dtype)
    return avg_acc, avg_loss
    ### END YOUR SOLUTION

### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)


### GCN / CORA inference utilities ###


def _named_parameters(module, prefix=""):
    """Recursively collect named parameters for a Module."""
    if not isinstance(module, nn.Module):
        return
    for attr, value in module.__dict__.items():
        name = f"{prefix}{attr}" if prefix else attr
        if isinstance(value, nn.Parameter):
            yield name, value
        elif isinstance(value, nn.Module):
            yield from _named_parameters(value, name + ".")
        elif isinstance(value, (list, tuple)):
            for idx, item in enumerate(value):
                list_name = f"{name}.{idx}"
                if isinstance(item, nn.Module):
                    yield from _named_parameters(item, list_name + ".")
                elif isinstance(item, nn.Parameter):
                    yield list_name, item


def load_npz_weights(model, weight_path, device=None, dtype="float32"):
    """
    Load weights stored in an .npz file into a Module.
    The archive should map parameter names (matching _named_parameters) to arrays.
    """
    weights = np.load(weight_path)
    params = dict(_named_parameters(model))
    missing = [name for name in params if name not in weights]
    if missing:
        raise KeyError(f"Weights missing for parameters: {missing}")
    for name, param in params.items():
        array = weights[name]
        param.data = ndl.Tensor(array, device=device or param.device, dtype=dtype)


def load_cora_graph(data_dir, device=None, dtype="float32"):
    dataset = ndl.data.CoraDataset(data_dir)
    graph = dataset.graph_tensors(device=device, dtype=dtype)
    return dataset, graph


def _build_normalized_adjacency(edge_src, edge_dst, inv_sqrt_deg, sparse=False, device=None, dtype="float32"):
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


def cora_gcn_inference(
    data_dir,
    weight_path=None,
    hidden_dim=16,
    num_layers=2,
    dropout=0.5,
    use_sparse=False,
    device=None,
    dtype="float32",
):
    """
    Runs a multi-layer GCN on the Cora graph for inference-only evaluation.
    If use_sparse=True, builds a SparseNDArray adjacency and uses spmm; otherwise uses
    a dense normalized adjacency matrix.
    Optionally loads pretrained weights from an .npz archive.
    """
    device = ndl.cpu() if device is None else device
    dataset, graph = load_cora_graph(data_dir, device=device, dtype=dtype)
    model = GCNModel(
        in_features=graph["features"].shape[1],
        hidden_features=hidden_dim,
        num_classes=dataset.num_classes,
        num_layers=num_layers,
        dropout=dropout,
        device=device,
        dtype=dtype,
        edge_index=graph["edge_index"],
        inv_sqrt_deg=graph["inv_sqrt_deg"],
        use_sparse=use_sparse,
        weight_path=weight_path,
    )
    model.eval()
    # print out actual inference time

    start_time = time.time()
    logits = model(graph["features"])
    elapsed_time = time.time() - start_time
    print(f"Inference time: {elapsed_time:.4f} seconds")
    predictions = logits.numpy().argmax(axis=1)
    labels = graph["labels"].numpy()
    test_mask = graph["masks"]["test"]
    test_accuracy = float((predictions[test_mask] == labels[test_mask]).mean())
    print(f"Cora GCN inference: test accuracy {test_accuracy:.4f}")
    return predictions, test_accuracy


from apps.unified_example import GCNModel as UnifiedGCN, build_needle_adjacency
from apps.simple_ml import load_cora_graph
from needle.data.datasets.cora_dataset import CoraDataset
import tvm


def benchmark_gcn(
    backend="needle",
    device_str="cpu",
    use_sparse=False,
    use_tensorcore=False,
    cpu_instruction=None,
    dtype="float32",
    runs=20,
    warmup=5,
    DATA_DIR = "data/cora",
    WEIGHT_PATH = "data/cora/gcn_pytorch.npz",
    model=None,  # Optional: pass pre-compiled model to use it directly
    use_tuning=False,  # Optional: if model is None, whether to use tuning when compiling
    tune_log_file=None,  # Optional: path to tuning log file
    ):
    # Convert device_str to Needle device object
    if device_str == "cuda":
        device = ndl.cuda()
        if not device.enabled():
            raise RuntimeError("CUDA device not available")
    else:
        device = ndl.cpu()
    
    if backend == "needle":
        # Needle backend: load data and build adjacency
        dataset, graph = load_cora_graph(DATA_DIR, device=device, dtype=dtype)
        adjacency = build_needle_adjacency(
            graph["edge_index"][0],
            graph["edge_index"][1],
            graph["inv_sqrt_deg"],
            sparse=use_sparse,
            device=device,
            dtype=dtype,
        )
        
        # Create model - UnifiedGCN handles device conversion internally
        model = UnifiedGCN(
            backend="needle",
            in_features=graph["features"].shape[1],
            hidden_features=32,
            num_classes=dataset.num_classes,
            num_layers=2,
            dropout=0.5,
            device=device,
            dtype=dtype,
            use_sparse=use_sparse,
        )
        
        # Load weights
        if os.path.exists(WEIGHT_PATH):
            model.load_weights(WEIGHT_PATH)
        
        # Prepare inputs
        features = graph["features"]
        
        # Warmup
        for _ in range(warmup):
            _ = model(features, adjacency)
        
        # Benchmark
        times = []
        for _ in range(runs):
            start = time.perf_counter()
            _ = model(features, adjacency)
            times.append((time.perf_counter() - start) * 1000)
        
        return float(np.mean(times)), float(np.std(times)), None
        
    elif backend == "tvm":
        # If model is provided, use it directly (assume it's already compiled)
        if model is not None:
            # Use provided model directly
            if not hasattr(model, '_compiled') or not model._compiled:
                raise ValueError("Provided model must be compiled. Call model.compile() first.")
            # Get features from model's dataset
            if hasattr(model, 'model') and hasattr(model.model, 'data'):
                features = model.model.data.features.numpy().astype(dtype)
            else:
                # Fallback: load dataset to get features
                dataset = CoraDataset(root=DATA_DIR)
                features = dataset.features.astype(dtype)
        else:
            # Load dataset - UnifiedGCN will convert to TVM format automatically
            dataset = CoraDataset(root=DATA_DIR)
            
            # Create model - UnifiedGCN handles all conversion internally
            model = UnifiedGCN(
                backend="tvm",
                dataset=dataset,  # UnifiedGCN converts this to TVM format automatically
                hidden_features=32,
                num_layers=2,
                device=device_str,  # Pass string, UnifiedGCN handles conversion
                opt_level=3,
                cpu_instruction_set=cpu_instruction,
                use_gpu_mma=use_tensorcore,
                dtype=dtype,
                activation=None,
            )
            
            # Set tuning log file if provided
            if tune_log_file is not None:
                model._tune_log_file = tune_log_file
            
            # Compile model (with or without tuning)
            model.compile(use_tuning=use_tuning)
            
            # Prepare features - UnifiedGCN.forward handles padding automatically
            features = dataset.features.astype(dtype)
        
        # Warmup
        for _ in range(warmup):
            _ = model.forward(features)
        
        # Benchmark
        times = []
        for _ in range(runs):
            if device_str == "cuda":
                tvm.cuda(0).sync()
            start = time.perf_counter()
            _ = model.forward(features)
            if device_str == "cuda":
                tvm.cuda(0).sync()
            times.append((time.perf_counter() - start) * 1000)
        
        # Get target from model
        target = model.target
        return float(np.mean(times)), float(np.std(times)), target
    
    else:
        raise ValueError(f"Unknown backend: {backend}")




if __name__ == "__main__":
    cora_dir = os.path.join("data", "cora")
    if os.path.exists(os.path.join(cora_dir, "cora.content")):
        cora_gcn_inference(cora_dir)
    else:
        print("Cora dataset not found. Please place cora.content and cora.cites under data/cora.")
