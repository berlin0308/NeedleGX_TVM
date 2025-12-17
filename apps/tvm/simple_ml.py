"""
Simple ML test script for GCN model with comprehensive benchmarking
"""
import argparse
import os
import sys
from pathlib import Path

# os.environ["TVM_CUDA_ARCH"] = "sm_75"

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Add TVM path (must be before any TVM imports)
_DEFAULT_TVM_PATHS = [
    "/home/nas/polin/cmu-berlin/sparse-gnn-tvm-tensorcore/tvm-0.10.0/python",
    "/home/nas/polin/cmu-berlin/tvm-0.10.0/python",
    str((REPO_ROOT.parent / "tvm-0.10.0" / "python").resolve()),
]
for tvm_python_path in _DEFAULT_TVM_PATHS:
    if os.path.exists(tvm_python_path) and tvm_python_path not in sys.path:
        sys.path.insert(0, tvm_python_path)
        break

# Remove user site-packages to avoid using pip-installed TVM
user_site = os.path.expanduser("~/.local/lib/python3.9/site-packages")
if user_site in sys.path:
    sys.path.remove(user_site)

data_dir = str((REPO_ROOT.parent / "data" / "mutag").resolve())

from dataset.graph_dataset import (
    load_cora_dataset,
    load_mutag_dataset,
    load_single_mutag_molecule,
    random_graph_dataset,
    random_block_cluster_dataset,
    plot_adj_matrix,
)
from apps.tvm.models import GraphConvNetwork
from apps.tvm.visualizer import visualize_graph_output, visualize_graph_output_comparison
import numpy as np
import time
import networkx as nx
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def get_dtype_from_data(data):
    """
    Extract dtype string from data.features
    
    Parameters
    ----------
    data: DataWrapper
        Data wrapper containing features
    
    Returns
    -------
    dtype: str
        "float32" or "float16"
    """
    if hasattr(data.features, 'dtype'):
        if data.features.dtype == torch.float16:
            return "float16"
        elif data.features.dtype == torch.float32:
            return "float32"
        else:
            # Try to convert numpy dtype
            if hasattr(data.features, 'numpy'):
                np_dtype = data.features.numpy().dtype
                if np_dtype == np.float16:
                    return "float16"
                elif np_dtype == np.float32:
                    return "float32"
    # Default to float32
    return "float32"

# try:
from apps.gnn_tvm_utils.spmm_cuda_alter_layout import register_spmm_cuda_alter_layout, register_fixed_sparse_dense_tir
register_spmm_cuda_alter_layout()
register_fixed_sparse_dense_tir()  # Fix float16 support in CUDA Core
print("✓ Registered custom sparse_dense_alter_layout for CUDA")
# print("✓ Registered fixed sparse_dense_tir (supports float16)")
# except ImportError:
#     print("⚠️  Using default sparse_dense (may have block size issues)")

def calculate_gcn_flops(graph, dataset_info, num_hidden, num_layers):
    """
    Calculate FLOPs for GCN model
    
    For each GraphConv layer:
    - Dense MM: N * F_in * F_out
    - Sparse MM: nnz * F_out (where nnz is number of non-zeros)
    - Norm multiply (before): N * F_in
    - Norm multiply (after): N * F_out
    - Bias add: N * F_out
    # - Activation (ReLU): N * F_out (if present)
    
    Parameters
    ----------
    graph: networkx.Graph
        Input graph
    dataset_info: dict
        Dataset information
    num_hidden: int
        Number of hidden units
    num_layers: int
        Number of layers
    
    Returns
    -------
    total_flops: int
        Total FLOPs for the model
    """
    num_nodes = dataset_info['num_nodes']
    infeat_dim = dataset_info['infeat_dim']
    num_classes = dataset_info['num_classes']
    
    # Get number of edges (non-zeros in adjacency matrix)
    num_edges = graph.number_of_edges()
    # For undirected graph, each edge appears twice in adjacency matrix
    # But we count it once, so nnz = num_edges (if self-loops are added)
    nnz = num_edges
    
    total_flops = 0
    
    # First layer: infeat_dim -> num_hidden
    # Dense MM: N * infeat_dim * num_hidden
    total_flops += num_nodes * infeat_dim * num_hidden
    # Sparse MM: nnz * num_hidden
    total_flops += nnz * num_hidden
    # Norm (before): N * infeat_dim
    total_flops += num_nodes * infeat_dim
    # Norm (after): N * num_hidden
    total_flops += num_nodes * num_hidden
    # Bias: N * num_hidden
    total_flops += num_nodes * num_hidden
    # ReLU: N * num_hidden
    total_flops += num_nodes * num_hidden
    
    # Hidden layers: num_hidden -> num_hidden
    for _ in range(num_layers - 2):
        # Dense MM: N * num_hidden * num_hidden
        total_flops += num_nodes * num_hidden * num_hidden
        # Sparse MM: nnz * num_hidden
        total_flops += nnz * num_hidden
        # Norm (before): N * num_hidden
        total_flops += num_nodes * num_hidden
        # Norm (after): N * num_hidden
        total_flops += num_nodes * num_hidden
        # Bias: N * num_hidden
        total_flops += num_nodes * num_hidden
        # ReLU: N * num_hidden
        total_flops += num_nodes * num_hidden
    
    # Output layer: num_hidden -> num_hidden (output dim is 32, not num_classes)
    # Dense MM: N * num_hidden * num_hidden
    total_flops += num_nodes * num_hidden * num_hidden
    # Sparse MM: nnz * num_hidden
    total_flops += nnz * num_hidden
    # Norm (before): N * num_hidden
    total_flops += num_nodes * num_hidden
    # Norm (after): N * num_hidden
    total_flops += num_nodes * num_hidden
    # Bias: N * num_hidden
    total_flops += num_nodes * num_hidden
    
    return total_flops


def benchmark_model(model, num_warmup=10, num_runs=10):
    """
    Benchmark model inference time
    
    Parameters
    ----------
    model: GraphConvNetwork
        The model to benchmark
    num_warmup: int
        Number of warmup runs
    num_runs: int
        Number of benchmark runs
    
    Returns
    -------
    mean_time_ms: float
        Mean inference time in milliseconds
    std_time_ms: float
        Standard deviation in milliseconds
    times_ms: list
        List of all inference times in milliseconds
    """
    # Warmup
    for _ in range(num_warmup):
        model.forward()
    
    # Synchronize if using GPU
    if model.device == "cuda":
        import tvm
        tvm.cuda(0).sync()
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        if model.device == "cuda":
            import tvm
            tvm.cuda(0).sync()
            start = time.perf_counter()
            model.forward()
            tvm.cuda(0).sync()
            end = time.perf_counter()
        else:
            start = time.perf_counter()
            model.forward()
            end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to milliseconds
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    
    return mean_time, std_time, times


def compare_outputs(results, rtol=1e-3, atol=1e-5):
    """
    Compare output results from different tests
    
    Parameters
    ----------
    results: dict
        Dictionary of test results, where each value is a dict with 'output' key
    rtol: float
        Relative tolerance for comparison
    atol: float
        Absolute tolerance for comparison
    
    Returns
    -------
    comparison_results: dict
        Dictionary with comparison results
    """
    # Filter out None results
    valid_results = {k: v for k, v in results.items() if v is not None and 'output' in v}
    
    if len(valid_results) < 2:
        print("⚠️  Not enough valid results to compare (need at least 2)")
        return {}
    
    comparison_results = {}
    test_names = list(valid_results.keys())
    
    print("\n" + "="*80)
    print("Output Comparison")
    print("="*80)
    
    # Compare each pair of results
    for i, name1 in enumerate(test_names):
        output1 = valid_results[name1]['output']
        for j, name2 in enumerate(test_names[i+1:], start=i+1):
            output2 = valid_results[name2]['output']
            
            # Check shapes match
            if output1.shape != output2.shape:
                match = False
                max_diff = float('inf')
                mean_diff = float('inf')
                shape_match = False
                print(f"✗ {name1} vs {name2}: Shape mismatch!")
                print(f"    {name1} shape: {output1.shape}")
                print(f"    {name2} shape: {output2.shape}")
            else:
                shape_match = True
                # Convert to float32 for comparison if needed
                if output1.dtype != output2.dtype:
                    output1_comp = output1.astype(np.float32)
                    output2_comp = output2.astype(np.float32)
                else:
                    output1_comp = output1.astype(np.float32)
                    output2_comp = output2.astype(np.float32)
                
                # Calculate differences
                diff = np.abs(output1_comp - output2_comp)
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)
                
                # Check if results match within tolerance
                match = np.allclose(output1_comp, output2_comp, rtol=rtol, atol=atol)
                
                status = "✓" if match else "✗"
                print(f"{status} {name1} vs {name2}:")
                print(f"    Match: {match}")
                print(f"    Max difference: {max_diff:.6e}")
                print(f"    Mean difference: {mean_diff:.6e}")
                print(f"    Tolerance: rtol={rtol}, atol={atol}")
            
            comparison_results[f"{name1}_vs_{name2}"] = {
                "match": match,
                "max_diff": max_diff,
                "mean_diff": mean_diff,
                "shape_match": shape_match
            }
    
    # Find reference result (first valid result)
    reference_name = test_names[0]
    reference_output = valid_results[reference_name]['output']
    
    print(f"\nReference: {reference_name} (shape: {reference_output.shape}, dtype: {reference_output.dtype})")
    
    # Compare all others against reference
    print("\nComparison against reference:")
    all_match = True
    for name in test_names[1:]:
        output = valid_results[name]['output']
        if output.shape != reference_output.shape:
            print(f"✗ {name}: Shape mismatch with reference")
            all_match = False
            continue
        
        # Convert to float32 for comparison
        ref_comp = reference_output.astype(np.float32)
        out_comp = output.astype(np.float32)
        
        diff = np.abs(ref_comp - out_comp)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        match = np.allclose(ref_comp, out_comp, rtol=rtol, atol=atol)
        
        status = "✓" if match else "✗"
        print(f"{status} {name}: match={match}, max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}")
        
        if not match:
            all_match = False
    
    if all_match:
        print("\n✓ All outputs match within tolerance!")
    else:
        print("\n⚠️  Some outputs do not match within tolerance")
    
    print("="*80)
    
    return comparison_results


def run_pytorch_reference(model):
    """
    Run a PyTorch reference GCN with the same weights/graph as the TVM model.
    Returns numpy output or None if the adjacency format is unsupported.
    """
    import torch

    if model.params.get("format", "csr") != "csr":
        print("⚠️  PyTorch reference only supports CSR adjacency; skipping")
        return None

    dtype = torch.float16 if model.dtype == "float16" else torch.float32
    x = torch.tensor(model.params["infeats"], dtype=dtype)

    indptr = torch.tensor(model.params["indptr"], dtype=torch.int64)
    indices = torch.tensor(model.params["indices"], dtype=torch.int64)
    data = torch.tensor(model.params["g_data"], dtype=dtype)
    num_nodes = model.dataset_info["num_nodes"]
    adj_dense = torch.sparse_csr_tensor(
        indptr, indices, data, size=(num_nodes, num_nodes), dtype=dtype
    ).to_dense()

    for layer_idx in range(model.num_layers):
        weight = torch.tensor(
            model.params[f"layers.{layer_idx}.weight"], dtype=dtype
        )
        x = adj_dense @ (x @ weight)

    return x.detach().cpu().numpy()


def test_cpu_llvm(g, data, dataset_info):
    """Test CPU with LLVM (default)"""
    # Detect dtype from data
    dtype = get_dtype_from_data(data)
    print("="*60)
    print(f"1. Test CPU LLVM (default) - {dtype}")
    print("="*60)
    
    print(f"Dataset: {dataset_info['num_nodes']} nodes, {dataset_info['num_edges']} edges")
    
    # Create model
    num_hidden = 32  # Use same as input dim
    num_layers = 2
    model = GraphConvNetwork(
        graph=g,
        data=data,
        dataset_info=dataset_info,
        num_hidden=num_hidden,
        num_layers=num_layers,
        device="cpu",
        opt_level=0,
        cpu_instruction_set=None,
        dtype=dtype,  # Use detected dtype
        # # activation="relu"  # Enable ReLU activation
    )

    ir_result = model.show_ir(
        show_relay=True,      # Skip Relay IR (high-level)
        show_tir=True,        # Skip TIR (can be verbose)
        show_source=True,      # Show generated CUDA code
        check_tensorcore=False, # Check for Tensor Core intrinsics
        verbose=False,          # Non-verbose mode (truncated output)
        save_to_file="ir_cpu_llvm.txt"
    )
    
    print(f"Model: {num_layers} layers, hidden={num_hidden}")
    print("Compiling...")
    model.compile()
    print("✓ Compiled")
    
    # Calculate FLOPs
    total_flops = calculate_gcn_flops(g, dataset_info, num_hidden, num_layers)
    print(f"Total FLOPs: {total_flops:,}")
    
    # Benchmark
    print("Benchmarking...")
    mean_time_ms, std_time_ms, times_ms = benchmark_model(model)
    
    # Calculate TFLOPs
    mean_time_s = mean_time_ms / 1000.0
    tflops = (total_flops / 1e12) / mean_time_s
    
    print(f"\nResults:")
    print(f"  Latency: {mean_time_ms:.3f} ± {std_time_ms:.3f} ms")
    print(f"  Throughput: {tflops:.4f} TFLOPs")
    
    # Get output result for comparison
    print("Getting output result...")
    output_result = model.forward()
    
    print("Running PyTorch reference for accuracy check...")
    pytorch_output = run_pytorch_reference(model)
    
    return {
        "latency_ms": mean_time_ms,
        "latency_std_ms": std_time_ms,
        "tflops": tflops,
        "flops": total_flops,
        "output": output_result,
        "pytorch_output": pytorch_output
    }


def test_cpu_llvm_avx2(g, data, dataset_info):
    """Test CPU with LLVM AVX2"""
    # Detect dtype from data
    dtype = get_dtype_from_data(data)
    print("\n" + "="*60)
    print(f"2. Test CPU LLVM AVX2 - {dtype}")
    print("="*60)
    
    print(f"Dataset: {dataset_info['num_nodes']} nodes, {dataset_info['num_edges']} edges")
    
    # Create model
    num_hidden = 32  # Use same as input dim
    num_layers = 2
    model = GraphConvNetwork(
        graph=g,
        data=data,
        dataset_info=dataset_info,
        num_hidden=num_hidden,
        num_layers=num_layers,
        device="cpu",
        opt_level=0,
        cpu_instruction_set="avx2",
        dtype=dtype,  # Use detected dtype
        # # activation="relu"  # Enable ReLU activation
    )
    ir_result = model.show_ir(
        show_relay=True,      # Skip Relay IR (high-level)
        show_tir=True,        # Skip TIR (can be verbose)
        show_source=True,      # Show generated CUDA code
        check_tensorcore=False, # Check for Tensor Core intrinsics
        verbose=False,          # Non-verbose mode (truncated output)
        save_to_file="ir_cpu_llvm_avx2.txt"
    )

    print(f"Model: {num_layers} layers, hidden={num_hidden}")
    print("Compiling...")
    model.compile()
    print("✓ Compiled")
    
    # Calculate FLOPs
    total_flops = calculate_gcn_flops(g, dataset_info, num_hidden, num_layers)
    print(f"Total FLOPs: {total_flops:,}")
    
    # Benchmark
    print("Benchmarking...")
    mean_time_ms, std_time_ms, times_ms = benchmark_model(model)
    
    # Calculate TFLOPs
    mean_time_s = mean_time_ms / 1000.0
    tflops = (total_flops / 1e12) / mean_time_s
    
    print(f"\nResults:")
    print(f"  Latency: {mean_time_ms:.3f} ± {std_time_ms:.3f} ms")
    print(f"  Throughput: {tflops:.4f} TFLOPs")
    
    # Get output result for comparison
    print("Getting output result...")
    output_result = model.forward()
    
    return {
        "latency_ms": mean_time_ms,
        "latency_std_ms": std_time_ms,
        "tflops": tflops,
        "flops": total_flops,
        "output": output_result
    }


def test_cpu_llvm_avx512(g, data, dataset_info):
    """Test CPU with LLVM AVX512"""
    # Detect dtype from data
    dtype = get_dtype_from_data(data)
    print("\n" + "="*60)
    print(f"3. Test CPU LLVM AVX512 - {dtype}")
    print("="*60)
    
    print(f"Dataset: {dataset_info['num_nodes']} nodes, {dataset_info['num_edges']} edges")
    
    # Create model
    num_hidden = 32  # Use same as input dim
    num_layers = 2
    model = GraphConvNetwork(
        graph=g,
        data=data,
        dataset_info=dataset_info,
        num_hidden=num_hidden,
        num_layers=num_layers,
        device="cpu",
        opt_level=0,
        cpu_instruction_set="avx512",
        dtype=dtype,  # Use detected dtype
        # # activation="relu"  # Enable ReLU activation
    )
    
    ir_result = model.show_ir(
        show_relay=True,      # Skip Relay IR (high-level)
        show_tir=True,        # Skip TIR (can be verbose)
        show_source=True,      # Show generated CUDA code
        check_tensorcore=False, # Check for Tensor Core intrinsics
        verbose=False,          # Non-verbose mode (truncated output)
        save_to_file="ir_cpu_llvm_avx512.txt"
    )
    
    print(f"Model: {num_layers} layers, hidden={num_hidden}")
    print("Compiling...")
    model.compile()
    print("✓ Compiled")
    
    # Calculate FLOPs
    total_flops = calculate_gcn_flops(g, dataset_info, num_hidden, num_layers)
    print(f"Total FLOPs: {total_flops:,}")
    
    # Benchmark
    print("Benchmarking...")
    mean_time_ms, std_time_ms, times_ms = benchmark_model(model)
    
    # Calculate TFLOPs
    mean_time_s = mean_time_ms / 1000.0
    tflops = (total_flops / 1e12) / mean_time_s
    
    print(f"\nResults:")
    print(f"  Latency: {mean_time_ms:.3f} ± {std_time_ms:.3f} ms")
    print(f"  Throughput: {tflops:.4f} TFLOPs")
    
    # Get output result for comparison
    print("Getting output result...")
    output_result = model.forward()
    
    return {
        "latency_ms": mean_time_ms,
        "latency_std_ms": std_time_ms,
        "tflops": tflops,
        "flops": total_flops,
        "output": output_result
    }


def test_cuda_core(g, data, dataset_info):

    # Detect dtype from data
    dtype = get_dtype_from_data(data)
    print("\n" + "="*60)
    print(f"4. Test CUDA Core (no Tensor Core) - {dtype}")
    print("="*60)
    
    
    # Check CUDA availability
    try:
        import tvm
        tvm.cuda(0)
    except:
        print("✗ CUDA not available, skipping")
        return None
    
    print(f"Dataset: {dataset_info['num_nodes']} nodes, {dataset_info['num_edges']} edges")
    print(f"  Sparsity: {dataset_info.get('sparsity', 0):.4f} ({dataset_info.get('sparsity', 0)*100:.2f}%)")
    print(f"  Non-zero elements (nnz): {dataset_info['num_edges']}")
    if 'cluster_count' in dataset_info:
        print(f"  Cluster structure: {dataset_info['cluster_count']} clusters")
        print(f"    -> This affects BSR block size in sparse_dense_padded")
        print(f"    -> CUDA Core performance depends on BSR block structure")
    
    num_hidden = 32
    
    num_layers = 2
    model = GraphConvNetwork(
        graph=g,
        data=data,
        dataset_info=dataset_info,
        num_hidden=num_hidden,
        num_layers=num_layers,
        device="cuda",
        opt_level=3,  # Need opt_level > 0 for alter_op_layout to be applied
        use_gpu_mma=False,
        dtype=dtype,  # Use detected dtype
        # # activation="relu"  # Enable ReLU activation
    )
    
    print(f"Model: {num_layers} layers, hidden={num_hidden}")
    
    # Register alter_op_layout RIGHT BEFORE compiling (not earlier)
    # This ensures it doesn't affect CPU tests
    # Also register fixed sparse_dense strategy as fallback for clustered graphs
    try:
        from tvm_implement.sparse_dense_fixed import register_fixed_sparse_dense_strategy
        register_fixed_sparse_dense_strategy()
        print("✓ Registered fixed sparse_dense strategy (fallback for block size issues)")
    except ImportError:
        print("⚠️  Could not register fixed sparse_dense strategy (may have block size issues)")
    
    print("Compiling...")
    try:
        model.compile()
        print("✓ Compiled")
    except Exception as e:
        error_str = str(e)
        import traceback
        full_traceback = traceback.format_exc()
        print(f"✗ Compilation failed:")
        print(f"  Error: {error_str[:500]}")
        print(f"  Full traceback (first 1000 chars):\n{full_traceback[:1000]}")
        return None
    
    # Show IR after compilation (so we can see the generated code)
    try:
        ir_result = model.show_ir(
                    show_relay=True,      # Skip Relay IR (high-level)
                    show_tir=True,        # Skip TIR (can be verbose)
                    show_source=True,      # Show generated CUDA code
                    check_tensorcore=True, # Check for Tensor Core intrinsics
                    verbose=False,          # Non-verbose mode (truncated output)
                    save_to_file="ir_cuda_core.txt"
                )
    except Exception as e:
        print(f"⚠️  Could not show IR: {e}")
        # Continue even if IR inspection fails
    
    # Calculate FLOPs
    total_flops = calculate_gcn_flops(g, dataset_info, num_hidden, num_layers)
    print(f"Total FLOPs: {total_flops:,}")
    
    # Benchmark
    print("Benchmarking...")
    try:
        mean_time_ms, std_time_ms, times_ms = benchmark_model(model)
        
        # Calculate TFLOPs
        mean_time_s = mean_time_ms / 1000.0
        tflops = (total_flops / 1e12) / mean_time_s
        
        print(f"\nResults:")
        print(f"  Latency: {mean_time_ms:.3f} ± {std_time_ms:.3f} ms")
        print(f"  Throughput: {tflops:.4f} TFLOPs")
        
        # Get output result for comparison
        print("Getting output result...")
        output_result = model.forward()
        
        return {
            "latency_ms": mean_time_ms,
            "latency_std_ms": std_time_ms,
            "tflops": tflops,
            "flops": total_flops,
            "output": output_result
        }
    except Exception as e:
        error_str = str(e)
        import traceback
        full_traceback = traceback.format_exc()
        print(f"\n✗ Benchmark/Execution failed:")
        print(f"  Error type: {type(e).__name__}")
        print(f"  Error message: {error_str[:500]}")
        if "block" in error_str.lower() or "cudalaunch" in error_str.lower() or "invalid_value" in error_str.lower():
            print(f"  Reason: TVM 0.10.0 block size bug (block size may be too large)")
        print(f"  Full traceback (first 2000 chars):\n{full_traceback[:2000]}")
        return None


def test_tensor_core(g, data, dataset_info):
    """Test Tensor Core implementation"""
    # Detect dtype from data
    dtype = get_dtype_from_data(data)
    print("\n" + "="*60)
    print(f"5. Test Tensor Core (GPU MMA) - {dtype}")
    print("="*60)
    
    # Check CUDA availability
    try:
        import tvm
        tvm.cuda(0)
    except:
        print("✗ CUDA not available, skipping")
        return None
    
    print(f"Dataset: {dataset_info['num_nodes']} nodes, {dataset_info['num_edges']} edges")
    
    num_layers = 2
    num_hidden = 32
    model = GraphConvNetwork(
        graph=g,
        data=data,
        dataset_info=dataset_info,
        num_hidden=num_hidden,
        num_layers=num_layers,
        device="cuda",
        opt_level=3,  # Need opt_level > 0 for optimizations
        use_gpu_mma=True,  # Enable Tensor Core
        dtype=dtype,  # Use detected dtype
        # # activation="relu"  # Enable ReLU activation (will be applied in Tensor Core computation)
    )
    
    print(f"Model: {num_layers} layers, hidden={num_hidden}")
    print(f"  Using {dtype} for all tensors")
    
    print("Compiling with Tensor Core...")
    try:
        model.compile()
        print("✓ Compiled with Tensor Core strategies")
    except Exception as e:
        error_str = str(e)
        import traceback
        full_traceback = traceback.format_exc()
        print(f"✗ Compilation failed: {error_str}")
        print(f"  Full traceback:\n{full_traceback}")
        return None
    
    # Calculate FLOPs
    total_flops = calculate_gcn_flops(g, dataset_info, num_hidden, num_layers)
    print(f"Total FLOPs: {total_flops:,}")
    
    # Benchmark
    print("Benchmarking...")
    try:
        mean_time_ms, std_time_ms, times_ms = benchmark_model(model)
        
        # Calculate TFLOPs
        mean_time_s = mean_time_ms / 1000.0
        tflops = (total_flops / 1e12) / mean_time_s
        
        print(f"\nResults:")
        print(f"  Latency: {mean_time_ms:.3f} ± {std_time_ms:.3f} ms")
        print(f"  Throughput: {tflops:.4f} TFLOPs")
        print(f"  ✓ Tensor Core implementation working!")
        
        # Get output result for comparison
        print("Getting output result...")
        output_result = model.forward()
        
        # Check Tensor Core usage using show_ir
        # This provides detailed IR inspection to verify Tensor Core usage
        print("\n" + "-"*60)
        print("IR Inspection (to verify Tensor Core usage)")
        print("-"*60)
        try:
            # Use show_ir to inspect low-level IR and check for Tensor Core
            ir_result = model.show_ir(
                show_relay=True,      # Skip Relay IR (high-level)
                show_tir=True,        # Skip TIR (can be verbose)
                show_source=True,      # Show generated CUDA code
                check_tensorcore=True, # Check for Tensor Core intrinsics
                verbose=False,          # Non-verbose mode (truncated output)
                save_to_file="ir_output.txt"
            )
            
            # Additional check using tensorcore_strategy
            try:
                from apps.gnn_tvm_utils.tensorcore_strategy import check_tensorcore_usage
                tensorcore_used = check_tensorcore_usage(model.lib, verbose=False)
                if tensorcore_used:
                    print(f"  ✓ Tensor Core intrinsics confirmed in generated code")
                else:
                    print(f"  ⚠️  Warning: Tensor Core intrinsics not found (may be using fallback)")
            except:
                pass
        except Exception as e:
            print(f"  ⚠️  Could not inspect IR: {e}")
            # Fallback to simple check
            try:
                from apps.gnn_tvm_utils.tensorcore_strategy import check_tensorcore_usage
                tensorcore_used = check_tensorcore_usage(model.lib, verbose=False)
                if tensorcore_used:
                    print(f"  ✓ Tensor Core intrinsics confirmed in generated code")
                else:
                    print(f"  ⚠️  Warning: Tensor Core intrinsics not found (may be using fallback)")
            except:
                pass
        
        return {
            "latency_ms": mean_time_ms,
            "latency_std_ms": std_time_ms,
            "tflops": tflops,
            "flops": total_flops,
            "output": output_result
        }
    except Exception as e:
        error_str = str(e)
        import traceback
        full_traceback = traceback.format_exc()
        print(f"✗ Execution failed: {error_str[:500]}")
        print(f"  Full traceback (first 1000 chars):\n{full_traceback[:1000]}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCN benchmark runner")
    parser.add_argument("--use-cora", action="store_true", help="Use Cora dataset instead of random synthetic graph")
    parser.add_argument("--cora-dir", type=str, default=os.path.join("data", "cora"), help="Path to Cora data directory")
    parser.add_argument("--pad-infeat-dim", action="store_true", help="Pad input feature dim to multiple of 64 (for Tensor Core)")
    parser.add_argument("--plot-max-nodes", type=int, default=512, help="Skip adjacency plot if nodes exceed this count")
    args = parser.parse_args()

    print("GCN Model Benchmark")
    print("="*60)
    print()
    print("Note: Float16 performance on CPU is typically slower than float32 because:")
    print("  1. CPUs lack native float16 arithmetic units")
    print("  2. Float16 values must be converted to float32 for computation, then back to float16")
    print("  3. These conversion overheads outweigh the memory bandwidth benefits")
    print("  4. AVX2/AVX512 instructions are optimized for float32, not float16")
    print("  Float16 is primarily beneficial on GPUs with Tensor Core support")
    print()
    
     # Load unified dataset (use smallest to ensure GPU compatibility)
    # Block size = num_nodes * num_hidden, must be <= 1024 for CUDA
    # Use num_nodes=32, num_hidden=32 (block size = 1024, at limit)
    print("Loading unified dataset...")
    
    # Option 1: Use single molecule (smaller graph, fewer edges)
    
    # MOLECULE_IDX = 65  # Index of molecule to load (0-187)
    # # 27 nodes (<=24 nodes will fail to compile)
    # # Not correct enough
    # g_f32, data_f32, dataset_info_f32 = load_single_mutag_molecule(
    #     molecule_idx=MOLECULE_IDX,
    #     self_loop=True,
    #     dtype="float32",  # Use float32 for CPU and CUDA Core tests
    #     pad_infeat_dim_to_32=True  # Don't pad for CPU/CUDA Core - padding only for Tensor Core
    # )
    # g_f16, data_f16, dataset_info_f16 = load_single_mutag_molecule(
    #     molecule_idx=MOLECULE_IDX,
    #     self_loop=True,
    #     dtype="float16",  # Use float16 for Tensor Core tests
    #     pad_infeat_dim_to_32=True  # Don't pad for CPU/CUDA Core - padding only for Tensor Core
    # )
    
    # # For Tensor Core tests, we need padded data
    # # Load separate datasets with padding for Tensor Core
    # g_f16_tc, data_f16_tc, dataset_info_f16_tc = load_single_mutag_molecule(
    #     molecule_idx=MOLECULE_IDX,
    #     self_loop=True,
    #     dtype="float16",
    #     pad_infeat_dim_to_32=True  # Pad ONLY for Tensor Core
    # )
    
    # print(f"Using single molecule (index {MOLECULE_IDX})")

    # Option 2: Use merged graph (all molecules, large graph)
    # g_f32, data_f32, dataset_info_f32 = load_mutag_dataset(
    #     data_dir=data_dir,
    #     dtype="float32"
    # )
    # g_f16, data_f16, dataset_info_f16 = load_mutag_dataset(
    #     data_dir=data_dir,
    #     dtype="float16"
    # )
    # print("Using merged graph (all molecules)")
    
    # Option 3: Use random dataset
    # Correct!


    # g_f32, data_f32, dataset_info_f32 = random_graph_dataset(
    #     num_nodes=128,
    #     infeat_dim=32,
    #     num_classes=32,
    #     sparsity=0.8358,
    #     dtype="float32"
    # )

    # g_f16, data_f16, dataset_info_f16 = random_graph_dataset(
    #     num_nodes=128,
    #     infeat_dim=32,
    #     num_classes=32,
    #     sparsity=0.8358,
    #     dtype="float16"
    # )

    print("Loading dataset...")
    if args.use_cora:
        print(f"Using Cora dataset from {args.cora_dir}")
        g_f32, data_f32, dataset_info_f32 = load_cora_dataset(
            data_dir=args.cora_dir,
            dtype="float32",
            pad_infeat_dim=args.pad_infeat_dim
        )
        g_f16, data_f16, dataset_info_f16 = load_cora_dataset(
            data_dir=args.cora_dir,
            dtype="float16",
            pad_infeat_dim=args.pad_infeat_dim
        )
    else:
        
        g_f32, data_f32, dataset_info_f32 = random_graph_dataset(
            num_nodes=2752,
            infeat_dim=1472,
            num_classes=32,
            sparsity=0.9987,
            dtype="float32"
        )
        g_f16, data_f16, dataset_info_f16 = random_graph_dataset(
            num_nodes=2752,
            infeat_dim=1472,
            num_classes=32,
            sparsity=0.9987,
            dtype="float16"
        )

    # g_f32, data_f32, dataset_info_f32 = random_block_cluster_dataset(
    #     num_nodes=2048,
    #     infeat_dim=32,
    #     num_classes=32,
    #     num_clusters=6,
    #     intra_density=0.98,  # Increased from 0.95 to make cluster blocks more dense
    #     inter_density=0.001,  # Keep inter-cluster sparse to maintain block structure
    #     seed=22,
    #     dtype="float32"
    # )
    # g_f16, data_f16, dataset_info_f16 = random_block_cluster_dataset(
    #     num_nodes=2048,
    #     infeat_dim=32,
    #     num_classes=32,
    #     num_clusters=6,
    #     intra_density=0.98,  # Increased from 0.95 to make cluster blocks more dense
    #     inter_density=0.001,  # Keep inter-cluster sparse to maintain block structure
    #     seed=22,
    #     dtype="float16"
    # )

    if dataset_info_f16["num_nodes"] <= args.plot_max_nodes:
        plot_adj_matrix(g_f16, "adj_matrix.png")
    else:
        print(f"Skipping adjacency plot (num_nodes={dataset_info_f16['num_nodes']} > {args.plot_max_nodes})")

    # plot_adj_matrix(g_f16, "adj_matrix.png")
    # print(f"Float16 dataset: {dataset_info_f16['num_nodes']} nodes, {dataset_info_f16['num_edges']} edges")
    print(f"Float16 dataset: {dataset_info_f16['num_nodes']} nodes, {dataset_info_f16['num_edges']} edges")
    print(f"  infeat_dim={dataset_info_f16['infeat_dim']}, num_classes={dataset_info_f16['num_classes']}")
    print()
    
    results = {}
    
    # IMPORTANT: Run ALL CPU tests FIRST before any CUDA registration
    # This ensures CPU tests use the original sparse_dense without any modifications
    # CUDA tests will register alter_op_layout inside test_cuda_core function
    
    
    # Test 1: CPU LLVM (both float32 and float16)
    print("\n" + "="*60)
    print("Testing CPU LLVM with float32 and float16")
    print("="*60)
    try:
        results["CPU_LLVM_float32"] = test_cpu_llvm(g_f32, data_f32, dataset_info_f32)
    except Exception as e:
        print(f"✗ CPU LLVM (float32) failed: {e}")
        import traceback
        traceback.print_exc()
        results["CPU_LLVM_float32"] = None
    
    try:
        results["CPU_LLVM_float16"] = test_cpu_llvm(g_f16, data_f16, dataset_info_f16)
    except Exception as e:
        print(f"✗ CPU LLVM (float16) failed: {e}")
        import traceback
        traceback.print_exc()
        results["CPU_LLVM_float16"] = None
    
    # Add PyTorch reference outputs (reuses CPU LLVM weights/graph)
    if results.get("CPU_LLVM_float32") and results["CPU_LLVM_float32"].get("pytorch_output") is not None:
        results["PYTORCH_REF_float32"] = {"output": results["CPU_LLVM_float32"]["pytorch_output"]}
    if results.get("CPU_LLVM_float16") and results["CPU_LLVM_float16"].get("pytorch_output") is not None:
        results["PYTORCH_REF_float16"] = {"output": results["CPU_LLVM_float16"]["pytorch_output"]}
    
    # Test 2: CPU LLVM AVX2 (both float32 and float16)
    print("\n" + "="*60)
    print("Testing CPU LLVM AVX2 with float32 and float16")
    print("="*60)
    try:
        results["CPU_LLVM_AVX2_float32"] = test_cpu_llvm_avx2(g_f32, data_f32, dataset_info_f32)
    except Exception as e:
        print(f"✗ CPU LLVM AVX2 (float32) failed: {e}")
        import traceback
        traceback.print_exc()
        results["CPU_LLVM_AVX2_float32"] = None
    
    try:
        results["CPU_LLVM_AVX2_float16"] = test_cpu_llvm_avx2(g_f16, data_f16, dataset_info_f16)
    except Exception as e:
        print(f"✗ CPU LLVM AVX2 (float16) failed: {e}")
        import traceback
        traceback.print_exc()
        results["CPU_LLVM_AVX2_float16"] = None
    
    # Test 3: CPU LLVM AVX512 (both float32 and float16)
    print("\n" + "="*60)
    print("Testing CPU LLVM AVX512 with float32 and float16")
    print("="*60)
    try:
        results["CPU_LLVM_AVX512_float32"] = test_cpu_llvm_avx512(g_f32, data_f32, dataset_info_f32)
    except Exception as e:
        print(f"✗ CPU LLVM AVX512 (float32) failed: {e}")
        import traceback
        traceback.print_exc()
        results["CPU_LLVM_AVX512_float32"] = None
    
    try:
        results["CPU_LLVM_AVX512_float16"] = test_cpu_llvm_avx512(g_f16, data_f16, dataset_info_f16)
    except Exception as e:
        print(f"✗ CPU LLVM AVX512 (float16) failed: {e}")
        import traceback
        traceback.print_exc()
        results["CPU_LLVM_AVX512_float16"] = None
    
    # Test 4: CUDA Core (both float32 and float16)
    # Registration happens inside this function, so it won't affect CPU tests
    print("\n" + "="*60)
    print("Testing CUDA Core with float32 and float16")
    print("="*60)

    try:
        results["CUDA_CORE_float32"] = test_cuda_core(g_f32, data_f32, dataset_info_f32)
    except Exception as e:
        print(f"✗ CUDA Core (float32) failed: {e}")
        import traceback
        traceback.print_exc()
        results["CUDA_CORE_float32"] = None
    
    try:
        results["CUDA_CORE_float16"] = test_cuda_core(g_f16, data_f16, dataset_info_f16)
    except Exception as e:
        print(f"✗ CUDA Core (float16) failed: {e}")
        import traceback
        traceback.print_exc()
        results["CUDA_CORE_float16"] = None
    

    # Test 5: Tensor Core (both float32 and float16)
    # Use padded data for Tensor Core tests
    print("\n" + "="*60)
    print("Testing Tensor Core with float32 and float16")
    print("="*60)
    results["TENSOR_CORE_float32"] = None
    
    try:
        # Use padded data for Tensor Core
        results["TENSOR_CORE_float16"] = test_tensor_core(g_f16, data_f16, dataset_info_f16)
    except Exception as e:
        print(f"✗ Tensor Core (float16) failed: {e}")
        import traceback
        traceback.print_exc()
        results["TENSOR_CORE_float16"] = None

    
    
    # Print summary
    print("\n" + "="*80)
    print("Benchmark Summary")
    print("="*80)
    print(f"{'Test':<35} {'Latency (ms)':<20} {'TFLOPs':<15}")
    print("-" * 80)
    
    # Group results by test type
    test_groups = {
        "CPU_LLVM": ["CPU_LLVM_float32", "CPU_LLVM_float16"],
        "CPU_LLVM_AVX2": ["CPU_LLVM_AVX2_float32", "CPU_LLVM_AVX2_float16"],
        "CPU_LLVM_AVX512": ["CPU_LLVM_AVX512_float32", "CPU_LLVM_AVX512_float16"],
        "CUDA_CORE": ["CUDA_CORE_float32", "CUDA_CORE_float16"],
        "TENSOR_CORE": ["TENSOR_CORE_float32", "TENSOR_CORE_float16"]
    }
    
    for group_name, test_keys in test_groups.items():
        print(f"\n{group_name}:")
        for test_name in test_keys:
            result = results.get(test_name)
            if result is not None:
                dtype_label = "float32" if "float32" in test_name else "float16"
                print(f"  {dtype_label:<10} {result['latency_ms']:>8.3f} ± {result['latency_std_ms']:>6.3f} ms  {result['tflops']:>10.4f} TFLOPs")
            else:
                dtype_label = "float32" if "float32" in test_name else "float16"
                print(f"  {dtype_label:<10} {'N/A':<20} {'N/A':<15}")
    
    print("="*80)
    
    # Compare outputs for float32 tests
    print("\n" + "="*80)
    print("Comparing Float32 Test Results")
    print("="*80)
    float32_results = {
        k: v for k, v in results.items() 
        if v is not None and "float32" in k and "output" in v
    }
    if len(float32_results) > 0:
        # Float32: Standard tolerance
        # rtol=1e-2 (1% relative error) - accounts for different optimization strategies
        # atol=1e-4 (0.0001 absolute error) - accounts for numerical precision differences
        compare_outputs(float32_results, rtol=1e-2, atol=1e-4)
    
    # Compare outputs for float16 tests
    print("\n" + "="*80)
    print("Comparing Float16 Test Results")
    print("="*80)
    float16_results = {
        k: v for k, v in results.items() 
        if v is not None and "float16" in k and "output" in v
    }
    if len(float16_results) > 0:
        # Float16: More lenient tolerance due to precision limitations
        # rtol=5e-2 (5% relative error) - float16 has only 3-4 significant digits
        # atol=1e-1 (0.1 absolute error) - accounts for float16 precision and optimization differences
        compare_outputs(float16_results, rtol=5e-2, atol=1e-1)
        
        # Detailed analysis for float16
        print("\n" + "="*80)
        print("Detailed Float16 Analysis")
        print("="*80)
        try:
            from analyze_tensorcore_accuracy import analyze_differences
            float16_outputs = {k: v['output'] for k, v in float16_results.items()}
            analyze_differences(float16_outputs)
        except ImportError:
            print("Could not import analyze_differences function")
        except Exception as e:
            print(f"Analysis failed: {e}")
    
    # Visualize outputs
    print("\n" + "="*80)
    print("Visualizing Outputs")
    print("="*80)
    try:
        # Visualize float32 outputs
        float32_outputs = {
            k: v['output'] for k, v in results.items() 
            if v is not None and "float32" in k and "output" in v
        }
        if len(float32_outputs) > 0:
            print("\nVisualizing Float32 outputs...")
            # Use the graph from the first test
            g_vis = g_f32 if g_f32 is not None else g_f16
            if g_vis is not None:
                visualize_graph_output_comparison(
                    g_vis, float32_outputs, 
                    reduction='max',
                    layout='spring',
                    save_path="output_visualization_float32.png"
                )
                # Also create individual visualizations
                for name, output in list(float32_outputs.items())[:3]:  # Limit to first 3
                    visualize_graph_output(
                        g_vis, output,
                        reduction='max',
                        layout='spring',
                        save_path=f"output_{name.replace(' ', '_')}.png",
                        title=name
                    )
        
        # Visualize float16 outputs
        float16_outputs = {
            k: v['output'] for k, v in results.items() 
            if v is not None and "float16" in k and "output" in v
        }
        if len(float16_outputs) > 0:
            print("\nVisualizing Float16 outputs...")
            g_vis = g_f16 if g_f16 is not None else g_f32
            if g_vis is not None:
                visualize_graph_output_comparison(
                    g_vis, float16_outputs,
                    reduction='max',
                    layout='spring',
                    save_path="output_visualization_float16.png"
                )
    except Exception as e:
        print(f"Visualization failed: {e}")
        import traceback
        traceback.print_exc()
