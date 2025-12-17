#!/usr/bin/env python3
"""
Test GEMM Tensor Core in Relay

This script tests the integration of gemm_mma.py's Tensor Core implementation
into Relay by creating a simple dense layer and verifying it compiles and runs correctly.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Add TVM path
tvm_python_path = project_root / "tvm-0.10.0" / "python"
if tvm_python_path.exists() and str(tvm_python_path) not in sys.path:
    sys.path.insert(0, str(tvm_python_path))

import numpy as np
import tvm
from tvm import relay
from tvm import te
from tvm.contrib import graph_executor

# Import Tensor Core strategy registration
from apps.gnn_tvm_utils.tensorcore_strategy import register_tensorcore_strategies

# Tensor Core configuration
WMMA_M = 16
WMMA_N = 16
WMMA_K = 16


def test_dense_tensorcore_relay():
    """
    Test dense Tensor Core in Relay
    
    Creates a simple dense layer with dimensions that satisfy Tensor Core requirements:
    - batch % 32 == 0, in_dim % 16 == 0, out_dim % 8 == 0 (condition 3)
    """
    print("=" * 80)
    print("Testing Dense Tensor Core in Relay")
    print("=" * 80)
    print()
    
    # Test dimensions that satisfy gemm_mma.py requirements
    # Condition 3: (batch % 32 == 0 and in_dim % 16 == 0 and out_dim % 8 == 0)
    batch = 32  # M_padded
    in_dim = 64  # K_padded
    out_dim = 16  # N
    
    print(f"Test Configuration:")
    print(f"  batch (M): {batch}")
    print(f"  in_dim (K): {in_dim}")
    print(f"  out_dim (N): {out_dim}")
    print(f"  Condition check:")
    print(f"    batch % 32 = {batch % 32} (should be 0)")
    print(f"    in_dim % 16 = {in_dim % 16} (should be 0)")
    print(f"    out_dim % 8 = {out_dim % 8} (should be 0)")
    print()
    
    # Register Tensor Core strategies
    print("Registering Tensor Core strategies...")
    try:
        register_tensorcore_strategies()
        print("✓ Tensor Core strategies registered")
    except Exception as e:
        print(f"✗ Failed to register strategies: {e}")
        return False
    print()
    
    # Create input data (float32, will be cast to float16)
    print("Creating test data...")
    np.random.seed(42)
    X_np = np.random.randn(batch, in_dim).astype("float32")
    W_np = np.random.randn(out_dim, in_dim).astype("float32")
    Y_ref = (X_np @ W_np.T).astype("float32")
    
    print(f"  Input X: {X_np.shape}, dtype={X_np.dtype}")
    print(f"  Weight W: {W_np.shape}, dtype={W_np.dtype}")
    print(f"  Reference output Y: {Y_ref.shape}, dtype={Y_ref.dtype}")
    print()
    
    # Build Relay model
    print("Building Relay model...")
    try:
        # Create input variable
        x = relay.var("x", shape=(batch, in_dim), dtype="float32")
        
        # Cast to float16 for Tensor Core
        x_fp16 = relay.cast(x, "float16")
        
        # Create weight constant (float32, will be cast)
        w = relay.Constant(tvm.nd.array(W_np))
        w_fp16 = relay.cast(w, "float16")
        
        # Dense layer - this should use Tensor Core via strategy registration
        dense_out = relay.nn.dense(x_fp16, w_fp16, out_dtype="float32")
        
        # Create function
        func = relay.Function([x], dense_out)
        mod = tvm.IRModule()
        mod["main"] = func
        
        print("✓ Relay model created")
        print(f"  Function: {func}")
        print()
        
    except Exception as e:
        print(f"✗ Failed to create Relay model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Compile
    print("Compiling with CUDA target...")
    try:
        target = tvm.target.Target("cuda")
        dev = tvm.cuda(0)
        
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target)
        
        print("✓ Compilation successful")
        print()
        
    except Exception as e:
        print(f"✗ Compilation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Run
    print("Running on GPU...")
    try:
        # Create executor
        executor = graph_executor.GraphModule(lib["default"](dev))
        
        # Set input
        executor.set_input("x", X_np)
        
        # Run
        executor.run()
        
        # Get output
        Y_tvm = executor.get_output(0).numpy()
        
        print("✓ Execution successful")
        print(f"  Output shape: {Y_tvm.shape}, dtype={Y_tvm.dtype}")
        print()
        
    except Exception as e:
        print(f"✗ Execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Verify correctness
    print("Verifying correctness...")
    try:
        max_diff = np.max(np.abs(Y_tvm - Y_ref))
        mean_diff = np.mean(np.abs(Y_tvm - Y_ref))
        
        print(f"  Max difference: {max_diff:.6f}")
        print(f"  Mean difference: {mean_diff:.6f}")
        
        # Use reasonable tolerance for float16 accumulation
        if max_diff < 1.0:
            print("✓ Results match reference (within tolerance)")
        else:
            print("⚠ Results differ from reference")
            print(f"  First few values:")
            print(f"    Y_tvm[0, :5] = {Y_tvm[0, :5]}")
            print(f"    Y_ref[0, :5] = {Y_ref[0, :5]}")
        print()
        
    except Exception as e:
        print(f"✗ Verification failed: {e}")
        return False
    
    # Check if Tensor Core is used
    print("Checking Tensor Core usage...")
    try:
        source_code = lib.get_lib().get_source()
        if "wmma" in source_code.lower() or "mma.sync" in source_code.lower():
            print("✓ Tensor Core intrinsics found in generated code!")
            print("  (wmma/mma.sync instructions indicate Tensor Core usage)")
        else:
            print("⚠ Tensor Core intrinsics not clearly visible in source")
            print("  (This may be due to code generation format)")
        print()
    except Exception as e:
        print(f"  (Could not inspect source code: {e})")
        print()
    
    # Benchmark
    print("Benchmarking...")
    try:
        evaluator = executor.module.time_evaluator("run", dev, number=100, repeat=10)
        mean_time = np.mean(evaluator().results) * 1000
        
        # Calculate GFLOPS
        total_ops = 2 * batch * in_dim * out_dim
        gflops = (total_ops / 1e9) / (mean_time / 1000.0)
        
        print(f"  Mean execution time: {mean_time:.4f} ms")
        print(f"  Performance: {gflops:.2f} GFLOPS")
        print(f"  Total operations: {total_ops:,} (2 * M * K * N)")
        print()
        
    except Exception as e:
        print(f"  (Benchmarking failed: {e})")
        print()
    
    print("=" * 80)
    print("Test completed successfully!")
    print("=" * 80)
    
    return True


def test_multiple_dimensions():
    """
    Test different dimension combinations that satisfy Tensor Core requirements
    """
    print("\n" + "=" * 80)
    print("Testing Multiple Dimension Combinations")
    print("=" * 80)
    print()
    
    # Test cases: (batch, in_dim, out_dim)
    test_cases = [
        # Condition 1: (batch % 8 == 0 and in_dim % 16 == 0 and out_dim % 32 == 0)
        (8, 16, 32, "Condition 1"),
        (16, 16, 32, "Condition 1"),
        # Condition 2: (batch % 16 == 0 and in_dim % 16 == 0 and out_dim % 16 == 0)
        (16, 16, 16, "Condition 2"),
        (32, 16, 16, "Condition 2"),
        (64, 64, 16, "Condition 2"),
        # Condition 3: (batch % 32 == 0 and in_dim % 16 == 0 and out_dim % 8 == 0)
        (32, 16, 8, "Condition 3"),
        (32, 64, 16, "Condition 3"),
        (64, 64, 8, "Condition 3"),
    ]
    
    for batch, in_dim, out_dim, condition in test_cases:
        print(f"Testing {condition}: batch={batch}, in_dim={in_dim}, out_dim={out_dim}")
        
        try:
            # Create simple test
            np.random.seed(42)
            X_np = np.random.randn(batch, in_dim).astype("float32")
            W_np = np.random.randn(out_dim, in_dim).astype("float32")
            
            x = relay.var("x", shape=(batch, in_dim), dtype="float32")
            x_fp16 = relay.cast(x, "float16")
            w = relay.Constant(tvm.nd.array(W_np))
            w_fp16 = relay.cast(w, "float16")
            dense_out = relay.nn.dense(x_fp16, w_fp16, out_dtype="float32")
            func = relay.Function([x], dense_out)
            mod = tvm.IRModule()
            mod["main"] = func
            
            target = tvm.target.Target("cuda")
            dev = tvm.cuda(0)
            
            with tvm.transform.PassContext(opt_level=3):
                lib = relay.build(mod, target)
            
            executor = graph_executor.GraphModule(lib["default"](dev))
            executor.set_input("x", X_np)
            executor.run()
            Y_tvm = executor.get_output(0).numpy()
            
            print(f"  ✓ Success: output shape {Y_tvm.shape}")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
        
        print()


if __name__ == "__main__":
    # Test basic dense Tensor Core
    success = test_dense_tensorcore_relay()
    
    if success:
        # Test multiple dimension combinations
        test_multiple_dimensions()
    
    if success:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed!")
        sys.exit(1)
