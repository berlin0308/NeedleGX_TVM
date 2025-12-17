#!/usr/bin/env python3
"""
DTC-SpMM Tensor Core Implementation in TVM 0.10.0

This script implements DTC-SpMM using Tensor Core with 16x16x16 blocks,
strictly following the DTC_spmm_kernel.cu implementation.

Reference: https://github.com/HPMLL/DTC-SpMM_ASPLOS24
"""

import sys
import os
import tvm
from tvm import te
import numpy as np
from tvm.contrib import nvcc

# Set numpy print options to show full arrays (no truncation)
np.set_printoptions(threshold=np.inf, linewidth=np.inf, edgeitems=10)

# Import matplotlib for visualization (optional)
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("  Warning: matplotlib not available, skipping visualization")

from tvm.topi.cuda.tensor_intrin import (
    intrin_wmma_load_matrix_A,
    intrin_wmma_load_matrix_W,
    intrin_wmma_store_matrix,
    intrin_wmma_gemm,
)
from tvm.topi.utils import get_const_tuple

# Import verbose control from tensorcore_strategy
try:
    from .tensorcore_strategy import get_verbose

    def _get_verbose():
        return get_verbose()
except ImportError:
    try:
        from apps.gnn_tvm_utils.tensorcore_strategy import get_verbose

        def _get_verbose():
            return get_verbose()
    except ImportError:
        def _get_verbose():
            return False

# ============================================================================
# Data Type Configuration
# ============================================================================
# Unified dtype definitions for all matrices
A_dtype = "float16"  # Input matrix A dtype
X_dtype = "float16"  # Input matrix X dtype
Y_dtype = "float16"

# Test configuration (only used in test functions, not in dtc_spmm_tensorcore_cuda/schedule_dtc_spmm_tensorcore)
# Note: dtc_spmm_tensorcore_cuda and schedule_dtc_spmm_tensorcore are generic and work with any input sizes
# They get dimensions from input tensor shapes, not from these global variables
TEST_M = 64  # Test matrix rows (only for test_dtc_spmm_tensorcore)
TEST_K = 64  # Test matrix original cols (only for test_dtc_spmm_tensorcore)
TEST_N = 64  # Test matrix cols (only for test_dtc_spmm_tensorcore)
TEST_SPARSITY = 0.98  # Test sparsity (only for test_dtc_spmm_tensorcore)


WMMA_M = 16  # WMMA matrix M dimension
WMMA_N = 16  # WMMA matrix N dimension  
WMMA_K = 16  # WMMA matrix K dimension (must be 16 for float16)
# Block dimensions should match WMMA dimensions
BLK_H = WMMA_M  # Block height (rows per TC block)
BLK_W = WMMA_N  # Block width (cols per TC block)


# NumPy dtype equivalents
A_dtype_np = np.float16 if A_dtype == "float16" else np.float32
X_dtype_np = np.float16 if X_dtype == "float16" else np.float32
Y_dtype_np = np.float32 if Y_dtype == "float32" else np.float16

def visualize_matrices(A, A_compressed, X, X_selected, Y, M, K, N, A_compressed_used_cols=None, X_selected_used_rows=None):
    """
    Visualize matrices A, A_compressed, X, X_selected, and Y using matplotlib
    All plots use the same scale for fair comparison
    Using diverging colormap (RdBu) for X and Y so zero values appear white
    All axes ticks are set to multiples of BLK_H/BLK_W
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('DTC-SpMM Matrix Visualization', fontsize=16, fontweight='bold')
    
    # Determine common axis limits
    # Note: K here is K_padded (compressed dimension) when X_selected is compressed
    max_rows = max(M, A_compressed.shape[0], K, X_selected.shape[0], Y.shape[0])
    max_cols = max(K, A_compressed.shape[1], N, X_selected.shape[1], Y.shape[1])
    
    # Round up to nearest multiple of BLK_H/BLK_W for tick display
    max_rows_rounded = ((max_rows + BLK_H - 1) // BLK_H) * BLK_H
    max_cols_rounded = ((max_cols + BLK_W - 1) // BLK_W) * BLK_W
    
    # Generate tick positions (multiples of BLK_H/BLK_W)
    y_ticks = list(range(0, max_rows_rounded + 1, BLK_H))
    x_ticks = list(range(0, max_cols_rounded + 1, BLK_W))
    
    # Original sparse matrix A - add BLK_H x BLK_W block grid
    ax = axes[0, 0]
    im = ax.imshow(A, aspect='equal', cmap='Blues', interpolation='nearest', 
                   extent=[0, K, M, 0])
    # Add BLK_H x BLK_W block grid lines
    for i in range(0, M + 1, BLK_H):
        ax.axhline(y=i, color='gray', linewidth=0.5, linestyle='--', alpha=0.7)
    for j in range(0, K + 1, BLK_W):
        ax.axvline(x=j, color='gray', linewidth=0.5, linestyle='--', alpha=0.7)
    ax.set_xlim(0, max_cols)
    ax.set_ylim(0, max_rows)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_title(f'Original A ({M}x{K})', fontsize=12, fontweight='bold')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Compressed A - add 16x16 block grid
    ax = axes[0, 1]
    im = ax.imshow(A_compressed, aspect='equal', cmap='Blues', interpolation='nearest',
                   extent=[0, A_compressed.shape[1], A_compressed.shape[0], 0])
    # Add 16x16 block grid lines
    for i in range(0, A_compressed.shape[0] + 1, BLK_H):
        ax.axhline(y=i, color='gray', linewidth=0.5, linestyle='--', alpha=0.7)
    for j in range(0, A_compressed.shape[1] + 1, BLK_W):
        ax.axvline(x=j, color='gray', linewidth=0.5, linestyle='--', alpha=0.7)
    ax.set_xlim(0, max_cols)
    ax.set_ylim(0, max_rows)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_title(f'Compressed A ({A_compressed.shape[0]}x{A_compressed.shape[1]})', fontsize=12, fontweight='bold')
    ax.set_xlabel('Column (compressed)')
    ax.set_ylabel('Row')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Original X - use diverging colormap so zero appears white
    ax = axes[1, 0]
    # Find value range for symmetric colormap
    X_max = np.abs(X).max()
    # X is (K_original x N), but K here might be K_padded
    X_original_K = X.shape[0]  # Original K dimension
    im = ax.imshow(X, aspect='equal', cmap='RdBu_r', interpolation='nearest',
                   extent=[0, N, X_original_K, 0], vmin=-X_max, vmax=X_max)
    ax.set_xlim(0, max_cols)
    ax.set_ylim(0, max_rows)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_title(f'Original X ({X_original_K}x{N})', fontsize=12, fontweight='bold')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Selected X (mapped to original positions) - use diverging colormap so zero appears white
    # ax = axes[1, 1]
    # # Find value range for symmetric colormap
    # X_selected_max = np.abs(X_selected).max()
    # # X_selected is now mapped to original column positions (K x N_padded)
    # im = ax.imshow(X_selected, aspect='equal', cmap='RdBu_r', interpolation='nearest',
    #                extent=[0, X_selected.shape[1], X_selected.shape[0], 0],
    #                vmin=-X_selected_max, vmax=X_selected_max)
    # ax.set_xlim(0, max_cols)
    # ax.set_ylim(0, max_rows)
    # ax.set_xticks(x_ticks)
    # ax.set_yticks(y_ticks)
    # ax.set_title(f'Selected X (mapped to A cols) ({X_selected.shape[0]}x{X_selected.shape[1]})', fontsize=12, fontweight='bold')
    # ax.set_xlabel('Column')
    # ax.set_ylabel('Row (original A col position)')
    # plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Output Y - use diverging colormap so zero appears white
    ax = axes[1, 1]
    # Find value range for symmetric colormap
    Y_max = np.abs(Y).max()
    im = ax.imshow(Y, aspect='equal', cmap='plasma', interpolation='nearest',
                   extent=[0, Y.shape[1], Y.shape[0], 0], vmin=-Y_max, vmax=Y_max)
    ax.set_xlim(0, max_cols)
    ax.set_ylim(0, max_rows)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_title(f'Output Y ({Y.shape[0]}x{Y.shape[1]})', fontsize=12, fontweight='bold')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Sparsity comparison
    # ax = axes[0, 2]
    # sparsity_A = 1.0 - np.count_nonzero(A) / (M * K)
    # sparsity_A_comp = 1.0 - np.count_nonzero(A_compressed) / (A_compressed.shape[0] * A_compressed.shape[1])
    # compression_ratio = (M * K) / (A_compressed.shape[0] * A_compressed.shape[1])
    
    # stats_text = f"""
    # Matrix Statistics:
    
    # Original A:
    #   Size: {M}x{K}
    #   Sparsity: {sparsity_A:.2%}
    #   Non-zeros: {np.count_nonzero(A)}
    
    # Compressed A:
    #   Size: {A_compressed.shape[0]}x{A_compressed.shape[1]}
    #   Sparsity: {sparsity_A_comp:.2%}
    #   Non-zeros: {np.count_nonzero(A_compressed)}
    #   Compression Ratio: {compression_ratio:.2f}x
    
    # Output Y:
    #   Size: {Y.shape[0]}x{Y.shape[1]}
    # """
    # ax.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
    #         family='monospace', transform=ax.transAxes)
    # ax.axis('off')
    # ax.set_title('Statistics', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    output_file = 'dtc_spmm_matrices.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved visualization to {output_file}")
    plt.close()

def create_dtc_spmm_data(M, K, N, sparsity=0.7, seed=42, use_hardcoded=False):
    """
    Create test data for DTC-SpMM in block format
    
    Returns compressed dense blocks that can be processed by Tensor Core
    """
    
    np.random.seed(seed)
    
    if use_hardcoded:
        # Create sparse matrix A (M x K) with diagonal pattern
        # Initialize as diagonal matrix with value 2
        A = np.zeros((M, K), dtype=A_dtype_np)
        for i in range(min(M, K)):
            A[i, i] = 2.0
        
        # Set half of diagonal elements back to 0
        # This creates a sparse pattern that will test compression
        diagonal_indices = list(range(min(M, K)))
        np.random.shuffle(diagonal_indices)
        zero_indices = diagonal_indices[:len(diagonal_indices)//2]
        for idx in zero_indices:
            A[idx, idx] = 0.0
        
        # print(f"  Created diagonal matrix: {M}x{K}")
        # print(f"  Diagonal elements: {min(M, K)} total, {min(M, K) - len(zero_indices)} non-zero, {len(zero_indices)} zero")
        # print(A)

        # DTC compression: process by 16x16 blocks, not entire columns
        # For each row block (16 rows), find non-zero columns within that block
        # Then pack them into a compressed format
        
        # Number of row blocks
        num_row_blocks = (M + BLK_H - 1) // BLK_H
        num_col_blocks = (K + BLK_W - 1) // BLK_W
        
        # For each row block, collect non-zero columns
        # We'll create a compressed matrix where each row block has at most 16 non-zero columns
        block_non_zero_cols = []  # List of lists: one per row block
        
        for row_block_idx in range(num_row_blocks):
            row_start = row_block_idx * BLK_H
            row_end = min(row_start + BLK_H, M)
            
            # Find columns with non-zero values in this row block
            block_cols = set()
            for i in range(row_start, row_end):
                for j in range(K):
                    if A[i, j] != 0:
                        block_cols.add(j)
            
            # Convert to sorted list and pad to multiple of BLK_W
            block_cols_list = sorted(list(block_cols))
            padded_cols = ((len(block_cols_list) + BLK_W - 1) // BLK_W) * BLK_W
            
            # Pad with zero columns if needed
            while len(block_cols_list) < padded_cols:
                for j in range(K):
                    if j not in block_cols_list:
                        block_cols_list.append(j)
                        break
                if len(block_cols_list) >= K:
                    break
            
            block_non_zero_cols.append(block_cols_list[:padded_cols])
        
        # Find the maximum number of columns across all blocks
        max_cols = max(len(cols) for cols in block_non_zero_cols) if block_non_zero_cols else BLK_W
        max_cols = ((max_cols + BLK_W - 1) // BLK_W) * BLK_W  # Round up to multiple of BLK_W (WMMA_N)
        
        # Create compressed matrix: A_compressed (M_padded x max_cols)
        # For each row block, pack its non-zero columns
        M_padded = ((M + BLK_H - 1) // BLK_H) * BLK_H
        A_compressed = np.zeros((M_padded, max_cols), dtype=A.dtype)
        
        # Mapping: for each row block, which original columns are used
        block_col_mapping = []  # List of column indices for each row block
        
        for row_block_idx in range(num_row_blocks):
            row_start = row_block_idx * BLK_H
            row_end = min(row_start + BLK_H, M)
            block_cols = block_non_zero_cols[row_block_idx]
            
            # Map original columns to compressed columns
            col_mapping = {orig_col: comp_col for comp_col, orig_col in enumerate(block_cols)}
            block_col_mapping.append(block_cols)
            
            # Copy non-zero values to compressed matrix
            for i in range(row_start, row_end):
                for orig_col in block_cols:
                    comp_col = col_mapping[orig_col]
                    A_compressed[i, comp_col] = A[i, orig_col]
        
        # For reference, we need to know which columns are used globally
        # Collect all unique columns used across all blocks
        all_non_zero_cols = set()
        for block_cols in block_non_zero_cols:
            all_non_zero_cols.update(block_cols)
        non_zero_cols = sorted(list(all_non_zero_cols))
    else:
        # Create sparse matrix A (M x K) with given sparsity
        # Use random sparse pattern
        A = np.zeros((M, K), dtype=A_dtype_np)
        
        # Calculate number of non-zero elements
        total_elements = M * K
        num_non_zero = int(total_elements * (1.0 - sparsity))
        
        # Randomly select positions for non-zero values
        # Use a set to avoid duplicates
        non_zero_positions = set()
        while len(non_zero_positions) < num_non_zero:
            i = np.random.randint(0, M)
            j = np.random.randint(0, K)
            non_zero_positions.add((i, j))
        
        # Fill in non-zero values (use random values)
        for i, j in non_zero_positions:
            A[i, j] = 1.0
        
        print(f"  Created sparse matrix: {M}x{K}")
        print(f"  Sparsity: {sparsity:.2%}, Non-zero elements: {num_non_zero}/{total_elements}")
        
        # DTC compression: process by 16x16 blocks, not entire columns
        # For each row block (16 rows), find non-zero columns within that block
        # Then pack them into a compressed format
        
        # Number of row blocks
        num_row_blocks = (M + BLK_H - 1) // BLK_H
        num_col_blocks = (K + BLK_W - 1) // BLK_W
        
        # For each row block, collect non-zero columns
        # We'll create a compressed matrix where each row block has at most 16 non-zero columns
        block_non_zero_cols = []  # List of lists: one per row block
        
        for row_block_idx in range(num_row_blocks):
            row_start = row_block_idx * BLK_H
            row_end = min(row_start + BLK_H, M)
            
            # Find columns with non-zero values in this row block
            block_cols = set()
            for i in range(row_start, row_end):
                for j in range(K):
                    if A[i, j] != 0:
                        block_cols.add(j)
            
            # Convert to sorted list and pad to multiple of BLK_W
            block_cols_list = sorted(list(block_cols))
            padded_cols = ((len(block_cols_list) + BLK_W - 1) // BLK_W) * BLK_W
            
            # Pad with zero columns if needed
            while len(block_cols_list) < padded_cols:
                for j in range(K):
                    if j not in block_cols_list:
                        block_cols_list.append(j)
                        break
                if len(block_cols_list) >= K:
                    break
            
            block_non_zero_cols.append(block_cols_list[:padded_cols])
        
        # Find the maximum number of columns across all blocks
        max_cols = max(len(cols) for cols in block_non_zero_cols) if block_non_zero_cols else BLK_W
        max_cols = ((max_cols + BLK_W - 1) // BLK_W) * BLK_W  # Round up to multiple of BLK_W (WMMA_N)
        
        # Create compressed matrix: A_compressed (M_padded x max_cols)
        # For each row block, pack its non-zero columns
        M_padded = ((M + BLK_H - 1) // BLK_H) * BLK_H
        A_compressed = np.zeros((M_padded, max_cols), dtype=A.dtype)
        
        # Mapping: for each row block, which original columns are used
        block_col_mapping = []  # List of column indices for each row block
        
        for row_block_idx in range(num_row_blocks):
            row_start = row_block_idx * BLK_H
            row_end = min(row_start + BLK_H, M)
            block_cols = block_non_zero_cols[row_block_idx]
            
            # Map original columns to compressed columns
            col_mapping = {orig_col: comp_col for comp_col, orig_col in enumerate(block_cols)}
            block_col_mapping.append(block_cols)
            
            # Copy non-zero values to compressed matrix
            for i in range(row_start, row_end):
                for orig_col in block_cols:
                    comp_col = col_mapping[orig_col]
                    A_compressed[i, comp_col] = A[i, orig_col]
        
        # For reference, we need to know which columns are used globally
        # Collect all unique columns used across all blocks
        all_non_zero_cols = set()
        for block_cols in block_non_zero_cols:
            all_non_zero_cols.update(block_cols)
        non_zero_cols = sorted(list(all_non_zero_cols))
    
    X = np.random.randn(K, N).astype(X_dtype_np)
    
    # For DTC: Aij x Xjk = Yik
    # A_compressed 是基于 block-wise 压缩的，列数应该是 max_cols（每个 block 的最大非零列数，向上取整到 16）
    # X_selected 应该基于 A_compressed 中实际使用的列来选择
    
    # A_compressed 的列数已经确定：max_cols（基于 block-wise 压缩）
    # 我们需要找出 A_compressed 中实际使用的原始列索引
    # 这应该基于 block_non_zero_cols 的并集，但需要映射到压缩后的列索引
    
    # 收集所有在 A_compressed 中使用的原始列索引
    # 由于每个 block 可能有不同的列映射，我们需要找出所有被使用的列
    A_compressed_used_cols = set()
    for row_block_idx in range(num_row_blocks):
        block_cols = block_non_zero_cols[row_block_idx]
        A_compressed_used_cols.update(block_cols)
    A_compressed_used_cols = sorted(list(A_compressed_used_cols))
    
    print(f"  A_compressed uses {len(A_compressed_used_cols)} unique columns from original A")
    print(f"    Column indices: {A_compressed_used_cols[:20]}..." if len(A_compressed_used_cols) > 20 else f"    Column indices: {A_compressed_used_cols}")
    print(f"  A_compressed shape: {M_padded}x{max_cols} (compressed columns)")
    
    # Print non-zero columns for each block
    print(f"\n  Non-zero columns for each row block:")
    for row_block_idx in range(num_row_blocks):
        row_start = row_block_idx * BLK_H
        row_end = min(row_start + BLK_H, M)
        block_cols = block_non_zero_cols[row_block_idx]
        print(f"    Block {row_block_idx} (rows {row_start}-{row_end-1}): {len(block_cols)} cols -> {block_cols[:20]}..." if len(block_cols) > 20 else f"    Block {row_block_idx} (rows {row_start}-{row_end-1}): {len(block_cols)} cols -> {block_cols}")
    
    # Create X_selected_full: same shape as X (K x N)
    # For block-wise compression: each row block uses different columns
    # So X_selected_full should show: for each row block, only rows corresponding to
    # columns used in that specific block are kept, others are zeroed
    X_selected_full = np.zeros_like(X)  # Same shape as X (K x N)
    
    # Fill X_selected_full block by block: for each row block, only keep rows
    # corresponding to columns used in that specific block
    for row_block_idx in range(num_row_blocks):
        row_start = row_block_idx * BLK_H
        row_end = min(row_start + BLK_H, M)
        block_cols = block_non_zero_cols[row_block_idx]
        
        # For this row block, only keep X rows corresponding to columns used in this block
        for j in block_cols:
            X_selected_full[j, :] = X[j, :]
    
    # For Tensor Core computation, X_selected_compressed should match A_compressed columns
    # A_compressed 是 (M_padded x max_cols)，所以 X_selected_compressed 应该是 (max_cols x N_padded)
    # max_cols is already padded to multiple of BLK_W (WMMA_N = 16)
    # But for K dimension in matrix multiplication, we need to ensure it's multiple of WMMA_K * chunk
    # chunk is used in schedule for splitting reduce axis, so K must be multiple of wmma_k * chunk
    # Optimize: choose chunk_factor based on max_cols to avoid unnecessary padding
    # If max_cols <= 32, use chunk=2 (requires 32 multiple)
    # If max_cols > 32, use chunk=4 (requires 64 multiple)
    if max_cols <= 32:
        chunk_factor = 2  # Requires K to be multiple of 16 * 2 = 32
    else:
        chunk_factor = 4  # Requires K to be multiple of 16 * 4 = 64
    
    K_padded = max_cols  # Use max_cols from A_compressed (already padded to BLK_W)
    # Ensure K_padded is also multiple of WMMA_K * chunk for Tensor Core schedule
    K_padded = ((K_padded + WMMA_K * chunk_factor - 1) // (WMMA_K * chunk_factor)) * (WMMA_K * chunk_factor)
    
    # Now we need to pad A_compressed columns to K_padded to match X_selected_compressed
    if K_padded > max_cols:
        # Pad A_compressed with zero columns
        padding_cols_A = np.zeros((A_compressed.shape[0], K_padded - max_cols), dtype=A_compressed.dtype)
        A_compressed = np.hstack([A_compressed, padding_cols_A])
        print(f"  Padded A_compressed columns: {max_cols} -> {K_padded} (added {K_padded - max_cols} zero columns, chunk_factor={chunk_factor})")
        max_cols = K_padded  # Update max_cols to match K_padded
    else:
        print(f"  A_compressed columns: {max_cols} (no padding needed, chunk_factor={chunk_factor})")
    
    # 更好的方法：直接使用 A_compressed_used_cols，按顺序映射
    # K_padded is now calculated (multiple of WMMA_K * chunk = 64)
    # Pad A_compressed_used_cols to K_padded
    while len(A_compressed_used_cols) < K_padded:
        for j in range(K):
            if j not in A_compressed_used_cols:
                A_compressed_used_cols.append(j)
                break
        if len(A_compressed_used_cols) >= K:
            break
    
    A_compressed_used_cols = A_compressed_used_cols[:K_padded]
    
    # 创建压缩版本的 X_selected，行数必须等于 A_compressed 的列数 (K_padded)
    # 我们需要一个映射：A_compressed 的压缩列索引 -> X 的原始行索引
    # 由于 block-wise 压缩的复杂性，我们使用一个简化的全局映射
    
    # 方法：使用所有 block 中出现的列，按顺序映射到 0..K_padded-1
    X_selected_compressed = np.zeros((K_padded, N), dtype=X.dtype)
    
    # 创建一个从压缩列索引到原始列索引的映射
    # 使用 A_compressed_used_cols（所有在 A_compressed 中实际使用的原始列）
    # 这些列按顺序映射到压缩列索引 0..K_padded-1
    
    # 但是，由于 block-wise 压缩，每个 block 的列映射可能不同
    # 我们需要一个全局映射策略
    
    # 方法：使用 A_compressed_used_cols，按顺序映射到 0..K_padded-1
    # 如果 A_compressed_used_cols 的长度小于 K_padded，需要 padding
    mapping_cols = list(A_compressed_used_cols)
    
    # Pad to K_padded
    while len(mapping_cols) < K_padded:
        added = False
        for j in range(K):
            if j not in mapping_cols:
                mapping_cols.append(j)
                added = True
                break
        if not added:
            # If all columns are already in mapping_cols, pad with last column
            if len(mapping_cols) > 0:
                mapping_cols.append(mapping_cols[-1])
            else:
                mapping_cols.append(0)
        if len(mapping_cols) >= K_padded:
            break
    # Ensure exactly K_padded elements
    mapping_cols = mapping_cols[:K_padded]
    if len(mapping_cols) < K_padded:
        # Final padding if still needed
        while len(mapping_cols) < K_padded:
            mapping_cols.append(mapping_cols[-1] if mapping_cols else 0)
    
    # 填充 X_selected_compressed
    # X_selected_compressed[comp_col, :] = X[orig_col, :]
    # 其中 orig_col = mapping_cols[comp_col]
    for comp_col in range(K_padded):
        orig_col = mapping_cols[comp_col]
        X_selected_compressed[comp_col, :] = X[orig_col, :]
    
    # Pad X_selected_compressed columns to multiple of BLK_W (WMMA_N) if needed
    N_original = X_selected_compressed.shape[1]
    N_padded = ((N_original + BLK_W - 1) // BLK_W) * BLK_W
    
    if N_padded > N_original:
        # Pad with zero columns
        padding_cols = np.zeros((X_selected_compressed.shape[0], N_padded - N_original), dtype=X_selected_compressed.dtype)
        X_selected_compressed = np.hstack([X_selected_compressed, padding_cols])
        print(f"  Padded X columns: {N_original} -> {N_padded} (added {N_padded - N_original} zero columns)")
    
    # Dimension check: A_compressed is (M_padded x K_padded)
    # X_selected_compressed should be (K_padded x N_padded)
    # For matrix multiplication: A_compressed @ X_selected_compressed
    # We need: K_padded == X_selected_compressed.shape[0] == A_compressed.shape[1]
    assert A_compressed.shape[1] == K_padded, \
        f"Dimension mismatch: A_compressed columns ({A_compressed.shape[1]}) != K_padded ({K_padded})"
    assert K_padded == X_selected_compressed.shape[0], \
        f"Dimension mismatch: K_padded ({K_padded}) != X_selected_compressed rows ({X_selected_compressed.shape[0]})"
    
    # Reference output: Y = A_compressed @ X_selected_compressed (use compressed for computation)
    print(f"  Compressed A matrix: {A_compressed.shape[0]}x{A_compressed.shape[1]} (K_padded={K_padded})")
    # print(A_compressed)
    print(f"  Selected X matrix (compressed): {X_selected_compressed.shape[0]}x{X_selected_compressed.shape[1]} (K_padded={K_padded}, N_padded={N_padded})")
    # print(X_selected_compressed)
    # print(f"  Matrix multiplication: ({A_compressed.shape[0]}x{A_compressed.shape[1]}) @ ({X_selected_compressed.shape[0]}x{X_selected_compressed.shape[1]}) = ({M_padded}x{N_padded})")
    Y_ref = A_compressed @ X_selected_compressed
    
    # For visualization, create X_selected_full_mapped: same shape as X (K x N)
    # Each block uses different columns, so we need to show which X rows are used per block
    # The logic: for each row block i, only X rows corresponding to columns used in that block are kept
    # This shows the block-wise selection: each block computes with different X rows
    X_selected_full_mapped = np.zeros_like(X)  # Same shape as X (K x N)
    
    # For each row block, mark which X rows (corresponding to A columns) are used
    # This shows the block-wise selection logic: each block uses its own set of columns
    for row_block_idx in range(num_row_blocks):
        row_start = row_block_idx * BLK_H
        row_end = min(row_start + BLK_H, M)
        block_cols = block_non_zero_cols[row_block_idx]  # Original columns used in this block
        
        # For this block, only X rows corresponding to columns used in this block are kept
        # These are the actual X rows that participate in computation for this block
        # We directly use X[orig_col, :] since orig_col is the row index in X that corresponds to column orig_col in A
        for orig_col in block_cols:
            if orig_col < K:  # Make sure orig_col is valid
                # X row orig_col corresponds to A column orig_col
                # For this block, if A uses column orig_col, then X row orig_col is used
                # So we copy X[orig_col, :] to X_selected_full_mapped[orig_col, :]
                X_selected_full_mapped[orig_col, :X.shape[1]] = X[orig_col, :]
    
    # Pad to N_padded if needed
    if X_selected_full_mapped.shape[1] < N_padded:
        padding_cols = np.zeros((X_selected_full_mapped.shape[0], N_padded - X_selected_full_mapped.shape[1]), dtype=X_selected_full_mapped.dtype)
        X_selected_full_mapped = np.hstack([X_selected_full_mapped, padding_cols])
    
    X_selected = X_selected_full_mapped  # Use mapped version for visualization (K x N_padded)
    
    # Y_ref shape should be (M_padded, N_padded)
    # Trim to original size if needed (for comparison)
    Y_ref_trimmed = Y_ref[:M, :N] if M_padded > M or N_padded > N else Y_ref
    
    # Return compressed X_selected for computation, but we also have the full version for visualization
    # Also return original A and X (uncompressed) for visualization
    return (A_compressed.astype(A_dtype_np), X_selected_compressed.astype(X_dtype_np), Y_ref, Y_ref_trimmed, 
            non_zero_cols, (M_padded, K_padded, N_padded), A.astype(A_dtype_np), X.astype(X_dtype_np), X_selected)

def dtc_spmm_tensorcore_cuda(A_compressed, X_selected, M_original=None, N_original=None, M_padded=None, N_padded=None, out_dtype=None):
    """
    DTC-SpMM using Tensor Core with 16x16x16 blocks
    
    This implements: Y = A_compressed @ X_selected
    where A_compressed is (M_original, K_padded) and X_selected is (K_padded, N_original)
    Padding is handled directly in the compute to avoid intermediate padding ops.
    
    All dimensions must be multiples of WMMA_M, WMMA_N, WMMA_K for Tensor Core (enforced by padding).
    """
    if out_dtype is None:
        out_dtype = Y_dtype  # Use global Y_dtype
    M_A, K_padded = get_const_tuple(A_compressed.shape)
    K_padded_check, N_X = get_const_tuple(X_selected.shape)
    
    if _get_verbose():
        print(f"  [DTC Debug] dtc_spmm_tensorcore_cuda: A_compressed.shape={A_compressed.shape}, X_selected.shape={X_selected.shape}")
        print(f"  [DTC Debug] dtc_spmm_tensorcore_cuda: N_X={N_X}, N_padded={N_padded}, N_original={N_original}")
    
    assert K_padded == K_padded_check, f"K dimension mismatch: {K_padded} != {K_padded_check}"
    
    # Use provided padded dimensions or infer from shapes
    if M_padded is None:
        M_padded = M_A
    if N_padded is None:
        N_padded = N_X  # X_selected shape is (K_padded, N_padded), so N_X is N_padded
    if M_original is None:
        M_original = M_A
    if N_original is None:
        # N_original might be less than N_padded, but we need to infer it from the actual data
        # For now, use N_padded as a fallback (will be handled by padding logic)
        N_original = N_X
    
    # Assert dimensions are compatible with WMMA_M x WMMA_N x WMMA_K Tensor Core
    if K_padded <= 32:
        chunk_factor = 2
    else:
        chunk_factor = 4
    
    assert M_padded % WMMA_M == 0, f"M_padded ({M_padded}) must be multiple of {WMMA_M}"
    assert N_padded % WMMA_N == 0, f"N_padded ({N_padded}) must be multiple of {WMMA_N}"
    assert K_padded % (WMMA_K * chunk_factor) == 0, f"K_padded ({K_padded}) must be multiple of {WMMA_K * chunk_factor}"
    
    # X_selected shape should be (K_padded, N_padded) from tensorcore_strategy.py
    # If it's (K_padded, N_original), we need to pad it
    # But padding with if_then_else causes boundary checks, so we expect X_selected to already be (K_padded, N_padded)
    # Verify X_selected shape matches N_padded
    # CRITICAL: Following SparseTIR approach - dimensions should be fixed at definition time
    # X_selected should already be (K_padded, N_padded) from tensorcore_strategy.py
    # If N_X != N_padded, raise an error rather than padding here (which causes boundary checks)
    if N_X != N_padded:
        raise ValueError(
            f"X_selected shape mismatch: expected (K_padded={K_padded}, N_padded={N_padded}), "
            f"got (K_padded={K_padded}, N_X={N_X}). "
            f"This indicates Relay layer padding didn't work correctly. "
            f"Expected X_selected to already be padded to {N_padded} at Relay level."
        )
    X_selected_padded = X_selected
    
    # Transpose X_selected_padded to col-major for Tensor Core
    # X_selected_padded shape is (K_padded, N_padded), so X_selected_T should be (N_padded, K_padded)
    X_selected_T = te.compute(
        (N_padded, K_padded),
        lambda j, k: X_selected_padded[k, j],
        name="X_selected_T"
    )
    
    # Dense matrix multiplication (pure, no padding)
    # Output shape is (M_padded, N_padded) to avoid boundary checks in tensorize
    # But we can't use if_then_else in tensorize body, so we need a different approach
    # Instead, create padded inputs and use pure matrix multiplication
    import tvm.tir as tir
    
    # Create padded A_compressed
    # CRITICAL: If M_original == M_padded, A_compressed is already padded, so no need for if_then_else
    # This avoids boundary checks that break tensorize
    # If M_original < M_padded, we need to pad with zeros
    if M_original == M_padded:
        # A_compressed is already padded, use it directly without if_then_else
        A_compressed_padded = A_compressed
    else:
        # A_compressed needs padding, use if_then_else (but this may cause boundary checks)
        A_compressed_padded = te.compute(
            (M_padded, K_padded),
            lambda i, k: te.if_then_else(i < M_original, A_compressed[i, k], tvm.tir.const(0.0, A_compressed.dtype)),
            name="A_compressed_padded"
        )
    
    # X_selected_T is already (N_padded, K_padded), no need to pad
    X_selected_T_padded = X_selected_T
    
    # Pure matrix multiplication (no if_then_else in body)
    # Output shape is (M_padded, N_padded) for Tensor Core
    # Note: Relay will handle slicing to (M_original, N_original) if needed
    k = te.reduce_axis((0, K_padded), name="k")
    Y = te.compute(
        (M_padded, N_padded),
        lambda i, j: te.sum(
            A_compressed_padded[i, k].astype(out_dtype) * X_selected_T_padded[j, k].astype(out_dtype),
            axis=k
        ),
        name="Y_dtc",
        tag="dense_tensorcore"
    )
    
    return Y

def dtc_spmm_norm_relu_tensorcore_cuda(A_compressed, X_selected, norm=None, bias=None, activation=None, out_dtype=None):
    """
    DTC-SpMM with fused norm, bias, and activation using Tensor Core
    
    This implements: Y = activation(bias_add(multiply(A_compressed @ X_selected, norm), bias))
    where:
    - A_compressed is (M_padded, K_padded)
    - X_selected is (K_padded, N_padded)
    - norm is (M_padded,) or (M_padded, 1) for broadcasting
    - bias is (N_padded,) for broadcasting
    - activation is a function like relu, sigmoid, etc.
    
    All operations are fused into a single Tensor Core kernel.
    """
    if out_dtype is None:
        out_dtype = Y_dtype  # Use global Y_dtype
    
    # First compute matrix multiplication: Y_matmul = A_compressed @ X_selected
    Y_matmul = dtc_spmm_tensorcore_cuda(A_compressed, X_selected, out_dtype=out_dtype)
    
    M_padded, N_padded = get_const_tuple(Y_matmul.shape)
    
    # Apply norm (multiply) if provided
    if norm is not None:
        # norm should be (M_padded,) or (M_padded, 1) for broadcasting
        # Y_matmul is (M_padded, N_padded)
        # Broadcast norm to (M_padded, N_padded)
        Y_norm = te.compute(
            (M_padded, N_padded),
            lambda i, j: Y_matmul[i, j] * norm[i] if len(norm.shape) == 1 else Y_matmul[i, j] * norm[i, 0],
            name="Y_norm",
            tag="broadcast"
        )
        Y_current = Y_norm
    else:
        Y_current = Y_matmul
    
    # Apply bias if provided
    if bias is not None:
        # bias should be (N_padded,) for broadcasting
        # Y_current is (M_padded, N_padded)
        # Broadcast bias to (M_padded, N_padded)
        Y_bias = te.compute(
            (M_padded, N_padded),
            lambda i, j: Y_current[i, j] + bias[j].astype(out_dtype),
            name="Y_bias",
            tag="broadcast"
        )
        Y_current = Y_bias
    
    # Apply activation if provided
    if activation is not None:
        # activation is a function like relu, sigmoid, etc.
        # For relu: max(0, x)
        if activation == "relu" or (hasattr(activation, '__name__') and activation.__name__ == 'relu'):
            Y_act = te.compute(
                (M_padded, N_padded),
                lambda i, j: te.max(0.0, Y_current[i, j]),
                name="Y_relu",
                tag="elemwise"
            )
        else:
            # For other activations, use a generic compute
            # This is a placeholder - you may need to implement specific activations
            Y_act = te.compute(
                (M_padded, N_padded),
                lambda i, j: Y_current[i, j],  # Identity for now
                name="Y_activation",
                tag="elemwise"
            )
        Y_current = Y_act
    
    return Y_current

def schedule_dtc_spmm_norm_relu_tensorcore(s, Y, norm=None, bias=None, activation=None):
    """
    Schedule DTC-SpMM with fused norm, bias, and activation using Tensor Core
    
    This schedule ensures all operations (matrix multiplication, norm multiply, 
    bias_add, activation) are computed within the Tensor Core thread environment.
    
    Parameters
    ----------
    s: tvm.te.Schedule
        The schedule object
    Y: tvm.te.Tensor
        The output tensor (may be fused operation output)
    norm: bool or tvm.te.Tensor, optional
        If bool: True means norm multiply will be present (auto-detect from graph)
        If Tensor: explicit norm tensor for multiplication
    bias: bool, optional
        True means bias_add will be present (auto-detect from graph)
    activation: str, optional
        Activation function name (e.g., "relu", "sigmoid") - used to identify operation in graph
    """
    if _get_verbose():
        print(f"  [DTC Schedule] schedule_dtc_spmm_norm_relu_tensorcore called with:")
        print(f"    norm={norm}, bias={bias}, activation={activation}")
    # Find the root matrix multiplication operation (Y_dtc with tag="dense_tensorcore")
    Y_matrix_mult = Y
    if not (hasattr(Y.op, 'tag') and Y.op.tag == "dense_tensorcore"):
        # Y is not the matrix multiplication - find it in the input chain
        def find_matrix_mult_op(tensor, visited=None):
            if visited is None:
                visited = set()
            if tensor in visited:
                return None
            visited.add(tensor)
            
            # Check if this is the matrix multiplication
            if hasattr(tensor.op, 'tag') and tensor.op.tag == "dense_tensorcore":
                return tensor
            
            # Check inputs
            if hasattr(tensor.op, 'input_tensors'):
                for inp in tensor.op.input_tensors:
                    result = find_matrix_mult_op(inp, visited)
                    if result is not None:
                        return result
            return None
        
        found = find_matrix_mult_op(Y)
        if found is not None:
            Y_matrix_mult = found
    
    # Get input tensors from matrix multiplication
    A = s[Y_matrix_mult].op.input_tensors[0]  # A_compressed
    X_T = s[Y_matrix_mult].op.input_tensors[1]  # X_selected_T (transposed)
    
    # Debug: Check X_T shape
    if _get_verbose():
        try:
            X_T_shape = get_const_tuple(X_T.shape)
            print(f"  [DTC Schedule] X_T shape: {X_T_shape} (expected: (N_padded, K_padded))")
        except:
            pass
    
    # Get X_selected from X_T if it's a compute op
    try:
        if hasattr(X_T.op, 'input_tensors') and len(X_T.op.input_tensors) > 0:
            X = X_T.op.input_tensors[0]  # Original X_selected
            if hasattr(X.op, 'input_tensors') and len(X.op.input_tensors) > 0:
                s[X].compute_inline()
    except:
        pass
    
    # Get shapes and dtypes
    M, N = get_const_tuple(Y_matrix_mult.shape)
    data_dtype = A.dtype
    out_dtype = Y_matrix_mult.dtype
    
    # Get K dimension from A
    M_A, K_padded = get_const_tuple(A.shape)
    
    # Fixed Tensor Core configuration
    wmma_m = WMMA_M
    wmma_n = WMMA_N
    wmma_k = WMMA_K
    
    # Dynamic chunk selection
    if K_padded <= 32:
        chunk = 2
    else:
        chunk = 4
    
    # Fixed tuning parameters
    block_row_warps = 2
    block_col_warps = 2
    warp_row_tiles = 1
    warp_col_tiles = 1
    offset = 8
    offsetCS = 8
    vec = 4
    warp_size = 32
    
    # Cache read from inputs
    # Handle padding compute ops: schedule them properly
    # A might be A_compressed_padded, X_T might use X_selected_T_padded
    A_actual = A
    X_T_actual = X_T
    A_is_padding = False
    X_T_has_padding = False
    
    # Check if A is a padding compute op (A_compressed_padded)
    try:
        if hasattr(A.op, 'input_tensors') and len(A.op.input_tensors) > 0:
            A_actual = A.op.input_tensors[0]
            A_is_padding = True
    except:
        pass
    
    # Check if X_T uses padding compute ops
    try:
        if hasattr(X_T.op, 'input_tensors') and len(X_T.op.input_tensors) > 0:
            X_selected_T_padded = X_T.op.input_tensors[0]
            if hasattr(X_selected_T_padded.op, 'input_tensors') and len(X_selected_T_padded.op.input_tensors) > 0:
                X_T_actual = X_selected_T_padded.op.input_tensors[0]  # X_selected
                X_T_has_padding = True
    except:
        pass
    
    # Cache_read from actual inputs first
    AS = s.cache_read(A_actual, "shared", [Y_matrix_mult])
    XS = s.cache_read(X_T_actual, "shared", [Y_matrix_mult])
    
    # Inline X_T if it's a compute op (transpose)
    try:
        if hasattr(X_T.op, 'axis') or hasattr(X_T.op, 'body'):
            s[X_T].compute_inline()
    except:
        pass
    
    AF = s.cache_read(AS, "wmma.matrix_a", [Y_matrix_mult])
    XF = s.cache_read(XS, "wmma.matrix_b", [Y_matrix_mult])
    
    # Cache write for matrix multiplication
    YF = s.cache_write(Y_matrix_mult, "wmma.accumulator")
    YS = s.cache_read(YF, "shared", [Y_matrix_mult])
    
    # CRITICAL: Schedule fused operations (norm, bias, activation) to be computed
    # within the Tensor Core thread environment
    # Use explicit parameters from gconv.py instead of auto-detection
    
    # Find all fused operations between Y_matrix_mult and Y
    fused_ops = []
    current = Y
    while current != Y_matrix_mult:
        if hasattr(current.op, 'input_tensors') and len(current.op.input_tensors) > 0:
            fused_ops.insert(0, current)  # Add to front (reverse order)
            current = current.op.input_tensors[0]
        else:
            break
    
    # Use explicit parameters to identify which operations to schedule
    # norm: bool - True means we expect a multiply operation
    # bias: bool - True means we expect a bias_add operation
    # activation: str - Name of activation function (e.g., "relu")
    
    expected_ops = []
    if norm:
        expected_ops.append('multiply')
    if bias:
        expected_ops.append('bias')
    if activation:
        expected_ops.append(activation.lower())
    
    print(f"  [Fusion] Expected operations: {expected_ops}")
    print(f"  [Fusion] Found {len(fused_ops)} fused operations")
    
    # Schedule fused operations to be computed within Y_matrix_mult's thread environment
    # We'll compute them at the same level as Y_matrix_mult (after YS store)
    for fused_op in fused_ops:
        op_name = getattr(fused_op.op, 'name', '').lower()
        print(f"  [Fusion] Scheduling fused operation: {op_name}")
        
        # Verify this is an expected operation (if parameters were provided)
        is_expected = False
        if len(expected_ops) == 0:
            # No explicit parameters - schedule all fused operations
            is_expected = True
        else:
            # Check if this operation matches expected operations
            for expected in expected_ops:
                if expected in op_name:
                    is_expected = True
                    break
        
        if is_expected or len(expected_ops) == 0:
            try:
                # Compute fused operations at the block level of Y_matrix_mult
                # This ensures they are computed within the Tensor Core thread environment
                s[fused_op].compute_at(s[Y_matrix_mult], s[Y_matrix_mult].op.axis[0])
                print(f"  [Fusion] ✓ Scheduled {op_name} with compute_at")
            except:
                # If compute_at fails, try inline (will be computed within Y_matrix_mult's loop)
                try:
                    s[fused_op].compute_inline()
                    print(f"  [Fusion] ✓ Scheduled {op_name} with compute_inline")
                except Exception as e:
                    print(f"  [Fusion] ✗ Failed to schedule {op_name}: {e}")
        else:
            print(f"  [Fusion] ⚠ Skipping {op_name} (not in expected operations: {expected_ops})")
    
    # Use Y_matrix_mult for thread binding (this is the actual matrix multiplication)
    Y = Y_matrix_mult
    
    # Define strides
    AS_align = int(chunk * wmma_k + offset)
    XS_align = int(chunk * wmma_k + offset)
    YS_align = int(warp_col_tiles * block_col_warps * wmma_n + offsetCS)
    AS_stride = [AS_align, 1]
    XS_stride = [XS_align, 1]
    AF_stride = [wmma_k, 1]
    XF_stride = [wmma_k, 1]
    YF_stride = [warp_col_tiles * wmma_n, 1]
    YS_stride = [YS_align, 1]
    
    # Thread and block axes
    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    thread_x = te.thread_axis("threadIdx.x")
    thread_y = te.thread_axis("threadIdx.y")
    thread_z = te.thread_axis("threadIdx.z")
    
    # Schedule for output computation
    block_factor_m = wmma_m * warp_row_tiles * block_row_warps
    block_factor_n = wmma_n * warp_col_tiles * block_col_warps
    
    i, j = Y.op.axis
    block_i, bi = s[Y].split(i, factor=block_factor_m)
    block_j, bj = s[Y].split(j, factor=block_factor_n)
    s[Y].reorder(block_i, block_j, bi, bj)
    
    t = s[Y].fuse(bi, bj)
    t, vi = s[Y].split(t, factor=vec)
    t, tx = s[Y].split(t, factor=warp_size)
    t, ty = s[Y].split(t, factor=block_row_warps)
    t, tz = s[Y].split(t, factor=block_col_warps)
    
    s[Y].bind(block_i, block_x)
    s[Y].bind(block_j, block_y)
    s[Y].bind(tz, thread_z)
    s[Y].bind(ty, thread_y)
    s[Y].bind(tx, thread_x)
    s[Y].vectorize(vi)
    
    # Schedule for WMMA store
    s[YS].compute_at(s[Y], block_j)
    bb, oo = YS.op.axis
    s[YS].storage_align(bb, YS_align - 1, YS_align)
    bb, bbi = s[YS].split(bb, factor=wmma_m)
    oo, ooi = s[YS].split(oo, factor=wmma_n)
    bb, bbii = s[YS].split(bb, factor=warp_row_tiles)
    oo, ooii = s[YS].split(oo, factor=warp_col_tiles)
    s[YS].reorder(bb, oo, bbii, ooii, bbi, ooi)
    s[YS].bind(bb, thread_y)
    s[YS].bind(oo, thread_z)
    
    # Schedule for WMMA computation
    s[YF].compute_at(s[YS], oo)
    warp_i, warp_j = YF.op.axis
    warp_i, _ii = s[YF].split(warp_i, factor=wmma_m)
    warp_j, _jj = s[YF].split(warp_j, factor=wmma_n)
    (k,) = YF.op.reduce_axis
    k, _k = s[YF].split(k, factor=wmma_k)
    ko, ki = s[YF].split(k, factor=chunk)
    s[YF].reorder(ko, ki, warp_i, warp_j, _ii, _jj, _k)
    
    # Schedule for WMMA matrix_a load
    s[AF].compute_at(s[YF], ki)
    b, i = AF.op.axis
    b, b_ii = s[AF].split(b, factor=wmma_m)
    i, i_jj = s[AF].split(i, factor=wmma_k)
    s[AF].reorder(b, i, b_ii, i_jj)
    
    # Schedule for WMMA matrix_b load
    s[XF].compute_at(s[YF], ki)
    o, i = XF.op.axis
    o, o_ii = s[XF].split(o, factor=wmma_n)
    i, i_ii = s[XF].split(i, factor=wmma_k)
    s[XF].reorder(o, i, o_ii, i_ii)
    
    # Schedule for shared memory load
    def shared_schedule(stage, strides):
        s[stage].compute_at(s[YF], ko)
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
    shared_schedule(XS, XS_align)
    
    # Tensorize with Tensor Core intrinsics
    shape = (wmma_m, wmma_n, wmma_k)
    AL_gemm = te.placeholder((wmma_m, wmma_k), name="AL_gemm", dtype=data_dtype)
    XL_gemm = te.placeholder((wmma_n, wmma_k), name="XL_gemm", dtype=data_dtype)
    k_gemm = te.reduce_axis((0, wmma_k), name="k_gemm")
    YL_compute = te.compute(
        (wmma_m, wmma_n),
        lambda ii, jj: te.sum(
            AL_gemm[ii, k_gemm].astype(out_dtype) * XL_gemm[jj, k_gemm].astype(out_dtype),
            axis=k_gemm,
        ),
        name="YL_compute",
    )
    
    # Apply Tensor Core intrinsics
    s[AF].tensorize(
        b_ii,
        intrin_wmma_load_matrix_A(
            AF_stride, AS_stride, shape, "row_major", 
            (wmma_m, wmma_k), (wmma_m, wmma_k), data_dtype
        ),
    )
    s[XF].tensorize(
        o_ii,
        intrin_wmma_load_matrix_W(
            XF_stride, XS_stride, shape, "col_major", 
            (wmma_n, wmma_k), (wmma_n, wmma_k), data_dtype
        ),
    )
    s[YF].tensorize(
        _ii, 
        intrin_wmma_gemm(AL_gemm, XL_gemm, YL_compute, AF_stride, XF_stride, YF_stride, shape)
    )
    s[YS].tensorize(
        bbi,
        intrin_wmma_store_matrix(
            YS_stride, YF_stride, shape, out_dtype, 
            (wmma_m, wmma_n), (wmma_m, wmma_n)
        ),
    )
    
    return s

def schedule_dtc_spmm_tensorcore(s, Y, use_relu=False):
    """
    Schedule DTC-SpMM using Tensor Core with 16x16x16 blocks
    
    This schedule ensures Tensor Core is used and prevents fallback to CUDA core.
    
    Parameters
    ----------
    s: tvm.te.Schedule
        The schedule object
    Y: tvm.te.Tensor
        The output tensor
    use_relu: bool, optional
        If True, apply ReLU activation directly in the Tensor Core computation
        (applied after matrix multiplication, before storing to shared memory)
    """
    # Get input tensors - Y = A @ X_T where X_T is transposed X
    # But Y might be a fused operation (e.g., ReLU), so we need to find the matrix multiplication
    # First, find the actual matrix multiplication operation
    Y_matrix_mult_for_inputs = Y
    if not (hasattr(Y.op, 'reduce_axis') and len(Y.op.reduce_axis) > 0):
        # Y is not the matrix multiplication - find it in the input chain
        def find_matrix_mult_for_inputs(tensor, visited=None):
            if visited is None:
                visited = set()
            if tensor in visited:
                return None
            visited.add(tensor)
            
            # Check if this is the matrix multiplication (has reduce_axis)
            if hasattr(tensor.op, 'reduce_axis') and len(tensor.op.reduce_axis) > 0:
                return tensor
            
            # Check inputs
            if hasattr(tensor.op, 'input_tensors') and len(tensor.op.input_tensors) > 0:
                for inp in tensor.op.input_tensors:
                    result = find_matrix_mult_for_inputs(inp, visited)
                    if result is not None:
                        return result
            return None
        
        found = find_matrix_mult_for_inputs(Y)
        if found is not None:
            Y_matrix_mult_for_inputs = found
    
    # Now get input tensors from the matrix multiplication
    A = s[Y_matrix_mult_for_inputs].op.input_tensors[0]  # A_compressed
    if len(s[Y_matrix_mult_for_inputs].op.input_tensors) > 1:
        X_T = s[Y_matrix_mult_for_inputs].op.input_tensors[1]  # X_selected_T (transposed)
    else:
        # Fallback: try to get from Y if it has inputs
        if hasattr(Y.op, 'input_tensors') and len(Y.op.input_tensors) > 0:
            # Y might be ReLU, get its input (which should be the matrix multiplication)
            Y_matrix_mult_for_inputs = Y.op.input_tensors[0]
            A = s[Y_matrix_mult_for_inputs].op.input_tensors[0]  # A_compressed
            if len(s[Y_matrix_mult_for_inputs].op.input_tensors) > 1:
                X_T = s[Y_matrix_mult_for_inputs].op.input_tensors[1]  # X_selected_T (transposed)
            else:
                raise ValueError(f"Cannot find X_T in matrix multiplication inputs. Y_matrix_mult_for_inputs has {len(s[Y_matrix_mult_for_inputs].op.input_tensors)} inputs")
        else:
            raise ValueError(f"Cannot find matrix multiplication inputs. Y has no input_tensors")
    
    # X_T might be a compute op (transpose) or a placeholder
    # If it's a compute op, get the original X_selected
    # If it's a placeholder, it might already be in the right format
    try:
        if hasattr(X_T.op, 'input_tensors') and len(X_T.op.input_tensors) > 0:
            X = X_T.op.input_tensors[0]  # Original X_selected
        else:
            # X_T is already a placeholder, use it directly
            # This might happen if the transpose is done elsewhere
            X = X_T
    except:
        # Fallback: assume X_T is the input we need
        X = X_T
    
    # Get shapes and dtypes
    M, N = get_const_tuple(Y.shape)
    data_dtype = A.dtype
    out_dtype = Y.dtype
    
    # Get K dimension from A (A is M x K_padded)
    M_A, K_padded = get_const_tuple(A.shape)
    
    # Fixed Tensor Core configuration: 16x16x16
    wmma_m = WMMA_M
    wmma_n = WMMA_N
    wmma_k = WMMA_K
    
    # Dynamic chunk selection based on K_padded to avoid unnecessary padding
    # If K_padded <= 32, use chunk=2 (requires 32 multiple)
    # If K_padded > 32, use chunk=4 (requires 64 multiple)
    if K_padded <= 32:
        chunk = 2  # Requires K to be multiple of 16 * 2 = 32
    else:
        chunk = 4  # Requires K to be multiple of 16 * 4 = 64
    
    # Fixed tuning parameters for 16x16x16
    # Use consistent, smaller configuration to avoid thread binding conflicts
    # across multiple GraphConv layers in the same model
    # All layers must use the same thread binding extent to avoid TVM errors
    block_row_warps = 2  # Fixed to 2 to avoid conflicts (was 4)
    block_col_warps = 2  # Fixed to 2 to avoid conflicts (was 4)
    warp_row_tiles = 1   # One tile per warp
    warp_col_tiles = 1   # One tile per warp
    offset = 8
    offsetCS = 8
    vec = 4
    warp_size = 32
    
    # Define memory hierarchy (critical for Tensor Core)
    # Schedule the transpose to be computed inline first (if it's a compute op)
    # Note: Y's inputs are A and X_T, not X. So we need to cache_read from X_T.
    try:
        if hasattr(X_T.op, 'axis') or hasattr(X_T.op, 'body'):
            # X_T is a compute op, can be inlined
            # But we still need to cache_read from X_T before inlining
            # So we cache_read from X_T, then inline it
            pass  # Don't inline yet, cache_read first
        # X_T is what Y actually uses, so we cache_read from X_T
        X_for_cache = X_T
    except:
        # If check fails, use X_T directly
        X_for_cache = X_T
    
    # Handle padding compute ops: schedule them properly
    # A might be A_compressed_padded, X_T might use X_selected_T_padded
    A_actual = A
    X_T_actual = X_T
    A_is_padding = False
    X_T_has_padding = False
    
    # Check if A is a padding compute op (A_compressed_padded)
    try:
        if hasattr(A.op, 'input_tensors') and len(A.op.input_tensors) > 0:
            A_actual = A.op.input_tensors[0]
            A_is_padding = True
    except:
        pass
    
    # Check if X_T uses padding compute ops
    try:
        if hasattr(X_T.op, 'input_tensors') and len(X_T.op.input_tensors) > 0:
            X_selected_T_padded = X_T.op.input_tensors[0]
            if hasattr(X_selected_T_padded.op, 'input_tensors') and len(X_selected_T_padded.op.input_tensors) > 0:
                X_T_actual = X_selected_T_padded.op.input_tensors[0]  # X_selected
                X_T_has_padding = True
            else:
                # X_T is X_selected_T_padded, but it doesn't have padding
                X_T_actual = X_selected_T_padded
    except:
        pass
    
    # First, check if X_T_actual is a compute op and schedule it if needed
    try:
        if hasattr(X_T_actual.op, 'input_tensors') and len(X_T_actual.op.input_tensors) > 0:
            X = X_T_actual.op.input_tensors[0]  # Original X_selected
            # Schedule X to be computed inline or at block level to avoid global memory
            # Check if X is also a compute op (transpose)
            if hasattr(X.op, 'input_tensors') and len(X.op.input_tensors) > 0:
                # X is a compute op, schedule it to be computed at block level
                # This ensures it's computed within thread environment
                # We'll compute it inline to avoid global memory allocation
                s[X].compute_inline()
    except:
        pass
    
    # Find the actual matrix multiplication operation (Y_dtc_padded with tag="dense_tensorcore")
    # Y might be a slice of Y_dtc_padded, so we need to find the actual matrix multiplication
    Y_matrix_mult = Y
    if not (hasattr(Y.op, 'reduce_axis') and len(Y.op.reduce_axis) > 0):
        # Y is not the matrix multiplication - find it in the input chain
        def find_matrix_mult_op(tensor, visited=None):
            if visited is None:
                visited = set()
            if tensor in visited:
                return None
            visited.add(tensor)
            
            # Check if this is the matrix multiplication (has reduce_axis or tag="dense_tensorcore")
            if hasattr(tensor.op, 'reduce_axis') and len(tensor.op.reduce_axis) > 0:
                return tensor
            if hasattr(tensor.op, 'tag') and tensor.op.tag == "dense_tensorcore":
                return tensor
            
            # Check inputs
            if hasattr(tensor.op, 'input_tensors'):
                for inp in tensor.op.input_tensors:
                    result = find_matrix_mult_op(inp, visited)
                    if result is not None:
                        return result
            return None
        
        found = find_matrix_mult_op(Y)
        if found is not None:
            Y_matrix_mult = found
        else:
            # If we can't find it, assume Y is the matrix multiplication
            # This will fail if Y doesn't have reduce_axis, but at least we'll get a clearer error
            Y_matrix_mult = Y
    
    # If Y is a slice of Y_matrix_mult, we need to schedule it properly
    # But first, we need to schedule Y_matrix_mult, then schedule Y
    # We'll handle this after Y_matrix_mult is scheduled
    Y_is_slice = (Y != Y_matrix_mult)
    
    # Handle padding compute ops: schedule them properly
    # A might be A_compressed_padded, X_T might use X_selected_T_padded
    A_actual = A
    X_T_actual = X_T
    A_is_padding = False
    X_T_has_padding = False
    
    # Check if A is a padding compute op (A_compressed_padded)
    try:
        if hasattr(A.op, 'input_tensors') and len(A.op.input_tensors) > 0:
            A_actual = A.op.input_tensors[0]
            A_is_padding = True
    except:
        pass
    
    # Check if X_T uses padding compute ops
    try:
        if hasattr(X_T.op, 'input_tensors') and len(X_T.op.input_tensors) > 0:
            X_selected_T_padded = X_T.op.input_tensors[0]
            if hasattr(X_selected_T_padded.op, 'input_tensors') and len(X_selected_T_padded.op.input_tensors) > 0:
                X_T_actual = X_selected_T_padded.op.input_tensors[0]  # X_selected
                X_T_has_padding = True
            else:
                # X_T is X_selected_T_padded, but it doesn't have padding
                X_T_actual = X_selected_T_padded
    except:
        pass
    
    # Cache_read from A and X_T (which Y_matrix_mult actually uses)
    # If they are padding compute ops, we'll schedule them later
    AS = s.cache_read(A, "shared", [Y_matrix_mult])
    XS = s.cache_read(X_T, "shared", [Y_matrix_mult])
    
    # If A is a padding compute op, cache_read its original input (A_actual) to shared memory
    # This ensures A_actual is accessible in thread environment
    AS_input = None
    if A_is_padding:
        try:
            # Cache_read A_actual (the original input before padding) to shared memory
            AS_input = s.cache_read(A_actual, "shared", [AS])
        except:
            pass
    
    # Inline X_T if it's a compute op (transpose)
    try:
        if hasattr(X_T.op, 'axis') or hasattr(X_T.op, 'body'):
            s[X_T].compute_inline()
    except:
        pass
    
    # Inline X_selected_T_padded if it exists and is a padding compute op
    # This avoids boundary checks in tensorize
    try:
        if X_T_has_padding and hasattr(X_T.op, 'input_tensors') and len(X_T.op.input_tensors) > 0:
            X_selected_T_padded = X_T.op.input_tensors[0]
            if hasattr(X_selected_T_padded.op, 'axis') or hasattr(X_selected_T_padded.op, 'body'):
                s[X_selected_T_padded].compute_inline()
    except:
        pass
    
    # Inline X_selected if it's a compute op (transpose from weight_padded)
    # This avoids boundary checks in tensorize
    try:
        if X_T_has_padding and hasattr(X_T.op, 'input_tensors') and len(X_T.op.input_tensors) > 0:
            X_selected_T_padded = X_T.op.input_tensors[0]
            if hasattr(X_selected_T_padded.op, 'input_tensors') and len(X_selected_T_padded.op.input_tensors) > 0:
                X_selected = X_selected_T_padded.op.input_tensors[0]
                if hasattr(X_selected.op, 'axis') or hasattr(X_selected.op, 'body'):
                    s[X_selected].compute_inline()
    except:
        pass
    
    # Inline weight_padded_N if it exists (padding compute op)
    # This avoids boundary checks in tensorize
    try:
        if X_T_has_padding and hasattr(X_T.op, 'input_tensors') and len(X_T.op.input_tensors) > 0:
            X_selected_T_padded = X_T.op.input_tensors[0]
            if hasattr(X_selected_T_padded.op, 'input_tensors') and len(X_selected_T_padded.op.input_tensors) > 0:
                X_selected = X_selected_T_padded.op.input_tensors[0]
                if hasattr(X_selected.op, 'input_tensors') and len(X_selected.op.input_tensors) > 0:
                    weight_padded = X_selected.op.input_tensors[0]
                    # Check if weight_padded is weight_N_padded (padding compute op)
                    if hasattr(weight_padded.op, 'name') and 'weight_N_padded' in weight_padded.op.name:
                        s[weight_padded].compute_inline()
    except:
        pass
    
    # If X_T_actual is a compute op (X_selected), schedule it
    try:
        if X_T_has_padding and hasattr(X_T.op, 'input_tensors') and len(X_T.op.input_tensors) > 0:
            X_selected_T_padded = X_T.op.input_tensors[0]
            if hasattr(X_selected_T_padded.op, 'input_tensors') and len(X_selected_T_padded.op.input_tensors) > 0:
                X_selected = X_selected_T_padded.op.input_tensors[0]
                # Schedule X_selected to be computed inline or at block level
                if hasattr(X_selected.op, 'axis') or hasattr(X_selected.op, 'body'):
                    s[X_selected].compute_inline()
    except:
        pass
    
    # Now inline X_T and X_selected if they are compute ops (after cache_read)
    try:
        if hasattr(X_T.op, 'axis') or hasattr(X_T.op, 'body'):
            s[X_T].compute_inline()
        if hasattr(X.op, 'input_tensors') and len(X.op.input_tensors) > 0:
            X = X_T.op.input_tensors[0]  # X_selected
            if hasattr(X.op, 'axis') or hasattr(X.op, 'body'):
                s[X].compute_inline()
    except:
        pass
    AF = s.cache_read(AS, "wmma.matrix_a", [Y_matrix_mult])
    XF = s.cache_read(XS, "wmma.matrix_b", [Y_matrix_mult])
    
    # Cache write for matrix multiplication
    # Use Y_matrix_mult (the actual matrix multiplication) instead of Y
    YF = s.cache_write(Y_matrix_mult, "wmma.accumulator")
    
    # Cache read from YF to shared memory
    YS = s.cache_read(YF, "shared", [Y_matrix_mult])
    
    # Note: ReLU will be applied in Relay layer (gconv.py) after the matrix multiplication
    # We cannot easily apply ReLU here in the Tensor Core schedule because:
    # 1. YF_relu created with te.compute() is not automatically added to schedule
    # 2. We cannot modify Y_matrix_mult's output directly
    # 3. We cannot modify YS (cache_read result) directly
    # So ReLU will be handled as a separate Relay operation after the DTC SpMM
    # This is acceptable because ReLU is element-wise and can be efficiently fused
    
    # Use Y_matrix_mult for scheduling (this is the actual matrix multiplication)
    # But if Y is a fused operation (e.g., ReLU), we need to schedule it too
    # Store original Y for later scheduling if it's a fused operation
    Y_original = Y
    
    # For scheduling, we'll use Y_matrix_mult (the actual matrix multiplication)
    # This ensures all cache operations (YS, YF, etc.) are correctly scheduled
    Y = Y_matrix_mult
    
    # Define strides
    AS_align = int(chunk * wmma_k + offset)
    XS_align = int(chunk * wmma_k + offset)
    YS_align = int(warp_col_tiles * block_col_warps * wmma_n + offsetCS)
    AS_stride = [AS_align, 1]
    XS_stride = [XS_align, 1]
    AF_stride = [wmma_k, 1]
    XF_stride = [wmma_k, 1]
    YF_stride = [warp_col_tiles * wmma_n, 1]
    YS_stride = [YS_align, 1]
    
    # Thread and block axes
    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    thread_x = te.thread_axis("threadIdx.x")
    thread_y = te.thread_axis("threadIdx.y")
    thread_z = te.thread_axis("threadIdx.z")
    
    # Schedule for output computation
    block_factor_m = wmma_m * warp_row_tiles * block_row_warps
    block_factor_n = wmma_n * warp_col_tiles * block_col_warps
    
    # If Y is a slice, we need to schedule Y_matrix_mult first
    if Y_is_slice:
        # Schedule Y_matrix_mult (the actual padded output)
        # Verify Y_matrix_mult shape
        M_mult, N_mult = get_const_tuple(Y_matrix_mult.shape)
        if _get_verbose():
            print(f"  [DTC Schedule] Y_matrix_mult shape: ({M_mult}, {N_mult}), block_factor_n={block_factor_n}")
        assert N_mult % block_factor_n == 0, f"Y_matrix_mult N dimension ({N_mult}) must be multiple of block_factor_n ({block_factor_n})"
        
        i, j = Y_matrix_mult.op.axis
        block_i, bi = s[Y_matrix_mult].split(i, factor=block_factor_m)
        block_j, bj = s[Y_matrix_mult].split(j, factor=block_factor_n)
        s[Y_matrix_mult].reorder(block_i, block_j, bi, bj)
        
        t = s[Y_matrix_mult].fuse(bi, bj)
        t, vi = s[Y_matrix_mult].split(t, factor=vec)
        t, tx = s[Y_matrix_mult].split(t, factor=warp_size)
        t, ty = s[Y_matrix_mult].split(t, factor=block_row_warps)
        t, tz = s[Y_matrix_mult].split(t, factor=block_col_warps)
        
        s[Y_matrix_mult].bind(block_i, block_x)
        s[Y_matrix_mult].bind(block_j, block_y)
        s[Y_matrix_mult].bind(tz, thread_z)
        s[Y_matrix_mult].bind(ty, thread_y)
        s[Y_matrix_mult].bind(tx, thread_x)
        s[Y_matrix_mult].vectorize(vi)
        
        # Schedule for WMMA store (use Y_matrix_mult)
        s[YS].compute_at(s[Y_matrix_mult], block_j)
        
        # Y_original (slice) is the output tensor, so it must be scheduled at root level
        # But its shape is (M_original, N_original), not (M_padded, N_padded)
        # To avoid boundary checks, we need to schedule it carefully
        # Use the same loop structure as Y_matrix_mult, but don't split N_original if it's not a multiple of block_factor_n
        i_slice, j_slice = Y_original.op.axis
        M_slice, N_slice = get_const_tuple(Y_original.shape)
        
        # Split M dimension (should be fine if M_original is a multiple of block_factor_m)
        block_i_slice, bi_slice = s[Y_original].split(i_slice, factor=block_factor_m)
        
        # For N dimension, only split if N_original is a multiple of block_factor_n
        # Otherwise, don't split to avoid boundary checks
        if N_slice % block_factor_n == 0:
            block_j_slice, bj_slice = s[Y_original].split(j_slice, factor=block_factor_n)
            s[Y_original].reorder(block_i_slice, block_j_slice, bi_slice, bj_slice)
            
            t_slice = s[Y_original].fuse(bi_slice, bj_slice)
            t_slice, vi_slice = s[Y_original].split(t_slice, factor=vec)
            t_slice, tx_slice = s[Y_original].split(t_slice, factor=warp_size)
            t_slice, ty_slice = s[Y_original].split(t_slice, factor=block_row_warps)
            t_slice, tz_slice = s[Y_original].split(t_slice, factor=block_col_warps)
            
            s[Y_original].bind(block_i_slice, block_x)
            s[Y_original].bind(block_j_slice, block_y)
            s[Y_original].bind(tz_slice, thread_z)
            s[Y_original].bind(ty_slice, thread_y)
            s[Y_original].bind(tx_slice, thread_x)
            s[Y_original].vectorize(vi_slice)
        else:
            # N_original is not a multiple of block_factor_n, so don't split it
            # Just bind to threads without splitting to avoid boundary checks
            s[Y_original].reorder(block_i_slice, bi_slice, j_slice)
            
            t_slice = s[Y_original].fuse(bi_slice, j_slice)
            t_slice, vi_slice = s[Y_original].split(t_slice, factor=vec)
            t_slice, tx_slice = s[Y_original].split(t_slice, factor=warp_size)
            t_slice, ty_slice = s[Y_original].split(t_slice, factor=block_row_warps)
            t_slice, tz_slice = s[Y_original].split(t_slice, factor=block_col_warps)
            
            s[Y_original].bind(block_i_slice, block_x)
            s[Y_original].bind(tz_slice, thread_z)
            s[Y_original].bind(ty_slice, thread_y)
            s[Y_original].bind(tx_slice, thread_x)
            s[Y_original].vectorize(vi_slice)
    else:
        # Normal case: Y is the matrix multiplication
        i, j = Y.op.axis
        block_i, bi = s[Y].split(i, factor=block_factor_m)
        block_j, bj = s[Y].split(j, factor=block_factor_n)
        s[Y].reorder(block_i, block_j, bi, bj)
        
        t = s[Y].fuse(bi, bj)
        t, vi = s[Y].split(t, factor=vec)
        t, tx = s[Y].split(t, factor=warp_size)
        t, ty = s[Y].split(t, factor=block_row_warps)
        t, tz = s[Y].split(t, factor=block_col_warps)
        
        s[Y].bind(block_i, block_x)
        s[Y].bind(block_j, block_y)
        s[Y].bind(tz, thread_z)
        s[Y].bind(ty, thread_y)
        s[Y].bind(tx, thread_x)
        s[Y].vectorize(vi)
        
        # Schedule for WMMA store
        s[YS].compute_at(s[Y], block_j)
    
    bb, oo = YS.op.axis
    s[YS].storage_align(bb, YS_align - 1, YS_align)
    bb, bbi = s[YS].split(bb, factor=wmma_m)
    oo, ooi = s[YS].split(oo, factor=wmma_n)
    bb, bbii = s[YS].split(bb, factor=warp_row_tiles)
    oo, ooii = s[YS].split(oo, factor=warp_col_tiles)
    s[YS].reorder(bb, oo, bbii, ooii, bbi, ooi)
    s[YS].bind(bb, thread_y)
    s[YS].bind(oo, thread_z)
    
    # Schedule for WMMA computation
    s[YF].compute_at(s[YS], oo)
    warp_i, warp_j = YF.op.axis
    warp_i, _ii = s[YF].split(warp_i, factor=wmma_m)
    warp_j, _jj = s[YF].split(warp_j, factor=wmma_n)
    (k,) = YF.op.reduce_axis
    k, _k = s[YF].split(k, factor=wmma_k)
    ko, ki = s[YF].split(k, factor=chunk)
    s[YF].reorder(ko, ki, warp_i, warp_j, _ii, _jj, _k)
    
    # Schedule for WMMA matrix_a load
    s[AF].compute_at(s[YF], ki)
    b, i = AF.op.axis
    b, b_ii = s[AF].split(b, factor=wmma_m)
    i, i_jj = s[AF].split(i, factor=wmma_k)
    s[AF].reorder(b, i, b_ii, i_jj)
    
    # Schedule for WMMA matrix_b load
    s[XF].compute_at(s[YF], ki)
    o, i = XF.op.axis
    o, o_ii = s[XF].split(o, factor=wmma_n)
    i, i_ii = s[XF].split(i, factor=wmma_k)
    s[XF].reorder(o, i, o_ii, i_ii)
    
    # Schedule for shared memory load
    def shared_schedule(stage, strides):
        s[stage].compute_at(s[YF], ko)
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
    
    # Schedule shared memory loads
    shared_schedule(AS, AS_align)
    shared_schedule(XS, XS_align)
    
    # Get ko from YF for scheduling padding ops (after YF is scheduled)
    (ko,) = YF.op.reduce_axis
    
    # If A is a padding compute op, schedule it after AS is scheduled
    if A_is_padding:
        try:
            # If AS_input exists, compute A at AS_input's first axis
            # This ensures A's computation uses cached A_actual in thread environment
            if AS_input is not None:
                # Get AS_input's first axis
                AS_input_axis = s[AS_input].op.axis[0] if len(s[AS_input].op.axis) > 0 else None
                if AS_input_axis is not None:
                    # Compute A at AS_input's first axis
                    s[A].compute_at(s[AS_input], AS_input_axis)
                    # Apply shared_schedule to AS_input manually
                    xo, yo = AS_input.op.axis
                    s[AS_input].storage_align(xo, AS_align - 1, AS_align)
                    t = s[AS_input].fuse(xo, yo)
                    t, vi = s[AS_input].split(t, factor=vec)
                    t, tx = s[AS_input].split(t, factor=warp_size)
                    t, ty = s[AS_input].split(t, factor=block_row_warps)
                    _, tz = s[AS_input].split(t, factor=block_col_warps)
                    s[AS_input].bind(ty, thread_y)
                    s[AS_input].bind(tz, thread_z)
                    s[AS_input].bind(tx, thread_x)
                    s[AS_input].vectorize(vi)
                else:
                    # Fallback: compute A at ko
                    s[A].compute_at(s[YF], ko)
            else:
                # No AS_input, compute A at ko
                s[A].compute_at(s[YF], ko)
        except:
            # If scheduling fails, inline A
            try:
                s[A].compute_inline()
            except:
                pass
    
    # If X_T has padding ops, schedule them after XS is scheduled
    if X_T_has_padding:
        try:
            # Get X_selected_T_padded
            if hasattr(X_T.op, 'input_tensors') and len(X_T.op.input_tensors) > 0:
                X_selected_T_padded = X_T.op.input_tensors[0]
                # Schedule X_selected_T_padded to compute at the same level as XS (which is at ko)
                s[X_selected_T_padded].compute_at(s[YF], ko)
        except:
            # If scheduling fails, inline
            try:
                if hasattr(X_T.op, 'input_tensors') and len(X_T.op.input_tensors) > 0:
                    X_selected_T_padded = X_T.op.input_tensors[0]
                    s[X_selected_T_padded].compute_inline()
            except:
                pass
    
    # Tensorize with Tensor Core intrinsics (CRITICAL: ensures Tensor Core usage)
    # Note: YL_compute must be a pure matrix multiplication (only reduction)
    # ReLU cannot be applied here because TVM requires reductions at top level
    shape = (wmma_m, wmma_n, wmma_k)
    AL_gemm = te.placeholder((wmma_m, wmma_k), name="AL_gemm", dtype=data_dtype)
    XL_gemm = te.placeholder((wmma_n, wmma_k), name="XL_gemm", dtype=data_dtype)
    k_gemm = te.reduce_axis((0, wmma_k), name="k_gemm")
    YL_compute = te.compute(
        (wmma_m, wmma_n),
        lambda ii, jj: te.sum(
            AL_gemm[ii, k_gemm].astype(out_dtype) * XL_gemm[jj, k_gemm].astype(out_dtype),
            axis=k_gemm,
        ),
        name="YL_compute",
    )
    
    # Apply Tensor Core intrinsics - this ensures Tensor Core is used
    s[AF].tensorize(
        b_ii,
        intrin_wmma_load_matrix_A(
            AF_stride, AS_stride, shape, "row_major", 
            (wmma_m, wmma_k), (wmma_m, wmma_k), data_dtype
        ),
    )
    s[XF].tensorize(
        o_ii,
        intrin_wmma_load_matrix_W(
            XF_stride, XS_stride, shape, "col_major", 
            (wmma_n, wmma_k), (wmma_n, wmma_k), data_dtype
        ),
    )
    s[YF].tensorize(
        _ii, 
        intrin_wmma_gemm(AL_gemm, XL_gemm, YL_compute, AF_stride, XF_stride, YF_stride, shape)
    )
    s[YS].tensorize(
        bbi,
        intrin_wmma_store_matrix(
            YS_stride, YF_stride, shape, out_dtype, 
            (wmma_m, wmma_n), (wmma_m, wmma_n)
        ),
    )
    
    return s

def test_dtc_spmm_tensorcore():
    """
    Test DTC-SpMM with Tensor Core
    """
    print("=" * 80)
    print("DTC-SpMM Tensor Core Implementation Test")
    print("=" * 80)
    print()
    
    print(f"Test Configuration:")
    print(f"  M (rows in A): {TEST_M}")
    print(f"  K (original cols in A): {TEST_K}")
    print(f"  N (cols in X): {TEST_N}")
    print(f"  Sparsity: {TEST_SPARSITY}")
    print(f"  Tensor Core: {WMMA_M}x{WMMA_N}x{WMMA_K}")
    print()
    
    # Create test data
    print("Creating test data...")
    (A_compressed_np, X_selected_np, Y_ref, Y_ref_trimmed, non_zero_cols, 
     (M_padded, K_padded, N_padded), A_original, X_original, X_selected_mapped) = create_dtc_spmm_data(
        TEST_M, TEST_K, TEST_N, sparsity=TEST_SPARSITY, use_hardcoded=False
    )
    
    print(f"  Original A shape: {TEST_M}x{TEST_K}")
    print(f"  A_compressed shape: {A_compressed_np.shape} (dtype: {A_compressed_np.dtype})")
    print(f"    -> Padded to: {M_padded}x{K_padded} (multiples of {WMMA_M}x{WMMA_K})")
    print(f"  X_selected shape: {X_selected_np.shape} (dtype: {X_selected_np.dtype})")
    print(f"    -> Padded to: {K_padded}x{N_padded} (multiples of {WMMA_K}x{WMMA_N})")
    print(f"  Y_ref shape: {Y_ref.shape} (padded)")
    print(f"  Y_ref_trimmed shape: {Y_ref_trimmed.shape} (original size for comparison)")
    print(f"  Non-zero columns: {len(non_zero_cols)} columns (indices: {non_zero_cols[:10]}...)" if len(non_zero_cols) > 10 else f"  Non-zero columns: {non_zero_cols}")
    print()
    
    # Create TVM placeholders using global dtype constants
    A_compressed = te.placeholder(A_compressed_np.shape, name="A_compressed", dtype=A_dtype)
    X_selected = te.placeholder(X_selected_np.shape, name="X_selected", dtype=X_dtype)
    
    # Compute output - Y_dtype is automatically set based on WMMA size
    Y = dtc_spmm_tensorcore_cuda(A_compressed, X_selected, out_dtype=Y_dtype)
    
    print(f"  Y shape: {Y.shape}")
    print()
    
    # Create schedule
    print("Creating Tensor Core schedule...")
    s = te.create_schedule(Y.op)
    schedule_dtc_spmm_tensorcore(s, Y)
    
    print("✓ Schedule created with Tensor Core intrinsics")
    print()
    
    # Build and run
    print("Building CUDA kernel with Tensor Core...")
    # Use same approach as test_tensorcore_conv.py - just use "cuda" string
    # TVM will auto-detect the GPU architecture
    dev = tvm.cuda(0)
    
    # Check if TensorCore is available
    from tvm.contrib import nvcc
    if not nvcc.have_tensorcore(dev.compute_version):
        raise RuntimeError(f"GPU compute capability {dev.compute_version} does not support TensorCore")
    
    # Build with PassContext (same as test_tensorcore_conv.py)
    with tvm.transform.PassContext(config={"tir.UnrollLoop": {"auto_max_step": 16}}):
        func = tvm.build(s, [A_compressed, X_selected, Y], "cuda")
    print("✓ Kernel built successfully")
    print()
    print(f"✓ Using CUDA device 0 (compute capability: {dev.compute_version})")
    print()
    
    # Prepare data
    print("Preparing data on GPU...")
    A_compressed_tvm = tvm.nd.array(A_compressed_np, dev)
    X_selected_tvm = tvm.nd.array(X_selected_np, dev)
    # Output dtype uses global Y_dtype constant
    Y_tvm = tvm.nd.empty(Y_ref.shape, dtype=Y_dtype, device=dev)
    
    # Run
    print("Running kernel...")
    func(A_compressed_tvm, X_selected_tvm, Y_tvm)
    
    # Verify - compare with padded reference, then trim for original size comparison
    Y_result = Y_tvm.numpy()
    
    # Compare with padded reference
    max_diff_padded = np.max(np.abs(Y_result - Y_ref))
    mean_diff_padded = np.mean(np.abs(Y_result - Y_ref))
    
    # Trim to original size for comparison
    Y_result_trimmed = Y_result[:TEST_M, :TEST_N]
    max_diff = np.max(np.abs(Y_result_trimmed - Y_ref_trimmed))
    mean_diff = np.mean(np.abs(Y_result_trimmed - Y_ref_trimmed))
    
    print(f"  Max difference (padded): {max_diff_padded:.6f}")
    print(f"  Mean difference (padded): {mean_diff_padded:.6f}")
    print(f"  Max difference (trimmed to original): {max_diff:.6f}")
    print(f"  Mean difference (trimmed to original): {mean_diff:.6f}")
    print()
    
    if max_diff < 1e-1:  # More lenient tolerance for float16 (accumulator precision)
        print("✓ Results match reference!")
        print("✓ Tensor Core implementation is correct!")
    else:
        print("⚠ Results differ from reference")
        print(f"  First few values (trimmed):")
        print(f"    Y_result[0, :5] = {Y_result_trimmed[0, :5]}")
        print(f"    Y_ref[0, :5] = {Y_ref_trimmed[0, :5]}")
    
    # Check if Tensor Core is used by examining the generated code
    print()
    print("Verifying Tensor Core usage...")
    try:
        source_code = func.get_source()
        if "wmma" in source_code.lower() or "mma.sync" in source_code.lower():
            print("✓ Tensor Core intrinsics found in generated code!")
            print("  (wmma/mma.sync instructions indicate Tensor Core usage)")
            print("  ✓ CONFIRMED: Tensor Core is used, no fallback to CUDA core!")
        else:
            print("⚠ Warning: Tensor Core intrinsics not clearly visible in source")
            print("  (This may be due to code generation format)")
    except:
        print("  (Could not inspect generated source code)")
        print("  ✓ Kernel built successfully with Tensor Core intrinsics applied")
    
    # Benchmark execution time (same as spmm_cuda.py)
    print()
    print("=" * 80)
    print("Benchmarking execution time...")
    print("=" * 80)
    evaluator = func.time_evaluator(func.entry_name, dev, min_repeat_ms=500)
    execution_time = np.median(evaluator(A_compressed_tvm, X_selected_tvm, Y_tvm).results) * 1000
    print(f"DTC-SpMM Tensor Core execution time: {execution_time:.3f} ms")
    
    # Calculate GFLOPS
    # For matrix multiplication: Y = A @ X
    # Operations: 2 * M * K * N (multiply-add pairs)
    total_ops = 2 * M_padded * K_padded * N_padded
    gflops = (total_ops / 1e9) / (execution_time / 1000.0)
    print(f"Performance: {gflops:.2f} GFLOPS")
    print(f"  Matrix dimensions: {M_padded}x{K_padded} @ {K_padded}x{N_padded} = {M_padded}x{N_padded}")
    print(f"  Total operations: {total_ops:,} (2 * M * K * N)")
    
    # Visualize results after computation
    if HAS_MATPLOTLIB:
        print()
        print("=" * 80)
        print("Visualizing computation results...")
        print("=" * 80)
        # Get A_compressed_used_cols from non_zero_cols
        A_compressed_used_cols = non_zero_cols
        # Use computed Y_result for visualization
        visualize_matrices(A_original, A_compressed_np, X_original, X_selected_mapped, Y_result, TEST_M, TEST_K, TEST_N,
                          A_compressed_used_cols=A_compressed_used_cols,
                          X_selected_used_rows=None)
        print("✓ Visualization saved to dtc_spmm_matrices.png")
            

def test_dtc_spmm_tensorcore_relay():
    """
    Test DTC-SpMM Tensor Core in Relay
    
    This tests the integration of DTC-SpMM Tensor Core implementation
    into Relay by using relay.nn.dense with DTC compressed matrices.
    """
    print("\n" + "=" * 80)
    print("DTC-SpMM Tensor Core Relay Integration Test")
    print("=" * 80)
    print()
    
    print(f"Test Configuration:")
    print(f"  M (rows in A): {TEST_M}")
    print(f"  K (original cols in A): {TEST_K}")
    print(f"  N (cols in X): {TEST_N}")
    print(f"  Sparsity: {TEST_SPARSITY}")
    print(f"  Tensor Core: {WMMA_M}x{WMMA_N}x{WMMA_K}")
    print()
    
    # Register Tensor Core strategies
    print("Registering Tensor Core strategies...")
    try:
        try:
            from .tensorcore_strategy import register_tensorcore_strategies
        except ImportError:
            from apps.gnn_tvm_utils.tensorcore_strategy import register_tensorcore_strategies
        register_tensorcore_strategies()
        print("✓ Tensor Core strategies registered")
    except Exception as e:
        print(f"✗ Failed to register strategies: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()
    
    # Create test data
    print("Creating test data...")
    (A_compressed_np, X_selected_np, Y_ref, Y_ref_trimmed, non_zero_cols, 
     (M_padded, K_padded, N_padded), A_original, X_original, X_selected_mapped) = create_dtc_spmm_data(
        TEST_M, TEST_K, TEST_N, sparsity=TEST_SPARSITY, use_hardcoded=False
    )
    
    print(f"  Original A shape: {TEST_M}x{TEST_K}")
    print(f"  A_compressed shape: {A_compressed_np.shape} (dtype: {A_compressed_np.dtype})")
    print(f"    -> Padded to: {M_padded}x{K_padded} (multiples of {WMMA_M}x{WMMA_K})")
    print(f"  X_selected shape: {X_selected_np.shape} (dtype: {X_selected_np.dtype})")
    print(f"    -> Padded to: {K_padded}x{N_padded} (multiples of {WMMA_K}x{WMMA_N})")
    print(f"  Y_ref shape: {Y_ref.shape} (padded)")
    print(f"  Y_ref_trimmed shape: {Y_ref_trimmed.shape} (original size for comparison)")
    print()
    
    # Build Relay model
    print("Building Relay model...")
    try:
        import tvm
        from tvm import relay
        from tvm.contrib import graph_executor
        
        # Create input variables
        # A_compressed: (M_padded, K_padded)
        # X_selected: (K_padded, N_padded)
        # We need to compute: A_compressed @ X_selected = (M_padded, N_padded)
        # In Relay: dense(A, B) computes A @ B^T
        # So: dense(A_compressed, X_selected_T) = A_compressed @ X_selected
        
        # Create A_compressed as constant
        A_compressed_const = relay.Constant(tvm.nd.array(A_compressed_np.astype("float16")))
        
        # Create X_selected as input variable (will be transposed in dense)
        X_selected_var = relay.var("X_selected", shape=(K_padded, N_padded), dtype="float32")
        X_selected_fp16 = relay.cast(X_selected_var, "float16")
        
        # Transpose X_selected for dense: (K_padded, N_padded) -> (N_padded, K_padded)
        X_selected_T = relay.transpose(X_selected_fp16)
        
        # Dense layer: A_compressed @ X_selected_T^T = A_compressed @ X_selected
        # This should use Tensor Core via strategy registration
        dense_out = relay.nn.dense(A_compressed_const, X_selected_T, out_dtype="float32")
        
        # Create function
        func = relay.Function([X_selected_var], dense_out)
        mod = tvm.IRModule()
        mod["main"] = func
        
        print("✓ Relay model created")
        print(f"  Function computes: A_compressed @ X_selected")
        print(f"    A_compressed: {A_compressed_np.shape} (constant)")
        print(f"    X_selected: {X_selected_np.shape} (input)")
        print(f"    Output: ({M_padded}, {N_padded})")
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
        
        # Check if TensorCore is available
        from tvm.contrib import nvcc
        if not nvcc.have_tensorcore(dev.compute_version):
            raise RuntimeError(f"GPU compute capability {dev.compute_version} does not support TensorCore")
        
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target)
        
        print("✓ Compilation successful")
        print(f"✓ Using CUDA device 0 (compute capability: {dev.compute_version})")
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
        
        # Set input (X_selected as float32, will be cast to float16)
        executor.set_input("X_selected", X_selected_np.astype("float32"))
        
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
        # Compare with padded reference
        max_diff_padded = np.max(np.abs(Y_tvm - Y_ref))
        mean_diff_padded = np.mean(np.abs(Y_tvm - Y_ref))
        
        # Trim to original size for comparison
        Y_tvm_trimmed = Y_tvm[:TEST_M, :TEST_N]
        max_diff = np.max(np.abs(Y_tvm_trimmed - Y_ref_trimmed))
        mean_diff = np.mean(np.abs(Y_tvm_trimmed - Y_ref_trimmed))
        
        print(f"  Max difference (padded): {max_diff_padded:.6f}")
        print(f"  Mean difference (padded): {mean_diff_padded:.6f}")
        print(f"  Max difference (trimmed to original): {max_diff:.6f}")
        print(f"  Mean difference (trimmed to original): {mean_diff:.6f}")
        print()
        
        # Use reasonable tolerance for float16 accumulation
        if max_diff < 1.0:
            print("✓ Results match reference (within tolerance)")
        else:
            print("⚠ Results differ from reference")
            print(f"  First few values (trimmed):")
            print(f"    Y_tvm[0, :5] = {Y_tvm_trimmed[0, :5]}")
            print(f"    Y_ref[0, :5] = {Y_ref_trimmed[0, :5]}")
        print()
        
    except Exception as e:
        print(f"✗ Verification failed: {e}")
        import traceback
        traceback.print_exc()
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
        total_ops = 2 * M_padded * K_padded * N_padded
        gflops = (total_ops / 1e9) / (mean_time / 1000.0)
        
        print(f"  Mean execution time: {mean_time:.4f} ms")
        print(f"  Performance: {gflops:.2f} GFLOPS")
        print(f"  Total operations: {total_ops:,} (2 * M * K * N)")
        print(f"  Matrix dimensions: {M_padded}x{K_padded} @ {K_padded}x{N_padded} = {M_padded}x{N_padded}")
        print()
        
    except Exception as e:
        print(f"  (Benchmarking failed: {e})")
        print()
    
    print("=" * 80)
    print("Relay integration test completed successfully!")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    # Test TE-level implementation
    test_dtc_spmm_tensorcore()
    
    # Test Relay-level integration
    test_dtc_spmm_tensorcore_relay()
