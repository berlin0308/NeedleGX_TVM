"""
CSR to DTC Format Conversion

This module provides functions to convert CSR (Compressed Sparse Row) format
or NetworkX graph directly to DTC (Dense Tensor Core) compressed format for Tensor Core acceleration.
"""

import numpy as np
import scipy.sparse as sp

# Tensor Core block dimensions
WMMA_M = 16  # WMMA matrix M dimension
WMMA_N = 16  # WMMA matrix N dimension
WMMA_K = 16  # WMMA matrix K dimension
BLK_H = WMMA_M  # Block height (rows per TC block)
BLK_W = WMMA_N  # Block width (cols per TC block)


def csr_to_dtc(adjacency_csr, dtype=np.float32):
    """
    Convert CSR sparse matrix to DTC compressed format
    
    DTC (Dense Tensor Core) format compresses sparse matrices by:
    1. Processing in 16x16 blocks
    2. For each row block (16 rows), finding non-zero columns
    3. Packing non-zero columns into a dense compressed matrix
    
    Parameters
    ----------
    adjacency_csr: scipy.sparse.csr_matrix or scipy.sparse.csr_array
        Input sparse matrix in CSR format
    dtype: numpy.dtype
        Data type for compressed matrix (default: float32)
    
    Returns
    -------
    dict: Dictionary containing DTC format data
        - 'A_compressed': np.ndarray, shape (M_padded, K_padded)
            Compressed dense matrix
        - 'block_col_mapping': list of lists
            For each row block, list of original column indices used
        - 'M': int, original number of rows
        - 'K': int, original number of columns
        - 'M_padded': int, padded number of rows (multiple of 16)
        - 'K_padded': int, padded number of columns (multiple of 16)
        - 'num_row_blocks': int, number of row blocks
    """
    # Convert to dense matrix for processing
    if sp.issparse(adjacency_csr):
        A = adjacency_csr.toarray().astype(dtype)
    else:
        A = np.asarray(adjacency_csr, dtype=dtype)
    
    M, K = A.shape
    
    # DTC compression: process by blocks
    # CRITICAL: Use block_factor_m = 32 for consistency with M_padded calculation
    block_factor_m = 64  # From schedule_dtc_spmm_tensorcore
    num_row_blocks = (M + block_factor_m - 1) // block_factor_m
    
    # For each row block, collect non-zero columns
    block_non_zero_cols = []  # List of lists: one per row block
    
    for row_block_idx in range(num_row_blocks):
        row_start = row_block_idx * block_factor_m
        row_end = min(row_start + block_factor_m, M)
        
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
    
    # CRITICAL: Build A_compressed with GLOBAL mapping (not block-wise)
    # This matches spmm_mma.py's global mapping strategy
    # 
    # Step 1: Collect all unique columns used across all blocks
    all_non_zero_cols = set()
    for block_cols in block_non_zero_cols:
        all_non_zero_cols.update(block_cols)
    A_compressed_used_cols = sorted(list(all_non_zero_cols))
    
    # Step 2: Determine K_padded (must be multiple of WMMA_K * chunk)
    # CRITICAL: Use len(A_compressed_used_cols) instead of max_cols per block
    # because we need to accommodate ALL unique columns used across ALL blocks,
    # not just the maximum columns in a single block
    chunk_factor = 4  # Must match chunk in schedule_dtc_spmm_tensorcore
    # Use the total number of unique columns, not the max per block
    num_unique_cols = len(A_compressed_used_cols)
    K_padded = ((num_unique_cols + BLK_W - 1) // BLK_W) * BLK_W  # Round up to multiple of BLK_W
    # Then pad to multiple of WMMA_K * chunk_factor (64)
    K_padded = ((K_padded + WMMA_K * chunk_factor - 1) // (WMMA_K * chunk_factor)) * (WMMA_K * chunk_factor)
    
    # Build global mapping: comp_col -> orig_col
    mapping_cols = list(A_compressed_used_cols)
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
    
    # Step 3: Build A_compressed with global mapping
    # A_compressed[i, comp_col] = A[i, mapping_cols[comp_col]]
    # This ensures that A_compressed uses the same global mapping as X_selected in gconv.py
    # CRITICAL: M_padded must be multiple of 32 for Tensor Core schedule (block_factor_m = 32)
    # block_factor_m = WMMA_M * warp_row_tiles * block_row_warps = 16 * 1 * 2 = 32
    block_factor_m = 32  # From schedule_dtc_spmm_tensorcore
    M_padded = ((M + block_factor_m - 1) // block_factor_m) * block_factor_m
    A_compressed = np.zeros((M_padded, K_padded), dtype=dtype)
    
    # Mapping: for each row block, which original columns are used (for reference)
    block_col_mapping = []  # List of column indices for each row block
    
    for row_block_idx in range(num_row_blocks):
        block_cols = block_non_zero_cols[row_block_idx]
        block_col_mapping.append(block_cols)
        
        # Build A_compressed using global mapping
        # CRITICAL: Use block_factor_m = 32 for consistency with M_padded calculation
        block_factor_m = 32  # From schedule_dtc_spmm_tensorcore
        row_start = row_block_idx * block_factor_m
        row_end = min(row_start + block_factor_m, M)
        for i in range(row_start, row_end):
            for comp_col in range(K_padded):
                orig_col = mapping_cols[comp_col]
                # Copy value from original matrix (will be 0 if not present)
                if orig_col < K:
                    A_compressed[i, comp_col] = A[i, orig_col]
    
    return {
        'A_compressed': A_compressed,
        'block_col_mapping': block_col_mapping,
        'M': M,
        'K': K,
        'M_padded': M_padded,
        'K_padded': K_padded,
        'num_row_blocks': num_row_blocks,
    }


def graph_to_dtc(graph, dtype=np.float32):
    """
    Convert NetworkX graph directly to DTC compressed format
    
    This function converts a NetworkX graph directly to DTC format without
    going through CSR, following the logic in spmm_mma.py for consistency.
    
    Parameters
    ----------
    graph: networkx.Graph
        Input NetworkX graph
    dtype: numpy.dtype
        Data type for compressed matrix (default: float32)
    
    Returns
    -------
    dict: Dictionary containing DTC format data
        - 'A_compressed': np.ndarray, shape (M_padded, K_padded)
            Compressed dense matrix with GLOBAL mapping (matching spmm_mma.py)
        - 'block_col_mapping': list of lists
            For each row block, list of original column indices used
        - 'M': int, original number of rows
        - 'K': int, original number of columns
        - 'M_padded': int, padded number of rows (multiple of 16)
        - 'K_padded': int, padded number of columns (multiple of 16)
        - 'num_row_blocks': int, number of row blocks
    """
    import networkx as nx
    
    # Get number of nodes
    num_nodes = graph.number_of_nodes()
    M = K = num_nodes
    
    # Build adjacency matrix directly from graph (following spmm_mma.py logic)
    # Create dense matrix A (M x K) from graph edges
    A = np.zeros((M, K), dtype=dtype)
    
    # Fill adjacency matrix from graph edges
    for edge in graph.edges():
        i, j = edge
        A[i, j] = 1.0
        # If undirected, also set symmetric entry
        if not graph.is_directed():
            A[j, i] = 1.0
    
    # DTC compression: process by blocks
    # CRITICAL: Use block_factor_m = 32 for consistency with M_padded calculation
    block_factor_m = 32  # From schedule_dtc_spmm_tensorcore
    num_row_blocks = (M + block_factor_m - 1) // block_factor_m
    
    # For each row block, collect non-zero columns
    block_non_zero_cols = []  # List of lists: one per row block
    
    for row_block_idx in range(num_row_blocks):
        row_start = row_block_idx * block_factor_m
        row_end = min(row_start + block_factor_m, M)
        
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
    
    # CRITICAL: Build A_compressed with GLOBAL mapping (matching spmm_mma.py)
    # Step 1: Collect all unique columns used across all blocks
    all_non_zero_cols = set()
    for block_cols in block_non_zero_cols:
        all_non_zero_cols.update(block_cols)
    A_compressed_used_cols = sorted(list(all_non_zero_cols))
    
    # Step 2: Determine K_padded (must be multiple of WMMA_K * chunk)
    # CRITICAL: Use len(A_compressed_used_cols) instead of max_cols per block
    # because we need to accommodate ALL unique columns used across ALL blocks,
    # not just the maximum columns in a single block
    chunk_factor = 4  # Must match chunk in schedule_dtc_spmm_tensorcore
    # Use the total number of unique columns, not the max per block
    num_unique_cols = len(A_compressed_used_cols)
    K_padded = ((num_unique_cols + BLK_W - 1) // BLK_W) * BLK_W  # Round up to multiple of BLK_W
    # Then pad to multiple of WMMA_K * chunk_factor (64)
    K_padded = ((K_padded + WMMA_K * chunk_factor - 1) // (WMMA_K * chunk_factor)) * (WMMA_K * chunk_factor)
    
    # Build global mapping: comp_col -> orig_col
    mapping_cols = list(A_compressed_used_cols)
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
    
    # Step 3: Build A_compressed with global mapping
    # A_compressed[i, comp_col] = A[i, mapping_cols[comp_col]]
    # This ensures that A_compressed uses the same global mapping as X_selected in gconv.py
    # CRITICAL: M_padded must be multiple of 32 for Tensor Core schedule (block_factor_m = 32)
    # block_factor_m = WMMA_M * warp_row_tiles * block_row_warps = 16 * 1 * 2 = 32
    block_factor_m = 32  # From schedule_dtc_spmm_tensorcore
    M_padded = ((M + block_factor_m - 1) // block_factor_m) * block_factor_m
    A_compressed = np.zeros((M_padded, K_padded), dtype=dtype)
    
    # Mapping: for each row block, which original columns are used (for reference)
    block_col_mapping = []  # List of column indices for each row block
    
    for row_block_idx in range(num_row_blocks):
        block_cols = block_non_zero_cols[row_block_idx]
        block_col_mapping.append(block_cols)
        
        # Build A_compressed using global mapping
        # CRITICAL: Use block_factor_m = 32 for consistency with M_padded calculation
        block_factor_m = 32  # From schedule_dtc_spmm_tensorcore
        row_start = row_block_idx * block_factor_m
        row_end = min(row_start + block_factor_m, M)
        for i in range(row_start, row_end):
            for comp_col in range(K_padded):
                orig_col = mapping_cols[comp_col]
                # Copy value from original matrix (will be 0 if not present)
                if orig_col < K:
                    A_compressed[i, comp_col] = A[i, orig_col]
    
    return {
        'A_compressed': A_compressed,
        'block_col_mapping': block_col_mapping,
        'M': M,
        'K': K,
        'M_padded': M_padded,
        'K_padded': K_padded,
        'num_row_blocks': num_row_blocks,
    }


if __name__ == "__main__":
    # Test conversion
    import scipy.sparse as sp
    
    # Create a test CSR matrix
    M, K = 32, 32
    A = sp.random(M, K, density=0.1, format='csr', dtype=np.float32)
    
    print(f"Original CSR matrix: {M}x{K}, {A.nnz} non-zeros")
    
    # Convert to DTC
    dtc_data = csr_to_dtc(A)
    
    print(f"DTC compressed matrix: {dtc_data['M_padded']}x{dtc_data['K_padded']}")
    print(f"Compression ratio: {(M * K) / (dtc_data['M_padded'] * dtc_data['K_padded']):.2f}x")
    print(f"Number of row blocks: {dtc_data['num_row_blocks']}")

