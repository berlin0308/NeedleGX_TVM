#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>

#include <cstdio>
#include <random>
#include <vector>
#include <algorithm>

#define CHECK_CUDA(expr)                                                        \
  do {                                                                          \
    cudaError_t _err = (expr);                                                  \
    if (_err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,             \
              cudaGetErrorString(_err));                                        \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

#define CHECK_CUSPARSE(expr)                                                    \
  do {                                                                          \
    cusparseStatus_t _err = (expr);                                             \
    if (_err != CUSPARSE_STATUS_SUCCESS) {                                      \
      fprintf(stderr, "cuSPARSE error %s:%d: %d\n", __FILE__, __LINE__, _err);  \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

#define CHECK_CUBLAS(expr)                                                      \
  do {                                                                          \
    cublasStatus_t _err = (expr);                                               \
    if (_err != CUBLAS_STATUS_SUCCESS) {                                        \
      fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, _err);    \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

// Simple timing helper using CUDA events.
float elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
  float ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
  return ms;
}

int main(int argc, char** argv) {
  int m = 1024;  // rows
  int n = 1024;  // cols
  int k = 256;   // feature dim
  float density = 0.02f;
  int runs = 50;
  if (argc > 1) m = std::atoi(argv[1]);
  if (argc > 2) n = std::atoi(argv[2]);
  if (argc > 3) k = std::atoi(argv[3]);
  if (argc > 4) density = std::atof(argv[4]);

  printf("Benchmark: dense vs SpMM (cuSPARSE)\n");
  printf("m=%d n=%d k=%d density=%.4f runs=%d\n", m, n, k, density, runs);

  // Generate random sparse matrix in COO.
  std::mt19937 rng(0);
  std::uniform_real_distribution<float> uni01(0.f, 1.f);
  std::normal_distribution<float> norm(0.f, 1.f);

  std::vector<int> h_rows;
  std::vector<int> h_cols;
  std::vector<float> h_vals;
  h_rows.reserve(static_cast<size_t>(m * n * density));
  h_cols.reserve(h_rows.capacity());
  h_vals.reserve(h_rows.capacity());
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      if (uni01(rng) < density) {
        h_rows.push_back(row);
        h_cols.push_back(col);
        h_vals.push_back(norm(rng));
      }
    }
  }
  int nnz = static_cast<int>(h_rows.size());
  printf("nnz=%d (actual density=%.4f)\n", nnz, nnz / float(m * n));

  // Host dense copy for cuBLAS path (column-major).
  std::vector<float> h_dense(static_cast<size_t>(m) * n, 0.f);
  for (int idx = 0; idx < nnz; ++idx) {
    // column-major: entry (row, col) stored at col*ld + row
    h_dense[h_cols[idx] * m + h_rows[idx]] = h_vals[idx];
  }

  // Device allocations
  int *d_rows = nullptr, *d_cols = nullptr;
  float *d_vals = nullptr, *d_B = nullptr, *d_C = nullptr, *d_dense = nullptr;
  CHECK_CUDA(cudaMalloc((void**)&d_rows, nnz * sizeof(int)));
  CHECK_CUDA(cudaMalloc((void**)&d_cols, nnz * sizeof(int)));
  CHECK_CUDA(cudaMalloc((void**)&d_vals, nnz * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void**)&d_B, n * k * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void**)&d_C, m * k * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void**)&d_dense, m * n * sizeof(float)));

  CHECK_CUDA(cudaMemcpy(d_rows, h_rows.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_cols, h_cols.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_vals, h_vals.data(), nnz * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_dense, h_dense.data(), m * n * sizeof(float), cudaMemcpyHostToDevice));

  // Random dense features B (column-major for cuBLAS)
  std::vector<float> h_B(static_cast<size_t>(n) * k);
  for (auto& v : h_B) v = norm(rng);
  CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), n * k * sizeof(float), cudaMemcpyHostToDevice));

  // cuSPARSE SpMM (COO x dense)
  cusparseHandle_t sp_handle;
  CHECK_CUSPARSE(cusparseCreate(&sp_handle));
  cusparseSpMatDescr_t matA;
  cusparseDnMatDescr_t matB, matC;
  CHECK_CUSPARSE(cusparseCreateCoo(&matA, m, n, nnz, d_rows, d_cols, d_vals,
                                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
  CHECK_CUSPARSE(cusparseCreateDnMat(&matB, n, k, k, d_B, CUDA_R_32F, CUSPARSE_ORDER_ROW));
  CHECK_CUSPARSE(cusparseCreateDnMat(&matC, m, k, k, d_C, CUDA_R_32F, CUSPARSE_ORDER_ROW));

  float alpha = 1.f, beta = 0.f;
  size_t bufferSize = 0;
  CHECK_CUSPARSE(cusparseSpMM_bufferSize(sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         &alpha, matA, matB, &beta, matC,
                                         CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
                                         &bufferSize));
  void* dBuffer = nullptr;
  CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

  // Warmup
  for (int i = 0; i < 5; ++i) {
    CHECK_CUSPARSE(cusparseSpMM(sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, matB, &beta, matC,
                                CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer));
  }
  cudaEvent_t ev_start, ev_stop;
  CHECK_CUDA(cudaEventCreate(&ev_start));
  CHECK_CUDA(cudaEventCreate(&ev_stop));
  CHECK_CUDA(cudaEventRecord(ev_start));
  for (int i = 0; i < runs; ++i) {
    CHECK_CUSPARSE(cusparseSpMM(sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, matB, &beta, matC,
                                CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer));
  }
  CHECK_CUDA(cudaEventRecord(ev_stop));
  CHECK_CUDA(cudaEventSynchronize(ev_stop));
  float spmm_ms = elapsed_ms(ev_start, ev_stop) / runs;
  printf("cuSPARSE SpMM: %.3f ms avg over %d runs\\n", spmm_ms, runs);

  // cuBLAS dense GEMM: C_col = B_col * A_col (column-major)
  cublasHandle_t blas_handle;
  CHECK_CUBLAS(cublasCreate(&blas_handle));
  const float one = 1.f, zero = 0.f;
  // Warmup
  for (int i = 0; i < 5; ++i) {
    CHECK_CUBLAS(cublasSgemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             k, m, n, &one, d_B, k, d_dense, n, &zero, d_C, k));
  }
  CHECK_CUDA(cudaEventRecord(ev_start));
  for (int i = 0; i < runs; ++i) {
    CHECK_CUBLAS(cublasSgemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             k, m, n, &one, d_B, k, d_dense, n, &zero, d_C, k));
  }
  CHECK_CUDA(cudaEventRecord(ev_stop));
  CHECK_CUDA(cudaEventSynchronize(ev_stop));
  float gemm_ms = elapsed_ms(ev_start, ev_stop) / runs;
  printf("cuBLAS dense GEMM: %.3f ms avg over %d runs\\n", gemm_ms, runs);

  // Cleanup
  cublasDestroy(blas_handle);
  cusparseDestroySpMat(matA);
  cusparseDestroyDnMat(matB);
  cusparseDestroyDnMat(matC);
  cusparseDestroy(sp_handle);
  cudaFree(dBuffer);
  cudaFree(d_rows);
  cudaFree(d_cols);
  cudaFree(d_vals);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_dense);
  cudaEventDestroy(ev_start);
  cudaEventDestroy(ev_stop);
  return 0;
}
