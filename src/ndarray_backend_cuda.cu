#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifdef USE_CUSPARSE_SPMM
#include <cusparse.h>
#endif

#include <cmath>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace needle {
namespace cuda {

#define BASE_THREAD_NUM 256

#define TILE 4
#define V 2
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }
  
  scalar_t* ptr;
  size_t size;
};

struct CudaDims {
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

// Runtime toggle for cuSPARSE SpMM
#ifdef USE_CUSPARSE_SPMM
static bool g_use_cusparse_spmm = true;
static bool g_use_tensor_core_spmm = false;
#else
static bool g_use_cusparse_spmm = false;
static bool g_use_tensor_core_spmm = false;
#endif

CudaVec VecToCuda(const std::vector<int32_t>& x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

#ifdef USE_CUSPARSE_SPMM
// Cache tensor core availability to avoid repeated device queries.
static bool TensorCoresAvailable() {
  static int cached = -1;
  if (cached >= 0) return cached;
  cudaDeviceProp prop{};
  int device_id = 0;
  cudaError_t err_dev = cudaGetDevice(&device_id);
  cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
  if (err_dev != cudaSuccess || err != cudaSuccess) {
    cached = 0;
  } else {
    cached = (prop.major > 7 || (prop.major == 7 && prop.minor >= 0)) ? 1 : 0;
  }
  return cached;
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t* out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = val;
}

void Fill(CudaArray* out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

// Untility function to convert contiguous index i to memory location from strides



__global__ void CompactKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the 
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   * 
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of out array
   *   offset: offset of out array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= size) return;
  /// BEGIN SOLUTION
  int ndim = shape.size;

  size_t cur = gid;
  size_t in_idx = offset;
  for (int i = ndim - 1; i >= 0; i--) {
      size_t idx_i = cur % shape.data[i];
      cur /= shape.data[i];
      in_idx += idx_i * strides.data[i];
  }

  out[gid] = a[in_idx]; // note there is no for loop over input a
  /// END SOLUTION
}

void Compact(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory.  Unlike the C++ version, in CUDA this will primarily call the 
   * relevant CUDA kernel.  In this case, we illustrate how you should set this up (i.e., we give 
   * you the code for this fuction, and also the prototype for the CompactKernel() function).  For
   * the functions after this, however, you'll need to define these kernels as you see fit to 
   * execute the underlying function.
   * 
   * Args:
   *   a: non-compact represntation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   */

  // Nothing needs to be added here
  CudaDims dim = CudaOneDim(out->size);
  CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
}


__global__ void EwiseSetitemKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the ewise setitem opeation.  This should effectively map a single entry in 
   * the compact input a, to the corresponding item in the non-compact array out.
   * 
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of a array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of out array
   *   offset: offset of out array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= size) return;
  /// BEGIN SOLUTION
  int ndim = shape.size;

  size_t cur = gid;
  size_t out_idx = offset;
  for (int i = ndim - 1; i >= 0; i--) {
      size_t idx_i = cur % shape.data[i];
      cur /= shape.data[i];
      out_idx += idx_i * strides.data[i];
  }

  out[out_idx] = a[gid]; // note there is no for loop over input a
  /// END SOLUTION
}

void EwiseSetitem(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array using CUDA.  Yyou will most likely want to implement a
   * EwiseSetitemKernel() function, similar to those above, that will do the actual work.
   * 
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(a.size);
  EwiseSetitemKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, a.size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
  /// END SOLUTION
}


__global__ void ScalarSetitemKernel(scalar_t val, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the scalar setitem opeation.  This should effectively map a single entry 
   * in the non-compact output out, to be set to the scalar value val.
   * 
   * Args:
   *   val: scalar value to write to
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of out array
   *   offset: offset of out array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= size) return;
  /// BEGIN SOLUTION
  int ndim = shape.size;

  size_t cur = gid;
  size_t out_idx = offset;
  for (int i = ndim - 1; i >= 0; i--) {
      size_t idx_i = cur % shape.data[i];
      cur /= shape.data[i];
      out_idx += idx_i * strides.data[i];
  }

  out[out_idx] = val; // note there is no for loop over input a
  /// END SOLUTION
}

void ScalarSetitem(size_t size, scalar_t val, CudaArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   * 
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the 
   *         product of items in shape, but covenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(size);
  ScalarSetitemKernel<<<dim.grid, dim.block>>>(val, out->ptr, size, VecToCuda(shape),
                                                VecToCuda(strides), offset);
  /// END SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////


__global__ void EwiseAddKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + b[gid];
}

void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Add together two CUDA arrays.
   * Args:
   *   a: Input array 'a' to be added
   *   b: Input array 'b' to be added
   *   out: Output array to store the result of 'a + b'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Kernel will execute on 'dim.grid' blocks, each containing 'dim.block' threads.
  EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarAddKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + val;
}

void ScalarAdd(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Add a scalar value to every element of a CUDA array.
   * Args:
   *   a: Input array 'a'
   *   val: Scalar value to be added
   *   out: Output array to store the result of 'a + val'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Launch the ScalarAddKernel that will add the scalar 'val' to each element of array 'a', 
  // and store the result in array 'out'.
  ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

/**
 * In the code the follows, use the above template to create analogous elementise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */

__global__ void EwiseMulKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] * b[gid];
}

void EwiseMul(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Multiply two CUDA arrays element-wise.
   * Args:
   *   a: Input array 'a' to be multiplied
   *   b: Input array 'b' to be multiplied
   *   out: Output array to store the result of 'a * b'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Kernel will execute on 'dim.grid' blocks, each containing 'dim.block' threads.
  EwiseMulKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}


__global__ void ScalarMulKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] * val;
}

void ScalarMul(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Multiply a scalar value to every element of a CUDA array.
   * Args:
   *   a: Input array 'a'
   *   val: Scalar value to be multiplied
   *   out: Output array to store the result of 'a * val'
   */
  CudaDims dim = CudaOneDim(out->size);

  ScalarMulKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseDivKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] / b[gid];
}

void EwiseDiv(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Divide two CUDA arrays element-wise.
   * Args:
   *   a: Input array 'a' (numerator)
   *   b: Input array 'b' (denominator)
   *   out: Output array to store the result of 'a / b'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Kernel will execute on 'dim.grid' blocks, each containing 'dim.block' threads.
  EwiseDivKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarDivKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] / val;
}

void ScalarDiv(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Divide every element of a CUDA array by a scalar value.
   * Args:
   *   a: Input array 'a'
   *   val: Scalar value (denominator)
   *   out: Output array to store the result of 'a / val'
   */
  CudaDims dim = CudaOneDim(out->size);

  ScalarDivKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void ScalarPowerKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = powf(a[gid], val);
}

void ScalarPower(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Raise every element of a CUDA array to the power of a scalar value.
   * Args:
   *   a: Input array 'a'
   *   val: Scalar exponent
   *   out: Output array to store the result of 'a ** val'
   */
  CudaDims dim = CudaOneDim(out->size);

  ScalarPowerKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}


__global__ void EwiseMaximumKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = fmaxf(a[gid], b[gid]);
}

void EwiseMaximum(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Compute the element-wise maximum of two CUDA arrays.
   * Args:
   *   a: Input array 'a'
   *   b: Input array 'b'
   *   out: Output array to store the result of element-wise maximum
   */
  CudaDims dim = CudaOneDim(out->size);

  // Kernel will execute on 'dim.grid' blocks, each containing 'dim.block' threads.
  EwiseMaximumKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarMaximumKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = fmaxf(a[gid], val);
}

void ScalarMaximum(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Compute the element-wise maximum of a CUDA array and a scalar value.
   * Args:
   *   a: Input array 'a'
   *   val: Scalar value
   *   out: Output array to store the result of element-wise maximum
   */
  CudaDims dim = CudaOneDim(out->size);

  ScalarMaximumKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseEqKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = (a[gid] == b[gid]) ? 1.0f : 0.0f;
}

void EwiseEq(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Compare two CUDA arrays for equality element-wise.
   * Args:
   *   a: Input array 'a'
   *   b: Input array 'b'
   *   out: Output array to store the result of element-wise equality comparison
   */
  CudaDims dim = CudaOneDim(out->size);

  // Kernel will execute on 'dim.grid' blocks, each containing 'dim.block' threads.
  EwiseEqKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarEqKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = (a[gid] == val) ? 1.0f : 0.0f;
}

void ScalarEq(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Compare a CUDA array for equality with a scalar value element-wise.
   * Args:
   *   a: Input array 'a'
   *   val: Scalar value
   *   out: Output array to store the result of element-wise equality comparison
   */
  CudaDims dim = CudaOneDim(out->size);

  ScalarEqKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseGeKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = (a[gid] >= b[gid]) ? 1.0f : 0.0f;
}

void EwiseGe(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Compare two CUDA arrays for greater-than-or-equal element-wise.
   * Args:
   *   a: Input array 'a'
   *   b: Input array 'b'
   *   out: Output array to store the result of element-wise greater-than-or-equal comparison
   */
  CudaDims dim = CudaOneDim(out->size);

  // Kernel will execute on 'dim.grid' blocks, each containing 'dim.block' threads.
  EwiseGeKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarGeKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = (a[gid] >= val) ? 1.0f : 0.0f;
}


void ScalarGe(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Compare a CUDA array for greater-than-or-equal with a scalar value element-wise.
   * Args:
   *   a: Input array 'a'
   *   val: Scalar value
   *   out: Output array to store the result of element-wise greater-than-or-equal comparison
   */
  CudaDims dim = CudaOneDim(out->size);

  ScalarGeKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseLogKernel(const scalar_t* a, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = logf(a[gid]);
}

void EwiseLog(const CudaArray& a, CudaArray* out) {
  /**
   * Compute the natural logarithm of each element in a CUDA array.
   * Args:
   *   a: Input array 'a'
   *   out: Output array to store the result of natural logarithm
   */
  CudaDims dim = CudaOneDim(out->size);

  // Kernel will execute on 'dim.grid' blocks, each containing 'dim.block' threads.
  EwiseLogKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}

__global__ void EwiseExpKernel(const scalar_t* a, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = expf(a[gid]);
}

void EwiseExp(const CudaArray& a, CudaArray* out) {
  /**
   * Compute the exponential of each element in a CUDA array.
   * Args:
   *   a: Input array 'a'
   *   out: Output array to store the result of exponential
   */
  CudaDims dim = CudaOneDim(out->size);

  // Kernel will execute on 'dim.grid' blocks, each containing 'dim.block' threads.
  EwiseExpKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}

__global__ void EwiseTanhKernel(const scalar_t* a, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = tanhf(a[gid]);
}

void EwiseTanh(const CudaArray& a, CudaArray* out) {
  /**
   * Compute the hyperbolic tangent of each element in a CUDA array.
   * Args:
   *   a: Input array 'a'
   *   out: Output array to store the result of hyperbolic tangent
   */
  CudaDims dim = CudaOneDim(out->size);

  // Kernel will execute on 'dim.grid' blocks, each containing 'dim.block' threads.
  EwiseTanhKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

__global__ void MatmulSlowKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, uint32_t M,
                             uint32_t N, uint32_t P) {
  
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  for (int i = row; i < M; i += blockDim.y * gridDim.y) {
      for (int j = col; j < P; j += blockDim.x * gridDim.x) {
          scalar_t sum = 0.0f;
          for (int k = 0; k < N; k++) {
              sum += a[i * N + k] * b[k * P + j];
          }
          out[i * P + j] = sum;
      }
  }
}


// __global__ void MatmulFastKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, uint32_t M,
//                              uint32_t N, uint32_t P) {
//   /**
//    * The CUDA kernel for the matrix multiplication opeation.  This should effectively map each
//    * (i,j) entry in the output matrix to the corresponding row i of a and column j of b.
//    * 
//    * Args:
//    *   a: CUDA pointer to a array
//    *   b: CUDA point to b array
//    *   out: CUDA point to out array
//    *   M: rows of a / out
//    *   N: columns of a / rows of b
//    *   P: columns of b / out
//    */
//   size_t row = blockIdx.y * blockDim.y + threadIdx.y;
//   size_t col = blockIdx.x * blockDim.x + threadIdx.x;

//   __shared__ float sA[16][16];
//   __shared__ float sB[16][16];

//   float c[TILE][TILE] = {0};
//   float a_frag[TILE], b_frag[TILE];
//   int yblock = blockIdx.y;
//   int xblock = blockIdx.x;

//   for (int k0 = 0; k0 < 16; k0 += TILE) {
//     __synthreads();
//     //cooperative loading
    
//     __syn
//   }



// void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
//             uint32_t P) {
//   /**
//    * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
//    * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
//    * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
//    * over (i,j) entries in the output array.  However, to really get the full benefit of this
//    * problem, we would encourage you to use cooperative fetching, shared memory register tiling, 
//    * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
//    * the CPU backend, here you should implement a single function that works across all size
//    * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
//    * implementations, this function here will largely just set up the kernel call, and you should
//    * implement the logic in a separate MatmulKernel() call.
//    * 
//    *
//    * Args:
//    *   a: compact 2D array of size m x n
//    *   b: comapct 2D array of size n x p
//    *   out: compact 2D array of size m x p to write the output to
//    *   M: rows of a / out
//    *   N: columns of a / rows of b
//    *   P: columns of b / out
//    */

//   /// BEGIN SOLUTION
//   dim3 block(16, 16);
//   dim3 grid((P + block.x - 1) / block.x, (M + block.y - 1) / block.y);
//   MatmulSlowKernel<<<grid, block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
//   /// END SOLUTION
// }

__global__ void MatmulKernel(const scalar_t* a, const scalar_t* b, scalar_t* c, uint32_t M, uint32_t N,
            uint32_t P){
#define V 2
#define TILE 4
  /**
   * a: M x N
   * b: N x P
   */
  int block_x = blockIdx.x;
  int block_y = blockIdx.y;
  int thread_x = threadIdx.x;
  int thread_y = threadIdx.y;
  int thread_id = thread_x + thread_y * blockDim.x;
  int nthreads = blockDim.x * blockDim.y;

  __shared__ scalar_t a_shared[TILE][TILE];
  __shared__ scalar_t b_shared[TILE][TILE];
  scalar_t c_reg[V][V] = {0};
  scalar_t a_reg[V]={0}, b_reg[V]={0};


  for(int start=0; start<N; start+=TILE){
    __syncthreads();
    
    for (int idx = thread_id; idx < TILE * TILE; idx += nthreads){
      int x = idx / TILE; 
      int y = idx % TILE; 
      
      if(x+block_x*TILE < M && y+start < N){
        a_shared[x][y] = a[(x+block_x*TILE)*N + y+start];
      }
      if(x+start < N && y+block_y*TILE < P){
        b_shared[x][y] = b[(x+start)*P + y+block_y*TILE];
      }
    }
    __syncthreads();

    int stripe_cnt = min(TILE, N-start);
    for(int stripe_i=0; stripe_i<stripe_cnt; stripe_i++){
      if(thread_x * V >= TILE || thread_y * V >= TILE)
        continue;
      for(int reg_x=0; reg_x<V; reg_x++){
        int shared_x = reg_x + thread_x * V;
        if(shared_x >= TILE){
          break;
        }
        a_reg[reg_x] = a_shared[shared_x][stripe_i];
        // b_reg[reg_x] = b_shared[stripe_i][shared_x];
      }
      for(int reg_y=0; reg_y<V; reg_y++){
        int shared_y = reg_y + thread_y * V;
        if(shared_y >= TILE){
          printf("quit: thread id: %d, shared_y: %d, TILE: %d\n", thread_id, shared_y, TILE);
          break;
        }
        // a_reg[reg_y] = a_shared[stripe_i][shared_y];
        b_reg[reg_y] = b_shared[stripe_i][shared_y];
      }
      for(int i=0; i<V; i++){
        for(int j=0; j<V; j++){
          c_reg[i][j] += a_reg[i] * b_reg[j];
        }
      }
    }
  }

  if(thread_x * V >= TILE || thread_y * V >= TILE)
    return;
  for(int i=0; i<V; i++){
    for(int j=0; j<V; j++){
      int x = block_x * TILE + thread_x * V + i;
      int y = block_y * TILE + thread_y * V + j;
      if(x < M && y < P){
        c[x*P + y] = c_reg[i][j];
      } else {
        break;
      }

    }
  }


}

void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
            uint32_t P) {
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
   * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
   * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
   * over (i,j) entries in the output array.  However, to really get the full benefit of this
   * problem, we would encourage you to use cooperative fetching, shared memory register tiling, 
   * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
   * the CPU backend, here you should implement a single function that works across all size
   * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
   * implementations, this function here will largely just set up the kernel call, and you should
   * implement the logic in a separate MatmulKernel() call.
   * 
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: comapct 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   M: rows of a / out
   *   N: columns of a / rows of b
   *   P: columns of b / out
   */

  /// BEGIN SOLUTION
  dim3 grid_dim = dim3((M + TILE - 1) / TILE, (P + TILE - 1) / TILE, 1);
  dim3 block_dim = dim3(16, 16, 1);
  // dim3 block_dim = dim3(2, 2, 1);
  MatmulKernel<<<grid_dim, block_dim>>>(a.ptr, b.ptr, out->ptr, M, N, P);
  /// END SOLUTION
}
////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////

__global__ void ReduceMaxKernel(const scalar_t* a, scalar_t* out, size_t reduce_size, size_t out_size) {
  /**
   * The CUDA kernel for the max reduction opeation.  This should effectively map a single entry 
   * in the output out, to the corresponding maximum value over `reduce_size` contiguous blocks 
   * in the input a.
   * 
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   reduce_size: size of the dimension to reduce over
   *   out_size: size of out array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= out_size) return;

  size_t start_idx = gid * reduce_size;
  scalar_t max_val = a[start_idx];
  for (size_t i = 1; i < reduce_size; i++) {
      scalar_t cur_val = a[start_idx + i];
      if (cur_val > max_val) {
          max_val = cur_val;
      }
  }
  out[gid] = max_val;

}

void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though it is inefficient,
   * for simplicity you can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  ReduceMaxKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, reduce_size, out->size);
  /// END SOLUTION
}


__global__ void ReduceSumKernel(const scalar_t* a, scalar_t* out, size_t reduce_size, size_t out_size) {
  /**
   * The CUDA kernel for the sum reduction opeation.  This should effectively map a single entry 
   * in the output out, to the corresponding summation over `reduce_size` contiguous blocks in 
   * the input a.
   * 
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   reduce_size: size of the dimension to reduce over
   *   out_size: size of out array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= out_size) return;
  /// BEGIN SOLUTION
  size_t start_idx = gid * reduce_size;
  scalar_t sum_val = 0.0f;
  for (size_t i = 0; i < reduce_size; i++) {
      sum_val += a[start_idx + i];
  }
  out[gid] = sum_val;
  /// END SOLUTION
}
  
void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking summation over `reduce_size` contiguous blocks.  Again, for simplicity you 
   * can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  ReduceSumKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, reduce_size, out->size);
  /// END SOLUTION
}

__global__ void GraphNeighborSumKernel(const scalar_t* node_feat, const scalar_t* src_index,
                                       const scalar_t* dst_index, scalar_t* out,
                                       size_t num_edges, size_t feature_dim,
                                       size_t num_nodes) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= num_edges) return;
  int src = static_cast<int>(roundf(src_index[gid]));
  int dst = static_cast<int>(roundf(dst_index[gid]));
  if (src < 0 || dst < 0) return;
  if (static_cast<size_t>(src) >= num_nodes || static_cast<size_t>(dst) >= num_nodes) return;
  size_t src_offset = static_cast<size_t>(src) * feature_dim;
  size_t dst_offset = static_cast<size_t>(dst) * feature_dim;
  for (size_t f = 0; f < feature_dim; f++) {
    atomicAdd(out + dst_offset + f, node_feat[src_offset + f]);
  }
}

void GraphNeighborSum(const CudaArray& node_feat, const CudaArray& src_index,
                      const CudaArray& dst_index, CudaArray* out, size_t num_nodes,
                      size_t feature_dim, size_t num_edges) {
  Fill(out, 0);
  CudaDims dim = CudaOneDim(num_edges);
  GraphNeighborSumKernel<<<dim.grid, dim.block>>>(node_feat.ptr, src_index.ptr, dst_index.ptr,
                                                  out->ptr, num_edges, feature_dim, num_nodes);
}

__global__ void SpMMKernel(const scalar_t* a_rows, const scalar_t* a_cols, const scalar_t* a_vals,
                           const scalar_t* dense, scalar_t* out, size_t k, size_t nnz,
                           size_t n, size_t m) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= nnz) return;
  int r = static_cast<int>(roundf(a_rows[gid]));
  int c = static_cast<int>(roundf(a_cols[gid]));
  if (r < 0 || c < 0) return;
  if (static_cast<size_t>(c) >= n || static_cast<size_t>(r) >= m) return;
  scalar_t val = a_vals[gid];
  size_t dense_offset = static_cast<size_t>(c) * k;
  size_t out_offset = static_cast<size_t>(r) * k;
  for (size_t j = 0; j < k; j++) {
    atomicAdd(out + out_offset + j, val * dense[dense_offset + j]);
  }
}

#ifdef USE_CUSPARSE_SPMM
__global__ void CastFloatToIntKernel(const scalar_t* in, int* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = static_cast<int>(roundf(in[gid]));
}

void CastFloatToInt(const CudaArray& in, int* out) {
  CudaDims dim = CudaOneDim(in.size);
  CastFloatToIntKernel<<<dim.grid, dim.block>>>(in.ptr, out, in.size);
}

void SpMMCusparse(const CudaArray& a_rows, const CudaArray& a_cols, const CudaArray& a_vals,
                  const CudaArray& dense, CudaArray* out, size_t m, size_t n, size_t k,
                  size_t nnz) {
  // Allocate temporary int32 indices.
  int* d_rows = nullptr;
  int* d_cols = nullptr;
  cudaMalloc(&d_rows, nnz * sizeof(int));
  cudaMalloc(&d_cols, nnz * sizeof(int));
  CastFloatToInt(a_rows, d_rows);
  CastFloatToInt(a_cols, d_cols);

  cusparseHandle_t handle;
  cusparseCreate(&handle);

  cusparseSpMatDescr_t matA;
  cusparseDnMatDescr_t matB, matC;
  // Use the legacy signature (single index type); this is compatible with the headers shipped in this environment.
  cusparseCreateCoo(&matA, m, n, nnz, d_rows, d_cols, a_vals.ptr,
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
  // B: n x k, row-major
  cusparseCreateDnMat(&matB, n, k, k, const_cast<scalar_t*>(dense.ptr), CUDA_R_32F, CUSPARSE_ORDER_ROW);
  // C: m x k, row-major
  cusparseCreateDnMat(&matC, m, k, k, out->ptr, CUDA_R_32F, CUSPARSE_ORDER_ROW);

  float alpha = 1.0f, beta = 0.0f;
  size_t bufferSize = 0;
  cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                          &alpha, matA, matB, &beta, matC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
                          &bufferSize);
  void* dBuffer = nullptr;
  cudaMalloc(&dBuffer, bufferSize);
  cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
               &alpha, matA, matB, &beta, matC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer);

  cudaFree(dBuffer);
  cusparseDestroySpMat(matA);
  cusparseDestroyDnMat(matB);
  cusparseDestroyDnMat(matC);
  cusparseDestroy(handle);
  cudaFree(d_rows);
  cudaFree(d_cols);
}

void SpMMTensorCore(const CudaArray& a_rows, const CudaArray& a_cols, const CudaArray& a_vals,
                    const CudaArray& dense, CudaArray* out, size_t m, size_t n, size_t k,
                    size_t nnz) {
  // Allocate temporary int32 indices.
  int* d_rows = nullptr;
  int* d_cols = nullptr;
  cudaMalloc(&d_rows, nnz * sizeof(int));
  cudaMalloc(&d_cols, nnz * sizeof(int));
  CastFloatToInt(a_rows, d_rows);
  CastFloatToInt(a_cols, d_cols);

  cusparseHandle_t handle;
  cusparseCreate(&handle);
#ifdef CUSPARSE_MATH_TENSOR_OP
  cusparseSetMathMode(handle, CUSPARSE_MATH_TENSOR_OP);
#endif

  cusparseSpMatDescr_t matA;
  cusparseDnMatDescr_t matB, matC;
  cusparseCreateCoo(&matA, m, n, nnz, d_rows, d_cols, a_vals.ptr,
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
  // B: n x k, row-major
  cusparseCreateDnMat(&matB, n, k, k, const_cast<scalar_t*>(dense.ptr), CUDA_R_32F, CUSPARSE_ORDER_ROW);
  // C: m x k, row-major
  cusparseCreateDnMat(&matC, m, k, k, out->ptr, CUDA_R_32F, CUSPARSE_ORDER_ROW);

  float alpha = 1.0f, beta = 0.0f;
  size_t bufferSize = 0;
  cusparseSpMMAlg_t alg = CUSPARSE_SPMM_ALG_DEFAULT;
#ifdef CUSPARSE_SPMM_ALG_DEFAULT_TENSOR_OP
  alg = CUSPARSE_SPMM_ALG_DEFAULT_TENSOR_OP;
#endif
  cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                          &alpha, matA, matB, &beta, matC, CUDA_R_32F, alg,
                          &bufferSize);
  void* dBuffer = nullptr;
  cudaMalloc(&dBuffer, bufferSize);
  cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
               &alpha, matA, matB, &beta, matC, CUDA_R_32F, alg, dBuffer);

  cudaFree(dBuffer);
  cusparseDestroySpMat(matA);
  cusparseDestroyDnMat(matB);
  cusparseDestroyDnMat(matC);
  cusparseDestroy(handle);
  cudaFree(d_rows);
  cudaFree(d_cols);
}
#endif

void SpMM(const CudaArray& a_rows, const CudaArray& a_cols, const CudaArray& a_vals,
          const CudaArray& dense, CudaArray* out, size_t m, size_t n, size_t k, size_t nnz) {
  cudaMemset(out->ptr, 0, out->size * ELEM_SIZE);
  // std::fprintf(stderr, "SpMM m=%zu n=%zu k=%zu nnz=%zu\n", m, n, k, nnz);
#if defined(USE_CUSPARSE_SPMM)
  // std::fprintf(stderr, "g_use_cusparse_spmm=%d, g_use_tensor_core_spmm=%d\n",
  //             g_use_cusparse_spmm, g_use_tensor_core_spmm);
  g_use_cusparse_spmm = true;  // Force enable for testing
  g_use_tensor_core_spmm = false;  // Force disable for testing
  if (g_use_cusparse_spmm) {
    // Emit a runtime hint that cuSPARSE path is being used.
    // std::fprintf(stderr, "[cuSPARSE] SpMM m=%zu n=%zu k=%zu nnz=%zu\n", m, n, k, nnz);
    if (g_use_tensor_core_spmm && TensorCoresAvailable()) {
      // std::fprintf(stderr, "[cuSPARSE Tensor Core!!] SpMM m=%zu n=%zu k=%zu nnz=%zu\n", m, n, k, nnz);
      SpMMTensorCore(a_rows, a_cols, a_vals, dense, out, m, n, k, nnz);
      return;
    }
    SpMMCusparse(a_rows, a_cols, a_vals, dense, out, m, n, k, nnz);
    return;
  }
#endif
  CudaDims dim = CudaOneDim(nnz);
  SpMMKernel<<<dim.grid, dim.block>>>(a_rows.ptr, a_cols.ptr, a_vals.ptr, dense.ptr, out->ptr, k,
                                      nnz, n, m);
}

std::tuple<pybind11::array_t<scalar_t>, pybind11::array_t<scalar_t>,
           pybind11::array_t<scalar_t>>
SparseEwiseMul(const CudaArray& a_rows, const CudaArray& a_cols, const CudaArray& a_vals,
               const CudaArray& b_rows, const CudaArray& b_cols, const CudaArray& b_vals) {
  // Host-side intersection for simplicity/correctness.
  std::vector<scalar_t> h_a_rows(a_rows.size), h_a_cols(a_cols.size), h_a_vals(a_vals.size);
  std::vector<scalar_t> h_b_rows(b_rows.size), h_b_cols(b_cols.size), h_b_vals(b_vals.size);
  cudaMemcpy(h_a_rows.data(), a_rows.ptr, a_rows.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_a_cols.data(), a_cols.ptr, a_cols.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_a_vals.data(), a_vals.ptr, a_vals.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_b_rows.data(), b_rows.ptr, b_rows.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_b_cols.data(), b_cols.ptr, b_cols.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_b_vals.data(), b_vals.ptr, b_vals.size * ELEM_SIZE, cudaMemcpyDeviceToHost);

  std::unordered_map<int64_t, scalar_t> amap;
  amap.reserve(h_a_vals.size());
  for (size_t i = 0; i < h_a_vals.size(); i++) {
    int64_t r = static_cast<int64_t>(roundf(h_a_rows[i]));
    int64_t c = static_cast<int64_t>(roundf(h_a_cols[i]));
    int64_t key = (r << 32) ^ c;
    amap[key] = h_a_vals[i];
  }

  std::vector<scalar_t> out_rows;
  std::vector<scalar_t> out_cols;
  std::vector<scalar_t> out_vals;
  out_rows.reserve(std::min(h_a_vals.size(), h_b_vals.size()));
  out_cols.reserve(std::min(h_a_vals.size(), h_b_vals.size()));
  out_vals.reserve(std::min(h_a_vals.size(), h_b_vals.size()));

  for (size_t i = 0; i < h_b_vals.size(); i++) {
    int64_t r = static_cast<int64_t>(roundf(h_b_rows[i]));
    int64_t c = static_cast<int64_t>(roundf(h_b_cols[i]));
    int64_t key = (r << 32) ^ c;
    auto it = amap.find(key);
    if (it != amap.end()) {
      out_rows.push_back(static_cast<scalar_t>(r));
      out_cols.push_back(static_cast<scalar_t>(c));
      out_vals.push_back(it->second * h_b_vals[i]);
    }
  }

  return {pybind11::array_t<scalar_t>(out_rows.size(), out_rows.data()),
          pybind11::array_t<scalar_t>(out_cols.size(), out_cols.data()),
          pybind11::array_t<scalar_t>(out_vals.size(), out_vals.data())};
}

}  // namespace cuda
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
  });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray* out) {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
  m.def("graph_neighbor_sum", GraphNeighborSum);
  m.def("spmm", SpMM);
  m.def("sparse_ewise_mul", SparseEwiseMul);

#ifdef USE_CUSPARSE_SPMM
  m.attr("__cusparse_enabled__") = true;
  m.def("set_use_cusparse_spmm", [](bool use) { g_use_cusparse_spmm = use; });
  m.def("use_cusparse_spmm", []() { return g_use_cusparse_spmm; });
  m.attr("__tensor_core_spmm_available__") = TensorCoresAvailable();
  m.def("set_use_tensor_core_spmm", [](bool use) { g_use_tensor_core_spmm = use; });
  m.def("use_tensor_core_spmm",
        []() { return g_use_tensor_core_spmm && TensorCoresAvailable(); });
#else
  m.attr("__cusparse_enabled__") = false;
  m.def("set_use_cusparse_spmm", [](bool) {});
  m.def("use_cusparse_spmm", []() { return false; });
  m.attr("__tensor_core_spmm_available__") = false;
  m.def("set_use_tensor_core_spmm", [](bool) {});
  m.def("use_tensor_core_spmm", []() { return false; });
#endif
}
