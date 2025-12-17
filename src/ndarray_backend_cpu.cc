#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <unordered_map>

namespace needle {
namespace cpu {

#define ALIGNMENT 256
#define TILE 8
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);


/**
 * This is a utility structure for maintaining an array aligned to ALIGNMENT boundaries in
 * memory.  This alignment should be at least TILE * ELEM_SIZE, though we make it even larger
 * here by default.
 */
struct AlignedArray {
  AlignedArray(const size_t size) {
    int ret = posix_memalign((void**)&ptr, ALIGNMENT, size * ELEM_SIZE);
    if (ret != 0) throw std::bad_alloc();
    this->size = size;
  }
  ~AlignedArray() { free(ptr); }
  size_t ptr_as_int() {return (size_t)ptr; }
  scalar_t* ptr;
  size_t size;
};



void Fill(AlignedArray* out, scalar_t val) {
  /**
   * Fill the values of an aligned array with val
   */
  for (int i = 0; i < out->size; i++) {
    out->ptr[i] = val;
  }
}



void Compact(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory
   *
   * Args:
   *   a: non-compact representation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   *
   * Returns:
   *  void (you need to modify out directly, rather than returning anything; this is true for all the
   *  function will implement here, so we won't repeat this note.)
   */
  /// BEGIN SOLUTION
  int num_dims = shape.size();
  std::vector<int32_t> idx(num_dims, 0);
  size_t out_idx = 0;
  bool compacting_flag = true;
  // printf("The size of size_t is %lu\n", sizeof(size_t));
  while (compacting_flag) {

    size_t a_curridx = offset;

    for (int i = 0; i < num_dims; i++) {
      a_curridx += idx[i] * strides[i];
    }
    out->ptr[out_idx++] = a.ptr[a_curridx];


    for (int i = num_dims - 1; i >= 0; i--) {
      idx[i]++;
      
      if (i == 0 && idx[i] == shape[i]) {
        compacting_flag = false;
        break;
      }
      if (idx[i] >= shape[i]) { // the index is smaller than the actual size 
        // idx[i]--;
        idx[i] = 0;
      } else {
        break;
      }
      
    }

  }
  /// END SOLUTION
}

void EwiseSetitem(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array
   *
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN SOLUTION
  int num_dims = shape.size();
  std::vector<int32_t> idx(num_dims, 0);
  bool isSetting = true;
  size_t a_idx = 0;
  while (isSetting) {
    size_t start = offset;

    for (int i = 0; i < num_dims; i++) {
      start += idx[i] * strides[i];
    }

    out->ptr[start] = a.ptr[a_idx++];

    for (int i = num_dims - 1; i >= 0; i--) {
      idx[i]++;
      if (idx[i] < shape[i]) break; // no overflow
      if (i == 0)  {
        // overflow occurs and it is the last dim
        isSetting = false;
        break;
      }
      idx[i] = 0; // reset this idm if overflow occurs at the non-last dim
    }

  }
  /// END SOLUTION
}

void ScalarSetitem(const size_t size, scalar_t val, AlignedArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   *
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the
   *         product of items in shape, but convenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */

  /// BEGIN SOLUTION
  int num_dims = shape.size();
  std::vector<int32_t> idx(num_dims, 0);
  bool isSetting = true;
  // int32_t a_idx = 0;
  while (isSetting) {
    size_t start = offset;

    for (int i = 0; i < num_dims; i++) {
      start += idx[i] * strides[i];
    }

    out->ptr[start] = val;

    for (int i = num_dims - 1; i >= 0; i--) {
      idx[i]++;
      if (idx[i] < shape[i]) break; // no overflow
      if (i == 0)  {
        // overflow occurs and it is the last dim
        isSetting = false;
        break;
      }
      idx[i] = 0; // reset this idm if overflow occurs at the non-last dim
    }

  }
                    


  /// END SOLUTION
}

void EwiseAdd(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of correspondings entires in a and b.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + b.ptr[i];
  }
}

void ScalarAdd(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of corresponding entry in a plus the scalar val.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + val;
  }
}


/**
 * In the code the follows, use the above template to create analogous element-wise
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

template <typename Func>
void EwiseBinaryOp(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, Func f) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = f(a.ptr[i], b.ptr[i]);
  }
}

template <typename Func>
void ScalarBinaryOp(const AlignedArray& a, scalar_t val, AlignedArray* out, Func f) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = f(a.ptr[i], val);
  }
}

void EwiseMul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  /**
   * Element-wise multiplication of a and b into out
   */
  /// BEGIN SOLUTION
  assert(a.size == b.size); // not strictly necessary, but good to check
  EwiseBinaryOp(a, b, out, [](scalar_t x, scalar_t y) { return x * y; });
  /// END SOLUTION
}

void ScalarMul(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  /**
   * Scalar multiplication of a and val into out
   */
  /// BEGIN SOLUTION
  ScalarBinaryOp(a, val, out, [](scalar_t x, scalar_t y) { return x * y; });
  /// END SOLUTION
}

void EwiseDiv(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  /**
   * Element-wise division of a and b into out
   */
  /// BEGIN SOLUTION
  EwiseBinaryOp(a, b, out, [](scalar_t x, scalar_t y) { return x / y; });
  /// END SOLUTION
}

void ScalarDiv(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  /**
   * Scalar division of a and val into out
   */
  /// BEGIN SOLUTION
  ScalarBinaryOp(a, val, out, [](scalar_t x, scalar_t y) { return x / y; });
  /// END SOLUTION
}

void ScalarPower(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  /**
   * Raise each element of a to the given power val, writing into out
   */
  /// BEGIN SOLUTION
  EwiseBinaryOp(a, a, out, [val](scalar_t x, scalar_t) { return std::pow(x, val); });
  /// END SOLUTION
}

void EwiseMaximum(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  /**
   * Element-wise maximum of a and b into out
   */
  /// BEGIN SOLUTION
  EwiseBinaryOp(a, b, out, [](scalar_t x, scalar_t y) { return x > y ? x : y; });
  /// END SOLUTION
}

void ScalarMaximum(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  /**
   * Scalar maximum of a and val into out
   */
  /// BEGIN SOLUTION
  ScalarBinaryOp(a, val, out, [](scalar_t x, scalar_t y) { return x > y ? x : y; });
  /// END SOLUTION
}

void EwiseEq(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  /**
   * Element-wise equality of a and b into out
   */
  /// BEGIN SOLUTION
  EwiseBinaryOp(a, b, out, [](scalar_t x, scalar_t y) { return x == y ? 1.0f : 0.0f; });
  /// END SOLUTION
}

void ScalarEq(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  /**
   * Scalar equality of a and val into out
   */
  /// BEGIN SOLUTION
  ScalarBinaryOp(a, val, out, [](scalar_t x, scalar_t y) { return x == y ? 1.0f : 0.0f; });
  /// END SOLUTION
}

void EwiseGe(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  /**
   * Element-wise greater-than-equal of a and b into out
   */
  /// BEGIN SOLUTION
  EwiseBinaryOp(a, b, out, [](scalar_t x, scalar_t y) { return x >= y ? 1.0f : 0.0f; });
  /// END SOLUTION
}

void ScalarGe(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  /**
   * Scalar greater-than-equal of a and val into out
   */
  /// BEGIN SOLUTION
  ScalarBinaryOp(a, val, out, [](scalar_t x, scalar_t y) { return x >= y ? 1.0f : 0.0f; });
  /// END SOLUTION
}

void EwiseLog(const AlignedArray& a, AlignedArray* out) {
  /**
   * Element-wise natural logarithm of a into out
   */
  /// BEGIN SOLUTION
  EwiseBinaryOp(a, a, out, [](scalar_t x, scalar_t) { return std::log(x); });
  /// END SOLUTION
}

void EwiseExp(const AlignedArray& a, AlignedArray* out) {
  /**
   * Element-wise exponential of a into out
   */
  /// BEGIN SOLUTION
  EwiseBinaryOp(a, a, out, [](scalar_t x, scalar_t) { return std::exp(x); });
  /// END SOLUTION
}

void EwiseTanh(const AlignedArray& a, AlignedArray* out) {
  /**
   * Element-wise hyperbolic tangent of a into out
   */
  /// BEGIN SOLUTION
  EwiseBinaryOp(a, a, out, [](scalar_t x, scalar_t) { return std::tanh(x); });
  /// END SOLUTION
}


void Matmul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m, uint32_t n,
            uint32_t p) {
  /**
   * Multiply two (compact) matrices into an output (also compact) matrix.  For this implementation
   * you can use the "naive" three-loop algorithm.
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: compact 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   */

  /// BEGIN SOLUTION
  for (size_t i = 0; i < m; i++) {
    for (size_t k = 0; k < p; k++) {

      scalar_t s = 0;
      for (size_t j = 0; j < n; j++) {
        s += a.ptr[i * n + j] * b.ptr[j * p + k];
      }
      out->ptr[i * p + k] = s;
    }
  }
  /// END SOLUTION
}

inline void AlignedDot(const float* __restrict__ a,
                       const float* __restrict__ b,
                       float* __restrict__ out) {

  /**
   * Multiply together two TILE x TILE matrices, and _add _the result to out (it is important to add
   * the result to the existing out, which you should not set to zero beforehand).  We are including
   * the compiler flags here that enable the compile to properly use vector operators to implement
   * this function.  Specifically, the __restrict__ keyword indicates to the compile that a, b, and
   * out don't have any overlapping memory (which is necessary in order for vector operations to be
   * equivalent to their non-vectorized counterparts (imagine what could happen otherwise if a, b,
   * and out had overlapping memory).  Similarly the __builtin_assume_aligned keyword tells the
   * compiler that the input array will be aligned to the appropriate blocks in memory, which also
   * helps the compiler vectorize the code.
   *
   * Args:
   *   a: compact 2D array of size TILE x TILE
   *   b: compact 2D array of size TILE x TILE
   *   out: compact 2D array of size TILE x TILE to write to
   */

  a = (const float*)__builtin_assume_aligned(a, TILE * ELEM_SIZE);
  b = (const float*)__builtin_assume_aligned(b, TILE * ELEM_SIZE);
  out = (float*)__builtin_assume_aligned(out, TILE * ELEM_SIZE);

  /// BEGIN SOLUTION
  // for (size_t i = 0; i < TILE; i++) {
  //   for (size_t k = 0; k < TILE; k++) {

  //     float s = 0;
  //     for (size_t j = 0; j < TILE; j++) {
  //       s += a[i * TILE + j] * b[j * TILE + k];
  //     }
  //     out[i * TILE + k] = s;
  //   }
  // }
  for (size_t i = 0; i < TILE; ++i) {
    for (size_t k = 0; k < TILE; ++k) {
      float s = 0.0f;
      for (size_t j = 0; j < TILE; ++j) {
        s += a[i * TILE + j] * b[j * TILE + k];
      }
      out[i * TILE + k] += s;
    }
  }
  /// END SOLUTION
}

void MatmulTiled(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m,
                 uint32_t n, uint32_t p) {
  /**
   * Matrix multiplication on tiled representations of array.  In this setting, a, b, and out
   * are all *4D* compact arrays of the appropriate size, e.g. a is an array of size
   *   a[m/TILE][n/TILE][TILE][TILE]
   * You should do the multiplication tile-by-tile to improve performance of the array (i.e., this
   * function should call `AlignedDot()` implemented above).
   *
   * Note that this function will only be called when m, n, p are all multiples of TILE, so you can
   * assume that this division happens without any remainder.
   *
   * Args:
   *   a: compact 4D array of size m/TILE x n/TILE x TILE x TILE
   *   b: compact 4D array of size n/TILE x p/TILE x TILE x TILE
   *   out: compact 4D array of size m/TILE x p/TILE x TILE x TILE to write to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   *
   */
  /// BEGIN SOLUTION
  // for (size_t i = 0; i < m / TILE; i++) {
  //   for (size_t j = 0; j < p / TILE; i++) {
  //     // float s = 0;
  //     float* out_tile = &out->ptr[(size_t)(i * p / TILE + j) * TILE * TILE];
  //     for (size_t k = 0; k < n / TILE; k++) {
  //       // float tmp = 0;
  //       const float* a_tile = &a.ptr[(size_t)(i * n / TILE + k) * TILE * TILE];
  //       const float* b_tile = &b.ptr[(size_t)(k * p / TILE + j) * TILE * TILE];
  //       AlignedDot(a_tile, b_tile, out_tile);
  //     }

  //     // out->ptr[i * n / TILE + j] = s;
  //   }
  // }
  const uint32_t Mt = m / TILE;
  const uint32_t Nt = n / TILE;
  const uint32_t Pt = p / TILE;

  std::fill(out->ptr, out->ptr + (size_t)Mt * Pt * TILE * TILE, 0.0f);

  for (uint32_t i = 0; i < Mt; ++i) {
    for (uint32_t j = 0; j < Pt; ++j) { 
      float* out_tile = &out->ptr[(size_t)(i * Pt + j) * TILE * TILE];
      for (uint32_t k = 0; k < Nt; ++k) {
        const float* a_tile = &a.ptr[(size_t)(i * Nt + k) * TILE * TILE];
        const float* b_tile = &b.ptr[(size_t)(k * Pt + j) * TILE * TILE];
        AlignedDot(a_tile, b_tile, out_tile);
      }
    }
  }
  /// END SOLUTION
}

void ReduceMax(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */

  /// BEGIN SOLUTION
  // printf("The tested reduce_size is %lu\n", reduce_size);
  size_t num_blocks = a.size / reduce_size; 

  for (size_t i = 0; i < num_blocks; i++) {
    scalar_t curr_max = a.ptr[i * reduce_size];
    for (size_t j = 0; j < reduce_size; j++) {
      curr_max = (curr_max > a.ptr[i * reduce_size + j]) ? curr_max : a.ptr[i * reduce_size + j];
    }

    out->ptr[i] = curr_max;

  }

  /// END SOLUTION
}

void ReduceSum(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Reduce by taking sum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */

  /// BEGIN SOLUTION
  size_t num_blocks = a.size / reduce_size; 
  // scalar_t s[num_blocks] = {0};

  for (size_t i = 0; i < num_blocks; i++) {
    // scalar_t curr_sum = a.ptr[i * reduce_size];
    scalar_t s = 0;
    for (size_t j = 0; j < reduce_size; j++) {
      // curr_sum += a.ptr[i * reduce_size + j];
      s += a.ptr[i * reduce_size + j];
    }

    out->ptr[i] = s;

  }
  /// END SOLUTION
}

void GraphNeighborSum(const AlignedArray& node_feat, const AlignedArray& src_index,
                      const AlignedArray& dst_index, AlignedArray* out, size_t num_nodes,
                      size_t feature_dim, size_t num_edges) {
  Fill(out, 0);
  for (size_t e = 0; e < num_edges; e++) {
    int src = static_cast<int>(std::round(src_index.ptr[e]));
    int dst = static_cast<int>(std::round(dst_index.ptr[e]));
    if (src < 0 || dst < 0) continue;
    if (static_cast<size_t>(src) >= num_nodes || static_cast<size_t>(dst) >= num_nodes) continue;
    const scalar_t* src_ptr = node_feat.ptr + static_cast<size_t>(src) * feature_dim;
    scalar_t* dst_ptr = out->ptr + static_cast<size_t>(dst) * feature_dim;
    for (size_t f = 0; f < feature_dim; f++) {
      dst_ptr[f] += src_ptr[f];
    }
  }
}

void SpMM(const AlignedArray& a_rows, const AlignedArray& a_cols, const AlignedArray& a_vals,
          const AlignedArray& dense, AlignedArray* out, size_t m, size_t n, size_t k,
          size_t nnz) {
  Fill(out, 0);
  for (size_t idx = 0; idx < nnz; idx++) {
    int r = static_cast<int>(std::round(a_rows.ptr[idx]));
    int c = static_cast<int>(std::round(a_cols.ptr[idx]));
    if (r < 0 || c < 0) continue;
    if (static_cast<size_t>(c) >= n || static_cast<size_t>(r) >= m) continue;
    scalar_t val = a_vals.ptr[idx];
    size_t dense_offset = static_cast<size_t>(c) * k;
    size_t out_offset = static_cast<size_t>(r) * k;
    for (size_t j = 0; j < k; j++) {
      out->ptr[out_offset + j] += val * dense.ptr[dense_offset + j];
    }
  }
}

std::tuple<pybind11::array_t<scalar_t>, pybind11::array_t<scalar_t>,
           pybind11::array_t<scalar_t>>
SparseEwiseMul(const AlignedArray& a_rows, const AlignedArray& a_cols, const AlignedArray& a_vals,
               const AlignedArray& b_rows, const AlignedArray& b_cols, const AlignedArray& b_vals) {
  // Build map from coordinates to value for A.
  std::unordered_map<int64_t, scalar_t> amap;
  amap.reserve(a_vals.size);
  for (size_t i = 0; i < a_vals.size; i++) {
    int64_t r = static_cast<int64_t>(std::round(a_rows.ptr[i]));
    int64_t c = static_cast<int64_t>(std::round(a_cols.ptr[i]));
    int64_t key = (r << 32) ^ c;
    amap[key] = a_vals.ptr[i];
  }

  std::vector<scalar_t> out_rows;
  std::vector<scalar_t> out_cols;
  std::vector<scalar_t> out_vals;
  out_rows.reserve(std::min(a_vals.size, b_vals.size));
  out_cols.reserve(std::min(a_vals.size, b_vals.size));
  out_vals.reserve(std::min(a_vals.size, b_vals.size));

  for (size_t i = 0; i < b_vals.size; i++) {
    int64_t r = static_cast<int64_t>(std::round(b_rows.ptr[i]));
    int64_t c = static_cast<int64_t>(std::round(b_cols.ptr[i]));
    int64_t key = (r << 32) ^ c;
    auto it = amap.find(key);
    if (it != amap.end()) {
      out_rows.push_back(static_cast<scalar_t>(r));
      out_cols.push_back(static_cast<scalar_t>(c));
      out_vals.push_back(it->second * b_vals.ptr[i]);
    }
  }

  return {pybind11::array_t<scalar_t>(out_rows.size(), out_rows.data()),
          pybind11::array_t<scalar_t>(out_cols.size(), out_cols.data()),
          pybind11::array_t<scalar_t>(out_vals.size(), out_vals.data())};
}

}  // namespace cpu
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cpu, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cpu;

  m.attr("__device_name__") = "cpu";
  m.attr("__tile_size__") = TILE;

  py::class_<AlignedArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def("ptr", &AlignedArray::ptr_as_int)
      .def_readonly("size", &AlignedArray::size);

  // return numpy array (with copying for simplicity, otherwise garbage
  // collection is a pain)
  m.def("to_numpy", [](const AlignedArray& a, std::vector<size_t> shape,
                       std::vector<size_t> strides, size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });
    return py::array_t<scalar_t>(shape, numpy_strides, a.ptr + offset);
  });

  // convert from numpy (with copying)
  m.def("from_numpy", [](py::array_t<scalar_t> a, AlignedArray* out) {
    std::memcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE);
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
  m.def("matmul_tiled", MatmulTiled);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
  m.def("graph_neighbor_sum", GraphNeighborSum);
  m.def("spmm", SpMM);
  m.def("sparse_ewise_mul", SparseEwiseMul);
}
