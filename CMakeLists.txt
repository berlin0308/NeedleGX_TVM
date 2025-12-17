cmake_minimum_required(VERSION 3.5)
project(needle C CXX)
if(POLICY CMP0146)
  cmake_policy(SET CMP0146 OLD)
endif()


# find correct version of Python
# Use 'python' command first (which respects conda environment)
# This ensures we use the Python from the current conda environment
execute_process(COMMAND python -c "import sys; print(sys.executable)"
  OUTPUT_VARIABLE PYTHON_EXECUTABLE
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET)
if(PYTHON_EXECUTABLE)
  # Use the Python executable we found
  set(Python3_EXECUTABLE ${PYTHON_EXECUTABLE})
  message(STATUS "Using Python: ${PYTHON_EXECUTABLE}")
else()
  # Fallback to python3
  find_program(PYTHON3_EXECUTABLE python3)
  if(PYTHON3_EXECUTABLE)
    set(Python3_EXECUTABLE ${PYTHON3_EXECUTABLE})
  endif()
endif()

# Find Python using the executable we determined
find_package(Python3 COMPONENTS Development Interpreter REQUIRED)
if(Python3_EXECUTABLE)
  set(Python_EXECUTABLE ${Python3_EXECUTABLE})
  set(Python_INCLUDE_DIRS ${Python3_INCLUDE_DIRS})
  set(Python_LIBRARIES ${Python3_LIBRARIES})
endif()
include_directories(${Python_INCLUDE_DIRS})

# find pybind
# Use 'python' command first (respects conda environment)
execute_process(COMMAND python -m pybind11 --cmakedir
  RESULT_VARIABLE __pybind_exit_code
  OUTPUT_VARIABLE __pybind_path
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET)
# Fallback to python3 if python fails
if(__pybind_exit_code OR NOT __pybind_path)
  execute_process(COMMAND python3 -m pybind11 --cmakedir
    RESULT_VARIABLE __pybind_exit_code
    OUTPUT_VARIABLE __pybind_path
    OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()
find_package(pybind11 PATHS ${__pybind_path})


if(NOT MSVC)
  set(CMAKE_CXX_FLAGS "-std=c++11 -O2 -march=native ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CUDA_STANDARD 14)
else()
  set(CMAKE_CXX_FLAGS "/std:c++11 -O2 -march=native ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CUDA_STANDARD 14)
endif()

include_directories(SYSTEM ${pybind11_INCLUDE_DIRS})
list(APPEND LINKER_LIBS ${pybind11_LIBRARIES})


###################
### CPU BACKEND ###
###################
add_library(ndarray_backend_cpu MODULE src/ndarray_backend_cpu.cc)
target_link_libraries(ndarray_backend_cpu PUBLIC ${LINKER_LIBS})
pybind11_extension(ndarray_backend_cpu)
pybind11_strip(ndarray_backend_cpu)


# directly output to ffi folder
set_target_properties(ndarray_backend_cpu
  PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/needle/backend_ndarray
  CXX_VISIBILITY_PRESET "hidden"
)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set_property(TARGET ndarray_backend_cpu PROPERTY LINK_OPTIONS -undefined dynamic_lookup)
endif()



####################
### CUDA BACKEND ###
####################
find_package(CUDA)
if(CUDA_FOUND)
  message(STATUS "Found cuda, building cuda backend")

  include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
  list(APPEND LINKER_LIBS ${CUDA_CUDART_LIBRARY})
  # Try to link cusparse for optional SpMM acceleration.
  find_library(CUDA_CUSPARSE_LIBRARY
    NAMES cusparse
    HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib
          /usr/local/cuda/lib64 /usr/local/cuda-*/lib64 /usr/lib/x86_64-linux-gnu
  )
  if(CUDA_CUSPARSE_LIBRARY)
    list(APPEND LINKER_LIBS ${CUDA_CUSPARSE_LIBRARY})
    set(USE_CUSPARSE_SPMM 1)
    message(STATUS "cuSPARSE found: ${CUDA_CUSPARSE_LIBRARY}")
  else()
    message(STATUS "cuSPARSE not found; building without cuSPARSE SpMM")
  endif()

  # invoke nvidia smi to detect if we really have a GPU
  execute_process(COMMAND "nvidia-smi" ERROR_QUIET  RESULT_VARIABLE NV_RET)
  if(NV_RET EQUAL "0")
    CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS Auto)
  else()
    # set to 3.7 the flag of K80
    CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS 3.7)
  endif()

  # set arch flags properly
  CUDA_ADD_LIBRARY(ndarray_backend_cuda MODULE src/ndarray_backend_cuda.cu OPTIONS ${ARCH_FLAGS})

  target_link_libraries(ndarray_backend_cuda ${LINKER_LIBS})
  if(USE_CUSPARSE_SPMM)
    target_compile_definitions(ndarray_backend_cuda PRIVATE USE_CUSPARSE_SPMM)
  endif()
  pybind11_extension(ndarray_backend_cuda)
  pybind11_strip(ndarray_backend_cuda)

  # directly output to ffi folder
  set_target_properties(ndarray_backend_cuda
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/needle/backend_ndarray
    CXX_VISIBILITY_PRESET "hidden"
    CUDA_VISIBILITY_PRESET "hidden"
)

endif()
