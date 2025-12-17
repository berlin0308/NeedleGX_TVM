# NeedleGX_TVM

NeedleGX is a Needle-based GNN inference acceleration extension. It provides both a **Needle Native Backend** and a **TVM Backend**, enabling backend switching and performance comparisons across CPU / CUDA Cores / Tensor Cores.

> For the full walkthrough and plots, see `needle_gx.ipynb`.

## Quick Start

### Build

```bash
make clean
make
```

### Run the Notebook

Use `needle_gx.ipynb` for:
- SpMM correctness (CPU / CUDA / cuSPARSE)
- End-to-end GCN inference correctness
- Performance comparison: Needle vs TVM (LLVM / AVX2 / AVX512 / CUDA / TensorCore)
- TVM auto-scheduling demo (`model.tune()`)

## Project Layout

- **`apps/`**: high-level examples and model interface (unified GCN interface + benchmark helpers)
- **`apps/gnn_tvm_utils/`**: TVM utilities for SpMM / TensorCore (MMA) strategies and layout/format tooling
- **`src/`**: C++/CUDA backend (NDArray, SpMM kernels, benchmark code)
- **`tests/`**: correctness tests (graph ops / sparse ops / GCN inference)
- **`tvm_ir/`**: generated (or collected) TVM IR output examples

## License / Notice

- **`python/needle/` contains CMU 10-714 (10714) homework code and is NOT open-sourced in this repository per course policy.**
