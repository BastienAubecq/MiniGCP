# MiniGCP

**MiniGCP** is a header-only C++23 Constraint Programming (CP) solver built on **SYCL** ([AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp)) for execution on both GPUs (NVIDIA, AMD, Intel, and Apple Silicon via Metal) and CPUs (via OpenMP).

---

## Features & Architecture

- **Header-Only C++23 Library** (`include/MiniGCP.hpp`): Zero-overhead compile-time polymorphism using C++20/23 concepts and templates instead of virtual dispatch on device kernels.
- **Three-Tier Memory Model (`Scope`)**:
  - `Scope::global`: Read-only data shared across all work-groups (e.g., distance matrices, constant tables).
  - `Scope::local`: Per-work-group non-backtracked scratch state (e.g., sparse-set permutation arrays like `notFixed`). Can be placed in fast GPU work-group local/shared memory (`localLocal = true`).
  - `Scope::state`: Per-work-group backtrackable state (variable domains, reversible counters). Can be placed in fast GPU work-group local/shared memory (`localState = true`) and copied to/from the global trails stack on `push`/`pop`.
- **Variables & Views (`include/MiniGCP/state/`)**:
  - `IntBitset` / `IntBitsetPtr`: Domain represented as a 32-bit word bitset with `[min, max, size]` header and lock-free work-group atomic updates.
  - `IntInterval` / `IntIntervalPtr`: Bound-consistent interval domain `[min, max]`.
  - `Bool` / `BoolPtr`: Compact boolean domain `{0, 1}`.
  - Views & constant adapters: `Add` / `AddVec` (`+`, `-`), `Neg` / `NegVec` (`-`), `Const`, `Same`, `Range`.
- **Constraints (`include/MiniGCP/constraints/`)**:
  - `AllDifferentFWC` (work-group parallel) & `AllDifferentFWCSeq` (single-thread): Forward-checking `AllDifferent`.
  - `Circuit` (work-group parallel) & `CircuitSeq` (single-thread): Subtour elimination over successor variables.
  - `Cumulative` (work-group parallel) & `CumulativeSeq` (single-thread): Time-Table filtering (Gay, Hartert, Schaus, CP 2015) combined with the event-point `Profile` structure (Gingras & Quimper, IJCAI 2016) built in GPU local memory via parallel Bitonic sort and prefix scan.
  - `Element1D`: 1D array indexing constraint $t[y] = z$ with sorted index bounds filtering.
  - `NotEqual` & `NotEqualVec`: Binary inequality $x \neq y$ and vectorized inequality over variable views.
  - `Precedence` & `PrecedenceVec`: Temporal precedence constraint $x + d \le y$ and work-group parallel vectorized precedences.
  - `Sum` (work-group parallel) & `SumSeq` (single-thread): Linear bound-consistent sum $\sum_i x_i = y$.
- **Search & Solvers (`include/MiniGCP/search/`, `include/MiniGCP/solver/`)**:
  - `FirstFail`: Parallel work-group reduction (`groupArgmin`) selecting the unfixed variable with the smallest domain and branching with `FixRemove` ($x = \min(D(x))$ left, $x \neq \min(D(x))$ right).
  - `EstFirst`: Parallel work-group reduction (`groupArgmin`) selecting the unfixed task variable with the smallest earliest start time ($\min(D(x))$).
  - Objectives: `Satisfaction`, `Minimization<IntVar>`, `Maximization<IntVar>`.
  - `DFSSolver`: Depth-First Search within a single work-group of `groupSize` threads.
  - `EPSSolver`: Embarrassingly Parallel Search decomposing the top `epsDepth` levels into $2^{\text{epsDepth}}$ subproblems solved dynamically across `nGroups` work-groups.

---

## Prerequisites & Installation

MiniGCP requires:
- A C++23-capable LLVM/Clang toolchain
- [AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp) (`acpp`) with its generic SSCP compiler enabled
- [CMake](https://cmake.org/) ($\ge 3.20$)
- [Catch2](https://github.com/catchorg/Catch2) (v3, for unit tests)

### macOS (Apple Silicon GPU via Metal + OpenMP CPU)

AdaptiveCpp includes an experimental **Metal backend** that compiles SYCL kernels to LLVM IR at compile time and JIT-translates them to Metal Shading Language (MSL) at runtime for Apple Silicon GPUs (M1/M2/M3/M4).

1. **Install dependencies via Homebrew**:
   ```bash
   brew install cmake llvm@21 lld@21 libomp catch2
   ```

2. **Build and install AdaptiveCpp with the Metal backend**:
   ```bash
   git clone https://github.com/AdaptiveCpp/AdaptiveCpp.git /tmp/AdaptiveCpp
   mkdir -p /tmp/AdaptiveCpp/build && cd /tmp/AdaptiveCpp/build

   cmake \
     -DCMAKE_INSTALL_PREFIX=/opt/homebrew \
     -DLLVM_DIR=/opt/homebrew/opt/llvm@21/lib/cmake/llvm \
     -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm@21/bin/clang \
     -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm@21/bin/clang++ \
     -DACPP_LLD_PATH=/opt/homebrew/opt/lld@21/bin/ld64.lld \
     -DCMAKE_OSX_SYSROOT="$(xcrun --show-sdk-path)" \
     -DACPP_COMPILER_FEATURE_PROFILE=full \
     -DWITH_METAL_BACKEND=ON \
     -DWITH_OPENCL_BACKEND=OFF \
     ..

   make -j$(sysctl -n hw.ncpu) install
   ```
   *(Note: If `metal-cpp` is not pre-installed, AdaptiveCpp's CMake configuration automatically fetches Apple's `metal-cpp` headers).*

3. **Verify detected CPU & Apple GPU backends**:
   ```bash
   acpp-info
   ```
   You should see both backends listed:
   ```text
   Loaded backend 0: OpenMP
     Found device: AdaptiveCpp OpenMP host device
   Loaded backend 1: Metal
     Found device: Apple M3 Pro
   ```

---

## Building & Running the Examples

The `examples/` directory includes:
- **`NQueens`** (`examples/NQueens.cpp`): Solves the $N$-Queens problem (includes `AllDifferentFWC`, binary `NotEqual`, and vectorized `NotEqualVec` formulations).
- **`RCPSP`** (`examples/RCPSP.cpp`): Solves the Resource-Constrained Project Scheduling Problem to optimality using `PrecedenceVec`, `Cumulative`, and `EstFirst`. Benchmark instances (including PSPLib `j30_1_1.rcp`) are provided in `data/RCPSP/`.
- **`TSP`** (`examples/TSP.cpp`): Solves the Travelling Salesperson Problem to optimality using `AllDifferentFWC`, `Circuit`, `Element1D`, and `Sum`. Benchmark instances are provided in `data/TSP/`.

```bash
mkdir -p examples/build && cd examples/build
cmake ..
make -j
```

### Run N-Queens
```bash
# Find all solutions for N = 12 queens
./NQueens 12

# Find the first solution for N = 50 queens
./NQueens 50 1
```
Example output (`./NQueens 12`):
```text
Solutions: 14200
Nodes: 232163
Failures: 101882
Objective: 0
Subproblem: 0
Completed: 1
11 9 7 1 4 2 0 8 10 5 3 6
```

### Run TSP
```bash
./TSP ../../data/TSP/tsp_15.txt
./TSP ../../data/TSP/tsp_17.txt
```
Example output (`./TSP ../../data/TSP/tsp_15.txt`):
```text
Solutions: 64
Nodes: 7033
Failures: 3453
Objective: 291
Subproblem: 0
Completed: 1
10 12 6 5 8 7 4 9 14 13 3 2 0 11 1
```

### Run RCPSP
```bash
./RCPSP ../../data/RCPSP/j30_1_1.rcp
```
Example output (`./RCPSP ../../data/RCPSP/j30_1_1.rcp`):
```text
Solutions: 8
Nodes: 347
Failures: 166
Objective: 43
Subproblem: 0
Completed: 1
0 4 0 0 9 31 4 4 12 6 12 16 4 18 12 13 23 10 13 23 29 29 36 38 30 33 15 31 9 41 40 43
```

---

## Building & Running the Tests

The `tests/` directory contains a comprehensive Catch2 test suite covering variable domains (`Bool`, `IntBitset`, `IntInterval`), constraints (`AllDifferentFWC`, `Circuit`, `Element1D`, `NotEqual`, `Sum`), search strategies (`FirstFail`, `FixRemove`), and end-to-end problem solving (`NQueens`, `TSP`) on the GPU (`sycl::gpu_selector_v` in `tests/utils.hpp`).

```bash
mkdir -p tests/build && cd tests/build
cmake ..
make -j
ctest --output-on-failure
```

---

## Quick Start Example

```cpp
#include <iostream>
#include "MiniGCP.hpp"

using namespace cp;

int main() {
    uint32_t n = 12;

    // 1. Declare the model with the constraint types it will use
    Model<
        AllDifferentFWC<IntBitsetPtr>,
        AllDifferentFWC<AddVec<IntBitsetPtr, Range>>,
        AllDifferentFWC<AddVec<NegVec<IntBitsetPtr>, Range>>
    > model;

    // 2. Allocate n bitset variables with domain {0, ..., n - 1}
    IntBitsetPtr queens(model, n, n);

    // 3. Post constraints using zero-overhead vector views
    model.constraint(AllDifferentFWC(model, queens, n));
    model.constraint(AllDifferentFWC(model, queens + Range(), n));
    model.constraint(AllDifferentFWC(model, -queens + Range(), n));

    // 4. Configure search strategy and GPU device parameters (work-group size 64, 1 group)
    FirstFail strategy(queens, n);
    DeviceParams<64> params(sycl::gpu_selector_v, 1);

    // 5. Solve on the GPU
    DFSSolver solver(model, Satisfaction(), queens, n, strategy, params);
    solver.solve(noLimit);

    std::cout << solver.getStatistics() << std::endl;
    return 0;
}
```
