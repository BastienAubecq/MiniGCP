# MiniGCP Copilot Instructions

## Build, test, and lint

MiniGCP is a header-only C++23 library compiled through SYCL. The project does not have a top-level `CMakeLists.txt`; configure the examples and tests independently from their own build directories. CMake defaults `CMAKE_CXX_COMPILER` to `acpp` (AdaptiveCpp) when no compiler is supplied, and both projects enable `-Wall -O3`. A C++23-capable SYCL compiler, CMake 3.20+, and Catch2 3 are required.

For debugging and validation, use Intel DPC++ on the CPU before trying AdaptiveCpp or a GPU. First source `/opt/intel/oneapi/setvars.sh`, configure CMake with `-DCMAKE_CXX_COMPILER=icpx -DCMAKE_CXX_FLAGS=-fsycl`, and compile with `icpx -fsycl`. Ensure the executable or test configuration uses `sycl::cpu_selector_v`; this avoids GPU crashes and the known AdaptiveCpp OpenMP backend issue.

Build and run the examples:

```bash
source /opt/intel/oneapi/setvars.sh
mkdir -p examples/build
cd examples/build
cmake .. -DCMAKE_CXX_COMPILER=icpx -DCMAKE_CXX_FLAGS=-fsycl
cmake --build . -j
./NQueens 12
./TSP ../../data/TSP/tsp_15.txt
```

Build and run all tests:

```bash
source /opt/intel/oneapi/setvars.sh
mkdir -p tests/build
cd tests/build
cmake .. -DCMAKE_CXX_COMPILER=icpx -DCMAKE_CXX_FLAGS=-fsycl
cmake --build . -j
ctest --output-on-failure
```

Run one discovered test case through CTest:

```bash
cd tests/build
ctest -R 'HostIntBitset allocation size simple' --output-on-failure
```

Run one Catch2 test case directly:

```bash
cd tests/build
./test_IntBitset "HostIntBitset allocation size simple"
```

There is no dedicated lint or formatting target. Keep the existing C++ style and use the compiler warnings enabled by the CMake files.

## Architecture

- The public API is header-only and is aggregated by `include/MiniGCP.hpp`. The implementation is organized into model allocation, state/variable domains, constraints, search, and solvers under `include/MiniGCP/`.
- `Model<Constraints...>` is compile-time typed: every constraint type used by a model must appear in its template parameter pack before constraints of that type can be posted. The model also owns the host-side `Allocator`, which assigns offsets for propagator data.
- Data is allocated into three `Scope`s: `global` is shared read-only data, `local` is per-work-group non-backtracked data, and `state` is per-work-group backtrackable data. `Context` resolves the allocated offsets to the current device/work-group storage. Prefer the existing allocation and context access patterns instead of adding raw device pointers.
- Domain implementations and views live in `state/`. `IntBitset`, `IntInterval`, and `Bool` provide domains; pointer/view adapters such as `Add`, `Neg`, `Const`, and `Range` let constraints operate on transformed variables without introducing runtime polymorphism.
- A constraint satisfies the `Constraint` concept, must be device-copyable, exposes a static `parallel` flag, and implements `propagate(Context) -> DomainChange`. Constraint propagation is performed by `ConstraintStore` until a fixpoint; parallel constraints use the whole work-group, while sequential variants are invoked per thread.
- Search strategies select and split variables into device-copyable decisions. `FirstFail` chooses the smallest non-singleton domain and branches through `FixRemove`; its `maxDepth()` determines the decision-stack capacity.
- `Solver` owns the SYCL queue and device-side `DeviceSolver`. `DFSSolver` requires one work-group and performs depth-first search; `EPSSolver` distributes the first `epsDepth` levels across multiple work-groups and then runs DFS within each subproblem. `DeviceParams<groupSize>` controls the work-group size, device selector, group count, and whether local/state data use local memory.
- Solutions and statistics are copied back from device storage after each kernel launch. Changes to objective handling, state restoration, decision stacks, or output selection must preserve work-group synchronization.

## Repository-specific conventions

- Prefer concepts, templates, and device-copyable value types over virtual interfaces or heap-owned polymorphic objects; solver and propagator types are instantiated at compile time for the model’s exact constraint set.
- When adding a new domain or constraint, integrate it with `Allocator`/`Scope`, `Context`, `DomainChange`, and the relevant `ConstraintStore`/model template requirements. Account for alignment and backtracking semantics rather than storing host-only state in the device object.
- Keep host allocation and device execution separate: host-side references use the allocator-backed data, while kernels access data through `Context`. Work-group operations must use the existing SYCL barriers/reductions and must not introduce divergent synchronization.
- Tests are split by subsystem under `tests/state`, `tests/constraints`, `tests/search`, and `tests/examples`. Most tests exercise device code through `tests/utils.hpp`, whose `testParams` uses `sycl::gpu_selector_v` with one group of 64 threads; host-only domain behavior is tagged separately (for example `[host]`).
- Example tests include the example implementation directly with `NO_MAIN` defined, so changes to an example’s solver factory or command-line entry point can affect both the executable and its tests.
- The CMake files define one executable per example or test source. Add new targets explicitly to `examples/CMakeLists.txt` or `tests/CMakeLists.txt`; do not assume a top-level build will discover them.
- Benchmark/problem data is stored under `data/TSP` and `data/RCPSP` and is passed to executables as a path relative to the current working directory.
