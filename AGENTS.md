# Repository Instructions

- Before running a build, long-running benchmark, or machine-heavy test command, ask the user for confirmation first.
- Small inspection commands such as `git status`, `find`, `grep`, `sed`, and short file reads do not require confirmation.
- Do not add fallback paths just to mask an error. Diagnose the root cause first, and only add fallback behavior when it is an explicit design choice.
- Do not manually copy build artifacts into the Python package/native directory to refresh pyuipc. Fix or invoke the proper build/install/sync target instead.
- For coordinate transforms, do not rotate or translate only the visualization/output layer unless the user explicitly asks for a viewer-only transform. Keep simulation state, exported state, gravity, ground, and collision geometry in one consistent coordinate frame, and state which frame is being changed before editing.
- When refactoring backend systems, first check ownership, friend/private access, and existing handle/slot usage patterns. Keep global state mutation in the system that already owns the permission; helper classes should mainly hold data or implement pure algorithm steps.
- When writing CUDA backend code, do not wrap `ParallelFor().apply(... __device__ lambda ...)` inside generic lambdas or heavily templated host lambdas. Prefer explicit helper functions or file-scope template functions so nvcc sees a simple kernel launch structure.
- In CUDA device lambdas, capture device viewers (`.viewer()` / `.cviewer()`), not host-side buffer views, unless existing code proves that exact type is device-callable.
- For Eigen atomic operations in CUDA kernels, bind mutable viewer elements to a local reference before calling `muda::eigen::atomic_add`, and materialize expression templates with `.eval()` when passing computed Eigen expressions.
