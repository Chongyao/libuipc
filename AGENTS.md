# Repository Instructions

- Before running a build, long-running benchmark, or machine-heavy test command, ask the user for confirmation first.
- Small inspection commands such as `git status`, `find`, `grep`, `sed`, and short file reads do not require confirmation.
- Do not add fallback paths just to mask an error. Diagnose the root cause first, and only add fallback behavior when it is an explicit design choice.
- Do not manually copy build artifacts into the Python package/native directory to refresh pyuipc. Fix or invoke the proper build/install/sync target instead.
- For coordinate transforms, do not rotate or translate only the visualization/output layer unless the user explicitly asks for a viewer-only transform. Keep simulation state, exported state, gravity, ground, and collision geometry in one consistent coordinate frame, and state which frame is being changed before editing.
- When refactoring backend systems, first check ownership, friend/private access, and existing handle/slot usage patterns. Keep global state mutation in the system that already owns the permission; helper classes should mainly hold data or implement pure algorithm steps.
