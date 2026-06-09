# Repository Instructions

- Before running a build, long-running benchmark, or machine-heavy test command, ask the user for confirmation first.
- Small inspection commands such as `git status`, `find`, `grep`, `sed`, and short file reads do not require confirmation.
- Do not add fallback paths just to mask an error. Diagnose the root cause first, and only add fallback behavior when it is an explicit design choice.
- Do not manually copy build artifacts into the Python package/native directory to refresh pyuipc. Fix or invoke the proper build/install/sync target instead.
