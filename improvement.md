# TWP Improvement Notes

## Current Status

- Done: `global_twp.cu` is now only responsible for system registration, dependency binding, projection orchestration, and state write-back.
- Done: Proximity search, edge constraint refresh, forward stepping, and debug reporting have been split into separate TWP translation units.
- Done: CUDA backend source discovery uses `CONFIGURE_DEPENDS`, so newly split backend files are picked up by the normal CMake regeneration path.
- Done: `TWPContext` keeps core projection state separate from `TWPDiagnostics`, which owns backward LCP diagnostics, forward-step statistics, and debug-only clearance workspace.
- Done: `TWPConstraintSet` exposes host-side contact/edge/total counts through semantic accessors instead of public `h_*` fields.
- Done: Existing half-plane and edge constraints are written through typed helper functions instead of hand-writing `weights`, `normals`, and `offsets` at each call site.
- Done: Edge reference lengths use the squared Eq. (13) form directly. TWP caches `||y_i^0 - y_j^0||^2` and avoids square roots in edge reference preparation and edge constraint assembly.
- Done: Function names now match behavior more closely, for example edge constraint update is named as a refresh operation instead of an append-only operation.

## High Priority

- `proximity_search()` currently searches from `context.target_y`, but Algorithm 1 uses `Proximity_Search(x^(l), Dmax)`. The proximity set can become inconsistent with the current forward state. It should search from `context.x`; `target_y` should remain the Newton target `y^0`.
- TWP currently only supports half-plane proximity. It does not use cloth self-proximity or existing simplex candidate systems, so it cannot prevent cloth self-intersection or self-compression.
- Done: The backward LCP solver now uses colored projected Gauss-Seidel for edge constraints. Edge colors are built from fixed mesh edge topology and reused across PH contact set changes. PH constraints are solved afterward as a separate parallel batch.
- Edge constraint refresh has separate timing, but still refreshes all surface edges every TWP iteration. It should eventually use active-region filtering.

## Architecture Remaining

- `TWPConstraintSet` still stores all constraints in one generic linear payload. Before adding vertex-triangle and edge-edge volume constraints, add typed builders or typed views for those constraint families.
- Half-plane contact is currently a special first constraint family. When adding cloth self-proximity, keep proximity generation and constraint encoding separate so `proximity_search()` does not silently become an edge/volume constraint assembly function.
- Backward solver diagnostics are separated from core context, but LCP solve policy is still hard-coded in `TWPBackwardSolver`. If adding alternative relaxations or switching between Jacobi/GS for experiments, make the solve policy explicit.
- PH constraints currently run as one parallel batch after edge constraints. This matches the current half-plane use case, but dynamic self-contact constraints will need their own coloring or conflict-free batching.
- Edge coloring currently uses a host-side greedy pass when edge count changes. Because mesh edge topology is fixed, this should be amortized; if mesh topology becomes dynamic, move edge coloring or active-set compaction to a GPU-side path.

## Verification Gaps

- Debug logs do not report mesh quality, such as minimum triangle area, area ratio, or minimum edge length after TWP writes back state.
- Logs only report final backward diagnostics for each projection call, not the full TWP outer-loop convergence history.
- There is no small deterministic test for the squared edge constraint linearization.
- There is no small deterministic test for typed half-plane and edge constraint writers.
