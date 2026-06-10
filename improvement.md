# TWP Improvement Notes

## High Priority

- `proximity_search()` currently searches from `context.target_y`, but Algorithm 1 uses `Proximity_Search(x^(l), Dmax)`. The proximity set can become inconsistent with the current forward state. It should search from `context.x`; `target_y` should remain the Newton target `y^0`.
- Done: Edge reference lengths now use the squared Eq. (13) form directly. TWP caches `||y_i^0 - y_j^0||^2` and avoids square roots in edge reference preparation and edge constraint assembly.
- TWP currently only supports half-plane proximity. It does not use cloth self-proximity or existing simplex candidate systems, so it cannot prevent cloth self-intersection or self-compression.
- The backward LCP solver is parallel projected Jacobi, while the paper recommends multi-color projected Gauss-Seidel. This can affect convergence and projection quality for coupled constraints.
- Partially done: Edge constraint refresh now has separate timing. It still refreshes all surface edges every TWP iteration and should eventually use active-region filtering.

## Architecture

- Done: `global_twp.cu` now owns system registration, dependency binding, projection orchestration, and state write-back only. Proximity search, edge constraint refresh, forward stepping, and debug reporting are split into separate TWP translation units.
- Done: CUDA backend source discovery now uses `CONFIGURE_DEPENDS`, so newly split backend files are picked up by the normal CMake regeneration path.
- Partially done: `TWPConstraintSet` now exposes host-side contact/edge/total counts through semantic accessors instead of public `h_*` fields. The linear constraint payload is still generic, so vertex-triangle and edge-edge constraints should eventually get typed builders or typed views instead of encoding semantics only in `weights`, `normals`, and `offsets`.
- `TWPContext` mixes algorithm state, diagnostics, temporary debug buffers, and forward-step flags. Diagnostics should eventually be separated from core solver state.
- Done: Function names must match behavior. For example, the edge constraint update is now named as a refresh operation instead of an append-only operation.

## Verification Gaps

- Debug logs do not report mesh quality, such as minimum triangle area, area ratio, or minimum edge length after TWP writes back state.
- Logs only report final backward diagnostics for each projection call, not the full TWP outer-loop convergence history.
- There is no small deterministic test for the squared edge constraint linearization.
