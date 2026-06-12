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

- Done: `proximity_search()` now searches from the current forward state `context.x`, matching Algorithm 1's `Proximity_Search(x^(l), Dmax)`. `context.target_y` remains the Newton target `y^0`.
- TWP currently only supports half-plane proximity. It does not use cloth self-proximity or existing simplex candidate systems, so it cannot prevent cloth self-intersection or self-compression.
- Done: The backward LCP solver now uses colored projected Gauss-Seidel for edge constraints. Edge colors are built from fixed mesh edge topology and reused across PH contact set changes. PH constraints are solved afterward as a separate parallel batch.
- Edge constraint refresh has separate timing, but still refreshes all surface edges every TWP iteration. It should eventually use active-region filtering.

## Architecture Remaining

- `GlobalTWP` currently calls `GlobalTrajectoryFilter::detect()` by adding `GlobalTWP`
  as a friend of `GlobalTrajectoryFilter`. This should be cleaned up later:
  keep candidate refresh on the owner side, preferably in the TWP advance path
  before `GlobalTWP::project()`, so TWP only reads existing simplex candidates
  and does not need private access to `GlobalTrajectoryFilter`.
- `TWPConstraintSet` still stores all constraints in one generic linear payload. Before adding vertex-triangle and edge-edge volume constraints, add typed builders or typed views for those constraint families.
- Half-plane contact is currently a special first constraint family. When adding cloth self-proximity, keep proximity generation and constraint encoding separate so `proximity_search()` does not silently become an edge/volume constraint assembly function.
- Backward solver diagnostics are separated from core context, but LCP solve policy is still hard-coded in `TWPBackwardSolver`. If adding alternative relaxations or switching between Jacobi/GS for experiments, make the solve policy explicit.
- PH constraints currently run as one parallel batch after edge constraints. This matches the current half-plane use case, but dynamic self-contact constraints will need their own coloring or conflict-free batching.
- Edge coloring currently uses a host-side greedy pass when edge count changes. Because mesh edge topology is fixed, this should be amortized; if mesh topology becomes dynamic, move edge coloring or active-set compaction to a GPU-side path.

## GPU Coloring Reference: Vivace

- Vivace uses vertex coloring, not constraint coloring. Graph vertices are solve unknowns/particles. An edge exists between two graph vertices when a constraint depends on both unknowns.
- The coloring procedure is fully GPU-oriented and runs as repeated parallel rounds over currently uncolored vertices:
  - tentative coloring: each uncolored vertex randomly picks a color from its available palette;
  - conflict resolution: each vertex checks neighbor tentative colors; on conflict, a deterministic Hungarian/Luby-style heuristic lets the higher-index vertex keep the color and losers retry;
  - palette update: accepted colors are removed from neighbors' palettes; empty palettes are fed with a new color.
- After coloring, colored Gauss-Seidel iterates over colors. For one color, all vertices in that partition are updated in parallel. Each thread owns one vertex and accumulates corrections from all incident constraints, so same-color updates have no write conflicts and do not need atomic adds.
- This is different from the current TWP implementation, which uses constraint coloring and constraint-owned updates. A full Vivace-style GPU path would require GPU CSR adjacency, per-vertex palettes/colors, conflict flags, and per-vertex incident constraint lists. It is a larger redesign than caching the existing constraint coloring.

## Planned: Contact Coloring Cache/Version

- Problem: `TWPBackwardSolver::update_contact_coloring()` rebuilds contact colors every `backward()` call by copying contact `vertex_ids` from device to host, running a CPU greedy coloring pass, then copying colored ids back to device. Within one TWP projection, the contact topology only changes when `proximity_search()` refreshes the active set. Rebuilding colors on every backward solve is unnecessary.
- Plan:
  - Add a host-side contact version counter to `TWPConstraintSet`, for example `contact_version()`.
  - Increment the version only when a new contact active set is finalized, i.e. in `set_host_contact_constraint_count()`. Edge-only refreshes must not change the contact version.
  - Rename `update_contact_coloring()` to `update_contact_coloring_if_needed()`.
  - Add `m_coloring_contact_version` and `m_coloring_contact_count` to `TWPBackwardSolver`.
  - Rebuild contact colors only when the version or contact count changes; otherwise reuse `m_colored_contact_ids` and `m_host_contact_color_offsets`.
  - Keep edge coloring behavior unchanged for now. Edge coloring is already keyed by fixed edge count.
- Expected benefit: removes repeated device-host-device coloring work when TWP performs many outer iterations using the same proximity set. This should reduce `TWP Build Contact LCP Colors` time and make backward cost more predictable.
- Boundary: this cache is still host-side coloring. It is a low-risk intermediate step before a possible GPU coloring implementation for dynamic PT/EE contact sets.

## Verification Gaps

- Debug logs do not report mesh quality, such as minimum triangle area, area ratio, or minimum edge length after TWP writes back state.
- Logs only report final backward diagnostics for each projection call, not the full TWP outer-loop convergence history.
- There is no small deterministic test for the squared edge constraint linearization.
- There is no small deterministic test for typed half-plane and edge constraint writers.
