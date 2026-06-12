# TWP GPU Coloring Implementation Plan

## Status
- Self-contact now uses a GPU randomized coloring path in the backward solver.
- Edge constraints still use the CPU greedy coloring cache.
- This document remains as the implementation/validation plan for the GPU self-contact coloring path.

## Objective
Replace the CPU self-contact coloring step with a GPU implementation that matches the randomized, iterative coloring structure from the Vivace paper, while preserving enough structure to compare against the old CPU approach if needed.

## What Must Stay
- Edge constraint coloring remains cached on CPU for now.
- PH support constraints remain separate and Jacobi-style.
- If CPU/GPU comparison is needed, restore the old self-contact CPU coloring as an explicit selectable path instead of mixing it into the GPU implementation.

## Data Model
- Nodes: self-contact constraints.
- Conflict relation: two constraints conflict if any vertex id overlaps.
- Device state:
  - `constraint_vertices`
  - `constraint_color`
  - `uncolored_mask`
  - `vertex_owner`
  - `vertex_incident_csr`
  - `color_counts`
  - `colored_constraint_ids`
  - `color_offsets`

## Proposed GPU Workflow
1. Build or reuse `vertex_incident_csr` on GPU.
2. Initialize all self-contact constraints as uncolored.
3. For each round, propose a color for every uncolored constraint.
4. Detect conflicts by walking only the incident constraints of the four vertices.
5. Apply a deterministic tie-break rule for conflicts.
6. Mark winners colored and remove them from the active set.
7. Stop when a round produces no new colors.
8. Compact `constraint_color` into contiguous color groups.
9. Expose `colored_constraint_ids + color_offsets` to backward GS.

## Efficiency Requirements
- No host round-trip in the coloring loop.
- No dense adjacency matrix.
- No fixed `MaxSelfContactColors` scan loop.
- Compact only once after convergence.
- Keep CSR cached across frames when the candidate set signature is unchanged.
- Favor one kernel per phase, not one kernel per constraint.

## Suggested Milestones
1. GPU `constraint_color` generation with deterministic tie-break.
2. GPU convergence detection for the active set.
3. GPU compaction into compact color groups.
4. Backward solver consumes GPU compact output.
5. Benchmark against CPU greedy coloring.

## Validation Criteria
- Color groups are complete and conflict-free.
- Self-contact constraints are all assigned.
- Solve results match CPU coloring within numerical tolerance.
- Runtime improves or stays competitive on large contact counts.
