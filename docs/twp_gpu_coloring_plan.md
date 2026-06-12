# TWP GPU Coloring Plan

## Goal
Implement self-contact coloring fully on GPU, following the Vivace-style randomized coloring workflow rather than CPU greedy coloring.

## Target Behavior
- Graph nodes are contact constraints.
- Two constraints conflict if they share any vertex.
- Build complete color groups on GPU.
- Return compact `colored_constraint_ids + color_offsets`.
- Backward solve keeps the existing colored GS structure.

## Required Device State
- `constraint_vertices`: `Vector4i` per constraint.
- `constraint_color`: one color id per constraint, initialized to `-1`.
- `uncolored_mask`: one flag per constraint.
- `vertex_owner`: one owner constraint id per vertex for the current trial color.
- `incident_constraints`: CSR from vertex to constraints.
- `color_histogram` or segmented counts for compaction.

## Algorithm Outline
1. Initialize all self-contact constraints as uncolored.
2. For each round, choose a trial color for each uncolored constraint.
3. Detect conflicts by scanning only incident constraints of the four vertices.
4. Resolve conflict with a deterministic tie-break rule.
5. Mark winners as colored and remove them from the uncolored set.
6. Repeat until no new constraints are colored in a round.
7. Compact `constraint_color` into `color_offsets` and `colored_constraint_ids`.

## Efficiency Rules
- Do not scan all constraints for every color.
- Use vertex incident CSR, not a dense global adjacency matrix.
- Keep the full flow on GPU, no host round-trips during coloring.
- Avoid fixed upper-bound rounds like `MaxSelfContactColors` unless it is only a safety cap.
- Compact only once after coloring converges.
- Cache CSR when the self-contact candidate set topology is unchanged.

## Integration Plan
1. Keep current CPU greedy path as fallback until GPU path is validated.
2. Implement GPU `constraint_color` first.
3. Add GPU compaction to produce contiguous color groups.
4. Switch backward solve to consume the compacted GPU result.
5. Compare color counts, solve counts, and runtime against CPU version.

## Open Questions
- Whether to rebuild CSR every frame or version-cache it by candidate set signature.
- Whether to use per-round random color selection or a hash-based deterministic proposal.
- Whether to keep edge constraints on CPU-cached coloring permanently.
