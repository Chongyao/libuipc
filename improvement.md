# TWP Improvement Notes

## Forward Step And Search Radius

1. Fix forward `D_i` semantics. Vertices without an actual proximity source should
   not be limited by the global `remaining_search_bound`; they should take the full
   step unless a real proximity constraint supplies a finite distance.

2. Compute half-plane forward distances directly from the fixed obstacle geometry.
   PH constraints should use the true distance to the plane for `D_i`, and fixed
   obstacles should keep the non-self-collision step factor.

4. Avoid treating `d_max` tuning as the primary fix. Increasing `d_max` can reduce
   proximity-search frequency, but it also broadens self-collision candidate sets
   and may increase backward/contact work. Use it only after the `D_i` and search
   budget semantics are correct.
