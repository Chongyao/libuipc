#pragma once
#include <type_define.h>
#include <muda/buffer/device_buffer.h>
#include <muda/buffer/device_var.h>

namespace uipc::backend::cuda
{
class GlobalVertexManager;
class FiniteElementMethod;
class FiniteElementVertexReporter;

struct TWPBackwardDiagnostics
{
    muda::DeviceBuffer<Float> backward_violations;
    muda::DeviceBuffer<Float> lcp_gaps;
    muda::DeviceBuffer<Float> lcp_complementarity;
    muda::DeviceBuffer<Float> lcp_projected_residual;

    muda::DeviceVar<Float> max_backward_violation;
    muda::DeviceVar<Float> min_lcp_gap;
    muda::DeviceVar<Float> max_lcp_complementarity;
    muda::DeviceVar<Float> max_lcp_projected_residual;

    Float  violation_inf          = 0.0;
    Float  lcp_min_gap            = 0.0;
    Float  lcp_complementarity_inf = 0.0;
    Float  lcp_projected_residual_inf = 0.0;
    IndexT iterations             = 0;
    bool   converged              = false;

    void ensure_constraint_storage(SizeT constraint_capacity);
    void reset();
};

struct TWPForwardDiagnostics
{
    muda::DeviceBuffer<Float>  safe_step_alphas;
    muda::DeviceBuffer<Float>  step_norms;
    muda::DeviceBuffer<IndexT> limited_flags;

    muda::DeviceVar<Float>  min_safe_step_alpha;
    muda::DeviceVar<Float>  max_residual;
    muda::DeviceVar<Float>  max_step_norm;
    muda::DeviceVar<IndexT> limited_count;

    Float  residual_inf     = 1.0;
    Float  max_step         = 0.0;
    Float  min_alpha        = 1.0;
    IndexT limited_vertices = 0;

    void ensure_vertex_storage(SizeT vertex_count);
    void reset();
};

struct TWPDebugWorkspace
{
    muda::DeviceBuffer<Float>  clearances;
    muda::DeviceBuffer<IndexT> penetration_flags;

    muda::DeviceVar<IndexT> penetration_count;
    muda::DeviceVar<Float>  min_clearance;
};

struct TWPDiagnostics
{
    TWPBackwardDiagnostics backward;
    TWPForwardDiagnostics  forward;
    TWPDebugWorkspace      debug;

    void ensure_vertex_storage(SizeT vertex_count);
    void ensure_constraint_storage(SizeT constraint_capacity);
    void reset();
};

struct TWPContext
{
    muda::DeviceBuffer<Vector3> x;
    muda::DeviceBuffer<Vector3> y;
    muda::DeviceBuffer<Vector3> target_y;
    muda::DeviceBuffer<Vector3> backward_corrections;
    muda::DeviceBuffer<Float>   residual;
    muda::DeviceBuffer<Float>   backward_lambdas;
    muda::DeviceBuffer<Float>   target_edge_length_squares;
    muda::DeviceBuffer<Float>   proximity_distances;
    muda::DeviceBuffer<IndexT>  proximity_constraint_ids;

    muda::DeviceBuffer<Float> remaining_obstacle_search_bounds;
    muda::DeviceBuffer<Float> remaining_self_collision_search_bounds;
    // Per-iteration device-side min of the above buffers, updated by
    // forward() so the host can decide whether to refresh proximity
    // search without an extra reduce in the outer loop.
    muda::DeviceVar<Float> min_remaining_obstacle_search_bound;
    muda::DeviceVar<Float> min_remaining_self_collision_search_bound;

    TWPDiagnostics diagnostics;

    void ensure_storage(SizeT vertex_count);
    void ensure_constraint_storage(SizeT constraint_capacity);
    void reset(GlobalVertexManager& global_vertex_manager);
};
}  // namespace uipc::backend::cuda
