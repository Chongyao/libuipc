#pragma once
#include <type_define.h>
#include <twp/twp_diagnostics.h>
#include <muda/buffer/device_buffer.h>
#include <muda/buffer/device_var.h>

namespace uipc::backend::cuda
{
class GlobalVertexManager;
class FiniteElementMethod;
class FiniteElementVertexReporter;

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
