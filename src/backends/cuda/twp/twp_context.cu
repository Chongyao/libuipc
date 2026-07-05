#include <twp/twp_context.h>
#include <global_geometry/global_vertex_manager.h>
#include <muda/buffer/buffer_launch.h>
#include <muda/launch/parallel_for.h>

namespace uipc::backend::cuda
{
void TWPContext::ensure_storage(SizeT vertex_count)
{
    x.resize(vertex_count);
    y.resize(vertex_count);
    target_y.resize(vertex_count);
    backward_corrections.resize(vertex_count);
    residual.resize(vertex_count);
    proximity_distances.resize(vertex_count);
    proximity_constraint_ids.resize(vertex_count);
    remaining_obstacle_search_bounds.resize(vertex_count);
    remaining_self_collision_search_bounds.resize(vertex_count);
    diagnostics.ensure_vertex_storage(vertex_count);
}

void TWPContext::ensure_constraint_storage(SizeT constraint_capacity)
{
    if(backward_lambdas.size() < constraint_capacity)
        backward_lambdas.resize(constraint_capacity);
    diagnostics.ensure_constraint_storage(constraint_capacity);
}

void TWPContext::reset(GlobalVertexManager& global_vertex_manager, Float start_toi)
{
    auto positions      = global_vertex_manager.positions();
    auto prev_positions = global_vertex_manager.prev_positions();

    ensure_storage(positions.size());

    muda::BufferLaunch().copy<Vector3>(y.view(), positions);
    muda::BufferLaunch().copy<Vector3>(target_y.view(), positions);

    Float clamped_toi = start_toi;
    if(clamped_toi < 0.0)
        clamped_toi = 0.0;
    if(clamped_toi > 1.0)
        clamped_toi = 1.0;

    if(clamped_toi == 0.0)
    {
        muda::BufferLaunch().copy<Vector3>(x.view(), prev_positions);
    }
    else if(clamped_toi == 1.0)
    {
        muda::BufferLaunch().copy<Vector3>(x.view(), positions);
    }
    else
    {
        using namespace muda;
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(x.size(),
                   [x = x.viewer().name("x"),
                    prev_positions = prev_positions.cviewer().name("prev_positions"),
                    positions = positions.cviewer().name("positions"),
                    clamped_toi] __device__(int i) mutable
                   {
                       x(i) = prev_positions(i)
                              + clamped_toi * (positions(i) - prev_positions(i));
                   });
    }

    backward_corrections.fill(Vector3::Zero());
    residual.fill(1.0);
    backward_lambdas.fill(0.0);
    proximity_distances.fill(Float{1e30});
    proximity_constraint_ids.fill(-1);
    diagnostics.reset();

    remaining_obstacle_search_bounds.fill(0.0);
    remaining_self_collision_search_bounds.fill(0.0);
    min_remaining_obstacle_search_bound = 0.0;
    min_remaining_self_collision_search_bound = 0.0;
}

}  // namespace uipc::backend::cuda
