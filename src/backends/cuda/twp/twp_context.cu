#include <twp/twp_context.h>
#include <global_geometry/global_vertex_manager.h>
#include <muda/buffer/buffer_launch.h>

namespace uipc::backend::cuda
{
void TWPContext::ensure_storage(SizeT vertex_count)
{
    x.resize(vertex_count);
    y.resize(vertex_count);
    target_y.resize(vertex_count);
    residual.resize(vertex_count);
    clearances.resize(vertex_count);
    backward_violations.resize(vertex_count);
    backward_corrections.resize(vertex_count);
    backward_correction_counts.resize(vertex_count);
    safe_step_alphas.resize(vertex_count);
    forward_step_norms.resize(vertex_count);
    penetration_flags.resize(vertex_count);
}

void TWPContext::ensure_constraint_storage(SizeT constraint_capacity)
{
    if(backward_violations.size() < constraint_capacity)
        backward_violations.resize(constraint_capacity);
    if(backward_lambdas.size() < constraint_capacity)
        backward_lambdas.resize(constraint_capacity);
}

void TWPContext::reset(GlobalVertexManager& global_vertex_manager)
{
    auto positions      = global_vertex_manager.positions();
    auto prev_positions = global_vertex_manager.prev_positions();

    ensure_storage(positions.size());

    muda::BufferLaunch().copy<Vector3>(x.view(), prev_positions);
    muda::BufferLaunch().copy<Vector3>(y.view(), positions);
    muda::BufferLaunch().copy<Vector3>(target_y.view(), positions);
    residual.fill(1.0);
    backward_violations.fill(0.0);
    backward_lambdas.fill(0.0);
    backward_corrections.fill(Vector3::Zero());
    backward_correction_counts.fill(0);
    safe_step_alphas.fill(1.0);
    forward_step_norms.fill(0.0);

    remaining_search_bound = 0.0;
    residual_inf           = 1.0;
    backward_violation_inf = 0.0;
    max_forward_step       = 0.0;
    min_forward_alpha      = 1.0;
    forward_limited_vertices = 0;
    backward_iterations    = 0;
    backward_converged     = false;
}

}  // namespace uipc::backend::cuda
