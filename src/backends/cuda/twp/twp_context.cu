#include <twp/twp_context.h>
#include <global_geometry/global_vertex_manager.h>
#include <muda/buffer/buffer_launch.h>

namespace uipc::backend::cuda
{
void TWPBackwardDiagnostics::ensure_constraint_storage(SizeT constraint_capacity)
{
    if(backward_violations.size() < constraint_capacity)
        backward_violations.resize(constraint_capacity);
    if(lcp_gaps.size() < constraint_capacity)
        lcp_gaps.resize(constraint_capacity);
    if(lcp_complementarity.size() < constraint_capacity)
        lcp_complementarity.resize(constraint_capacity);
    if(lcp_projected_residual.size() < constraint_capacity)
        lcp_projected_residual.resize(constraint_capacity);
}

void TWPBackwardDiagnostics::reset()
{
    backward_violations.fill(0.0);
    lcp_gaps.fill(0.0);
    lcp_complementarity.fill(0.0);
    lcp_projected_residual.fill(0.0);

    violation_inf = 0.0;
    lcp_min_gap = 0.0;
    lcp_complementarity_inf = 0.0;
    lcp_projected_residual_inf = 0.0;
    iterations = 0;
    converged = false;
}

void TWPForwardDiagnostics::ensure_vertex_storage(SizeT vertex_count)
{
    safe_step_alphas.resize(vertex_count);
    step_norms.resize(vertex_count);
    limited_flags.resize(vertex_count);
}

void TWPForwardDiagnostics::reset()
{
    safe_step_alphas.fill(1.0);
    step_norms.fill(0.0);
    limited_flags.fill(0);

    residual_inf = 1.0;
    max_step = 0.0;
    min_alpha = 1.0;
    limited_vertices = 0;
}

void TWPDiagnostics::ensure_vertex_storage(SizeT vertex_count)
{
    forward.ensure_vertex_storage(vertex_count);
}

void TWPDiagnostics::ensure_constraint_storage(SizeT constraint_capacity)
{
    backward.ensure_constraint_storage(constraint_capacity);
}

void TWPDiagnostics::reset()
{
    backward.reset();
    forward.reset();
}

void TWPContext::ensure_storage(SizeT vertex_count)
{
    x.resize(vertex_count);
    y.resize(vertex_count);
    target_y.resize(vertex_count);
    backward_corrections.resize(vertex_count);
    residual.resize(vertex_count);
    proximity_distances.resize(vertex_count);
    diagnostics.ensure_vertex_storage(vertex_count);
}

void TWPContext::ensure_constraint_storage(SizeT constraint_capacity)
{
    if(backward_lambdas.size() < constraint_capacity)
        backward_lambdas.resize(constraint_capacity);
    diagnostics.ensure_constraint_storage(constraint_capacity);
}

void TWPContext::reset(GlobalVertexManager& global_vertex_manager)
{
    auto positions      = global_vertex_manager.positions();
    auto prev_positions = global_vertex_manager.prev_positions();

    ensure_storage(positions.size());

    muda::BufferLaunch().copy<Vector3>(x.view(), prev_positions);
    muda::BufferLaunch().copy<Vector3>(y.view(), positions);
    muda::BufferLaunch().copy<Vector3>(target_y.view(), positions);
    backward_corrections.fill(Vector3::Zero());
    residual.fill(1.0);
    backward_lambdas.fill(0.0);
    proximity_distances.fill(Float{1e30});
    diagnostics.reset();

    remaining_search_bound = 0.0;
}

}  // namespace uipc::backend::cuda
