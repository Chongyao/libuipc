#include <twp/twp_backward_solver.h>
#include <twp/twp_backward_solver_lcp.h>
#include <twp/twp_constraint_set.h>
#include <twp/twp_context.h>
#include <global_geometry/global_vertex_manager.h>
#include <finite_element/finite_element_method.h>
#include <finite_element/finite_element_vertex_reporter.h>
#include <uipc/common/timer.h>
#include <muda/buffer/buffer_launch.h>
#include <muda/cub/device/device_reduce.h>
#include <utility>

namespace uipc::backend::cuda
{
using namespace twp_backward_solver_lcp;

void TWPBackwardSolver::solve(SolveInfo info)
{
    Timer timer{"TWP Backward"};

    auto& context     = *info.context;
    auto& constraints = *info.constraints;
    const IndexT constraint_count = constraints.host_total_constraint_count();

    muda::BufferLaunch().copy<Vector3>(context.y.view(),
                                       std::as_const(context.target_y).view());
    context.backward_lambdas.fill(0.0);
    context.diagnostics.backward.iterations = 0;
    context.diagnostics.backward.converged  = true;

    if(constraint_count == 0)
    {
        context.diagnostics.backward.violation_inf = 0.0;
        context.diagnostics.backward.lcp_min_gap = 0.0;
        context.diagnostics.backward.lcp_complementarity_inf = 0.0;
        context.diagnostics.backward.lcp_projected_residual_inf = 0.0;
        return;
    }

    update_edge_coloring_if_needed(constraints);
    update_self_contact_coloring(constraints, info.use_gpu_self_contact_coloring);

    bool use_fem_mass = info.finite_element_method && info.finite_element_vertex_reporter;
    const IndexT max_iterations = info.max_iterations > 0 ? info.max_iterations : 1;
    for(IndexT iter = 0; iter < max_iterations; ++iter)
    {
        if(use_fem_mass)
        {
            TWPFEMMassInfo mass_info{
                info.finite_element_method->masses().cviewer().name("masses"),
                info.finite_element_method->is_fixed().cviewer().name("is_fixed"),
                info.finite_element_vertex_reporter->vertex_offset(),
                info.finite_element_method->xs().size()};
            solve_constraints_colored(context,
                                      constraints,
                                      std::as_const(m_colored_edge_ids).view(),
                                      m_host_edge_color_offsets,
                                      constraints.host_edge_constraint_offset(),
                                      mass_info);
            solve_obstacle_constraints_jacobi(
                context, constraints, 0, constraints.host_obstacle_constraint_count(), mass_info);
            solve_constraints_colored(context,
                                      constraints,
                                      std::as_const(m_colored_contact_ids).view(),
                                      m_host_self_contact_color_offsets,
                                      constraints.host_self_contact_constraint_offset(),
                                      mass_info);
            if(info.check_convergence)
                compute_lcp_diagnostics(context, constraints, mass_info);
        }
        else
        {
            TWPUnitMassInfo mass_info;
            solve_constraints_colored(context,
                                      constraints,
                                      std::as_const(m_colored_edge_ids).view(),
                                      m_host_edge_color_offsets,
                                      constraints.host_edge_constraint_offset(),
                                      mass_info);
            solve_obstacle_constraints_jacobi(
                context, constraints, 0, constraints.host_obstacle_constraint_count(), mass_info);
            solve_constraints_colored(context,
                                      constraints,
                                      std::as_const(m_colored_contact_ids).view(),
                                      m_host_self_contact_color_offsets,
                                      constraints.host_self_contact_constraint_offset(),
                                      mass_info);
            if(info.check_convergence)
                compute_lcp_diagnostics(context, constraints, mass_info);
        }

        context.diagnostics.backward.iterations = iter + 1;
        if(!info.check_convergence)
            continue;

        muda::DeviceReduce().Max(
            context.diagnostics.backward.backward_violations.data(),
            context.diagnostics.backward.max_backward_violation.data(),
            constraint_count);
        muda::DeviceReduce().Min(context.diagnostics.backward.lcp_gaps.data(),
                                 context.diagnostics.backward.min_lcp_gap.data(),
                                 constraint_count);
        muda::DeviceReduce().Max(
            context.diagnostics.backward.lcp_complementarity.data(),
            context.diagnostics.backward.max_lcp_complementarity.data(),
            constraint_count);
        muda::DeviceReduce().Max(
            context.diagnostics.backward.lcp_projected_residual.data(),
            context.diagnostics.backward.max_lcp_projected_residual.data(),
            constraint_count);
        context.diagnostics.backward.violation_inf =
            context.diagnostics.backward.max_backward_violation;
        context.diagnostics.backward.lcp_min_gap = context.diagnostics.backward.min_lcp_gap;
        context.diagnostics.backward.lcp_complementarity_inf =
            context.diagnostics.backward.max_lcp_complementarity;
        context.diagnostics.backward.lcp_projected_residual_inf =
            context.diagnostics.backward.max_lcp_projected_residual;
        if(context.diagnostics.backward.violation_inf < BackwardTolerance
           && context.diagnostics.backward.lcp_projected_residual_inf < BackwardTolerance)
        {
            context.diagnostics.backward.converged = true;
            break;
        }
        context.diagnostics.backward.converged = false;
    }

    if(!info.check_convergence)
    {
        context.diagnostics.backward.converged = true;
        context.diagnostics.backward.violation_inf = 0.0;
        context.diagnostics.backward.lcp_min_gap = 0.0;
        context.diagnostics.backward.lcp_complementarity_inf = 0.0;
        context.diagnostics.backward.lcp_projected_residual_inf = 0.0;
    }
}
}  // namespace uipc::backend::cuda
