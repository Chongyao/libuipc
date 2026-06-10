#include <twp/twp_backward_solver.h>
#include <twp/twp_constraint_set.h>
#include <twp/twp_context.h>
#include <global_geometry/global_vertex_manager.h>
#include <finite_element/finite_element_method.h>
#include <finite_element/finite_element_vertex_reporter.h>
#include <uipc/common/timer.h>
#include <muda/buffer/buffer_launch.h>
#include <muda/ext/eigen/atomic.h>
#include <muda/launch/parallel_for.h>
#include <muda/cub/device/device_reduce.h>
#include <utility>

namespace uipc::backend::cuda
{
namespace
{
constexpr IndexT BackwardMaxIterations = 32;
constexpr Float  BackwardTolerance     = 1e-8;
constexpr Float  LCPRelaxation         = 1.0;
}  // namespace

struct TWPUnitMassInfo
{
    MUDA_GENERIC Float inv_mass(IndexT) const { return 1.0; }
};

struct TWPFEMMassInfo
{
    muda::CDense1D<Float>  masses;
    muda::CDense1D<IndexT> is_fixed;
    IndexT                 fem_vertex_offset;
    SizeT                  fem_vertex_count;

    MUDA_GENERIC Float inv_mass(IndexT v) const
    {
        if(v < fem_vertex_offset || v >= fem_vertex_offset + fem_vertex_count)
            return 0.0;

        IndexT fem_v = v - fem_vertex_offset;
        if(is_fixed(fem_v))
            return 0.0;

        Float mass = masses(fem_v);
        return mass > 0.0 ? 1.0 / mass : 0.0;
    }
};

template <typename MassInfo>
void solve_lcp_iteration(TWPContext&        context,
                         TWPConstraintSet& constraints,
                         MassInfo          mass_info)
{
    const IndexT constraint_count = constraints.host_total_constraint_count();
    context.backward_corrections.fill(Vector3::Zero());

    using namespace muda;
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(constraint_count,
               [vertex_ids = constraints.vertex_ids.viewer().name("vertex_ids"),
                weights = constraints.weights.viewer().name("weights"),
                normals = constraints.normals.viewer().name("normals"),
                offsets = constraints.offsets.viewer().name("offsets"),
                y = context.y.viewer().name("y"),
                lambdas = context.backward_lambdas.viewer().name("backward_lambdas"),
                corrections =
                    context.backward_corrections.viewer().name("backward_corrections"),
                mass_info] __device__(int c) mutable
               {
                   Vector4i ids = vertex_ids(c);
                   Vector4  ws  = weights(c);
                   Vector3  N   = normals(c);
                   Float    C   = -offsets(c);
                   Float    diag = 0.0;
                   Float    n2   = N.squaredNorm();

                   if(n2 <= 1e-24)
                       return;

                   for(IndexT local_i = 0; local_i < 4; ++local_i)
                   {
                       IndexT v = ids(local_i);
                       if(v < 0 || ws(local_i) == 0.0)
                           continue;

                       C += ws(local_i) * y(v).dot(N);

                       Float inv_mass = mass_info.inv_mass(v);
                       if(inv_mass > 0.0)
                           diag += ws(local_i) * ws(local_i) * inv_mass * n2;
                   }

                   if(diag <= 0.0)
                       return;

                   Float old_lambda = lambdas(c);
                   Float new_lambda = old_lambda - LCPRelaxation * C / diag;
                   new_lambda       = new_lambda > 0.0 ? new_lambda : 0.0;
                   Float delta_lambda = new_lambda - old_lambda;
                   lambdas(c)         = new_lambda;

                   if(delta_lambda == 0.0)
                       return;

                   for(IndexT local_i = 0; local_i < 4; ++local_i)
                   {
                       IndexT v = ids(local_i);
                       if(v < 0 || ws(local_i) == 0.0)
                           continue;

                       Float inv_mass = mass_info.inv_mass(v);
                       if(inv_mass > 0.0)
                       {
                           auto& correction = corrections(v);
                           muda::eigen::atomic_add(
                               correction,
                               ((ws(local_i) * inv_mass * delta_lambda) * N).eval());
                       }
                   }
               });

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(context.y.size(),
               [y = context.y.viewer().name("y"),
                corrections =
                    context.backward_corrections.viewer().name("backward_corrections")] __device__(int v) mutable
               {
                   y(v) += corrections(v);
               });
}

template <typename MassInfo>
void compute_lcp_diagnostics(TWPContext&        context,
                             TWPConstraintSet& constraints,
                             MassInfo          mass_info)
{
    const IndexT constraint_count = constraints.host_total_constraint_count();
    context.diagnostics.backward.backward_violations.fill(0.0);
    context.diagnostics.backward.lcp_gaps.fill(0.0);
    context.diagnostics.backward.lcp_complementarity.fill(0.0);
    context.diagnostics.backward.lcp_projected_residual.fill(0.0);

    using namespace muda;
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(constraint_count,
               [vertex_ids = constraints.vertex_ids.viewer().name("vertex_ids"),
                weights = constraints.weights.viewer().name("weights"),
                normals = constraints.normals.viewer().name("normals"),
                offsets = constraints.offsets.viewer().name("offsets"),
                y = context.y.viewer().name("y"),
                lambdas = context.backward_lambdas.viewer().name("backward_lambdas"),
                backward_violations =
                    context.diagnostics.backward.backward_violations.viewer().name(
                        "backward_violations"),
                gaps = context.diagnostics.backward.lcp_gaps.viewer().name("lcp_gaps"),
                complementarity =
                    context.diagnostics.backward.lcp_complementarity.viewer().name(
                        "lcp_complementarity"),
                projected_residual =
                    context.diagnostics.backward.lcp_projected_residual.viewer().name(
                        "lcp_projected_residual"),
                mass_info] __device__(
                   int c) mutable
               {
                   Vector4i ids = vertex_ids(c);
                   Vector4  ws  = weights(c);
                   Vector3  N   = normals(c);
                   Float    C   = -offsets(c);
                   Float    diag = 0.0;
                   Float    n2   = N.squaredNorm();

                   if(n2 <= 1e-24)
                       return;

                   for(IndexT local_i = 0; local_i < 4; ++local_i)
                   {
                       IndexT v = ids(local_i);
                       if(v >= 0 && ws(local_i) != 0.0)
                       {
                           C += ws(local_i) * y(v).dot(N);

                           Float inv_mass = mass_info.inv_mass(v);
                           if(inv_mass > 0.0)
                               diag += ws(local_i) * ws(local_i) * inv_mass * n2;
                       }
                   }

                   Float lambda = lambdas(c);
                   backward_violations(c) = C < 0.0 ? -C : 0.0;
                   gaps(c) = C;
                   complementarity(c) = lambda * C < 0.0 ? -lambda * C : lambda * C;

                   if(diag > 0.0)
                   {
                       Float projected =
                           lambda - LCPRelaxation * C / diag;
                       projected = projected > 0.0 ? projected : 0.0;
                       Float residual = lambda - projected;
                       projected_residual(c) = residual < 0.0 ? -residual : residual;
                   }
               });
}

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

    bool use_fem_mass = info.finite_element_method && info.finite_element_vertex_reporter;
    for(IndexT iter = 0; iter < BackwardMaxIterations; ++iter)
    {
        if(use_fem_mass)
        {
            TWPFEMMassInfo mass_info{
                info.finite_element_method->masses().cviewer().name("masses"),
                info.finite_element_method->is_fixed().cviewer().name("is_fixed"),
                info.finite_element_vertex_reporter->vertex_offset(),
                info.finite_element_method->xs().size()};
            solve_lcp_iteration(context, constraints, mass_info);
            compute_lcp_diagnostics(context, constraints, mass_info);
        }
        else
        {
            TWPUnitMassInfo mass_info;
            solve_lcp_iteration(context, constraints, mass_info);
            compute_lcp_diagnostics(context, constraints, mass_info);
        }

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
        context.diagnostics.backward.iterations = iter + 1;
        if(context.diagnostics.backward.violation_inf < BackwardTolerance
           && context.diagnostics.backward.lcp_projected_residual_inf < BackwardTolerance)
        {
            context.diagnostics.backward.converged = true;
            break;
        }
        context.diagnostics.backward.converged = false;
    }
}
}  // namespace uipc::backend::cuda
