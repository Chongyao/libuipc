#pragma once
#include <twp/twp_constraint_set.h>
#include <twp/twp_context.h>
#include <muda/atomic.h>
#include <muda/launch/parallel_for.h>
#include <muda/cub/device/device_reduce.h>
#include <muda/ext/eigen/atomic.h>
#include <vector>

namespace uipc::backend::cuda
{
namespace twp_backward_solver_lcp
{
inline constexpr Float BackwardTolerance = 1e-8;
inline constexpr Float LCPRelaxation     = 1.0;

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

template <typename VertexIdView,
          typename OffsetView,
          typename GradientView,
          typename PositionView,
          typename LambdaView,
          typename MassInfo>
MUDA_GENERIC void solve_lcp_constraint(IndexT        c,
                                       VertexIdView& vertex_ids,
                                       OffsetView&   offsets,
                                       GradientView& gradients,
                                       PositionView& y,
                                       LambdaView&   lambdas,
                                       MassInfo      mass_info)
{
    Vector4i ids = vertex_ids(c);
    Vector12 G   = gradients(c);
    Float    C   = -offsets(c);
    Float    diag = 0.0;

    for(IndexT local_i = 0; local_i < 4; ++local_i)
    {
        IndexT v = ids(local_i);
        Vector3 g = G.segment<3>(local_i * 3);
        if(v < 0 || g.squaredNorm() <= 1e-24)
            continue;

        C += g.dot(y(v));

        Float inv_mass = mass_info.inv_mass(v);
        if(inv_mass > 0.0)
            diag += inv_mass * g.squaredNorm();
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
        Vector3 g = G.segment<3>(local_i * 3);
        if(v < 0 || g.squaredNorm() <= 1e-24)
            continue;

        Float inv_mass = mass_info.inv_mass(v);
        if(inv_mass > 0.0)
            y(v) += inv_mass * delta_lambda * g;
    }
}

template <typename MassInfo>
void solve_obstacle_constraints_jacobi(TWPContext&       context,
                                       TWPConstraintSet& constraints,
                                       IndexT            constraint_offset,
                                       IndexT            constraint_count,
                                       MassInfo          mass_info)
{
    if(constraint_count <= 0)
        return;

    context.backward_corrections.fill(Vector3::Zero());

    using namespace muda;
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(constraint_count,
               [vertex_ids = constraints.vertex_ids.viewer().name("vertex_ids"),
                offsets = constraints.offsets.viewer().name("offsets"),
                gradients = constraints.gradients.viewer().name("gradients"),
                y = context.y.viewer().name("y"),
                corrections =
                    context.backward_corrections.viewer().name("backward_corrections"),
                lambdas = context.backward_lambdas.viewer().name("backward_lambdas"),
                constraint_offset,
                mass_info] __device__(int local_c) mutable
               {
                   IndexT   c   = constraint_offset + local_c;
                   Vector4i ids = vertex_ids(c);
                   Vector12 G   = gradients(c);
                   Float    C   = -offsets(c);
                   Float    diag = 0.0;

                   for(IndexT local_i = 0; local_i < 4; ++local_i)
                   {
                       IndexT v = ids(local_i);
                       Vector3 g = G.segment<3>(local_i * 3);
                       if(v < 0 || g.squaredNorm() <= 1e-24)
                           continue;

                       C += g.dot(y(v));

                       Float inv_mass = mass_info.inv_mass(v);
                       if(inv_mass > 0.0)
                           diag += inv_mass * g.squaredNorm();
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
                       Vector3 g = G.segment<3>(local_i * 3);
                       if(v < 0 || g.squaredNorm() <= 1e-24)
                           continue;

                       Float inv_mass = mass_info.inv_mass(v);
                       if(inv_mass <= 0.0)
                           continue;

                       auto& dst = corrections(v);
                       muda::eigen::atomic_add(
                           dst, (inv_mass * delta_lambda * g).eval());
                   }
               });

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(context.y.size(),
               [y = context.y.viewer().name("y"),
                corrections = context.backward_corrections.viewer().name(
                    "backward_corrections")] __device__(int v) mutable
               {
                   y(v) += corrections(v);
               });
}

template <typename MassInfo>
void solve_constraints_colored(TWPContext&               context,
                               TWPConstraintSet&         constraints,
                               muda::CBufferView<IndexT> colored_constraint_ids,
                               const std::vector<IndexT>& color_offsets,
                               IndexT                    constraint_offset,
                               MassInfo                  mass_info)
{
    for(SizeT color = 0; color + 1 < color_offsets.size(); ++color)
    {
        IndexT begin = color_offsets[color];
        IndexT end   = color_offsets[color + 1];
        IndexT count = end - begin;
        if(count == 0)
            continue;

        using namespace muda;
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(count,
                   [constraint_ids =
                        colored_constraint_ids.viewer().name("colored_constraint_ids"),
                    vertex_ids = constraints.vertex_ids.viewer().name("vertex_ids"),
                    offsets = constraints.offsets.viewer().name("offsets"),
                    gradients = constraints.gradients.viewer().name("gradients"),
                    y = context.y.viewer().name("y"),
                    lambdas =
                        context.backward_lambdas.viewer().name("backward_lambdas"),
                    constraint_offset,
                    begin,
                    mass_info] __device__(int local_c) mutable
                   {
                       IndexT constraint_id = constraint_ids(begin + local_c);
                       solve_lcp_constraint(constraint_offset + constraint_id,
                                            vertex_ids,
                                            offsets,
                                            gradients,
                                            y,
                                            lambdas,
                                            mass_info);
                   });
    }
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
                offsets = constraints.offsets.viewer().name("offsets"),
                gradients = constraints.gradients.viewer().name("gradients"),
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
                   Vector12 G   = gradients(c);
                   Float    C   = -offsets(c);
                   Float    diag = 0.0;

                   for(IndexT local_i = 0; local_i < 4; ++local_i)
                   {
                       IndexT v = ids(local_i);
                       Vector3 g = G.segment<3>(local_i * 3);
                       if(v >= 0 && g.squaredNorm() > 1e-24)
                       {
                           C += g.dot(y(v));

                           Float inv_mass = mass_info.inv_mass(v);
                           if(inv_mass > 0.0)
                               diag += inv_mass * g.squaredNorm();
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

}  // namespace twp_backward_solver_lcp
}  // namespace uipc::backend::cuda
