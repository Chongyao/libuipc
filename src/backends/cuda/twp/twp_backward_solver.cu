#include <twp/twp_backward_solver.h>
#include <twp/twp_constraint_set.h>
#include <twp/twp_context.h>
#include <global_geometry/global_vertex_manager.h>
#include <finite_element/finite_element_method.h>
#include <finite_element/finite_element_vertex_reporter.h>
#include <uipc/common/timer.h>
#include <muda/buffer/buffer_launch.h>
#include <muda/launch/parallel_for.h>
#include <muda/cub/device/device_reduce.h>
#include <muda/ext/eigen/atomic.h>
#include <unordered_set>
#include <utility>

namespace uipc::backend::cuda
{
namespace
{
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

void TWPBackwardSolver::update_edge_coloring_if_needed(TWPConstraintSet& constraints)
{
    const IndexT edge_count = constraints.host_edge_constraint_count();
    if(m_coloring_edge_count == edge_count)
        return;

    Timer timer{"TWP Build Edge LCP Colors"};

    m_host_edge_vertex_ids.resize(edge_count);
    if(edge_count > 0)
    {
        muda::BufferLaunch()
            .copy<Vector4i>(m_host_edge_vertex_ids.data(),
                            std::as_const(constraints.vertex_ids)
                                .view(constraints.host_edge_constraint_offset(),
                                      edge_count))
            .wait();
    }

    std::vector<std::vector<IndexT>>       color_constraint_ids;
    std::vector<std::unordered_set<IndexT>> color_vertices;

    for(IndexT e = 0; e < edge_count; ++e)
    {
        Vector4i ids = m_host_edge_vertex_ids[e];
        IndexT   selected_color = -1;

        for(IndexT color = 0; color < static_cast<IndexT>(color_vertices.size());
            ++color)
        {
            bool conflict = false;
            for(IndexT local_i = 0; local_i < 4; ++local_i)
            {
                IndexT v = ids(local_i);
                if(v >= 0 && color_vertices[color].contains(v))
                {
                    conflict = true;
                    break;
                }
            }

            if(!conflict)
            {
                selected_color = color;
                break;
            }
        }

        if(selected_color < 0)
        {
            selected_color = static_cast<IndexT>(color_vertices.size());
            color_vertices.emplace_back();
            color_constraint_ids.emplace_back();
        }

        color_constraint_ids[selected_color].push_back(e);
        for(IndexT local_i = 0; local_i < 4; ++local_i)
        {
            IndexT v = ids(local_i);
            if(v >= 0)
                color_vertices[selected_color].insert(v);
        }
    }

    m_host_edge_color_offsets.clear();
    m_host_edge_color_offsets.reserve(color_constraint_ids.size() + 1);
    m_host_edge_color_offsets.push_back(0);

    m_host_colored_edge_ids.clear();
    m_host_colored_edge_ids.reserve(edge_count);
    for(const auto& ids : color_constraint_ids)
    {
        m_host_colored_edge_ids.insert(m_host_colored_edge_ids.end(),
                                       ids.begin(),
                                       ids.end());
        m_host_edge_color_offsets.push_back(
            static_cast<IndexT>(m_host_colored_edge_ids.size()));
    }

    m_colored_edge_ids.resize(m_host_colored_edge_ids.size());
    if(!m_host_colored_edge_ids.empty())
    {
        muda::BufferLaunch()
            .copy<IndexT>(m_colored_edge_ids.view(), m_host_colored_edge_ids.data())
            .wait();
    }

    m_coloring_edge_count = edge_count;
}

void TWPBackwardSolver::update_self_contact_coloring(TWPConstraintSet& constraints)
{
    const IndexT contact_count = constraints.host_self_contact_constraint_count();
    if(contact_count == 0)
    {
        m_host_colored_self_contact_ids.clear();
        m_host_self_contact_color_offsets.clear();
        m_host_self_contact_color_offsets.push_back(0);
        m_colored_contact_ids.resize(0);
        return;
    }

    Timer        timer{"TWP Build Self Contact LCP Colors"};

    m_host_self_contact_vertex_ids.resize(contact_count);
    muda::BufferLaunch()
        .copy<Vector4i>(m_host_self_contact_vertex_ids.data(),
                        std::as_const(constraints.vertex_ids)
                            .view(constraints.host_self_contact_constraint_offset(),
                                  contact_count))
        .wait();

    std::vector<std::vector<IndexT>>        color_constraint_ids;
    std::vector<std::unordered_set<IndexT>> color_vertices;

    for(IndexT c = 0; c < contact_count; ++c)
    {
        Vector4i ids = m_host_self_contact_vertex_ids[c];
        IndexT   selected_color = -1;

        for(IndexT color = 0; color < static_cast<IndexT>(color_vertices.size());
            ++color)
        {
            bool conflict = false;
            for(IndexT local_i = 0; local_i < 4; ++local_i)
            {
                IndexT v = ids(local_i);
                if(v >= 0 && color_vertices[color].contains(v))
                {
                    conflict = true;
                    break;
                }
            }

            if(!conflict)
            {
                selected_color = color;
                break;
            }
        }

        if(selected_color < 0)
        {
            selected_color = static_cast<IndexT>(color_vertices.size());
            color_vertices.emplace_back();
            color_constraint_ids.emplace_back();
        }

        color_constraint_ids[selected_color].push_back(c);
        for(IndexT local_i = 0; local_i < 4; ++local_i)
        {
            IndexT v = ids(local_i);
            if(v >= 0)
                color_vertices[selected_color].insert(v);
        }
    }

    m_host_self_contact_color_offsets.clear();
    m_host_self_contact_color_offsets.reserve(color_constraint_ids.size() + 1);
    m_host_self_contact_color_offsets.push_back(0);

    m_host_colored_self_contact_ids.clear();
    m_host_colored_self_contact_ids.reserve(contact_count);
    for(const auto& ids : color_constraint_ids)
    {
        m_host_colored_self_contact_ids.insert(m_host_colored_self_contact_ids.end(),
                                               ids.begin(),
                                               ids.end());
        m_host_self_contact_color_offsets.push_back(
            static_cast<IndexT>(m_host_colored_self_contact_ids.size()));
    }

    m_colored_contact_ids.resize(m_host_colored_self_contact_ids.size());
    if(!m_host_colored_self_contact_ids.empty())
    {
        muda::BufferLaunch()
            .copy<IndexT>(m_colored_contact_ids.view(),
                          m_host_colored_self_contact_ids.data())
            .wait();
    }
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

    update_edge_coloring_if_needed(constraints);
    update_self_contact_coloring(constraints);

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
        context.diagnostics.backward.backward_violations.fill(0.0);
        context.diagnostics.backward.lcp_gaps.fill(0.0);
        context.diagnostics.backward.lcp_complementarity.fill(0.0);
        context.diagnostics.backward.lcp_projected_residual.fill(0.0);
        context.diagnostics.backward.converged = true;
        context.diagnostics.backward.violation_inf = 0.0;
        context.diagnostics.backward.lcp_min_gap = 0.0;
        context.diagnostics.backward.lcp_complementarity_inf = 0.0;
        context.diagnostics.backward.lcp_projected_residual_inf = 0.0;
    }
}
}  // namespace uipc::backend::cuda
