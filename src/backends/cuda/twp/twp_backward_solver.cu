#include <twp/twp_backward_solver.h>
#include <twp/twp_constraint_set.h>
#include <twp/twp_context.h>
#include <global_geometry/global_vertex_manager.h>
#include <finite_element/finite_element_method.h>
#include <finite_element/finite_element_vertex_reporter.h>
#include <uipc/common/timer.h>
#include <muda/atomic.h>
#include <muda/buffer/buffer_launch.h>
#include <muda/launch/parallel_for.h>
#include <muda/cub/device/device_reduce.h>
#include <muda/cub/device/device_scan.h>
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

namespace
{
MUDA_GENERIC IndexT hash_contact_color(IndexT constraint_id, IndexT round, IndexT palette_size)
{
    U64 x = static_cast<U64>(constraint_id) + 0x9e3779b97f4a7c15ull;
    x ^= static_cast<U64>(round) + 0xbf58476d1ce4e5b9ull + (x << 6) + (x >> 2);
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ull;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebull;
    x ^= x >> 31;
    return palette_size > 0 ? static_cast<IndexT>(x % static_cast<U64>(palette_size)) : 0;
}

template <typename VertexIdsView, typename CountsView, typename OffsetsView>
void build_self_contact_vertex_incidents(VertexIdsView vertex_ids,
                                         IndexT        constraint_offset,
                                         IndexT        contact_count,
                                         CountsView    counts,
                                         OffsetsView   offsets)
{
    using namespace muda;
    if(contact_count <= 0)
        return;

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(contact_count,
               [vertex_ids = vertex_ids.viewer().name("vertex_ids"),
                counts = counts.viewer().name("counts"),
                constraint_offset] __device__(int local_c) mutable
               {
                   Vector4i ids = vertex_ids(constraint_offset + local_c);
                   for(IndexT i = 0; i < 4; ++i)
                   {
                       IndexT v = ids(i);
                       if(v >= 0)
                           atomicAdd(counts.data() + v, 1);
                   }
               });

    muda::DeviceScan().ExclusiveSum(counts.data(), offsets.data(), counts.size());
}

template <typename VertexIdsView, typename CountsView, typename OffsetsView, typename IncidentsView>
void fill_self_contact_vertex_incidents(VertexIdsView vertex_ids,
                                        IndexT        constraint_offset,
                                        IndexT        contact_count,
                                        CountsView    counts,
                                        OffsetsView   offsets,
                                        IncidentsView incidents)
{
    using namespace muda;
    if(contact_count <= 0)
        return;

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(contact_count,
               [vertex_ids = vertex_ids.viewer().name("vertex_ids"),
                counts = counts.viewer().name("counts"),
                offsets = offsets.viewer().name("offsets"),
                incidents = incidents.viewer().name("incidents"),
                constraint_offset] __device__(int local_c) mutable
               {
                   Vector4i ids = vertex_ids(constraint_offset + local_c);
                   for(IndexT i = 0; i < 4; ++i)
                   {
                       IndexT v = ids(i);
                       if(v < 0)
                           continue;

                       IndexT slot = atomicAdd(counts.data() + v, 1);
                       incidents(offsets(v) + slot) = local_c;
                   }
               });
}

template <typename VertexIdsView,
          typename OffsetsView,
          typename IncidentsView,
          typename ColorsView,
          typename ActiveFlagsView,
          typename CandidateColorView,
          typename PriorityView>
void propose_self_contact_colors(VertexIdsView       vertex_ids,
                                 IndexT             constraint_offset,
                                 IndexT             contact_count,
                                 IndexT             round,
                                 IndexT             palette_size,
                                 OffsetsView        vertex_offsets,
                                 IncidentsView      vertex_incidents,
                                 ColorsView         colors,
                                 ActiveFlagsView    active_flags,
                                 CandidateColorView candidates,
                                 PriorityView       priorities)
{
    using namespace muda;
    if(contact_count <= 0)
        return;

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(contact_count,
               [vertex_ids = vertex_ids.viewer().name("vertex_ids"),
                vertex_offsets = vertex_offsets.viewer().name("vertex_offsets"),
                vertex_incidents = vertex_incidents.viewer().name("vertex_incidents"),
                colors = colors.viewer().name("colors"),
                active_flags = active_flags.viewer().name("active_flags"),
                candidates = candidates.viewer().name("candidates"),
                priorities = priorities.viewer().name("priorities"),
                constraint_offset,
                contact_count,
                round,
                palette_size] __device__(int local_c) mutable
               {
                   if(active_flags(local_c) == 0 || colors(local_c) >= 0)
                   {
                       candidates(local_c) = -1;
                       priorities(local_c) = -1;
                       return;
                   }

                   candidates(local_c) = hash_contact_color(local_c, round, palette_size);
                   priorities(local_c) = local_c;
               });
}

template <typename VertexIdsView,
          typename OffsetsView,
          typename IncidentsView,
          typename ColorsView,
          typename ActiveFlagsView,
          typename CandidateColorView,
          typename PriorityView>
void resolve_self_contact_color_conflicts(VertexIdsView       vertex_ids,
                                          IndexT             constraint_offset,
                                          IndexT             contact_count,
                                          OffsetsView        vertex_offsets,
                                          IncidentsView      vertex_incidents,
                                          ColorsView         colors,
                                          ActiveFlagsView    active_flags,
                                          CandidateColorView candidates,
                                          PriorityView       priorities)
{
    using namespace muda;
    if(contact_count <= 0)
        return;

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(contact_count,
               [vertex_ids = vertex_ids.viewer().name("vertex_ids"),
                vertex_offsets = vertex_offsets.viewer().name("vertex_offsets"),
                vertex_incidents = vertex_incidents.viewer().name("vertex_incidents"),
                colors = colors.viewer().name("colors"),
                active_flags = active_flags.viewer().name("active_flags"),
                candidates = candidates.viewer().name("candidates"),
                priorities = priorities.viewer().name("priorities"),
                constraint_offset] __device__(int local_c) mutable
               {
                   if(active_flags(local_c) == 0 || colors(local_c) >= 0)
                       return;

                   Vector4i ids = vertex_ids(constraint_offset + local_c);
                   IndexT   candidate = candidates(local_c);
                   IndexT   priority  = priorities(local_c);
                   if(candidate < 0)
                       return;

                   bool     winner = true;

                   for(IndexT local_i = 0; local_i < 4; ++local_i)
                   {
                       IndexT v = ids(local_i);
                       if(v < 0)
                           continue;

                       IndexT begin = vertex_offsets(v);
                       IndexT end   = vertex_offsets(v + 1);
                       for(IndexT p = begin; p < end; ++p)
                       {
                           IndexT other = vertex_incidents(p);
                           if(other == local_c)
                               continue;

                           Vector4i other_ids = vertex_ids(constraint_offset + other);
                           bool     share     = false;
                           for(IndexT k = 0; k < 4; ++k)
                           {
                               IndexT ov = other_ids(k);
                               if(ov < 0)
                                   continue;
                               for(IndexT j = 0; j < 4; ++j)
                               {
                                   if(ids(j) == ov)
                                   {
                                       share = true;
                                       break;
                                   }
                               }
                               if(share)
                                   break;
                           }

                           if(!share)
                               continue;

                           IndexT other_color = colors(other);
                           if(other_color == candidate)
                           {
                               winner = false;
                               break;
                           }

                           if(other_color >= 0)
                               continue;

                           if(active_flags(other) == 0)
                               continue;

                           if(candidates(other) == candidate
                              && priorities(other) > priority)
                           {
                               winner = false;
                               break;
                           }
                       }

                       if(!winner)
                           break;
                   }

                   if(winner && atomicCAS(colors.data() + local_c, IndexT{-1}, candidate) == -1)
                   {
                       active_flags(local_c) = 0;
                   }
               });
}

template <typename ColorsView, typename ActiveFlagsView>
IndexT count_active_uncolored(ColorsView colors, ActiveFlagsView active_flags, IndexT count)
{
    using namespace muda;
    DeviceVar<IndexT> active_count{0};
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(count,
               [colors = colors.viewer().name("colors"),
                active_flags = active_flags.viewer().name("active_flags"),
                active_count = active_count.viewer().name("active_count")] __device__(
                   int i) mutable
               {
                   if(active_flags(i) != 0 && colors(i) < 0)
                       atomicAdd(active_count.data(), 1);
               });
    return active_count;
}

template <typename ColorsView, typename CountsView>
void count_self_contact_colors(ColorsView colors, IndexT contact_count, CountsView counts)
{
    using namespace muda;
    if(contact_count <= 0)
        return;

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(contact_count,
               [colors = colors.viewer().name("colors"),
                counts = counts.viewer().name("counts")] __device__(int local_c) mutable
               {
                   IndexT color = colors(local_c);
                   if(color >= 0)
                       atomicAdd(counts.data() + color, 1);
               });
}

template <typename ColorsView, typename OffsetsView, typename CursorsView, typename CompactIdsView>
void scatter_self_contact_color_groups(ColorsView    colors,
                                       IndexT        contact_count,
                                       OffsetsView   offsets,
                                       CursorsView   cursors,
                                       CompactIdsView compact_ids)
{
    using namespace muda;
    if(contact_count <= 0)
        return;

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(contact_count,
               [colors = colors.viewer().name("colors"),
                offsets = offsets.viewer().name("offsets"),
                cursors = cursors.viewer().name("cursors"),
                compact_ids = compact_ids.viewer().name("compact_ids")] __device__(
                   int local_c) mutable
               {
                   IndexT color = colors(local_c);
                   if(color < 0)
                       return;

                   IndexT slot = atomicAdd(cursors.data() + color, 1);
                   compact_ids(offsets(color) + slot) = local_c;
               });
}

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

void TWPBackwardSolver::update_self_contact_coloring_cpu(TWPConstraintSet& constraints)
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

    Timer timer{"TWP Build Self Contact LCP Colors CPU"};

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

void TWPBackwardSolver::update_self_contact_coloring_gpu(TWPConstraintSet& constraints)
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

    Timer timer{"TWP Build Self Contact LCP Colors GPU"};
    const IndexT constraint_offset = constraints.host_self_contact_constraint_offset();
    m_host_self_contact_vertex_ids.resize(contact_count);
    muda::BufferLaunch()
        .copy<Vector4i>(m_host_self_contact_vertex_ids.data(),
                        std::as_const(constraints.vertex_ids)
                            .view(constraint_offset, contact_count))
        .wait();

    IndexT max_vertex = -1;
    for(const auto& ids : m_host_self_contact_vertex_ids)
        for(IndexT i = 0; i < 4; ++i)
            if(ids(i) > max_vertex)
                max_vertex = ids(i);
    if(max_vertex < 0)
        max_vertex = 0;

    m_self_contact_colors.resize(contact_count);
    m_self_contact_colors.fill(-1);
    m_self_contact_active_flags.resize(contact_count);
    m_self_contact_active_flags.fill(1);
    m_self_contact_vertex_counts.resize(static_cast<SizeT>(max_vertex) + 2);
    m_self_contact_vertex_counts.fill(0);
    m_self_contact_vertex_offsets.resize(static_cast<SizeT>(max_vertex) + 2);
    m_self_contact_vertex_incidents.resize(contact_count * 4);
    m_self_contact_candidate_colors.resize(contact_count);
    m_self_contact_candidate_priorities.resize(contact_count);
    m_self_contact_color_counts.resize(contact_count + 1);
    m_self_contact_color_offsets.resize(contact_count + 1);
    m_self_contact_color_cursors.resize(contact_count + 1);

    m_host_self_contact_color_offsets.clear();
    m_host_self_contact_color_offsets.push_back(0);

    build_self_contact_vertex_incidents(std::as_const(constraints.vertex_ids),
                                        constraint_offset,
                                        contact_count,
                                        m_self_contact_vertex_counts.view(),
                                        m_self_contact_vertex_offsets.view());
    m_self_contact_vertex_counts.fill(0);
    fill_self_contact_vertex_incidents(std::as_const(constraints.vertex_ids),
                                       constraint_offset,
                                       contact_count,
                                       m_self_contact_vertex_counts.view(),
                                       m_self_contact_vertex_offsets.view(),
                                       m_self_contact_vertex_incidents.view());

    const IndexT palette_size = contact_count;
    const IndexT max_rounds   = contact_count > 0 ? contact_count : 1;
    for(IndexT round = 0; round < max_rounds; ++round)
    {
        propose_self_contact_colors(std::as_const(constraints.vertex_ids),
                                    constraint_offset,
                                    contact_count,
                                    round,
                                    palette_size,
                                    m_self_contact_vertex_offsets.view(),
                                    m_self_contact_vertex_incidents.view(),
                                    m_self_contact_colors.view(),
                                    m_self_contact_active_flags.view(),
                                    m_self_contact_candidate_colors.view(),
                                    m_self_contact_candidate_priorities.view());
        resolve_self_contact_color_conflicts(std::as_const(constraints.vertex_ids),
                                             constraint_offset,
                                             contact_count,
                                             m_self_contact_vertex_offsets.view(),
                                             m_self_contact_vertex_incidents.view(),
                                             m_self_contact_colors.view(),
                                             m_self_contact_active_flags.view(),
                                             m_self_contact_candidate_colors.view(),
                                             m_self_contact_candidate_priorities.view());

        IndexT active_left = count_active_uncolored(
            m_self_contact_colors.view(), m_self_contact_active_flags.view(), contact_count);
        if(active_left == 0)
            break;
    }

    IndexT max_color = -1;
    muda::DeviceReduce().Max(m_self_contact_colors.data(), &max_color, contact_count);
    IndexT color_count = max_color >= 0 ? max_color + 1 : 0;
    m_host_self_contact_color_offsets.assign(static_cast<SizeT>(color_count) + 1, 0);
    if(color_count <= 0)
    {
        m_colored_contact_ids.resize(0);
        return;
    }

    m_self_contact_color_counts.fill(0);
    count_self_contact_colors(m_self_contact_colors.view(),
                              contact_count,
                              m_self_contact_color_counts.view());
    muda::DeviceScan().ExclusiveSum(m_self_contact_color_counts.data(),
                                    m_self_contact_color_offsets.data(),
                                    color_count + 1);
    muda::BufferLaunch()
        .copy<IndexT>(m_host_self_contact_color_offsets.data(),
                      std::as_const(m_self_contact_color_offsets).view(0, color_count + 1))
        .wait();

    m_self_contact_color_cursors.fill(0);
    m_colored_contact_ids.resize(contact_count);
    scatter_self_contact_color_groups(m_self_contact_colors.view(),
                                      contact_count,
                                      m_self_contact_color_offsets.view(),
                                      m_self_contact_color_cursors.view(),
                                      m_colored_contact_ids.view());
}

void TWPBackwardSolver::update_self_contact_coloring(TWPConstraintSet& constraints,
                                                     bool              use_gpu_coloring)
{
    if(use_gpu_coloring)
        update_self_contact_coloring_gpu(constraints);
    else
        update_self_contact_coloring_cpu(constraints);
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
