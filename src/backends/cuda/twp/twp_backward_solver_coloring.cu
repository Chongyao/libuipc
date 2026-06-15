#include <twp/twp_backward_solver.h>
#include <twp/twp_constraint_set.h>
#include <twp/twp_host_constraint_helpers.h>
#include <uipc/common/timer.h>
#include <muda/atomic.h>
#include <muda/buffer/buffer_launch.h>
#include <muda/launch/parallel_for.h>
#include <muda/cub/device/device_reduce.h>
#include <muda/cub/device/device_scan.h>
#include <unordered_set>

namespace uipc::backend::cuda
{
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

void validate_self_contact_coloring(const std::vector<Vector4i>& vertex_ids,
                                    const std::vector<IndexT>&   colors,
                                    const std::vector<IndexT>&   color_offsets,
                                    const std::vector<IndexT>&   colored_ids,
                                    IndexT                       contact_count)
{
    UIPC_ASSERT(static_cast<IndexT>(vertex_ids.size()) == contact_count,
                "TWP GPU self-contact coloring validation expected {} vertex ids, got {}.",
                contact_count,
                vertex_ids.size());
    UIPC_ASSERT(static_cast<IndexT>(colors.size()) == contact_count,
                "TWP GPU self-contact coloring validation expected {} colors, got {}.",
                contact_count,
                colors.size());
    UIPC_ASSERT(!color_offsets.empty(),
                "TWP GPU self-contact coloring produced no color offsets.");

    const IndexT color_count = static_cast<IndexT>(color_offsets.size()) - 1;
    const IndexT compact_count = color_offsets.back();
    UIPC_ASSERT(compact_count == contact_count,
                "TWP GPU self-contact coloring is incomplete: compacted {}, expected {}.",
                compact_count,
                contact_count);
    UIPC_ASSERT(static_cast<IndexT>(colored_ids.size()) == contact_count,
                "TWP GPU self-contact coloring expected {} compact ids, got {}.",
                contact_count,
                colored_ids.size());

    std::vector<IndexT> seen(static_cast<SizeT>(contact_count), 0);
    for(IndexT c = 0; c < contact_count; ++c)
    {
        IndexT color = colors[static_cast<SizeT>(c)];
        UIPC_ASSERT(color >= 0 && color < color_count,
                    "TWP GPU self-contact constraint {} has invalid color {} "
                    "(color_count={}).",
                    c,
                    color,
                    color_count);
    }

    for(IndexT color = 0; color < color_count; ++color)
    {
        IndexT begin = color_offsets[static_cast<SizeT>(color)];
        IndexT end   = color_offsets[static_cast<SizeT>(color + 1)];
        UIPC_ASSERT(begin <= end,
                    "TWP GPU self-contact color {} has invalid range [{}, {}).",
                    color,
                    begin,
                    end);
        UIPC_ASSERT(begin >= 0 && end <= contact_count,
                    "TWP GPU self-contact color {} range [{}, {}) is out of compact "
                    "count {}.",
                    color,
                    begin,
                    end,
                    contact_count);

        std::unordered_set<IndexT> color_vertices;
        for(IndexT p = begin; p < end; ++p)
        {
            IndexT c = colored_ids[static_cast<SizeT>(p)];
            UIPC_ASSERT(c >= 0 && c < contact_count,
                        "TWP GPU self-contact compact slot {} has invalid constraint "
                        "id {}.",
                        p,
                        c);
            UIPC_ASSERT(colors[static_cast<SizeT>(c)] == color,
                        "TWP GPU self-contact compact slot {} stores constraint {} "
                        "with color {}, expected {}.",
                        p,
                        c,
                        colors[static_cast<SizeT>(c)],
                        color);
            ++seen[static_cast<SizeT>(c)];

            const Vector4i ids = vertex_ids[static_cast<SizeT>(c)];
            for(IndexT local_i = 0; local_i < 4; ++local_i)
            {
                IndexT v = ids(local_i);
                if(v < 0)
                    continue;
                bool inserted = color_vertices.insert(v).second;
                UIPC_ASSERT(inserted,
                            "TWP GPU self-contact color {} is not conflict-free: "
                            "constraint {} shares vertex {} with another constraint "
                            "in the same color.",
                            color,
                            c,
                            v);
            }
        }
    }

    for(IndexT c = 0; c < contact_count; ++c)
    {
        UIPC_ASSERT(seen[static_cast<SizeT>(c)] == 1,
                    "TWP GPU self-contact constraint {} appears {} times in compact "
                    "color groups.",
                    c,
                    seen[static_cast<SizeT>(c)]);
    }
}

}  // namespace

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

    build_host_greedy_coloring(m_host_edge_vertex_ids,
                               edge_count,
                               m_host_edge_color_offsets,
                               m_host_colored_edge_ids);

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

    build_host_greedy_coloring(m_host_self_contact_vertex_ids,
                               contact_count,
                               m_host_self_contact_color_offsets,
                               m_host_colored_self_contact_ids);

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

    std::vector<IndexT> host_colors(static_cast<SizeT>(contact_count));
    std::vector<IndexT> host_colored_ids(static_cast<SizeT>(contact_count));
    muda::BufferLaunch()
        .copy<IndexT>(host_colors.data(),
                      std::as_const(m_self_contact_colors).view(0, contact_count))
        .copy<IndexT>(host_colored_ids.data(),
                      std::as_const(m_colored_contact_ids).view(0, contact_count))
        .wait();
    validate_self_contact_coloring(m_host_self_contact_vertex_ids,
                                   host_colors,
                                   m_host_self_contact_color_offsets,
                                   host_colored_ids,
                                   contact_count);
}

void TWPBackwardSolver::update_self_contact_coloring(TWPConstraintSet& constraints,
                                                     bool              use_gpu_coloring)
{
    if(use_gpu_coloring)
        update_self_contact_coloring_gpu(constraints);
    else
        update_self_contact_coloring_cpu(constraints);
}

}  // namespace uipc::backend::cuda
