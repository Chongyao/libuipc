#include <twp/global_twp.h>
#include <global_geometry/global_vertex_manager.h>
#include <global_geometry/global_simplicial_surface_manager.h>
#include <collision_detection/global_trajectory_filter.h>
#include <collision_detection/simplex_trajectory_filter.h>
#include <contact_system/global_contact_manager.h>
#include <implicit_geometry/half_plane.h>
#include <implicit_geometry/half_plane_vertex_reporter.h>
#include <uipc/common/timer.h>
#include <muda/launch/parallel_for.h>

namespace uipc::backend::cuda
{
void GlobalTWP::Impl::refresh_self_collision_candidates(Float search_bound)
{
    if(!global_trajectory_filter || !simplex_trajectory_filter)
        return;

    Timer timer{"TWP Refresh Self Collision Candidates"};

    global_vertex_manager->setup_ccd(context.x.view());
    global_trajectory_filter->detect(0.0, search_bound);
    global_vertex_manager->restore_ccd();
}

void GlobalTWP::Impl::proximity_search(Float obstacle_search_bound,
                                       Float self_collision_search_bound,
                                       bool  refresh_self_collision)
{
    Timer timer{"TWP Proximity Search"};

    constraints.clear();

    bool self_collision_enabled =
        !self_collision_enable_attr || self_collision_enable_attr->view()[0] != 0;
    if(self_collision_enabled && refresh_self_collision)
        refresh_self_collision_candidates(self_collision_search_bound);

    SizeT vertex_count = context.x.size();
    SizeT plane_count  = half_plane ? half_plane->positions().size() : 0;
    SizeT pt_count =
        simplex_trajectory_filter ? simplex_trajectory_filter->candidate_PTs().size() : 0;
    SizeT ee_count =
        simplex_trajectory_filter ? simplex_trajectory_filter->candidate_EEs().size() : 0;
    SizeT edge_count = global_simplicial_surface_manager ?
                           global_simplicial_surface_manager->surf_edges().size() :
                           0;
    SizeT max_count    = vertex_count * plane_count + pt_count + ee_count + edge_count;
    if(constraints.types.size() < max_count)
        constraints.resize(max_count);
    context.ensure_constraint_storage(max_count);

    if(half_plane && half_plane_vertex_reporter && plane_count > 0)
    {
        IndexT plane_vertex_offset = half_plane_vertex_reporter->vertex_offset();

        using namespace muda;
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(vertex_count,
                   [count = constraints.count.viewer().name("constraint_count"),
                    types = constraints.types.viewer().name("constraint_types"),
                    vertex_ids =
                        constraints.vertex_ids.viewer().name("constraint_vertex_ids"),
                    primitive_ids = constraints.primitive_ids.viewer().name(
                        "constraint_primitive_ids"),
                    weights = constraints.weights.viewer().name("constraint_weights"),
                    normals = constraints.normals.viewer().name("constraint_normals"),
                    offsets = constraints.offsets.viewer().name("constraint_offsets"),
                    gradients =
                        constraints.gradients.viewer().name("constraint_gradients"),
                    x = context.x.viewer().name("x"),
                    thicknesses =
                        global_vertex_manager->thicknesses().viewer().name("thicknesses"),
                    contact_ids =
                        global_vertex_manager->contact_element_ids().viewer().name("contact_ids"),
                    subscene_ids =
                        global_vertex_manager->subscene_element_ids().viewer().name("subscene_ids"),
                    contact_mask =
                        global_contact_manager->contact_mask_tabular().viewer().name("contact_mask"),
                    subscene_mask =
                        global_contact_manager->subscene_mask_tabular().viewer().name("subscene_mask"),
                    plane_positions = half_plane->positions().viewer().name("plane_positions"),
                    plane_normals = half_plane->normals().viewer().name("plane_normals"),
                    plane_vertex_offset,
                    plane_count,
                    obstacle_search_bound] __device__(int v) mutable
                   {
                       if(v >= plane_vertex_offset && v < plane_vertex_offset + plane_count)
                           return;

                       for(IndexT h = 0; h < plane_count; ++h)
                       {
                           IndexT plane_v = plane_vertex_offset + h;

                           IndexT L = contact_ids(v);
                           IndexT R = contact_ids(plane_v);
                           if(contact_mask(L, R) == 0)
                               continue;

                           IndexT sL = subscene_ids(v);
                           IndexT sR = subscene_ids(plane_v);
                           if(subscene_mask(sL, sR) == 0)
                               continue;

                           const Vector3& P = plane_positions(h);
                           const Vector3& N = plane_normals(h);

                           Float signed_dist = (x(v) - P).dot(N);
                           Float min_dist    = thicknesses(v);

                           if(signed_dist < min_dist + obstacle_search_bound)
                           {
                               IndexT I = atomic_add(count.data(), 1);
                               write_vertex_half_plane_constraint(types,
                                                                  vertex_ids,
                                                                  primitive_ids,
                                                                  weights,
                                                                  normals,
                                                                  offsets,
                                                                  gradients,
                                                                  I,
                                                                  v,
                                                                  h,
                                                                  N,
                                                                  P.dot(N) + min_dist);
                           }
                       }
                   });
    }

    IndexT obstacle_count = constraints.count;
    if(self_collision_enabled)
        append_simplex_contact_constraints(self_collision_search_bound);
    IndexT contact_count = constraints.count;
    constraints.set_host_contact_constraint_count(obstacle_count,
                                                  contact_count - obstacle_count);
}
}  // namespace uipc::backend::cuda
