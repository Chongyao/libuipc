#include <twp/global_twp.h>
#include <global_geometry/global_vertex_manager.h>
#include <global_geometry/global_simplicial_surface_manager.h>
#include <contact_system/global_contact_manager.h>
#include <implicit_geometry/half_plane.h>
#include <implicit_geometry/half_plane_vertex_reporter.h>
#include <uipc/common/timer.h>
#include <muda/launch/parallel_for.h>

namespace uipc::backend::cuda
{
void GlobalTWP::Impl::proximity_search(Float search_bound)
{
    Timer timer{"TWP Proximity Search"};

    constraints.clear();

    if(!half_plane || !half_plane_vertex_reporter || half_plane->positions().size() == 0)
        return;

    SizeT vertex_count = context.target_y.size();
    SizeT plane_count  = half_plane->positions().size();
    SizeT edge_count = global_simplicial_surface_manager ?
                           global_simplicial_surface_manager->surf_edges().size() :
                           0;
    SizeT max_count    = vertex_count * plane_count + edge_count;
    if(constraints.types.size() < max_count)
        constraints.resize(max_count);
    context.ensure_constraint_storage(max_count);

    IndexT plane_vertex_offset = half_plane_vertex_reporter->vertex_offset();

    using namespace muda;
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(vertex_count,
               [count = constraints.count.viewer().name("constraint_count"),
                types = constraints.types.viewer().name("constraint_types"),
                vertex_ids =
                    constraints.vertex_ids.viewer().name("constraint_vertex_ids"),
                weights = constraints.weights.viewer().name("constraint_weights"),
                normals = constraints.normals.viewer().name("constraint_normals"),
                offsets = constraints.offsets.viewer().name("constraint_offsets"),
                y = context.target_y.viewer().name("target_y"),
                thicknesses = global_vertex_manager->thicknesses().viewer().name("thicknesses"),
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
                search_bound] __device__(int v) mutable
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

                       Float signed_dist = (y(v) - P).dot(N);
                       Float min_dist    = thicknesses(v);

                       if(signed_dist < min_dist + search_bound)
                       {
                           IndexT I = atomic_add(count.data(), 1);
                           types(I) = TWPConstraintType::VertexHalfPlane;
                           vertex_ids(I) = Vector4i{v, -1, -1, -1};
                           weights(I)    = Vector4{1.0, 0.0, 0.0, 0.0};
                           normals(I)    = N;
                           offsets(I)    = P.dot(N) + min_dist;
                       }
                   }
               });

    constraints.set_host_contact_constraint_count(constraints.count);
}
}  // namespace uipc::backend::cuda
