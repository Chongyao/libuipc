#include <twp/global_twp.h>
#include <twp/global_twp_impl.h>
#include <collision_detection/simplex_trajectory_filter.h>
#include <global_geometry/global_vertex_manager.h>
#include <utils/codim_thickness.h>
#include <utils/primitive_d_hat.h>
#include <utils/distance/distance_flagged.h>
#include <uipc/common/timer.h>
#include <muda/atomic.h>
#include <muda/launch/parallel_for.h>

namespace uipc::backend::cuda
{
namespace
{
MUDA_GENERIC Float simplex_linearized_offset(const Vector12& gradient,
                                             const Vector3&  x0,
                                             const Vector3&  x1,
                                             const Vector3&  x2,
                                             const Vector3&  x3,
                                             Float           distance2,
                                             Float           min_distance)
{
    Float constant = distance2 - min_distance * min_distance;
    constant -= gradient.segment<3>(0).dot(x0);
    constant -= gradient.segment<3>(3).dot(x1);
    constant -= gradient.segment<3>(6).dot(x2);
    constant -= gradient.segment<3>(9).dot(x3);
    return -constant;
}
}  // namespace

void GlobalTWP::Impl::append_simplex_contact_constraints(Float search_bound)
{
    Timer timer{"TWP Append Simplex Contact Constraints"};

    if(!simplex_trajectory_filter || !global_simplicial_surface_manager)
        return;

    auto PTs = simplex_trajectory_filter->candidate_PTs();
    auto EEs = simplex_trajectory_filter->candidate_EEs();
    if(PTs.size() == 0 && EEs.size() == 0)
        return;

    using namespace muda;
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(PTs.size(),
               [count = constraints.count.viewer().name("constraint_count"),
                types = constraints.types.viewer().name("constraint_types"),
                vertex_ids =
                    constraints.vertex_ids.viewer().name("constraint_vertex_ids"),
                weights = constraints.weights.viewer().name("constraint_weights"),
                normals = constraints.normals.viewer().name("constraint_normals"),
                offsets = constraints.offsets.viewer().name("constraint_offsets"),
                gradients =
                    constraints.gradients.viewer().name("constraint_gradients"),
                PTs = PTs.viewer().name("candidate_PTs"),
                surf_vertices =
                    global_simplicial_surface_manager->surf_vertices().viewer().name(
                        "surf_vertices"),
                surf_triangles =
                    global_simplicial_surface_manager->surf_triangles().viewer().name(
                        "surf_triangles"),
                x = context.x.viewer().name("x"),
                thicknesses =
                    global_vertex_manager->thicknesses().viewer().name("thicknesses"),
                d_hats =
                    global_vertex_manager->d_hats().viewer().name("d_hats"),
                search_bound] __device__(int idx) mutable
               {
                   Vector2i candidate = PTs(idx);
                   Vector3i tri       = surf_triangles(candidate(1));
                   Vector4i PT{surf_vertices(candidate(0)), tri(0), tri(1), tri(2)};

                   const Vector3& P  = x(PT(0));
                   const Vector3& T0 = x(PT(1));
                   const Vector3& T1 = x(PT(2));
                   const Vector3& T2 = x(PT(3));

                   Float thickness = PT_thickness(thicknesses(PT(0)),
                                                  thicknesses(PT(1)),
                                                  thicknesses(PT(2)),
                                                  thicknesses(PT(3)));
                   Float d_hat = PT_d_hat(
                       d_hats(PT(0)), d_hats(PT(1)), d_hats(PT(2)), d_hats(PT(3)));

                   Vector4i flag = distance::point_triangle_distance_flag(P, T0, T1, T2);
                   if(distance::detail::active_count(flag) != 4)
                       return;

                   Float distance2 = 0.0;
                   distance::point_triangle_distance2(flag, P, T0, T1, T2, distance2);

                   Float min_distance = thickness + d_hat;
                   Float search_radius = min_distance + search_bound;
                   if(distance2 > search_radius * search_radius)
                       return;

                   Vector12 gradient = Vector12::Zero();
                   distance::point_triangle_distance2_gradient(
                       flag, P, T0, T1, T2, gradient);

                   Float offset = simplex_linearized_offset(
                       gradient, P, T0, T1, T2, distance2, min_distance);

                   IndexT I = atomic_add(count.data(), 1);
                   write_simplex_contact_constraint(types,
                                                    vertex_ids,
                                                    weights,
                                                    normals,
                                                    offsets,
                                                    gradients,
                                                    I,
                                                    TWPConstraintType::PointTriangle,
                                                    PT,
                                                    gradient,
                                                    offset);
               });

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(EEs.size(),
               [count = constraints.count.viewer().name("constraint_count"),
                types = constraints.types.viewer().name("constraint_types"),
                vertex_ids =
                    constraints.vertex_ids.viewer().name("constraint_vertex_ids"),
                weights = constraints.weights.viewer().name("constraint_weights"),
                normals = constraints.normals.viewer().name("constraint_normals"),
                offsets = constraints.offsets.viewer().name("constraint_offsets"),
                gradients =
                    constraints.gradients.viewer().name("constraint_gradients"),
                EEs = EEs.viewer().name("candidate_EEs"),
                surf_edges =
                    global_simplicial_surface_manager->surf_edges().viewer().name(
                        "surf_edges"),
                x = context.x.viewer().name("x"),
                thicknesses =
                    global_vertex_manager->thicknesses().viewer().name("thicknesses"),
                d_hats =
                    global_vertex_manager->d_hats().viewer().name("d_hats"),
                search_bound] __device__(int idx) mutable
               {
                   Vector2i candidate = EEs(idx);
                   Vector2i e0        = surf_edges(candidate(0));
                   Vector2i e1        = surf_edges(candidate(1));
                   Vector4i EE{e0(0), e0(1), e1(0), e1(1)};

                   const Vector3& E0 = x(EE(0));
                   const Vector3& E1 = x(EE(1));
                   const Vector3& E2 = x(EE(2));
                   const Vector3& E3 = x(EE(3));

                   Float thickness = EE_thickness(thicknesses(EE(0)),
                                                  thicknesses(EE(1)),
                                                  thicknesses(EE(2)),
                                                  thicknesses(EE(3)));
                   Float d_hat = EE_d_hat(
                       d_hats(EE(0)), d_hats(EE(1)), d_hats(EE(2)), d_hats(EE(3)));

                   Vector4i flag = distance::edge_edge_distance_flag(E0, E1, E2, E3);
                   if(distance::detail::active_count(flag) != 4)
                       return;

                   Float distance2 = 0.0;
                   distance::edge_edge_distance2(flag, E0, E1, E2, E3, distance2);

                   Float min_distance = thickness + d_hat;
                   Float search_radius = min_distance + search_bound;
                   if(distance2 > search_radius * search_radius)
                       return;

                   Vector12 gradient = Vector12::Zero();
                   distance::edge_edge_distance2_gradient(
                       flag, E0, E1, E2, E3, gradient);

                   Float offset = simplex_linearized_offset(
                       gradient, E0, E1, E2, E3, distance2, min_distance);

                   IndexT I = atomic_add(count.data(), 1);
                   write_simplex_contact_constraint(types,
                                                    vertex_ids,
                                                    weights,
                                                    normals,
                                                    offsets,
                                                    gradients,
                                                    I,
                                                    TWPConstraintType::EdgeEdge,
                                                    EE,
                                                    gradient,
                                                    offset);
               });
}
}  // namespace uipc::backend::cuda
