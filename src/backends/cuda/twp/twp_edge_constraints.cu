#include <twp/global_twp.h>
#include <global_geometry/global_simplicial_surface_manager.h>
#include <uipc/common/timer.h>
#include <muda/launch/parallel_for.h>

namespace uipc::backend::cuda
{
void GlobalTWP::Impl::prepare_edge_reference_length_squares()
{
    Timer timer{"TWP Prepare Edge Reference Length Squares"};

    if(!global_simplicial_surface_manager)
        return;

    auto surf_edges = global_simplicial_surface_manager->surf_edges();
    context.target_edge_length_squares.resize(surf_edges.size());

    using namespace muda;
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(surf_edges.size(),
               [target_edge_length_squares = context.target_edge_length_squares.viewer().name(
                    "target_edge_length_squares"),
                edges = surf_edges.viewer().name("surf_edges"),
                target_y = context.target_y.viewer().name("target_y")] __device__(
                   int e) mutable
               {
                   Vector2i E = edges(e);
                   target_edge_length_squares(e) =
                       (target_y(E.x()) - target_y(E.y())).squaredNorm();
               });
}

void GlobalTWP::Impl::refresh_edge_constraints()
{
    Timer timer{"TWP Refresh Edge Constraints"};

    if(!global_simplicial_surface_manager
       || constraints.host_contact_constraint_count() == 0)
        return;

    auto  surf_edges = global_simplicial_surface_manager->surf_edges();
    SizeT edge_count = surf_edges.size();
    if(edge_count == 0)
        return;

    UIPC_ASSERT(context.target_edge_length_squares.size() == edge_count,
                "TWP target edge length squares must be prepared before edge constraints.");

    IndexT edge_offset = constraints.host_edge_constraint_offset();
    Float  sigma = edge_sigma_attr ? edge_sigma_attr->view()[0] : 1.1;
    UIPC_ASSERT(sigma > 0.0, "contact/twp/edge_sigma must be positive.");

    using namespace muda;
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(edge_count,
               [types = constraints.types.viewer().name("constraint_types"),
                vertex_ids =
                    constraints.vertex_ids.viewer().name("constraint_vertex_ids"),
                weights = constraints.weights.viewer().name("constraint_weights"),
                normals = constraints.normals.viewer().name("constraint_normals"),
                offsets = constraints.offsets.viewer().name("constraint_offsets"),
                edges = surf_edges.viewer().name("surf_edges"),
                x = context.x.viewer().name("x"),
                target_edge_length_squares = context.target_edge_length_squares.viewer().name(
                    "target_edge_length_squares"),
                edge_offset,
                sigma] __device__(int e) mutable
               {
                   IndexT I = edge_offset + e;
                   Vector2i E = edges(e);
                   Float target_len2 = target_edge_length_squares(e);
                   types(I) = TWPConstraintType::EdgeLengthUpperBound;
                   vertex_ids(I) = Vector4i{E.x(), E.y(), -1, -1};
                   if(target_len2 > 1e-24)
                   {
                       Vector3 d = x(E.x()) - x(E.y());
                       Float d2 = d.squaredNorm();
                       Float rhs = sigma * sigma * target_len2 + d2;
                       weights(I) = Vector4{-2.0, 2.0, 0.0, 0.0};
                       normals(I) = d;
                       offsets(I) = -rhs;
                   }
                   else
                   {
                       weights(I) = Vector4::Zero();
                       normals(I) = Vector3::Zero();
                       offsets(I) = 0.0;
                   }
               });

    constraints.set_host_edge_constraint_count(edge_count);
}
}  // namespace uipc::backend::cuda
