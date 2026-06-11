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

    // Paper constraint:
    //   c_i(x_i, x_j) = sigma - ||x_i - x_j|| / ||y_i^{k+1} - y_j^{k+1}|| >= 0.
    // In the backward step, x is a constant, while y is the optimization variable.
    // For this edge constraint we use l = ||y_i^0 - y_j^0|| as the constant
    // reference length, giving the squared lower-bound form:
    //   F(y) = sigma^2 ||y_i - y_j||^2 - l^2 >= 0.
    //
    // The LCP stores a linearized constraint C(y) >= 0. At the current backward
    // iterate y^k, let d_k = y_i^k - y_j^k. Linearizing F gives:
    //   F(y) ~= 2 sigma^2 d_k dot (y_i - y_j)
    //           - (sigma^2 ||d_k||^2 + l^2) >= 0.
    // Therefore the gradient is built from the current y, and l^2 remains the
    // fixed target/reference length. The mutable forward state x is not used here.
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
                gradients =
                    constraints.gradients.viewer().name("constraint_gradients"),
                edges = surf_edges.viewer().name("surf_edges"),
                y = context.y.viewer().name("y"),
                target_edge_length_squares =
                    context.target_edge_length_squares.viewer().name(
                        "target_edge_length_squares"),
                edge_offset,
                sigma] __device__(int e) mutable
               {
                   IndexT I = edge_offset + e;
                   Vector2i E = edges(e);
                   Float target_len2 = target_edge_length_squares(e);
                   if(target_len2 > 1e-24)
                   {
                       Float sigma2 = sigma * sigma;
                       Vector3 y_d = y(E.x()) - y(E.y());
                       Float y_len2 = y_d.squaredNorm();
                       Vector3 gradient_d = sigma2 * y_d;
                       Float rhs = sigma2 * y_len2 + target_len2;
                       write_edge_length_lower_bound_constraint(types,
                                                                vertex_ids,
                                                                weights,
                                                                normals,
                                                                offsets,
                                                                gradients,
                                                                I,
                                                                E,
                                                                gradient_d,
                                                                rhs);
                   }
                   else
                   {
                       write_disabled_edge_length_lower_bound_constraint(types,
                                                                         vertex_ids,
                                                                         weights,
                                                                         normals,
                                                                         offsets,
                                                                         gradients,
                                                                         I);
                   }
               });

    constraints.set_host_edge_constraint_count(edge_count);
}
}  // namespace uipc::backend::cuda
