#include <twp/global_twp.h>
#include <global_geometry/global_vertex_manager.h>
#include <utils/distance/distance_flagged.h>
#include <uipc/common/timer.h>
#include <muda/launch/parallel_for.h>
#include <muda/cub/device/device_reduce.h>

namespace uipc::backend::cuda
{
namespace
{
__forceinline__ __device__ bool atomic_min_positive(Float* address, Float value)
{
    if constexpr(sizeof(Float) == sizeof(double))
    {
        auto address_as_ull = reinterpret_cast<unsigned long long int*>(address);
        auto old            = *address_as_ull;
        auto assumed        = old;
        auto value_as_ull =
            static_cast<unsigned long long int>(__double_as_longlong(value));
        while(value_as_ull < old)
        {
            assumed = old;
            old     = atomicCAS(address_as_ull, assumed, value_as_ull);
            if(old == assumed)
                return true;
        }
    }
    else
    {
        auto address_as_ui = reinterpret_cast<unsigned int*>(address);
        auto old           = *address_as_ui;
        auto assumed       = old;
        auto value_as_ui   = __float_as_uint(static_cast<float>(value));
        while(value_as_ui < old)
        {
            assumed = old;
            old     = atomicCAS(address_as_ui, assumed, value_as_ui);
            if(old == assumed)
                return true;
        }
    }
    return false;
}

template <typename DistanceView, typename SourceView>
__forceinline__ __device__ void update_proximity_distance(DistanceView& distances,
                                                          SourceView&   sources,
                                                          IndexT        vertex,
                                                          Float         distance,
                                                          IndexT        constraint)
{
    if(atomic_min_positive(&distances(vertex), distance))
        sources(vertex) = constraint;
}
}  // namespace

void GlobalTWP::Impl::forward()
{
    Timer timer{"TWP Forward"};

    constexpr Float ForwardSafety = 0.99;

    context.proximity_distances.fill(Float{1e30});
    context.proximity_constraint_ids.fill(-1);
    context.diagnostics.forward.safe_step_alphas.fill(1.0);

    const IndexT constraint_count = constraints.host_total_constraint_count();
    if(constraint_count > 0)
    {
        using namespace muda;
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(constraint_count,
                   [types = constraints.types.viewer().name("constraint_types"),
                    vertex_ids =
                        constraints.vertex_ids.viewer().name("constraint_vertex_ids"),
                    normals = constraints.normals.viewer().name("constraint_normals"),
                    offsets = constraints.offsets.viewer().name("constraint_offsets"),
                    x = context.x.viewer().name("x"),
                    proximity_distances =
                        context.proximity_distances.viewer().name("proximity_distances"),
                    proximity_constraint_ids =
                        context.proximity_constraint_ids.viewer().name(
                            "proximity_constraint_ids"),
                    thicknesses =
                        global_vertex_manager->thicknesses().viewer().name("thicknesses")] __device__(
                       int i) mutable
                   {
                       TWPConstraintType type = types(i);
                       Vector4i ids = vertex_ids(i);

                       if(type == TWPConstraintType::VertexHalfPlane)
                       {
                           IndexT  v = ids.x();
                           Vector3 N = normals(i);
                           Float   plane_offset = offsets(i) - thicknesses(v);
                           Float   distance = x(v).dot(N) - plane_offset;
                           distance = distance > 0.0 ? distance : 0.0;
                           update_proximity_distance(proximity_distances,
                                                     proximity_constraint_ids,
                                                     v,
                                                     distance,
                                                     i);
                       }
                       else if(type == TWPConstraintType::PointTriangle)
                       {
                           const Vector3& P  = x(ids(0));
                           const Vector3& T0 = x(ids(1));
                           const Vector3& T1 = x(ids(2));
                           const Vector3& T2 = x(ids(3));

                           Vector4i flag =
                               distance::point_triangle_distance_flag(P, T0, T1, T2);
                           Float distance2 = 0.0;
                           distance::point_triangle_distance2(
                               flag, P, T0, T1, T2, distance2);
                           Float distance =
                               distance2 > 0.0 ? sqrt(distance2) : Float{0.0};

                           for(IndexT local_i = 0; local_i < 4; ++local_i)
                               update_proximity_distance(proximity_distances,
                                                         proximity_constraint_ids,
                                                         ids(local_i),
                                                         distance,
                                                         i);
                       }
                       else if(type == TWPConstraintType::EdgeEdge)
                       {
                           const Vector3& E0 = x(ids(0));
                           const Vector3& E1 = x(ids(1));
                           const Vector3& E2 = x(ids(2));
                           const Vector3& E3 = x(ids(3));

                           Vector4i flag =
                               distance::edge_edge_distance_flag(E0, E1, E2, E3);
                           Float distance2 = 0.0;
                           distance::edge_edge_distance2(
                               flag, E0, E1, E2, E3, distance2);
                           Float distance =
                               distance2 > 0.0 ? sqrt(distance2) : Float{0.0};

                           for(IndexT local_i = 0; local_i < 4; ++local_i)
                               update_proximity_distance(proximity_distances,
                                                         proximity_constraint_ids,
                                                         ids(local_i),
                                                         distance,
                                                         i);
                       }
                   });
    }

    using namespace muda;
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(context.x.size(),
               [x = context.x.viewer().name("x"),
                y = context.y.viewer().name("y"),
                residual = context.residual.viewer().name("residual"),
                forward_step_norms =
                    context.diagnostics.forward.step_norms.viewer().name("forward_step_norms"),
                limited_flags =
                    context.diagnostics.forward.limited_flags.viewer().name(
                        "forward_limited_flags"),
                proximity_distances =
                    context.proximity_distances.viewer().name("proximity_distances"),
                safe_step_alphas =
                    context.diagnostics.forward.safe_step_alphas.viewer().name(
                        "safe_step_alphas")] __device__(int i) mutable
               {
                   Vector3 old_x = x(i);
                   Vector3 dir   = y(i) - old_x;
                   Float   norm  = dir.norm();
                   Float   alpha_i = 1.0;

                   if(norm > 1e-24)
                   {
                       Float D_i = proximity_distances(i);
                       if(D_i < Float{1e29})
                           alpha_i = min(Float{1.0},
                                         Float{0.5} * ForwardSafety * D_i / norm);
                   }

                   safe_step_alphas(i) = alpha_i;
                   Vector3 step  = alpha_i * dir;
                   Vector3 new_x = old_x + step;
                   x(i)          = new_x;
                   residual(i) *= (1.0 - alpha_i);
                   forward_step_norms(i) = step.norm();
                   limited_flags(i) = alpha_i < 1.0 ? 1 : 0;
               });

    DeviceReduce().Min(context.diagnostics.forward.safe_step_alphas.data(),
                       context.diagnostics.forward.min_safe_step_alpha.data(),
                       context.diagnostics.forward.safe_step_alphas.size());
    DeviceReduce().Max(context.residual.data(),
                       context.diagnostics.forward.max_residual.data(),
                       context.residual.size());
    DeviceReduce().Max(context.diagnostics.forward.step_norms.data(),
                       context.diagnostics.forward.max_step_norm.data(),
                       context.diagnostics.forward.step_norms.size());
    DeviceReduce().Sum(context.diagnostics.forward.limited_flags.data(),
                       context.diagnostics.forward.limited_count.data(),
                       context.diagnostics.forward.limited_flags.size());

    context.diagnostics.forward.residual_inf     = context.diagnostics.forward.max_residual;
    context.diagnostics.forward.max_step = context.diagnostics.forward.max_step_norm;
    context.diagnostics.forward.min_alpha = context.diagnostics.forward.min_safe_step_alpha;
    context.diagnostics.forward.limited_vertices = context.diagnostics.forward.limited_count;
}
}  // namespace uipc::backend::cuda
