#include <twp/twp_backward_solver.h>
#include <twp/twp_constraint_set.h>
#include <twp/twp_context.h>
#include <global_geometry/global_vertex_manager.h>
#include <finite_element/finite_element_method.h>
#include <finite_element/finite_element_vertex_reporter.h>
#include <uipc/common/timer.h>
#include <muda/launch/parallel_for.h>
#include <muda/cub/device/device_reduce.h>

namespace uipc::backend::cuda
{
void TWPBackwardSolver::solve(SolveInfo info)
{
    Timer timer{"TWP Backward"};

    auto& context     = *info.context;
    auto& constraints = *info.constraints;

    if(constraints.h_count == 0)
    {
        context.backward_violation_inf = 0.0;
        return;
    }

    context.backward_violations.fill(0.0);

    using namespace muda;
    if(info.finite_element_method && info.finite_element_vertex_reporter)
    {
        IndexT fem_vertex_offset = info.finite_element_vertex_reporter->vertex_offset();
        SizeT  fem_vertex_count  = info.finite_element_method->xs().size();

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(constraints.h_count,
                   [types = constraints.types.viewer().name("types"),
                    vertex_ids = constraints.vertex_ids.viewer().name("vertex_ids"),
                    weights = constraints.weights.viewer().name("weights"),
                    normals = constraints.normals.viewer().name("normals"),
                    offsets = constraints.offsets.viewer().name("offsets"),
                    y = context.y.viewer().name("y"),
                    backward_violations =
                        context.backward_violations.viewer().name("backward_violations"),
                    masses = info.finite_element_method->masses().viewer().name("masses"),
                    is_fixed =
                        info.finite_element_method->is_fixed().viewer().name("is_fixed"),
                    fem_vertex_offset,
                    fem_vertex_count] __device__(int c) mutable
                   {
                       if(types(c) != TWPConstraintType::VertexHalfPlane)
                           return;

                       Vector4i ids = vertex_ids(c);
                       Vector4  ws  = weights(c);
                       Vector3  N   = normals(c);
                       Float    C   = -offsets(c);

                       Float denom = 0.0;
                       for(IndexT local_i = 0; local_i < 4; ++local_i)
                       {
                           IndexT v = ids(local_i);
                           if(v < 0 || ws(local_i) == 0.0)
                               continue;

                           C += ws(local_i) * y(v).dot(N);

                           if(v < fem_vertex_offset
                              || v >= fem_vertex_offset + fem_vertex_count)
                               continue;

                           IndexT fem_v = v - fem_vertex_offset;
                           if(is_fixed(fem_v))
                               continue;

                           Float mass = masses(fem_v);
                           if(mass > 0.0)
                               denom += ws(local_i) * ws(local_i) / mass;
                       }

                       if(C >= 0.0 || denom <= 0.0)
                           return;

                       Float violation = -C;
                       Float lambda    = violation / denom;

                       for(IndexT local_i = 0; local_i < 4; ++local_i)
                       {
                           IndexT v = ids(local_i);
                           if(v < 0 || ws(local_i) == 0.0)
                               continue;

                           if(v < fem_vertex_offset
                              || v >= fem_vertex_offset + fem_vertex_count)
                               continue;

                           IndexT fem_v = v - fem_vertex_offset;
                           if(is_fixed(fem_v))
                               continue;

                           Float mass = masses(fem_v);
                           if(mass > 0.0)
                               y(v) += (ws(local_i) / mass) * lambda * N;
                       }

                       backward_violations(c) = violation;
                   });
    }
    else
    {
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(constraints.h_count,
                   [types = constraints.types.viewer().name("types"),
                    vertex_ids = constraints.vertex_ids.viewer().name("vertex_ids"),
                    weights = constraints.weights.viewer().name("weights"),
                    normals = constraints.normals.viewer().name("normals"),
                    offsets = constraints.offsets.viewer().name("offsets"),
                    y = context.y.viewer().name("y"),
                    backward_violations =
                        context.backward_violations.viewer().name("backward_violations")] __device__(
                       int c) mutable
                   {
                       if(types(c) != TWPConstraintType::VertexHalfPlane)
                           return;

                       Vector4i ids = vertex_ids(c);
                       Vector4  ws  = weights(c);
                       Vector3  N   = normals(c);
                       Float    C   = -offsets(c);

                       for(IndexT local_i = 0; local_i < 4; ++local_i)
                       {
                           IndexT v = ids(local_i);
                           if(v >= 0 && ws(local_i) != 0.0)
                               C += ws(local_i) * y(v).dot(N);
                       }

                       if(C < 0.0)
                       {
                           IndexT v = ids(0);
                           Float violation = -C;
                           y(v) += violation * N;
                           backward_violations(c) = violation;
                       }
                   });
    }

    DeviceReduce().Max(context.backward_violations.data(),
                       context.max_backward_violation.data(),
                       constraints.h_count);
    context.backward_violation_inf = context.max_backward_violation;
}
}  // namespace uipc::backend::cuda
