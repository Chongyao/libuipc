#include <twp/global_twp.h>
#include <global_geometry/global_vertex_manager.h>
#include <finite_element/finite_element_method.h>
#include <finite_element/finite_element_vertex_reporter.h>
#include <implicit_geometry/half_plane.h>
#include <implicit_geometry/half_plane_vertex_reporter.h>
#include <muda/buffer/buffer_launch.h>
#include <muda/launch/parallel_for.h>
#include <muda/cub/device/device_reduce.h>
#include <array>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace uipc::backend::cuda
{
namespace
{
const char* constraint_type_name(TWPConstraintType type)
{
    switch(type)
    {
        case TWPConstraintType::VertexHalfPlane:
            return "PH";
        case TWPConstraintType::EdgeLengthLowerBound:
            return "EDGE";
        case TWPConstraintType::PointTriangle:
            return "PT";
        case TWPConstraintType::EdgeEdge:
            return "EE";
        default:
            return "UNKNOWN";
    }
}

bool has_duplicate_valid_vertex(const Vector4i& vertices)
{
    for(IndexT i = 0; i < 4; ++i)
    {
        IndexT vi = vertices(i);
        if(vi < 0)
            continue;
        for(IndexT j = i + 1; j < 4; ++j)
        {
            if(vi == vertices(j))
                return true;
        }
    }
    return false;
}

Vector4i display_vertices(TWPConstraintType type,
                          const Vector4i&   vertex_ids,
                          const Vector4i&   primitive_ids)
{
    if(type == TWPConstraintType::VertexHalfPlane)
        return Vector4i{vertex_ids.x(), primitive_ids.x(), -1, -1};
    return vertex_ids;
}

struct TWPHostDebugSummary
{
    std::array<IndexT, 4> type_counts = {0, 0, 0, 0};

    IndexT worst_violation_constraint = -1;
    TWPConstraintType worst_violation_type = TWPConstraintType::VertexHalfPlane;
    Vector4i worst_violation_vertices = Vector4i{-1, -1, -1, -1};
    Float worst_violation = 0.0;
    Float worst_violation_gap = 0.0;
    Float worst_violation_lambda = 0.0;
    Float worst_violation_offset = 0.0;

    IndexT min_gap_constraint = -1;
    TWPConstraintType min_gap_type = TWPConstraintType::VertexHalfPlane;
    Vector4i min_gap_vertices = Vector4i{-1, -1, -1, -1};
    Float min_gap = 0.0;
    Float min_gap_lambda = 0.0;

    IndexT min_proximity_vertex = -1;
    IndexT min_proximity_constraint = -1;
    TWPConstraintType min_proximity_type = TWPConstraintType::VertexHalfPlane;
    Vector4i min_proximity_vertices = Vector4i{-1, -1, -1, -1};
    bool min_proximity_has_duplicate_vertex = false;
    Float min_proximity_distance = 0.0;
    IndexT finite_proximity_vertices = 0;
    IndexT near_zero_proximity_vertices = 0;

    IndexT min_alpha_vertex = -1;
    IndexT min_alpha_constraint = -1;
    TWPConstraintType min_alpha_type = TWPConstraintType::VertexHalfPlane;
    Vector4i min_alpha_vertices = Vector4i{-1, -1, -1, -1};
    bool min_alpha_has_duplicate_vertex = false;
    Float min_alpha = 1.0;
    Float min_alpha_proximity_distance = 0.0;
    Float min_alpha_step_norm = 0.0;

    IndexT max_step_vertex = -1;
    Float max_step_norm = 0.0;
    Float max_step_alpha = 1.0;

    IndexT max_backward_displacement_vertex = -1;
    Float max_backward_displacement = 0.0;
    Float max_backward_displacement_clearance_x = 0.0;
    Float max_backward_displacement_clearance_y = 0.0;
};

TWPHostDebugSummary collect_host_debug_summary(TWPContext&        context,
                                               TWPConstraintSet& constraints)
{
    TWPHostDebugSummary summary;

    const IndexT constraint_count = constraints.host_total_constraint_count();
    if(constraint_count > 0)
    {
        std::vector<TWPConstraintType> types(constraint_count);
        std::vector<Vector4i>          vertex_ids(constraint_count);
        std::vector<Vector4i>          primitive_ids(constraint_count);
        std::vector<Float>             offsets(constraint_count);
        std::vector<Float>             lambdas(constraint_count);
        std::vector<Float>             violations(constraint_count);
        std::vector<Float>             gaps(constraint_count);

        muda::BufferLaunch()
            .copy<TWPConstraintType>(types.data(),
                                     std::as_const(constraints.types)
                                         .view(0, constraint_count))
            .copy<Vector4i>(vertex_ids.data(),
                            std::as_const(constraints.vertex_ids)
                                .view(0, constraint_count))
            .copy<Vector4i>(primitive_ids.data(),
                            std::as_const(constraints.primitive_ids)
                                .view(0, constraint_count))
            .copy<Float>(offsets.data(),
                         std::as_const(constraints.offsets).view(0, constraint_count))
            .copy<Float>(lambdas.data(),
                         std::as_const(context.backward_lambdas)
                             .view(0, constraint_count))
            .copy<Float>(violations.data(),
                         std::as_const(
                             context.diagnostics.backward.backward_violations)
                             .view(0, constraint_count))
            .copy<Float>(gaps.data(),
                         std::as_const(context.diagnostics.backward.lcp_gaps)
                             .view(0, constraint_count))
            .wait();

        summary.min_gap = std::numeric_limits<Float>::infinity();
        for(IndexT c = 0; c < constraint_count; ++c)
        {
            IndexT type_index = static_cast<IndexT>(types[c]);
            if(type_index >= 0
               && type_index < static_cast<IndexT>(summary.type_counts.size()))
                ++summary.type_counts[type_index];

            if(std::isfinite(violations[c])
               && violations[c] > summary.worst_violation)
            {
                summary.worst_violation_constraint = c;
                summary.worst_violation_type       = types[c];
                summary.worst_violation_vertices =
                    display_vertices(types[c], vertex_ids[c], primitive_ids[c]);
                summary.worst_violation            = violations[c];
                summary.worst_violation_gap        = gaps[c];
                summary.worst_violation_lambda     = lambdas[c];
                summary.worst_violation_offset     = offsets[c];
            }

            if(gaps[c] < summary.min_gap)
            {
                summary.min_gap_constraint = c;
                summary.min_gap_type       = types[c];
                summary.min_gap_vertices =
                    display_vertices(types[c], vertex_ids[c], primitive_ids[c]);
                summary.min_gap            = gaps[c];
                summary.min_gap_lambda     = lambdas[c];
            }
        }

        if(!std::isfinite(summary.min_gap))
            summary.min_gap = 0.0;
    }

    const IndexT vertex_count = context.x.size();
    if(vertex_count > 0)
    {
        std::vector<Float> proximity_distances(vertex_count);
        std::vector<IndexT> proximity_constraint_ids(vertex_count);
        std::vector<Float> safe_step_alphas(vertex_count);
        std::vector<Float> step_norms(vertex_count);
        std::vector<Vector3> x(vertex_count);
        std::vector<Vector3> y(vertex_count);
        std::vector<Vector3> target_y(vertex_count);

        muda::BufferLaunch()
            .copy<Float>(proximity_distances.data(),
                         std::as_const(context.proximity_distances)
                             .view(0, vertex_count))
            .copy<IndexT>(proximity_constraint_ids.data(),
                          std::as_const(context.proximity_constraint_ids)
                              .view(0, vertex_count))
            .copy<Float>(safe_step_alphas.data(),
                         std::as_const(
                             context.diagnostics.forward.safe_step_alphas)
                             .view(0, vertex_count))
            .copy<Float>(step_norms.data(),
                         std::as_const(context.diagnostics.forward.step_norms)
                             .view(0, vertex_count))
            .copy<Vector3>(x.data(), std::as_const(context.x).view(0, vertex_count))
            .copy<Vector3>(y.data(), std::as_const(context.y).view(0, vertex_count))
            .copy<Vector3>(target_y.data(),
                           std::as_const(context.target_y).view(0, vertex_count))
            .wait();

        summary.min_proximity_distance = std::numeric_limits<Float>::infinity();
        for(IndexT v = 0; v < vertex_count; ++v)
        {
            Float D = proximity_distances[v];
            if(std::isfinite(D) && D < Float{1e29})
            {
                ++summary.finite_proximity_vertices;
                if(D < Float{1e-8})
                    ++summary.near_zero_proximity_vertices;
                if(D < summary.min_proximity_distance)
                {
                    summary.min_proximity_distance = D;
                    summary.min_proximity_vertex   = v;
                    summary.min_proximity_constraint =
                        proximity_constraint_ids[v];
                }
            }

            Float alpha = safe_step_alphas[v];
            if(alpha < summary.min_alpha)
            {
                summary.min_alpha = alpha;
                summary.min_alpha_vertex = v;
                summary.min_alpha_proximity_distance =
                    std::isfinite(D) && D < Float{1e29} ? D : Float{-1.0};
                summary.min_alpha_step_norm = step_norms[v];
                summary.min_alpha_constraint = proximity_constraint_ids[v];
            }

            Float step = step_norms[v];
            if(std::isfinite(step) && step > summary.max_step_norm)
            {
                summary.max_step_norm = step;
                summary.max_step_vertex = v;
                summary.max_step_alpha = alpha;
            }

            Float backward_displacement = (y[v] - target_y[v]).norm();
            if(std::isfinite(backward_displacement)
               && backward_displacement > summary.max_backward_displacement)
            {
                summary.max_backward_displacement = backward_displacement;
                summary.max_backward_displacement_vertex = v;
            }
        }

        if(!std::isfinite(summary.min_proximity_distance))
            summary.min_proximity_distance = -1.0;
    }

    auto fill_constraint_metadata = [&](IndexT             constraint,
                                        TWPConstraintType& type,
                                        Vector4i&         vertices)
    {
        if(constraint < 0 || constraint >= constraint_count)
            return;

        TWPConstraintType host_type;
        Vector4i          host_vertices;
        Vector4i          host_primitives;
        muda::BufferLaunch()
            .copy<TWPConstraintType>(&host_type,
                                     std::as_const(constraints.types).view(constraint, 1))
            .copy<Vector4i>(&host_vertices,
                            std::as_const(constraints.vertex_ids)
                                .view(constraint, 1))
            .copy<Vector4i>(&host_primitives,
                            std::as_const(constraints.primitive_ids)
                                .view(constraint, 1))
            .wait();
        type     = host_type;
        vertices = display_vertices(host_type, host_vertices, host_primitives);
    };

    fill_constraint_metadata(summary.min_proximity_constraint,
                             summary.min_proximity_type,
                             summary.min_proximity_vertices);
    fill_constraint_metadata(summary.min_alpha_constraint,
                             summary.min_alpha_type,
                             summary.min_alpha_vertices);
    summary.min_proximity_has_duplicate_vertex =
        has_duplicate_valid_vertex(summary.min_proximity_vertices);
    summary.min_alpha_has_duplicate_vertex =
        has_duplicate_valid_vertex(summary.min_alpha_vertices);

    return summary;
}
}  // namespace

Float GlobalTWP::Impl::compute_min_clearance(muda::CBufferView<Vector3> positions,
                                             IndexT global_vertex_offset)
{
    if(!half_plane || !half_plane_vertex_reporter || half_plane->positions().size() == 0)
        return 0.0;

    context.diagnostics.debug.clearances.resize(positions.size());

    SizeT plane_count = half_plane->positions().size();
    IndexT plane_vertex_offset = half_plane_vertex_reporter->vertex_offset();

    using namespace muda;
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(positions.size(),
               [clearances = context.diagnostics.debug.clearances.viewer().name("clearances"),
                positions = positions.viewer().name("positions"),
                thicknesses = global_vertex_manager->thicknesses().viewer().name("thicknesses"),
                plane_positions = half_plane->positions().viewer().name("plane_positions"),
                plane_normals = half_plane->normals().viewer().name("plane_normals"),
                plane_vertex_offset,
                plane_count,
                global_vertex_offset] __device__(int v) mutable
               {
                   IndexT global_v = v + global_vertex_offset;
                   if(global_v >= plane_vertex_offset
                      && global_v < plane_vertex_offset + plane_count)
                   {
                       clearances(v) = Float{1e30};
                       return;
                   }

                   Float min_clearance = Float{1e30};
                   for(IndexT h = 0; h < plane_count; ++h)
                   {
                       const Vector3& P = plane_positions(h);
                       const Vector3& N = plane_normals(h);
                       Float clearance = (positions(v) - P).dot(N) - thicknesses(global_v);
                       min_clearance = clearance < min_clearance ? clearance : min_clearance;
                   }
                   clearances(v) = min_clearance;
               });

    DeviceReduce().Min(context.diagnostics.debug.clearances.data(),
                       context.diagnostics.debug.min_clearance.data(),
                       positions.size());

    return context.diagnostics.debug.min_clearance;
}

IndexT GlobalTWP::Impl::count_penetrated_vertices(muda::CBufferView<Vector3> positions,
                                                  IndexT global_vertex_offset)
{
    if(!half_plane || !half_plane_vertex_reporter || half_plane->positions().size() == 0)
        return 0;

    context.diagnostics.debug.penetration_flags.resize(positions.size());

    SizeT plane_count = half_plane->positions().size();
    IndexT plane_vertex_offset = half_plane_vertex_reporter->vertex_offset();

    using namespace muda;
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(positions.size(),
               [flags = context.diagnostics.debug.penetration_flags.viewer().name(
                    "penetration_flags"),
                positions = positions.viewer().name("positions"),
                thicknesses = global_vertex_manager->thicknesses().viewer().name("thicknesses"),
                plane_positions = half_plane->positions().viewer().name("plane_positions"),
                plane_normals = half_plane->normals().viewer().name("plane_normals"),
                plane_vertex_offset,
                plane_count,
                global_vertex_offset] __device__(int v) mutable
               {
                   IndexT global_v = v + global_vertex_offset;
                   if(global_v >= plane_vertex_offset
                      && global_v < plane_vertex_offset + plane_count)
                   {
                       flags(v) = 0;
                       return;
                   }

                   IndexT penetrated = 0;
                   for(IndexT h = 0; h < plane_count; ++h)
                   {
                       const Vector3& P = plane_positions(h);
                       const Vector3& N = plane_normals(h);
                       Float clearance = (positions(v) - P).dot(N) - thicknesses(global_v);
                       if(clearance < 0.0)
                       {
                           penetrated = 1;
                           break;
                       }
                   }
                   flags(v) = penetrated;
               });

    DeviceReduce().Sum(context.diagnostics.debug.penetration_flags.data(),
                       context.diagnostics.debug.penetration_count.data(),
                       context.diagnostics.debug.penetration_flags.size());
    return context.diagnostics.debug.penetration_count;
}

void GlobalTWP::Impl::debug_log_state(std::string_view stage)
{
    if(!debug_enabled())
        return;

    Float target_min_clearance = 0.0;
    IndexT target_penetration_count = 0;
    if(context.target_y.size() == global_vertex_manager->positions().size())
    {
        target_min_clearance     = compute_min_clearance(context.target_y.view());
        target_penetration_count = count_penetrated_vertices(context.target_y.view());
    }

    Float context_x_min_clearance = 0.0;
    IndexT context_x_penetration_count = 0;
    if(context.x.size() == global_vertex_manager->positions().size())
    {
        context_x_min_clearance     = compute_min_clearance(context.x.view());
        context_x_penetration_count = count_penetrated_vertices(context.x.view());
    }

    Float context_y_min_clearance = 0.0;
    IndexT context_y_penetration_count = 0;
    if(context.y.size() == global_vertex_manager->positions().size())
    {
        context_y_min_clearance     = compute_min_clearance(context.y.view());
        context_y_penetration_count = count_penetrated_vertices(context.y.view());
    }

    Float global_min_clearance = compute_min_clearance(global_vertex_manager->positions());
    IndexT global_penetration_count =
        count_penetrated_vertices(global_vertex_manager->positions());

    Float  fem_min_clearance = 0.0;
    IndexT fem_penetration_count = 0;
    IndexT fem_vertex_offset = -1;
    SizeT  fem_vertex_count  = 0;
    if(finite_element_method)
    {
        UIPC_ASSERT(finite_element_vertex_reporter,
                    "FiniteElementVertexReporter is required for TWP debug mapping.");
        fem_vertex_offset = finite_element_vertex_reporter->vertex_offset();
        fem_vertex_count  = finite_element_method->xs().size();
        fem_min_clearance =
            compute_min_clearance(finite_element_method->xs(), fem_vertex_offset);
        fem_penetration_count =
            count_penetrated_vertices(finite_element_method->xs(), fem_vertex_offset);
    }

    bool  has_half_plane = static_cast<bool>(half_plane);
    bool  has_half_plane_vertex_reporter =
        static_cast<bool>(half_plane_vertex_reporter);
    SizeT plane_count = half_plane ? half_plane->positions().size() : 0;
    IndexT plane_vertex_offset =
        half_plane_vertex_reporter ? half_plane_vertex_reporter->vertex_offset() : -1;
    if(constraints.host_total_constraint_count() > 0 || target_penetration_count > 0
       || global_penetration_count > 0
       || fem_penetration_count > 0)
    {
        TWPHostDebugSummary host_summary =
            collect_host_debug_summary(context, constraints);

        logger::warn(
            "TWP Debug[{}]: planes={}({}), plane_offset={}, contact={}, edge={}, "
            "type_count[PH={}, PT={}, EE={}, EDGE={}], "
            "target[min={}, pen={}], ctx_x[min={}, pen={}], ctx_y[min={}, pen={}], "
            "global[min={}, pen={}], "
            "fem[offset={}, count={}, min={}, pen={}], "
            "backward[violation={}, lcp_gap={}, lcp_comp={}, lcp_proj_res={}, "
            "iter={}, converged={}], "
            "worst_violation[id={}, type={}, vertices=({}, {}, {}, {}), gap={}, "
            "lambda={}, offset={}], "
            "min_gap[id={}, type={}, vertices=({}, {}, {}, {}), gap={}, lambda={}], "
            "forward[residual={}, max_step={}, min_alpha={}, limited={}, "
            "min_D_vertex={}, min_D={}, finite_D={}, near_zero_D={}, "
            "min_D_constraint[id={}, type={}, vertices=({}, {}, {}, {}), duplicate_vertex={}], "
            "min_alpha_vertex={}, min_alpha_D={}, min_alpha_step={}, "
            "min_alpha_constraint[id={}, type={}, vertices=({}, {}, {}, {}), duplicate_vertex={}], "
            "max_step_vertex={}, max_step_alpha={}, "
            "max_backward_displacement_vertex={}, max_backward_displacement={}]",
            stage,
            plane_count,
            has_half_plane && has_half_plane_vertex_reporter,
            plane_vertex_offset,
            constraints.host_contact_constraint_count(),
            constraints.host_edge_constraint_count(),
            host_summary.type_counts[static_cast<IndexT>(
                TWPConstraintType::VertexHalfPlane)],
            host_summary.type_counts[static_cast<IndexT>(
                TWPConstraintType::PointTriangle)],
            host_summary.type_counts[static_cast<IndexT>(
                TWPConstraintType::EdgeEdge)],
            host_summary.type_counts[static_cast<IndexT>(
                TWPConstraintType::EdgeLengthLowerBound)],
            target_min_clearance,
            target_penetration_count,
            context_x_min_clearance,
            context_x_penetration_count,
            context_y_min_clearance,
            context_y_penetration_count,
            global_min_clearance,
            global_penetration_count,
            fem_vertex_offset,
            fem_vertex_count,
            fem_min_clearance,
            fem_penetration_count,
            context.diagnostics.backward.violation_inf,
            context.diagnostics.backward.lcp_min_gap,
            context.diagnostics.backward.lcp_complementarity_inf,
            context.diagnostics.backward.lcp_projected_residual_inf,
            context.diagnostics.backward.iterations,
            context.diagnostics.backward.converged,
            host_summary.worst_violation_constraint,
            constraint_type_name(host_summary.worst_violation_type),
            host_summary.worst_violation_vertices(0),
            host_summary.worst_violation_vertices(1),
            host_summary.worst_violation_vertices(2),
            host_summary.worst_violation_vertices(3),
            host_summary.worst_violation_gap,
            host_summary.worst_violation_lambda,
            host_summary.worst_violation_offset,
            host_summary.min_gap_constraint,
            constraint_type_name(host_summary.min_gap_type),
            host_summary.min_gap_vertices(0),
            host_summary.min_gap_vertices(1),
            host_summary.min_gap_vertices(2),
            host_summary.min_gap_vertices(3),
            host_summary.min_gap,
            host_summary.min_gap_lambda,
            context.diagnostics.forward.residual_inf,
            context.diagnostics.forward.max_step,
            context.diagnostics.forward.min_alpha,
            context.diagnostics.forward.limited_vertices,
            host_summary.min_proximity_vertex,
            host_summary.min_proximity_distance,
            host_summary.finite_proximity_vertices,
            host_summary.near_zero_proximity_vertices,
            host_summary.min_proximity_constraint,
            constraint_type_name(host_summary.min_proximity_type),
            host_summary.min_proximity_vertices(0),
            host_summary.min_proximity_vertices(1),
            host_summary.min_proximity_vertices(2),
            host_summary.min_proximity_vertices(3),
            host_summary.min_proximity_has_duplicate_vertex,
            host_summary.min_alpha_vertex,
            host_summary.min_alpha_proximity_distance,
            host_summary.min_alpha_step_norm,
            host_summary.min_alpha_constraint,
            constraint_type_name(host_summary.min_alpha_type),
            host_summary.min_alpha_vertices(0),
            host_summary.min_alpha_vertices(1),
            host_summary.min_alpha_vertices(2),
            host_summary.min_alpha_vertices(3),
            host_summary.min_alpha_has_duplicate_vertex,
            host_summary.max_step_vertex,
            host_summary.max_step_alpha,
            host_summary.max_backward_displacement_vertex,
            host_summary.max_backward_displacement);
    }
}
}  // namespace uipc::backend::cuda
