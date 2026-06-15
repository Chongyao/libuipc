#include <twp/twp_debug_summary.h>
#include <muda/buffer/buffer_launch.h>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace uipc::backend::cuda
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
            .copy<TWPConstraintType>(types.data(), std::as_const(constraints.types).view(0, constraint_count))
            .copy<Vector4i>(vertex_ids.data(), std::as_const(constraints.vertex_ids).view(0, constraint_count))
            .copy<Vector4i>(primitive_ids.data(), std::as_const(constraints.primitive_ids).view(0, constraint_count))
            .copy<Float>(offsets.data(), std::as_const(constraints.offsets).view(0, constraint_count))
            .copy<Float>(lambdas.data(), std::as_const(context.backward_lambdas).view(0, constraint_count))
            .copy<Float>(violations.data(), std::as_const(context.diagnostics.backward.backward_violations).view(0, constraint_count))
            .copy<Float>(gaps.data(), std::as_const(context.diagnostics.backward.lcp_gaps).view(0, constraint_count))
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
                summary.worst_violation        = violations[c];
                summary.worst_violation_gap    = gaps[c];
                summary.worst_violation_lambda = lambdas[c];
                summary.worst_violation_offset = offsets[c];
            }

            if(gaps[c] < summary.min_gap)
            {
                summary.min_gap_constraint = c;
                summary.min_gap_type       = types[c];
                summary.min_gap_vertices =
                    display_vertices(types[c], vertex_ids[c], primitive_ids[c]);
                summary.min_gap        = gaps[c];
                summary.min_gap_lambda = lambdas[c];
            }
        }

        if(!std::isfinite(summary.min_gap))
            summary.min_gap = 0.0;
    }

    const IndexT vertex_count = context.x.size();
    if(vertex_count > 0)
    {
        std::vector<Float>  proximity_distances(vertex_count);
        std::vector<IndexT> proximity_constraint_ids(vertex_count);
        std::vector<Float>  safe_step_alphas(vertex_count);
        std::vector<Float>  step_norms(vertex_count);
        std::vector<Vector3> x(vertex_count);
        std::vector<Vector3> y(vertex_count);
        std::vector<Vector3> target_y(vertex_count);

        muda::BufferLaunch()
            .copy<Float>(proximity_distances.data(), std::as_const(context.proximity_distances).view(0, vertex_count))
            .copy<IndexT>(proximity_constraint_ids.data(), std::as_const(context.proximity_constraint_ids).view(0, vertex_count))
            .copy<Float>(safe_step_alphas.data(), std::as_const(context.diagnostics.forward.safe_step_alphas).view(0, vertex_count))
            .copy<Float>(step_norms.data(), std::as_const(context.diagnostics.forward.step_norms).view(0, vertex_count))
            .copy<Vector3>(x.data(), std::as_const(context.x).view(0, vertex_count))
            .copy<Vector3>(y.data(), std::as_const(context.y).view(0, vertex_count))
            .copy<Vector3>(target_y.data(), std::as_const(context.target_y).view(0, vertex_count))
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
                    summary.min_proximity_constraint = proximity_constraint_ids[v];
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
            .copy<TWPConstraintType>(&host_type, std::as_const(constraints.types).view(constraint, 1))
            .copy<Vector4i>(&host_vertices, std::as_const(constraints.vertex_ids).view(constraint, 1))
            .copy<Vector4i>(&host_primitives, std::as_const(constraints.primitive_ids).view(constraint, 1))
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
}  // namespace uipc::backend::cuda
