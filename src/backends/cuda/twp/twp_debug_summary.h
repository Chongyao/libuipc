#pragma once
#include <twp/twp_constraint_set.h>
#include <twp/twp_context.h>
#include <array>

namespace uipc::backend::cuda
{
const char* constraint_type_name(TWPConstraintType type);
bool has_duplicate_valid_vertex(const Vector4i& vertices);
Vector4i display_vertices(TWPConstraintType type,
                          const Vector4i&   vertex_ids,
                          const Vector4i&   primitive_ids);

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
                                               TWPConstraintSet& constraints);
}  // namespace uipc::backend::cuda
