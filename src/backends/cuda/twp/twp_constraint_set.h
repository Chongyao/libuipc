#pragma once
#include <type_define.h>
#include <muda/buffer/device_buffer.h>
#include <muda/buffer/device_var.h>

namespace uipc::backend::cuda
{
enum class TWPConstraintType : IndexT
{
    VertexHalfPlane = 0,
    EdgeLengthUpperBound = 1,
};

template <typename TypeView,
          typename VertexIdView,
          typename WeightView,
          typename NormalView,
          typename OffsetView>
MUDA_GENERIC void write_vertex_half_plane_constraint(TypeView&       types,
                                                     VertexIdView&   vertex_ids,
                                                     WeightView&     weights,
                                                     NormalView&     normals,
                                                     OffsetView&     offsets,
                                                     IndexT          constraint_id,
                                                     IndexT          vertex_id,
                                                     const Vector3&  normal,
                                                     Float           offset)
{
    types(constraint_id)      = TWPConstraintType::VertexHalfPlane;
    vertex_ids(constraint_id) = Vector4i{vertex_id, -1, -1, -1};
    weights(constraint_id)    = Vector4{1.0, 0.0, 0.0, 0.0};
    normals(constraint_id)    = normal;
    offsets(constraint_id)    = offset;
}

template <typename TypeView,
          typename VertexIdView,
          typename WeightView,
          typename NormalView,
          typename OffsetView>
MUDA_GENERIC void write_edge_length_upper_bound_constraint(TypeView&       types,
                                                           VertexIdView&   vertex_ids,
                                                           WeightView&     weights,
                                                           NormalView&     normals,
                                                           OffsetView&     offsets,
                                                           IndexT          constraint_id,
                                                           const Vector2i& edge,
                                                           const Vector3&  direction,
                                                           Float           rhs)
{
    types(constraint_id)      = TWPConstraintType::EdgeLengthUpperBound;
    vertex_ids(constraint_id) = Vector4i{edge.x(), edge.y(), -1, -1};
    weights(constraint_id)    = Vector4{-2.0, 2.0, 0.0, 0.0};
    normals(constraint_id)    = direction;
    offsets(constraint_id)    = -rhs;
}

template <typename TypeView,
          typename VertexIdView,
          typename WeightView,
          typename NormalView,
          typename OffsetView>
MUDA_GENERIC void write_disabled_edge_length_upper_bound_constraint(
    TypeView&     types,
    VertexIdView& vertex_ids,
    WeightView&   weights,
    NormalView&   normals,
    OffsetView&   offsets,
    IndexT        constraint_id)
{
    types(constraint_id)      = TWPConstraintType::EdgeLengthUpperBound;
    vertex_ids(constraint_id) = Vector4i{-1, -1, -1, -1};
    weights(constraint_id)    = Vector4::Zero();
    normals(constraint_id)    = Vector3::Zero();
    offsets(constraint_id)    = 0.0;
}

struct TWPConstraintSet
{
    muda::DeviceBuffer<TWPConstraintType> types;
    muda::DeviceBuffer<Vector4i>          vertex_ids;
    muda::DeviceBuffer<Vector4>           weights;
    muda::DeviceBuffer<Vector3>           normals;
    muda::DeviceBuffer<Float>             offsets;
    muda::DeviceVar<IndexT>               count;

    void resize(SizeT capacity);
    void clear();
    void set_host_contact_constraint_count(IndexT contact_count);
    void set_host_edge_constraint_count(IndexT edge_count);

    IndexT host_total_constraint_count() const;
    IndexT host_contact_constraint_count() const;
    IndexT host_edge_constraint_count() const;
    IndexT host_edge_constraint_offset() const;

  private:
    IndexT m_host_total_count   = 0;
    IndexT m_host_contact_count = 0;
    IndexT m_host_edge_count    = 0;
};
}  // namespace uipc::backend::cuda
