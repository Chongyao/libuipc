#pragma once
#include <type_define.h>
#include <muda/buffer/device_buffer.h>
#include <muda/buffer/device_var.h>

namespace uipc::backend::cuda
{
enum class TWPConstraintType : IndexT
{
    VertexHalfPlane = 0,
    EdgeLengthLowerBound = 1,
    PointTriangle = 2,
    EdgeEdge = 3,
};

template <typename TypeView,
          typename VertexIdView,
          typename PrimitiveIdView,
          typename WeightView,
          typename NormalView,
          typename OffsetView,
          typename GradientView>
MUDA_GENERIC void write_vertex_half_plane_constraint(TypeView&       types,
                                                     VertexIdView&   vertex_ids,
                                                     PrimitiveIdView& primitive_ids,
                                                     WeightView&     weights,
                                                     NormalView&     normals,
                                                     OffsetView&     offsets,
                                                     GradientView&   gradients,
                                                     IndexT          constraint_id,
                                                     IndexT          vertex_id,
                                                     IndexT          half_plane_id,
                                                     const Vector3&  normal,
                                                     Float           offset)
{
    types(constraint_id)      = TWPConstraintType::VertexHalfPlane;
    // vertex_ids contains only LCP solve variables. The half-plane id is metadata;
    // keeping it out of vertex_ids prevents coloring from serializing all PH
    // constraints against the same plane id.
    vertex_ids(constraint_id) = Vector4i{vertex_id, -1, -1, -1};
    primitive_ids(constraint_id) = Vector4i{half_plane_id, -1, -1, -1};
    weights(constraint_id)    = Vector4{1.0, 0.0, 0.0, 0.0};
    normals(constraint_id)    = normal;
    offsets(constraint_id)    = offset;
    Vector12 G                = Vector12::Zero();
    G.segment<3>(0)           = normal;
    gradients(constraint_id)  = G;
}

template <typename TypeView,
          typename VertexIdView,
          typename WeightView,
          typename NormalView,
          typename OffsetView,
          typename GradientView>
MUDA_GENERIC void write_edge_length_lower_bound_constraint(TypeView&       types,
                                                           VertexIdView&   vertex_ids,
                                                           WeightView&     weights,
                                                           NormalView&     normals,
                                                           OffsetView&     offsets,
                                                           GradientView&   gradients,
                                                           IndexT          constraint_id,
                                                           const Vector2i& edge,
                                                           const Vector3&  direction,
                                                           Float           rhs)
{
    types(constraint_id)      = TWPConstraintType::EdgeLengthLowerBound;
    vertex_ids(constraint_id) = Vector4i{edge.x(), edge.y(), -1, -1};
    weights(constraint_id)    = Vector4{2.0, -2.0, 0.0, 0.0};
    normals(constraint_id)    = direction;
    offsets(constraint_id)    = rhs;
    Vector12 G                = Vector12::Zero();
    G.segment<3>(0)           = 2.0 * direction;
    G.segment<3>(3)           = -2.0 * direction;
    gradients(constraint_id)  = G;
}

template <typename TypeView,
          typename VertexIdView,
          typename WeightView,
          typename NormalView,
          typename OffsetView,
          typename GradientView>
MUDA_GENERIC void write_disabled_edge_length_lower_bound_constraint(
    TypeView&     types,
    VertexIdView& vertex_ids,
    WeightView&   weights,
    NormalView&   normals,
    OffsetView&   offsets,
    GradientView& gradients,
    IndexT        constraint_id)
{
    types(constraint_id)      = TWPConstraintType::EdgeLengthLowerBound;
    vertex_ids(constraint_id) = Vector4i{-1, -1, -1, -1};
    weights(constraint_id)    = Vector4::Zero();
    normals(constraint_id)    = Vector3::Zero();
    offsets(constraint_id)    = 0.0;
    gradients(constraint_id)  = Vector12::Zero();
}

template <typename TypeView,
          typename VertexIdView,
          typename WeightView,
          typename NormalView,
          typename OffsetView,
          typename GradientView>
MUDA_GENERIC void write_simplex_contact_constraint(TypeView&              types,
                                                   VertexIdView&          vertex_ids,
                                                   WeightView&            weights,
                                                   NormalView&            normals,
                                                   OffsetView&            offsets,
                                                   GradientView&          gradients,
                                                   IndexT                 constraint_id,
                                                   TWPConstraintType      type,
                                                   const Vector4i&        vertices,
                                                   const Vector12&        gradient,
                                                   Float                  offset)
{
    types(constraint_id)      = type;
    vertex_ids(constraint_id) = vertices;
    weights(constraint_id)    = Vector4::Zero();
    normals(constraint_id)    = Vector3::Zero();
    offsets(constraint_id)    = offset;
    gradients(constraint_id)  = gradient;
}

struct TWPConstraintSet
{
    muda::DeviceBuffer<TWPConstraintType> types;
    muda::DeviceBuffer<Vector4i>          vertex_ids;
    muda::DeviceBuffer<Vector4i>          primitive_ids;
    muda::DeviceBuffer<Vector4>           weights;
    muda::DeviceBuffer<Vector3>           normals;
    muda::DeviceBuffer<Float>             offsets;
    muda::DeviceBuffer<Vector12>          gradients;
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
