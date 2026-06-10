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
