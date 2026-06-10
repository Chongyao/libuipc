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
    IndexT                                h_count = 0;
    IndexT                                h_contact_count = 0;
    IndexT                                h_edge_count = 0;

    void resize(SizeT capacity);
    void clear();
};
}  // namespace uipc::backend::cuda
