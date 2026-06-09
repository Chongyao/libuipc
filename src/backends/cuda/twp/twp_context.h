#pragma once
#include <type_define.h>
#include <muda/buffer/device_buffer.h>
#include <muda/buffer/device_var.h>

namespace uipc::backend::cuda
{
class GlobalVertexManager;
class FiniteElementMethod;
class FiniteElementVertexReporter;

struct TWPContext
{
    muda::DeviceBuffer<Vector3> x;
    muda::DeviceBuffer<Vector3> y;
    muda::DeviceBuffer<Vector3> target_y;
    muda::DeviceBuffer<Float>   residual;
    muda::DeviceBuffer<Float>   clearances;
    muda::DeviceBuffer<Float>   backward_violations;
    muda::DeviceBuffer<Float>   backward_lambdas;
    muda::DeviceBuffer<Vector3> backward_corrections;
    muda::DeviceBuffer<IndexT>  backward_correction_counts;
    muda::DeviceBuffer<Float>   safe_step_alphas;
    muda::DeviceBuffer<Float>   forward_step_norms;
    muda::DeviceBuffer<IndexT>  contact_vertex_flags;
    muda::DeviceBuffer<IndexT>  penetration_flags;

    muda::DeviceVar<IndexT> penetration_count;
    muda::DeviceVar<Float>  min_clearance;
    muda::DeviceVar<Float>  max_backward_violation;
    muda::DeviceVar<Float>  min_safe_step_alpha;
    muda::DeviceVar<Float>  max_residual;
    muda::DeviceVar<Float>  max_step_norm;

    Float remaining_search_bound = 0.0;
    Float residual_inf           = 1.0;
    Float backward_violation_inf = 0.0;
    Float max_forward_step       = 0.0;
    IndexT backward_iterations   = 0;
    bool   backward_converged    = false;

    void ensure_storage(SizeT vertex_count);
    void ensure_constraint_storage(SizeT constraint_capacity);
    void reset(GlobalVertexManager& global_vertex_manager);
};
}  // namespace uipc::backend::cuda
