#pragma once
#include <sim_system.h>
#include <uipc/geometry/attribute_slot.h>
#include <muda/buffer/device_buffer.h>
#include <utility>

namespace uipc::backend::cuda
{
class GlobalVertexManager;
class GlobalTrajectoryFilter;
class GlobalContactManager;

class GlobalTWP final : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    class Impl
    {
      public:
        void init();
        void project();

        void ensure_storage(SizeT vertex_count);
        void reset_algorithm_state();
        void proximity_search(Float search_bound);
        void backward();
        void forward();

        SimSystemSlot<GlobalVertexManager>    global_vertex_manager;
        SimSystemSlot<GlobalTrajectoryFilter> global_trajectory_filter;
        SimSystemSlot<GlobalContactManager>   global_contact_manager;

        S<const geometry::AttributeSlot<IndexT>> max_iter_attr;
        S<const geometry::AttributeSlot<Float>>  eps_attr;
        S<const geometry::AttributeSlot<Float>>  d_min_attr;
        S<const geometry::AttributeSlot<Float>>  d_max_attr;

        muda::DeviceBuffer<Vector3> x;
        muda::DeviceBuffer<Vector3> y;
        muda::DeviceBuffer<Vector3> target_y;
        muda::DeviceBuffer<Float>   residual;

        Float remaining_search_bound = 0.0;
        Float residual_inf           = 1.0;
        Float max_forward_step       = 0.0;
    };

  private:
    friend class SimEngine;

    void do_build() override;
    void init();
    void project();

    Impl m_impl;
};
}  // namespace uipc::backend::cuda
