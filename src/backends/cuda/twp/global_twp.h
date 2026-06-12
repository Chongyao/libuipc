#pragma once
#include <sim_system.h>
#include <uipc/geometry/attribute_slot.h>
#include <muda/buffer/device_buffer.h>
#include <muda/buffer/device_var.h>
#include <twp/twp_backward_solver.h>
#include <twp/twp_constraint_set.h>
#include <twp/twp_context.h>
#include <string_view>
#include <utility>

namespace uipc::backend::cuda
{
class GlobalVertexManager;
class GlobalSimplicialSurfaceManager;
class GlobalTrajectoryFilter;
class VertexHalfPlaneTrajectoryFilter;
class SimplexTrajectoryFilter;
class GlobalContactManager;
class HalfPlane;
class HalfPlaneVertexReporter;
class FiniteElementMethod;
class FiniteElementVertexReporter;

class GlobalTWP final : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    class Impl
    {
      public:
        void init();
        void project();
        void debug_log_state(std::string_view stage);

        void ensure_storage(SizeT vertex_count);
        Float compute_full_step_toi();
        void reset_algorithm_state();
        void prepare_edge_reference_length_squares();
        void proximity_search(Float search_bound);
        void refresh_self_collision_candidates(Float search_bound);
        void append_simplex_contact_constraints(Float search_bound);
        void refresh_edge_constraints();
        void backward();
        void forward();
        void sync_half_plane_support_set();
        void sync_simplex_support_set();
        Float  compute_min_clearance(muda::CBufferView<Vector3> positions,
                                      IndexT global_vertex_offset = 0);
        IndexT count_penetrated_vertices(muda::CBufferView<Vector3> positions,
                                         IndexT global_vertex_offset = 0);
        bool  debug_enabled() const;

        SimSystemSlot<GlobalVertexManager>    global_vertex_manager;
        SimSystemSlot<GlobalSimplicialSurfaceManager> global_simplicial_surface_manager;
        SimSystemSlot<GlobalTrajectoryFilter> global_trajectory_filter;
        SimSystemSlot<VertexHalfPlaneTrajectoryFilter> vertex_half_plane_trajectory_filter;
        SimSystemSlot<SimplexTrajectoryFilter> simplex_trajectory_filter;
        SimSystemSlot<GlobalContactManager>   global_contact_manager;
        SimSystemSlot<FiniteElementMethod>    finite_element_method;
        SimSystemSlot<FiniteElementVertexReporter> finite_element_vertex_reporter;
        SimSystemSlot<HalfPlane> half_plane;
        SimSystemSlot<HalfPlaneVertexReporter> half_plane_vertex_reporter;

        S<const geometry::AttributeSlot<IndexT>> max_iter_attr;
        S<const geometry::AttributeSlot<Float>>  eps_attr;
        S<const geometry::AttributeSlot<Float>>  d_min_attr;
        S<const geometry::AttributeSlot<Float>>  d_max_attr;
        S<const geometry::AttributeSlot<Float>>  edge_sigma_attr;
        S<const geometry::AttributeSlot<Float>>  repulsion_stiffness_attr;
        S<const geometry::AttributeSlot<IndexT>> backward_max_iter_attr;
        S<const geometry::AttributeSlot<IndexT>> backward_check_convergence_attr;
        S<const geometry::AttributeSlot<IndexT>> self_collision_enable_attr;
        S<const geometry::AttributeSlot<IndexT>> self_contact_coloring_attr;
        S<const geometry::AttributeSlot<IndexT>> debug_attr;

        TWPContext        context;
        TWPConstraintSet  constraints;
        TWPBackwardSolver backward_solver;
        muda::DeviceBuffer<Vector2i> support_PHs;
        muda::DeviceVar<IndexT>      support_PH_count;
        muda::DeviceBuffer<Vector4i> support_PTs;
        muda::DeviceBuffer<Vector4i> support_EEs;
        muda::DeviceBuffer<Vector3i> empty_PEs;
        muda::DeviceBuffer<Vector2i> empty_PPs;
        muda::DeviceVar<IndexT>      support_PT_count;
        muda::DeviceVar<IndexT>      support_EE_count;
        bool                         has_support_contact = false;
    };

  private:
    friend class SimEngine;

    void do_build() override;
    void init();
    void project();
    void debug_log_state(std::string_view stage);

    Impl m_impl;
};
}  // namespace uipc::backend::cuda
