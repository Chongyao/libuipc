#pragma once
#include <type_define.h>
#include <muda/buffer/device_buffer.h>
#include <muda/buffer/device_var.h>
#include <vector>

namespace uipc::backend::cuda
{
class GlobalVertexManager;
class FiniteElementMethod;
class FiniteElementVertexReporter;
struct TWPConstraintSet;
struct TWPContext;

class TWPBackwardSolver
{
  public:
    struct SolveInfo
    {
        TWPContext*                  context = nullptr;
        TWPConstraintSet*            constraints = nullptr;
        GlobalVertexManager*         global_vertex_manager = nullptr;
        FiniteElementMethod*         finite_element_method = nullptr;
        FiniteElementVertexReporter* finite_element_vertex_reporter = nullptr;
        IndexT                       max_iterations = 32;
        bool                         check_convergence = true;
        bool                         use_gpu_self_contact_coloring = false;
    };

    void solve(SolveInfo info);

  private:
    void update_self_contact_coloring(TWPConstraintSet& constraints,
                                      bool              use_gpu_coloring);
    void update_self_contact_coloring_cpu(TWPConstraintSet& constraints);
    void update_self_contact_coloring_gpu(TWPConstraintSet& constraints);
    void update_edge_coloring_if_needed(TWPConstraintSet& constraints);

    muda::DeviceBuffer<IndexT> m_self_contact_colors;
    muda::DeviceBuffer<IndexT> m_self_contact_active_flags;
    muda::DeviceBuffer<IndexT> m_self_contact_vertex_counts;
    muda::DeviceBuffer<IndexT> m_self_contact_vertex_offsets;
    muda::DeviceBuffer<IndexT> m_self_contact_vertex_incidents;
    muda::DeviceBuffer<IndexT> m_self_contact_candidate_colors;
    muda::DeviceBuffer<IndexT> m_self_contact_candidate_priorities;
    muda::DeviceBuffer<IndexT> m_self_contact_color_counts;
    muda::DeviceBuffer<IndexT> m_self_contact_color_offsets;
    muda::DeviceBuffer<IndexT> m_self_contact_color_cursors;

    muda::DeviceBuffer<IndexT> m_colored_contact_ids;
    std::vector<Vector4i>      m_host_self_contact_vertex_ids;
    std::vector<IndexT>        m_host_self_contact_color_offsets;
    std::vector<IndexT>        m_host_colored_self_contact_ids;

    muda::DeviceBuffer<IndexT> m_colored_edge_ids;
    std::vector<IndexT>        m_host_colored_edge_ids;
    std::vector<IndexT>        m_host_edge_color_offsets;
    std::vector<Vector4i>      m_host_edge_vertex_ids;
    IndexT                     m_coloring_edge_count = -1;
};
}  // namespace uipc::backend::cuda
