#pragma once
#include <type_define.h>
#include <muda/buffer/device_buffer.h>
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
    };

    void solve(SolveInfo info);

  private:
    void update_edge_coloring_if_needed(TWPConstraintSet& constraints);

    muda::DeviceBuffer<IndexT> m_colored_edge_ids;
    std::vector<IndexT>        m_host_colored_edge_ids;
    std::vector<IndexT>        m_host_edge_color_offsets;
    std::vector<Vector4i>      m_host_edge_vertex_ids;
    IndexT                     m_coloring_edge_count = -1;
};
}  // namespace uipc::backend::cuda
