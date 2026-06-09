#pragma once
#include <type_define.h>

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
};
}  // namespace uipc::backend::cuda
