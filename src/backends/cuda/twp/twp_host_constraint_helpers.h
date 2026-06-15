#pragma once
#include <type_define.h>
#include <vector>

namespace uipc::backend::cuda
{
void build_host_greedy_coloring(const std::vector<Vector4i>& constraint_vertex_ids,
                                IndexT                       constraint_count,
                                std::vector<IndexT>&         color_offsets,
                                std::vector<IndexT>&         colored_constraint_ids);
}  // namespace uipc::backend::cuda
