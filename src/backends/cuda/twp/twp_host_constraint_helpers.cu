#include <twp/twp_host_constraint_helpers.h>
#include <unordered_set>

namespace uipc::backend::cuda
{
void build_host_greedy_coloring(const std::vector<Vector4i>& constraint_vertex_ids,
                                IndexT                       constraint_count,
                                std::vector<IndexT>&         color_offsets,
                                std::vector<IndexT>&         colored_constraint_ids)
{
    std::vector<std::vector<IndexT>>        color_constraint_ids;
    std::vector<std::unordered_set<IndexT>> color_vertices;

    for(IndexT c = 0; c < constraint_count; ++c)
    {
        Vector4i ids = constraint_vertex_ids[c];
        IndexT   selected_color = -1;

        for(IndexT color = 0; color < static_cast<IndexT>(color_vertices.size()); ++color)
        {
            bool conflict = false;
            for(IndexT local_i = 0; local_i < 4; ++local_i)
            {
                IndexT v = ids(local_i);
                if(v >= 0 && color_vertices[color].contains(v))
                {
                    conflict = true;
                    break;
                }
            }

            if(!conflict)
            {
                selected_color = color;
                break;
            }
        }

        if(selected_color < 0)
        {
            selected_color = static_cast<IndexT>(color_vertices.size());
            color_vertices.emplace_back();
            color_constraint_ids.emplace_back();
        }

        color_constraint_ids[selected_color].push_back(c);
        for(IndexT local_i = 0; local_i < 4; ++local_i)
        {
            IndexT v = ids(local_i);
            if(v >= 0)
                color_vertices[selected_color].insert(v);
        }
    }

    color_offsets.clear();
    color_offsets.reserve(color_constraint_ids.size() + 1);
    color_offsets.push_back(0);

    colored_constraint_ids.clear();
    colored_constraint_ids.reserve(constraint_count);
    for(const auto& ids : color_constraint_ids)
    {
        colored_constraint_ids.insert(colored_constraint_ids.end(), ids.begin(), ids.end());
        color_offsets.push_back(static_cast<IndexT>(colored_constraint_ids.size()));
    }
}
}  // namespace uipc::backend::cuda
