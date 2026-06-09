#include <twp/twp_constraint_set.h>

namespace uipc::backend::cuda
{
void TWPConstraintSet::resize(SizeT capacity)
{
    types.resize(capacity);
    vertex_ids.resize(capacity);
    weights.resize(capacity);
    normals.resize(capacity);
    offsets.resize(capacity);
}

void TWPConstraintSet::clear()
{
    count   = 0;
    h_count = 0;
}
}  // namespace uipc::backend::cuda
