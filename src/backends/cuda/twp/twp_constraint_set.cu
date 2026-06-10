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
    count                = 0;
    m_host_total_count   = 0;
    m_host_contact_count = 0;
    m_host_edge_count    = 0;
}

void TWPConstraintSet::set_host_contact_constraint_count(IndexT contact_count)
{
    m_host_contact_count = contact_count;
    m_host_edge_count    = 0;
    m_host_total_count   = contact_count;
    count                = contact_count;
}

void TWPConstraintSet::set_host_edge_constraint_count(IndexT edge_count)
{
    m_host_edge_count  = edge_count;
    m_host_total_count = m_host_contact_count + edge_count;
    count              = m_host_total_count;
}

IndexT TWPConstraintSet::host_total_constraint_count() const
{
    return m_host_total_count;
}

IndexT TWPConstraintSet::host_contact_constraint_count() const
{
    return m_host_contact_count;
}

IndexT TWPConstraintSet::host_edge_constraint_count() const
{
    return m_host_edge_count;
}

IndexT TWPConstraintSet::host_edge_constraint_offset() const
{
    return m_host_contact_count;
}
}  // namespace uipc::backend::cuda
