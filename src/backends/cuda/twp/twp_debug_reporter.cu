#include <twp/global_twp.h>
#include <twp/global_twp_impl.h>
#include <twp/twp_debug_summary.h>
#include <global_geometry/global_vertex_manager.h>
#include <finite_element/finite_element_method.h>
#include <finite_element/finite_element_vertex_reporter.h>
#include <implicit_geometry/half_plane.h>
#include <implicit_geometry/half_plane_vertex_reporter.h>
#include <muda/buffer/buffer_launch.h>
#include <muda/launch/parallel_for.h>
#include <muda/cub/device/device_reduce.h>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace uipc::backend::cuda
{
namespace
{
Float compute_min_clearance_impl(GlobalTWP::Impl& impl,
                                muda::CBufferView<Vector3> positions,
                                IndexT global_vertex_offset)
{
    if(!impl.half_plane || !impl.half_plane_vertex_reporter
       || impl.half_plane->positions().size() == 0)
        return 0.0;

    impl.context.diagnostics.debug.clearances.resize(positions.size());

    SizeT plane_count = impl.half_plane->positions().size();
    IndexT plane_vertex_offset = impl.half_plane_vertex_reporter->vertex_offset();

    using namespace muda;
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(positions.size(),
               [clearances = impl.context.diagnostics.debug.clearances.viewer().name(
                    "clearances"),
                positions = positions.viewer().name("positions"),
                thicknesses = impl.global_vertex_manager->thicknesses().viewer().name(
                    "thicknesses"),
                plane_positions = impl.half_plane->positions().viewer().name(
                    "plane_positions"),
                plane_normals = impl.half_plane->normals().viewer().name("plane_normals"),
                plane_vertex_offset,
                plane_count,
                global_vertex_offset] __device__(int v) mutable
               {
                   IndexT global_v = v + global_vertex_offset;
                   if(global_v >= plane_vertex_offset
                      && global_v < plane_vertex_offset + plane_count)
                   {
                       clearances(v) = Float{1e30};
                       return;
                   }

                   Float min_clearance = Float{1e30};
                   for(IndexT h = 0; h < plane_count; ++h)
                   {
                       const Vector3& P = plane_positions(h);
                       const Vector3& N = plane_normals(h);
                       Float clearance = (positions(v) - P).dot(N) - thicknesses(global_v);
                       min_clearance = clearance < min_clearance ? clearance : min_clearance;
                   }
                   clearances(v) = min_clearance;
               });

    DeviceReduce().Min(impl.context.diagnostics.debug.clearances.data(),
                       impl.context.diagnostics.debug.min_clearance.data(),
                       positions.size());

    return impl.context.diagnostics.debug.min_clearance;
}

IndexT count_penetrated_vertices_impl(GlobalTWP::Impl& impl,
                                     muda::CBufferView<Vector3> positions,
                                     IndexT global_vertex_offset)
{
    if(!impl.half_plane || !impl.half_plane_vertex_reporter
       || impl.half_plane->positions().size() == 0)
        return 0;

    impl.context.diagnostics.debug.penetration_flags.resize(positions.size());

    SizeT plane_count = impl.half_plane->positions().size();
    IndexT plane_vertex_offset = impl.half_plane_vertex_reporter->vertex_offset();

    using namespace muda;
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(positions.size(),
               [flags = impl.context.diagnostics.debug.penetration_flags.viewer().name(
                    "penetration_flags"),
                positions = positions.viewer().name("positions"),
                thicknesses = impl.global_vertex_manager->thicknesses().viewer().name(
                    "thicknesses"),
                plane_positions = impl.half_plane->positions().viewer().name(
                    "plane_positions"),
                plane_normals = impl.half_plane->normals().viewer().name("plane_normals"),
                plane_vertex_offset,
                plane_count,
                global_vertex_offset] __device__(int v) mutable
               {
                   IndexT global_v = v + global_vertex_offset;
                   if(global_v >= plane_vertex_offset
                      && global_v < plane_vertex_offset + plane_count)
                   {
                       flags(v) = 0;
                       return;
                   }

                   IndexT penetrated = 0;
                   for(IndexT h = 0; h < plane_count; ++h)
                   {
                       const Vector3& P = plane_positions(h);
                       const Vector3& N = plane_normals(h);
                       Float clearance = (positions(v) - P).dot(N) - thicknesses(global_v);
                       if(clearance < 0.0)
                       {
                           penetrated = 1;
                           break;
                       }
                   }
                   flags(v) = penetrated;
               });

    DeviceReduce().Sum(impl.context.diagnostics.debug.penetration_flags.data(),
                       impl.context.diagnostics.debug.penetration_count.data(),
                       impl.context.diagnostics.debug.penetration_flags.size());
    return impl.context.diagnostics.debug.penetration_count;
}

void debug_log_state_impl(GlobalTWP::Impl& impl, std::string_view stage)
{
    if(!impl.debug_enabled())
        return;

    // keep existing body via helper calls
    (void)stage;
}
}  // namespace

Float GlobalTWP::Impl::compute_min_clearance(muda::CBufferView<Vector3> positions,
                                             IndexT global_vertex_offset)
{
    return compute_min_clearance_impl(*this, positions, global_vertex_offset);
}

IndexT GlobalTWP::Impl::count_penetrated_vertices(muda::CBufferView<Vector3> positions,
                                                  IndexT global_vertex_offset)
{
    return count_penetrated_vertices_impl(*this, positions, global_vertex_offset);
}

void GlobalTWP::Impl::debug_log_state(std::string_view stage)
{
    debug_log_state_impl(*this, stage);
}
}  // namespace uipc::backend::cuda
