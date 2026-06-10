#include <twp/global_twp.h>
#include <global_geometry/global_vertex_manager.h>
#include <finite_element/finite_element_method.h>
#include <finite_element/finite_element_vertex_reporter.h>
#include <implicit_geometry/half_plane.h>
#include <implicit_geometry/half_plane_vertex_reporter.h>
#include <muda/launch/parallel_for.h>
#include <muda/cub/device/device_reduce.h>

namespace uipc::backend::cuda
{
Float GlobalTWP::Impl::compute_min_clearance(muda::CBufferView<Vector3> positions,
                                             IndexT global_vertex_offset)
{
    if(!half_plane || !half_plane_vertex_reporter || half_plane->positions().size() == 0)
        return 0.0;

    context.clearances.resize(positions.size());

    SizeT plane_count = half_plane->positions().size();
    IndexT plane_vertex_offset = half_plane_vertex_reporter->vertex_offset();

    using namespace muda;
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(positions.size(),
               [clearances = context.clearances.viewer().name("clearances"),
                positions = positions.viewer().name("positions"),
                thicknesses = global_vertex_manager->thicknesses().viewer().name("thicknesses"),
                plane_positions = half_plane->positions().viewer().name("plane_positions"),
                plane_normals = half_plane->normals().viewer().name("plane_normals"),
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

    DeviceReduce().Min(context.clearances.data(),
                       context.min_clearance.data(),
                       positions.size());

    return context.min_clearance;
}

IndexT GlobalTWP::Impl::count_penetrated_vertices(muda::CBufferView<Vector3> positions,
                                                  IndexT global_vertex_offset)
{
    if(!half_plane || !half_plane_vertex_reporter || half_plane->positions().size() == 0)
        return 0;

    context.penetration_flags.resize(positions.size());

    SizeT plane_count = half_plane->positions().size();
    IndexT plane_vertex_offset = half_plane_vertex_reporter->vertex_offset();

    using namespace muda;
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(positions.size(),
               [flags = context.penetration_flags.viewer().name("penetration_flags"),
                positions = positions.viewer().name("positions"),
                thicknesses = global_vertex_manager->thicknesses().viewer().name("thicknesses"),
                plane_positions = half_plane->positions().viewer().name("plane_positions"),
                plane_normals = half_plane->normals().viewer().name("plane_normals"),
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

    DeviceReduce().Sum(context.penetration_flags.data(),
                       context.penetration_count.data(),
                       context.penetration_flags.size());
    return context.penetration_count;
}

void GlobalTWP::Impl::debug_log_state(std::string_view stage)
{
    if(!debug_enabled())
        return;

    Float target_min_clearance = 0.0;
    IndexT target_penetration_count = 0;
    if(context.target_y.size() == global_vertex_manager->positions().size())
    {
        target_min_clearance     = compute_min_clearance(context.target_y.view());
        target_penetration_count = count_penetrated_vertices(context.target_y.view());
    }

    Float global_min_clearance = compute_min_clearance(global_vertex_manager->positions());
    IndexT global_penetration_count =
        count_penetrated_vertices(global_vertex_manager->positions());

    Float  fem_min_clearance = 0.0;
    IndexT fem_penetration_count = 0;
    IndexT fem_vertex_offset = -1;
    SizeT  fem_vertex_count  = 0;
    if(finite_element_method)
    {
        UIPC_ASSERT(finite_element_vertex_reporter,
                    "FiniteElementVertexReporter is required for TWP debug mapping.");
        fem_vertex_offset = finite_element_vertex_reporter->vertex_offset();
        fem_vertex_count  = finite_element_method->xs().size();
        fem_min_clearance =
            compute_min_clearance(finite_element_method->xs(), fem_vertex_offset);
        fem_penetration_count =
            count_penetrated_vertices(finite_element_method->xs(), fem_vertex_offset);
    }

    bool  has_half_plane = static_cast<bool>(half_plane);
    bool  has_half_plane_vertex_reporter =
        static_cast<bool>(half_plane_vertex_reporter);
    SizeT plane_count = half_plane ? half_plane->positions().size() : 0;
    IndexT plane_vertex_offset =
        half_plane_vertex_reporter ? half_plane_vertex_reporter->vertex_offset() : -1;
    if(constraints.host_total_constraint_count() > 0 || target_penetration_count > 0
       || global_penetration_count > 0
       || fem_penetration_count > 0)
    {
        logger::warn(
            "TWP Debug[{}]: planes={}({}), plane_offset={}, PH={}, edge={}, "
            "target[min={}, pen={}], global[min={}, pen={}], "
            "fem[offset={}, count={}, min={}, pen={}], "
            "backward[violation={}, lcp_gap={}, lcp_comp={}, lcp_proj_res={}, "
            "iter={}, converged={}], "
            "forward[residual={}, max_step={}, min_alpha={}, limited={}]",
            stage,
            plane_count,
            has_half_plane && has_half_plane_vertex_reporter,
            plane_vertex_offset,
            constraints.host_contact_constraint_count(),
            constraints.host_edge_constraint_count(),
            target_min_clearance,
            target_penetration_count,
            global_min_clearance,
            global_penetration_count,
            fem_vertex_offset,
            fem_vertex_count,
            fem_min_clearance,
            fem_penetration_count,
            context.backward_violation_inf,
            context.lcp_min_gap,
            context.lcp_complementarity_inf,
            context.lcp_projected_residual_inf,
            context.backward_iterations,
            context.backward_converged,
            context.residual_inf,
            context.max_forward_step,
            context.min_forward_alpha,
            context.forward_limited_vertices);
    }
}
}  // namespace uipc::backend::cuda
