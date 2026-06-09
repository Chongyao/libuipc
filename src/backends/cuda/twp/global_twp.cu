#include <twp/global_twp.h>
#include <pipeline/twp_pipeline_flag.h>
#include <sim_engine.h>
#include <global_geometry/global_vertex_manager.h>
#include <collision_detection/global_trajectory_filter.h>
#include <contact_system/global_contact_manager.h>
#include <finite_element/finite_element_method.h>
#include <finite_element/finite_element_vertex_reporter.h>
#include <implicit_geometry/half_plane.h>
#include <implicit_geometry/half_plane_vertex_reporter.h>
#include <uipc/common/timer.h>
#include <muda/buffer/buffer_launch.h>
#include <muda/launch/parallel_for.h>
#include <muda/cub/device/device_reduce.h>
#include <string_view>

namespace uipc::backend
{
template <>
class SimSystemCreator<cuda::GlobalTWP>
{
  public:
    static U<cuda::GlobalTWP> create(cuda::SimEngine& engine)
    {
        auto scene = engine.world().scene();
        auto ctype_attr = scene.config().find<std::string>("contact/constitution");
        if(ctype_attr->view()[0] != "twp")
            return nullptr;
        return make_unique<cuda::GlobalTWP>(engine);
    }
};
}  // namespace uipc::backend

namespace uipc::backend::cuda
{
REGISTER_SIM_SYSTEM(GlobalTWP);

void GlobalTWP::do_build()
{
    require<TWPPipelineFlag>();
    m_impl.global_vertex_manager    = require<GlobalVertexManager>();
    m_impl.global_trajectory_filter = find<GlobalTrajectoryFilter>();
    m_impl.global_contact_manager   = find<GlobalContactManager>();
    m_impl.finite_element_method    = find<FiniteElementMethod>();
    m_impl.finite_element_vertex_reporter = find<FiniteElementVertexReporter>();
    m_impl.half_plane               = find<HalfPlane>();
    m_impl.half_plane_vertex_reporter = find<HalfPlaneVertexReporter>();

    auto& config = world().scene().config();
    m_impl.max_iter_attr = config.find<IndexT>("contact/twp/max_iter");
    m_impl.eps_attr      = config.find<Float>("contact/twp/eps");
    m_impl.d_min_attr    = config.find<Float>("contact/twp/d_min");
    m_impl.d_max_attr    = config.find<Float>("contact/twp/d_max");
    m_impl.debug_attr    = config.find<IndexT>("contact/twp/debug");

    on_write_scene([this] { debug_log_state("retrieve"); });
}

void GlobalTWP::Impl::init()
{
    ensure_storage(global_vertex_manager->positions().size());
}

void GlobalTWP::Impl::ensure_storage(SizeT vertex_count)
{
    x.resize(vertex_count);
    y.resize(vertex_count);
    target_y.resize(vertex_count);
    residual.resize(vertex_count);
    clearances.resize(vertex_count);
    penetration_flags.resize(vertex_count);

    SizeT plane_count = half_plane ? half_plane->positions().size() : 0;
    PHs.resize(vertex_count * plane_count);
}

bool GlobalTWP::Impl::debug_enabled() const
{
    return debug_attr && debug_attr->view()[0] != 0;
}

void GlobalTWP::Impl::reset_algorithm_state()
{
    auto positions      = global_vertex_manager->positions();
    auto prev_positions = global_vertex_manager->prev_positions();

    ensure_storage(positions.size());

    muda::BufferLaunch().copy<Vector3>(x.view(), prev_positions);
    muda::BufferLaunch().copy<Vector3>(y.view(), positions);
    muda::BufferLaunch().copy<Vector3>(target_y.view(), positions);
    residual.fill(1.0);

    remaining_search_bound = 0.0;
    residual_inf           = 1.0;
    max_forward_step       = 0.0;
}

void GlobalTWP::Impl::proximity_search(Float search_bound)
{
    Timer timer{"TWP Proximity Search"};

    if(!half_plane || !half_plane_vertex_reporter || half_plane->positions().size() == 0)
        return;

    SizeT vertex_count = target_y.size();
    SizeT plane_count  = half_plane->positions().size();
    SizeT max_count    = vertex_count * plane_count;
    if(PHs.size() < max_count)
        PHs.resize(max_count);

    PH_count = 0;

    IndexT plane_vertex_offset = half_plane_vertex_reporter->vertex_offset();

    using namespace muda;
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(vertex_count,
               [count = PH_count.viewer().name("PH_count"),
                PHs = PHs.viewer().name("PHs"),
                y = target_y.viewer().name("target_y"),
                thicknesses = global_vertex_manager->thicknesses().viewer().name("thicknesses"),
                contact_ids =
                    global_vertex_manager->contact_element_ids().viewer().name("contact_ids"),
                subscene_ids =
                    global_vertex_manager->subscene_element_ids().viewer().name("subscene_ids"),
                contact_mask =
                    global_contact_manager->contact_mask_tabular().viewer().name("contact_mask"),
                subscene_mask =
                    global_contact_manager->subscene_mask_tabular().viewer().name("subscene_mask"),
                plane_positions = half_plane->positions().viewer().name("plane_positions"),
                plane_normals = half_plane->normals().viewer().name("plane_normals"),
                plane_vertex_offset,
                plane_count,
                search_bound] __device__(int v) mutable
               {
                   if(v >= plane_vertex_offset && v < plane_vertex_offset + plane_count)
                       return;

                   for(IndexT h = 0; h < plane_count; ++h)
                   {
                       IndexT plane_v = plane_vertex_offset + h;

                       IndexT L = contact_ids(v);
                       IndexT R = contact_ids(plane_v);
                       if(contact_mask(L, R) == 0)
                           continue;

                       IndexT sL = subscene_ids(v);
                       IndexT sR = subscene_ids(plane_v);
                       if(subscene_mask(sL, sR) == 0)
                           continue;

                       const Vector3& P = plane_positions(h);
                       const Vector3& N = plane_normals(h);

                       Float signed_dist = (y(v) - P).dot(N);
                       Float min_dist    = thicknesses(v);

                       if(signed_dist < min_dist + search_bound)
                       {
                           IndexT I = atomic_add(count.data(), 1);
                           PHs(I)   = Vector2i{v, h};
                       }
                   }
               });

    h_PH_count = PH_count;
}

void GlobalTWP::Impl::backward()
{
    Timer timer{"TWP Backward"};
    muda::BufferLaunch().copy<Vector3>(y.view(), std::as_const(target_y).view());

    if(!half_plane)
        return;

    if(h_PH_count == 0)
        return;

    using namespace muda;
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(h_PH_count,
               [PHs = PHs.viewer().name("PHs"),
                y = y.viewer().name("y"),
                plane_positions = half_plane->positions().viewer().name("plane_positions"),
                plane_normals = half_plane->normals().viewer().name("plane_normals"),
                thicknesses = global_vertex_manager->thicknesses().viewer().name("thicknesses")] __device__(
                   int i) mutable
               {
                   Vector2i PH = PHs(i);
                   IndexT   v  = PH.x();
                   IndexT   h  = PH.y();

                   const Vector3& P = plane_positions(h);
                   const Vector3& N = plane_normals(h);

                   Float signed_dist = (y(v) - P).dot(N);
                   Float min_dist    = thicknesses(v);

                   if(signed_dist < min_dist)
                   {
                       y(v) += (min_dist - signed_dist) * N;
                   }
               });
}

Float GlobalTWP::Impl::compute_min_clearance(muda::CBufferView<Vector3> positions,
                                             IndexT global_vertex_offset)
{
    if(!half_plane || !half_plane_vertex_reporter || half_plane->positions().size() == 0)
        return 0.0;

    clearances.resize(positions.size());

    SizeT plane_count = half_plane->positions().size();
    IndexT plane_vertex_offset = half_plane_vertex_reporter->vertex_offset();

    using namespace muda;
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(positions.size(),
               [clearances = clearances.viewer().name("clearances"),
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

    DeviceReduce().Min(clearances.data(), min_clearance.data(), positions.size());

    return min_clearance;
}

IndexT GlobalTWP::Impl::count_penetrated_vertices(muda::CBufferView<Vector3> positions,
                                                  IndexT global_vertex_offset)
{
    if(!half_plane || !half_plane_vertex_reporter || half_plane->positions().size() == 0)
        return 0;

    penetration_flags.resize(positions.size());

    SizeT plane_count = half_plane->positions().size();
    IndexT plane_vertex_offset = half_plane_vertex_reporter->vertex_offset();

    using namespace muda;
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(positions.size(),
               [flags = penetration_flags.viewer().name("penetration_flags"),
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

    DeviceReduce().Sum(penetration_flags.data(),
                       penetration_count.data(),
                       penetration_flags.size());
    return penetration_count;
}

void GlobalTWP::Impl::forward()
{
    Timer timer{"TWP Forward"};
    // Placeholder for Algorithm 1 line 12:
    // x^(l+1), r^(l+1) <- Forward(x^(l), r^(l), y^(l+1) - x^(l), P)
    //
    // With an empty P, the full step is safe, so accept y and terminate.
    muda::BufferLaunch().copy<Vector3>(x.view(), std::as_const(y).view());
    residual.fill(0.0);
    residual_inf     = 0.0;
    max_forward_step = 0.0;
}

void GlobalTWP::Impl::project()
{
    Timer timer{"TWP"};

    reset_algorithm_state();

    const IndexT max_iter = max_iter_attr->view()[0];
    const Float  eps      = eps_attr->view()[0];
    const Float  d_min    = d_min_attr->view()[0];
    const Float  d_max    = d_max_attr->view()[0];

    for(IndexT l = 0; l <= max_iter; ++l)
    {
        if(remaining_search_bound < d_min)
        {
            proximity_search(d_max);
            remaining_search_bound = d_max;
        }

        backward();
        forward();

        remaining_search_bound -= 2.0 * max_forward_step;

        if(residual_inf < eps)
            break;
    }

    global_vertex_manager->overwrite_positions(x.view());
    if(finite_element_method)
    {
        UIPC_ASSERT(finite_element_vertex_reporter,
                    "FiniteElementVertexReporter is required to map global TWP positions "
                    "back to FEM positions.");
        auto fem_vertex_count = finite_element_method->xs().size();
        auto fem_vertex_offset = finite_element_vertex_reporter->vertex_offset();
        UIPC_ASSERT(fem_vertex_offset + fem_vertex_count <= x.size(),
                    "FEM global vertex range [{}, {}) exceeds TWP vertex count {}.",
                    fem_vertex_offset,
                    fem_vertex_offset + fem_vertex_count,
                    x.size());
        finite_element_method->overwrite_xs(x.view(fem_vertex_offset, fem_vertex_count));
    }

    debug_log_state("post-project");
}

void GlobalTWP::Impl::debug_log_state(std::string_view stage)
{
    if(!debug_enabled())
        return;

    Float target_min_clearance = 0.0;
    IndexT target_penetration_count = 0;
    if(target_y.size() == global_vertex_manager->positions().size())
    {
        target_min_clearance     = compute_min_clearance(target_y.view());
        target_penetration_count = count_penetrated_vertices(target_y.view());
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

    if(debug_enabled())
    {
        bool  has_half_plane = static_cast<bool>(half_plane);
        bool  has_half_plane_vertex_reporter =
            static_cast<bool>(half_plane_vertex_reporter);
        SizeT plane_count = half_plane ? half_plane->positions().size() : 0;
        IndexT plane_vertex_offset =
            half_plane_vertex_reporter ? half_plane_vertex_reporter->vertex_offset() : -1;
        if(h_PH_count > 0 || target_penetration_count > 0 || global_penetration_count > 0
           || fem_penetration_count > 0)
        {
            logger::warn(
                "TWP Debug[{}]: planes={}({}), plane_offset={}, PH={}, "
                "target[min={}, pen={}], global[min={}, pen={}], "
                "fem[offset={}, count={}, min={}, pen={}]",
                stage,
                plane_count,
                has_half_plane && has_half_plane_vertex_reporter,
                plane_vertex_offset,
                h_PH_count,
                target_min_clearance,
                target_penetration_count,
                global_min_clearance,
                global_penetration_count,
                fem_vertex_offset,
                fem_vertex_count,
                fem_min_clearance,
                fem_penetration_count);
        }
    }
}

void GlobalTWP::init()
{
    m_impl.init();
}

void GlobalTWP::project()
{
    m_impl.project();
}

void GlobalTWP::debug_log_state(std::string_view stage)
{
    m_impl.debug_log_state(stage);
}
}  // namespace uipc::backend::cuda
