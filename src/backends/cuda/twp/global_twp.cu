#include <twp/global_twp.h>
#include <pipeline/twp_pipeline_flag.h>
#include <sim_engine.h>
#include <global_geometry/global_vertex_manager.h>
#include <collision_detection/global_trajectory_filter.h>
#include <contact_system/global_contact_manager.h>
#include <uipc/common/timer.h>
#include <muda/buffer/buffer_launch.h>

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

    auto& config = world().scene().config();
    m_impl.max_iter_attr = config.find<IndexT>("contact/twp/max_iter");
    m_impl.eps_attr      = config.find<Float>("contact/twp/eps");
    m_impl.d_min_attr    = config.find<Float>("contact/twp/d_min");
    m_impl.d_max_attr    = config.find<Float>("contact/twp/d_max");
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
    // Placeholder for Algorithm 1 line 8:
    // P <- Proximity_Search(x^(l), Dmax)
    //
    // Later this should reuse GlobalTrajectoryFilter/contact candidate buffers,
    // but the first framework pass intentionally keeps P empty.
}

void GlobalTWP::Impl::backward()
{
    Timer timer{"TWP Backward"};
    // Placeholder for Algorithm 1 line 11:
    // y^(l+1) <- Backward(y^(l), x^(l), y^[k+1], P)
    //
    // With an empty P, y remains the target state.
    muda::BufferLaunch().copy<Vector3>(y.view(), std::as_const(target_y).view());
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
}

void GlobalTWP::init()
{
    m_impl.init();
}

void GlobalTWP::project()
{
    m_impl.project();
}
}  // namespace uipc::backend::cuda
