#include <twp/global_twp.h>
#include <pipeline/twp_pipeline_flag.h>
#include <sim_engine.h>
#include <global_geometry/global_vertex_manager.h>
#include <global_geometry/global_simplicial_surface_manager.h>
#include <collision_detection/global_trajectory_filter.h>
#include <collision_detection/simplex_trajectory_filter.h>
#include <contact_system/global_contact_manager.h>
#include <finite_element/finite_element_method.h>
#include <finite_element/finite_element_vertex_reporter.h>
#include <implicit_geometry/half_plane.h>
#include <implicit_geometry/half_plane_vertex_reporter.h>
#include <uipc/common/timer.h>
#include <sstream>
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
    m_impl.global_simplicial_surface_manager = find<GlobalSimplicialSurfaceManager>();
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
    m_impl.edge_sigma_attr = config.find<Float>("contact/twp/edge_sigma");
    m_impl.debug_attr    = config.find<IndexT>("contact/twp/debug");

    on_init_scene(
        [this]
        {
            if(m_impl.global_trajectory_filter)
                m_impl.simplex_trajectory_filter =
                    m_impl.global_trajectory_filter->find<SimplexTrajectoryFilter>();
        });
    on_write_scene([this] { debug_log_state("retrieve"); });
}

void GlobalTWP::Impl::init()
{
    ensure_storage(global_vertex_manager->positions().size());
}

void GlobalTWP::Impl::ensure_storage(SizeT vertex_count)
{
    context.ensure_storage(vertex_count);

    SizeT plane_count = half_plane ? half_plane->positions().size() : 0;
    SizeT edge_count  = global_simplicial_surface_manager ?
                           global_simplicial_surface_manager->surf_edges().size() :
                           0;
    constraints.resize(vertex_count * plane_count + edge_count);
}

bool GlobalTWP::Impl::debug_enabled() const
{
    return debug_attr && debug_attr->view()[0] != 0;
}

void GlobalTWP::Impl::reset_algorithm_state()
{
    context.reset(*global_vertex_manager.view());
    constraints.clear();
}

void GlobalTWP::Impl::backward()
{
    TWPBackwardSolver::SolveInfo info;
    info.context                        = &context;
    info.constraints                    = &constraints;
    info.global_vertex_manager          = global_vertex_manager.view();
    info.finite_element_method          = finite_element_method.view();
    info.finite_element_vertex_reporter = finite_element_vertex_reporter.view();
    backward_solver.solve(info);
}

void GlobalTWP::Impl::project()
{
    Timer timer{"TWP"};

    reset_algorithm_state();
    prepare_edge_reference_length_squares();

    const IndexT max_iter = max_iter_attr->view()[0];
    const Float  eps      = eps_attr->view()[0];
    const Float  d_min    = d_min_attr->view()[0];
    const Float  d_max    = d_max_attr->view()[0];

    bool   converged = false;
    IndexT step_count = 0;
    IndexT abnormal_debug_count = 0;
    for(IndexT l = 0; l <= max_iter; ++l)
    {
        step_count = l + 1;

        if(context.remaining_search_bound < d_min)
        {
            proximity_search(d_max);
            context.remaining_search_bound = d_max;
        }

        refresh_edge_constraints();
        backward();
        forward();

        if(debug_enabled())
        {
            bool abnormal_forward =
                context.diagnostics.forward.max_step > d_max * Float{100.0}
                || context.diagnostics.forward.min_alpha == 0.0
                || context.diagnostics.forward.residual_inf > Float{0.5};
            bool abnormal_backward =
                !context.diagnostics.backward.converged
                && context.diagnostics.backward.violation_inf > Float{1e-4};
            if((abnormal_forward || abnormal_backward)
               && (abnormal_debug_count < 16 || l % 32 == 0))
            {
                std::ostringstream ss;
                ss << "iter-" << l;
                debug_log_state(ss.str());
                ++abnormal_debug_count;
            }
        }

        context.remaining_search_bound -= 2.0 * context.diagnostics.forward.max_step;

        if(context.diagnostics.forward.residual_inf < eps)
        {
            converged = true;
            break;
        }
    }

    if(!converged && debug_enabled())
    {
        logger::warn(
            "TWP exhausted: residual={}, eps={}, steps={}, max_iter={}, "
            "max_step={}, min_alpha={}, limited={}",
            context.diagnostics.forward.residual_inf,
            eps,
            step_count,
            max_iter,
            context.diagnostics.forward.max_step,
            context.diagnostics.forward.min_alpha,
            context.diagnostics.forward.limited_vertices);
    }

    global_vertex_manager->overwrite_positions(context.x.view());
    if(finite_element_method)
    {
        UIPC_ASSERT(finite_element_vertex_reporter,
                    "FiniteElementVertexReporter is required to map global TWP positions "
                    "back to FEM positions.");
        auto fem_vertex_count  = finite_element_method->xs().size();
        auto fem_vertex_offset = finite_element_vertex_reporter->vertex_offset();
        UIPC_ASSERT(fem_vertex_offset + fem_vertex_count <= context.x.size(),
                    "FEM global vertex range [{}, {}) exceeds TWP vertex count {}.",
                    fem_vertex_offset,
                    fem_vertex_offset + fem_vertex_count,
                    context.x.size());
        finite_element_method->overwrite_xs(
            context.x.view(fem_vertex_offset, fem_vertex_count));
    }

    debug_log_state("post-project");
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
