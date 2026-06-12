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
#include <collision_detection/vertex_half_plane_trajectory_filter.h>
#include <uipc/common/timer.h>
#include <muda/atomic.h>
#include <muda/launch/parallel_for.h>
#include <cmath>
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

namespace
{
bool is_finite(const TWPBackwardDiagnostics& backward,
               const TWPForwardDiagnostics&  forward)
{
    return std::isfinite(backward.violation_inf)
           && std::isfinite(backward.lcp_min_gap)
           && std::isfinite(backward.lcp_complementarity_inf)
           && std::isfinite(backward.lcp_projected_residual_inf)
           && std::isfinite(forward.residual_inf)
           && std::isfinite(forward.max_step)
           && std::isfinite(forward.min_alpha);
}
}  // namespace

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
    m_impl.repulsion_stiffness_attr =
        config.find<Float>("contact/twp/repulsion_stiffness");
    m_impl.backward_max_iter_attr =
        config.find<IndexT>("contact/twp/backward_max_iter");
    m_impl.backward_check_convergence_attr =
        config.find<IndexT>("contact/twp/backward_check_convergence");
    m_impl.self_collision_enable_attr =
        config.find<IndexT>("contact/twp/self_collision_enable");
    m_impl.self_contact_coloring_attr =
        config.find<IndexT>("contact/twp/self_contact_coloring");
    m_impl.debug_attr    = config.find<IndexT>("contact/twp/debug");

    on_init_scene(
        [this]
        {
            if(m_impl.global_trajectory_filter)
            {
                m_impl.simplex_trajectory_filter =
                    m_impl.global_trajectory_filter->find<SimplexTrajectoryFilter>();
                m_impl.vertex_half_plane_trajectory_filter =
                    m_impl.global_trajectory_filter->find<VertexHalfPlaneTrajectoryFilter>();
            }
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

Float GlobalTWP::Impl::compute_full_step_toi()
{
    constexpr Float ClearanceTolerance = 1e-12;
    const Float repulsion_stiffness =
        repulsion_stiffness_attr ? repulsion_stiffness_attr->view()[0] : Float{1e9};
    if(has_support_contact && repulsion_stiffness > 0.0)
        return 0.0;

    if(half_plane && half_plane_vertex_reporter
       && half_plane->positions().size() > 0)
    {
        Float target_clearance =
            compute_min_clearance(global_vertex_manager->positions());
        if(target_clearance <= ClearanceTolerance)
            return 0.0;
    }

    bool self_collision_enabled =
        !self_collision_enable_attr || self_collision_enable_attr->view()[0] != 0;
    if(!self_collision_enabled)
        return 1.0;

    if(!global_trajectory_filter)
        return 1.0;

    Timer timer{"TWP Full Step CCD Gate"};

    global_vertex_manager->setup_ccd(global_vertex_manager->prev_positions());
    global_trajectory_filter->detect(1.0);
    Float toi = global_trajectory_filter->filter_toi(1.0);
    global_vertex_manager->restore_ccd();

    return toi < 1.0 ? toi : 1.0;
}

bool GlobalTWP::Impl::debug_enabled() const
{
    return debug_attr && debug_attr->view()[0] != 0;
}

void GlobalTWP::Impl::reset_algorithm_state()
{
    context.reset(*global_vertex_manager.view());
    constraints.clear();
    has_support_contact = false;
}

void GlobalTWP::Impl::backward()
{
    TWPBackwardSolver::SolveInfo info;
    info.context                        = &context;
    info.constraints                    = &constraints;
    info.global_vertex_manager          = global_vertex_manager.view();
    info.finite_element_method          = finite_element_method.view();
    info.finite_element_vertex_reporter = finite_element_vertex_reporter.view();
    info.max_iterations = backward_max_iter_attr ? backward_max_iter_attr->view()[0] : 32;
    info.check_convergence = !backward_check_convergence_attr
                             || backward_check_convergence_attr->view()[0] != 0;
    info.use_gpu_self_contact_coloring =
        self_contact_coloring_attr && self_contact_coloring_attr->view()[0] != 0;
    backward_solver.solve(info);
}

void GlobalTWP::Impl::sync_half_plane_support_set()
{
    if(!vertex_half_plane_trajectory_filter)
        return;

    const IndexT contact_count = constraints.host_contact_constraint_count();
    support_PHs.resize(contact_count);
    support_PH_count = 0;

    if(contact_count > 0)
    {
        using namespace muda;
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(contact_count,
                   [types = constraints.types.viewer().name("constraint_types"),
                    vertex_ids =
                        constraints.vertex_ids.viewer().name("constraint_vertex_ids"),
                    primitive_ids = constraints.primitive_ids.viewer().name(
                        "constraint_primitive_ids"),
                    support_PHs = support_PHs.viewer().name("support_PHs"),
                    support_PH_count =
                        support_PH_count.viewer().name("support_PH_count")] __device__(
                       int c) mutable
                   {
                       if(types(c) != TWPConstraintType::VertexHalfPlane)
                           return;

                       Vector4i ids = vertex_ids(c);
                       Vector4i primitives = primitive_ids(c);
                       if(ids.x() < 0 || primitives.x() < 0)
                           return;

                       IndexT dst = atomic_add(support_PH_count.data(), 1);
                       support_PHs(dst) = Vector2i{ids.x(), primitives.x()};
                   });
    }

    IndexT host_count = 0;
    support_PH_count.view().copy_to(&host_count);
    support_PHs.resize(host_count);
    vertex_half_plane_trajectory_filter->replace_PHs(support_PHs.view());
    has_support_contact = host_count > 0;
}

void GlobalTWP::Impl::project()
{
    Timer timer{"TWP"};

    Float full_step_toi = compute_full_step_toi();
    if(full_step_toi >= 1.0)
    {
        if(debug_enabled())
            logger::warn("TWP skipped: full-step CCD is collision free.");
        return;
    }

    if(debug_enabled())
        logger::warn("TWP active: full-step CCD toi={}.", full_step_toi);

    reset_algorithm_state();
    prepare_edge_reference_length_squares();

    const IndexT max_iter = max_iter_attr->view()[0];
    const Float  eps      = eps_attr->view()[0];
    const Float  d_min    = d_min_attr->view()[0];
    const Float  d_max    = d_max_attr->view()[0];

    bool   converged = false;
    bool   nonfinite = false;
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

        if(!is_finite(context.diagnostics.backward, context.diagnostics.forward))
        {
            nonfinite = true;
            if(debug_enabled())
            {
                std::ostringstream ss;
                ss << "nonfinite-" << l;
                debug_log_state(ss.str());
            }
            break;
        }

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

    UIPC_ASSERT(!nonfinite,
                "TWP produced non-finite diagnostics before convergence. "
                "Refusing to write invalid projected positions back to the global state.");

    sync_half_plane_support_set();

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
