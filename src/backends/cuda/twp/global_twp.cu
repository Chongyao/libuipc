#include <twp/global_twp.h>
#include <pipeline/twp_pipeline_flag.h>
#include <sim_engine.h>
#include <global_geometry/global_vertex_manager.h>
#include <global_geometry/global_simplicial_surface_manager.h>
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

namespace
{
__forceinline__ __device__ void atomic_min_positive(Float* address, Float value)
{
    if constexpr(sizeof(Float) == sizeof(double))
    {
        auto address_as_ull = reinterpret_cast<unsigned long long int*>(address);
        auto old            = *address_as_ull;
        auto assumed        = old;
        auto value_as_ull =
            static_cast<unsigned long long int>(__double_as_longlong(value));
        while(value_as_ull < old)
        {
            assumed = old;
            old     = atomicCAS(address_as_ull, assumed, value_as_ull);
            if(old == assumed)
                break;
        }
    }
    else
    {
        auto address_as_ui = reinterpret_cast<unsigned int*>(address);
        auto old           = *address_as_ui;
        auto assumed       = old;
        auto value_as_ui   = __float_as_uint(static_cast<float>(value));
        while(value_as_ui < old)
        {
            assumed = old;
            old     = atomicCAS(address_as_ui, assumed, value_as_ui);
            if(old == assumed)
                break;
        }
    }
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
    m_impl.debug_attr    = config.find<IndexT>("contact/twp/debug");

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

void GlobalTWP::Impl::proximity_search(Float search_bound)
{
    Timer timer{"TWP Proximity Search"};

    constraints.clear();

    if(!half_plane || !half_plane_vertex_reporter || half_plane->positions().size() == 0)
        return;

    SizeT vertex_count = context.target_y.size();
    SizeT plane_count  = half_plane->positions().size();
    auto  surf_edges   = global_simplicial_surface_manager ?
                           global_simplicial_surface_manager->surf_edges() :
                           muda::CBufferView<Vector2i>{};
    SizeT edge_count   = surf_edges.size();
    SizeT max_count    = vertex_count * plane_count + edge_count;
    if(constraints.types.size() < max_count)
        constraints.resize(max_count);
    context.ensure_constraint_storage(max_count);
    context.contact_vertex_flags.fill(0);

    IndexT plane_vertex_offset = half_plane_vertex_reporter->vertex_offset();

    using namespace muda;
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(vertex_count,
               [count = constraints.count.viewer().name("constraint_count"),
                types = constraints.types.viewer().name("constraint_types"),
                vertex_ids =
                    constraints.vertex_ids.viewer().name("constraint_vertex_ids"),
                weights = constraints.weights.viewer().name("constraint_weights"),
                normals = constraints.normals.viewer().name("constraint_normals"),
                offsets = constraints.offsets.viewer().name("constraint_offsets"),
                contact_vertex_flags =
                    context.contact_vertex_flags.viewer().name("contact_vertex_flags"),
                y = context.target_y.viewer().name("target_y"),
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
                           types(I) = TWPConstraintType::VertexHalfPlane;
                           vertex_ids(I) = Vector4i{v, -1, -1, -1};
                           weights(I)    = Vector4{1.0, 0.0, 0.0, 0.0};
                           normals(I)    = N;
                           offsets(I)    = P.dot(N) + min_dist;
                           contact_vertex_flags(v) = 1;
                       }
                   }
               });

    constraints.h_contact_count = constraints.count;

    if(global_simplicial_surface_manager && edge_count > 0
       && constraints.h_contact_count > 0)
    {
        Float  sigma = edge_sigma_attr ? edge_sigma_attr->view()[0] : 1.1;
        UIPC_ASSERT(sigma > 0.0, "contact/twp/edge_sigma must be positive.");

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(edge_count,
                   [count = constraints.count.viewer().name("constraint_count"),
                    types = constraints.types.viewer().name("constraint_types"),
                    vertex_ids =
                        constraints.vertex_ids.viewer().name("constraint_vertex_ids"),
                    weights = constraints.weights.viewer().name("constraint_weights"),
                    normals = constraints.normals.viewer().name("constraint_normals"),
                    offsets = constraints.offsets.viewer().name("constraint_offsets"),
                    edges = surf_edges.viewer().name("surf_edges"),
                    contact_vertex_flags =
                        context.contact_vertex_flags.viewer().name("contact_vertex_flags"),
                    x = context.x.viewer().name("x"),
                    target_y = context.target_y.viewer().name("target_y"),
                    sigma] __device__(int e) mutable
                   {
                       Vector2i E = edges(e);
                       if(contact_vertex_flags(E.x()) == 0
                          && contact_vertex_flags(E.y()) == 0)
                           return;

                       IndexT I = atomic_add(count.data(), 1);
                       Vector3 x_ij = x(E.x()) - x(E.y());
                       Float ref_len = x_ij.norm();
                       Vector3 y_ij = target_y(E.x()) - target_y(E.y());
                       Float y_len = y_ij.norm();
                       Vector3 dir = Vector3::Zero();
                       Float min_len = 0.0;
                       if(ref_len > 1e-12)
                       {
                           dir     = y_len > 1e-12 ? y_ij / y_len : x_ij / ref_len;
                           min_len = ref_len / sigma;
                       }

                       types(I) = TWPConstraintType::EdgeLengthLowerBound;
                       vertex_ids(I) = Vector4i{E.x(), E.y(), -1, -1};
                       weights(I)    = Vector4{1.0, -1.0, 0.0, 0.0};
                       normals(I)    = dir;
                       offsets(I)    = min_len;
                   });

        constraints.h_count      = constraints.count;
        constraints.h_edge_count = constraints.h_count - constraints.h_contact_count;
    }

    constraints.h_count = constraints.count;
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

void GlobalTWP::Impl::forward()
{
    Timer timer{"TWP Forward"};

    constexpr Float ForwardSafety = 0.99;

    context.safe_step_alphas.fill(1.0);

    if(constraints.h_count > 0)
    {
        using namespace muda;
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(constraints.h_count,
                   [types = constraints.types.viewer().name("constraint_types"),
                    vertex_ids =
                        constraints.vertex_ids.viewer().name("constraint_vertex_ids"),
                    normals = constraints.normals.viewer().name("constraint_normals"),
                    offsets = constraints.offsets.viewer().name("constraint_offsets"),
                    safe_step_alphas =
                        context.safe_step_alphas.viewer().name("safe_step_alphas"),
                    x = context.x.viewer().name("x"),
                    y = context.y.viewer().name("y")] __device__(int i) mutable
                   {
                       if(types(i) != TWPConstraintType::VertexHalfPlane)
                           return;

                       IndexT  v = vertex_ids(i).x();
                       Vector3 N = normals(i);

                       Vector3 dir = y(v) - x(v);
                       Float   normal_dir = dir.dot(N);
                       Float   alpha_i    = 1.0;

                       if(normal_dir < 0.0)
                       {
                           Float clearance = x(v).dot(N) - offsets(i);
                           alpha_i = clearance / (-normal_dir);
                           alpha_i = max(Float{0.0}, min(Float{1.0}, alpha_i));
                       }

                       if(alpha_i < 1.0)
                       {
                           alpha_i = max(Float{0.0}, min(Float{1.0}, ForwardSafety * alpha_i));
                           atomic_min_positive(&safe_step_alphas(v), alpha_i);
                       }
                   });
    }

    using namespace muda;
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(context.x.size(),
               [x = context.x.viewer().name("x"),
                y = context.y.viewer().name("y"),
                residual = context.residual.viewer().name("residual"),
                forward_step_norms =
                    context.forward_step_norms.viewer().name("forward_step_norms"),
                safe_step_alphas =
                    context.safe_step_alphas.viewer().name("safe_step_alphas")] __device__(int i) mutable
               {
                   Float   alpha_i = safe_step_alphas(i);
                   Vector3 old_x = x(i);
                   Vector3 step  = alpha_i * (y(i) - old_x);
                   Vector3 new_x = old_x + step;
                   x(i)          = new_x;
                   residual(i) *= (1.0 - alpha_i);
                   forward_step_norms(i) = step.norm();
               });

    DeviceReduce().Max(context.residual.data(),
                       context.max_residual.data(),
                       context.residual.size());
    DeviceReduce().Max(context.forward_step_norms.data(),
                       context.max_step_norm.data(),
                       context.forward_step_norms.size());

    context.residual_inf     = context.max_residual;
    context.max_forward_step = context.max_step_norm;
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
        if(context.remaining_search_bound < d_min)
        {
            proximity_search(d_max);
            context.remaining_search_bound = d_max;
        }

        backward();
        forward();

        context.remaining_search_bound -= 2.0 * context.max_forward_step;

        if(context.residual_inf < eps)
            break;
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

    if(debug_enabled())
    {
        bool  has_half_plane = static_cast<bool>(half_plane);
        bool  has_half_plane_vertex_reporter =
            static_cast<bool>(half_plane_vertex_reporter);
        SizeT plane_count = half_plane ? half_plane->positions().size() : 0;
        IndexT plane_vertex_offset =
            half_plane_vertex_reporter ? half_plane_vertex_reporter->vertex_offset() : -1;
        if(constraints.h_count > 0 || target_penetration_count > 0 || global_penetration_count > 0
           || fem_penetration_count > 0)
        {
            logger::warn(
                "TWP Debug[{}]: planes={}({}), plane_offset={}, PH={}, edge={}, "
                "target[min={}, pen={}], global[min={}, pen={}], "
                "fem[offset={}, count={}, min={}, pen={}], "
                "backward[violation={}, iter={}, converged={}], "
                "forward[residual={}, max_step={}]",
                stage,
                plane_count,
                has_half_plane && has_half_plane_vertex_reporter,
                plane_vertex_offset,
                constraints.h_contact_count,
                constraints.h_edge_count,
                target_min_clearance,
                target_penetration_count,
                global_min_clearance,
                global_penetration_count,
                fem_vertex_offset,
                fem_vertex_count,
                fem_min_clearance,
                fem_penetration_count,
                context.backward_violation_inf,
                context.backward_iterations,
                context.backward_converged,
                context.residual_inf,
                context.max_forward_step);
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
