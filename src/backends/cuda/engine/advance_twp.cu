#include <sim_engine.h>
#include <uipc/common/range.h>
#include <global_geometry/global_vertex_manager.h>
#include <global_geometry/global_simplicial_surface_manager.h>
#include <dytopo_effect_system/global_dytopo_effect_manager.h>
#include <contact_system/global_contact_manager.h>
#include <collision_detection/global_trajectory_filter.h>
#include <line_search/line_searcher.h>
#include <linear_system/global_linear_system.h>
#include <animator/global_animator.h>
#include <external_force/global_external_force_manager.h>
#include <diff_sim/global_diff_sim_manager.h>
#include <newton_tolerance/newton_tolerance_manager.h>
#include <time_integrator/time_integrator_manager.h>
#include <twp/global_twp.h>
#include <energy_component_flags.h>
#include <cstdlib>

namespace uipc::backend::cuda
{
void SimEngine::advance_twp()
{
    Float alpha = 1.0;
    Float beta  = 1.0;

    auto compute_dytopo_effect = [this]
    {
        if(m_global_dytopo_effect_manager)
        {
            Timer timer{"Compute DyTopo Effect"};
            GlobalDyTopoEffectManager::ComputeDyTopoEffectInfo info;
            info.component_flags(EnergyComponentFlags::All);
            m_global_dytopo_effect_manager->compute_dytopo_effect(info);
        }
    };

    auto compute_energy = [this](Float alpha) -> Float
    {
        m_global_vertex_manager->step_forward(alpha);
        m_line_searcher->step_forward(alpha);

        if(m_dump_surface->view()[0])
        {
            dump_global_surface();
        }

        return m_line_searcher->compute_energy(false);
    };

    auto step_animation = [this]()
    {
        if(m_global_external_force_manager)
        {
            Timer timer{"Clear External Forces"};
            m_global_external_force_manager->clear();
        }

        if(m_global_animator)
        {
            Timer timer{"Step Animation"};
            m_global_animator->step();
        }

        if(m_global_external_force_manager)
        {
            Timer timer{"Compute External Force Accelerations"};
            m_global_external_force_manager->step();
        }
    };

    auto compute_animation_substep_ratio = [this](SizeT newton_iter)
    {
        if(m_global_animator)
        {
            m_global_animator->compute_substep_ratio(newton_iter);
            logger::info("Animation Substep Ratio: {}", m_global_animator->substep_ratio());
        }
    };

    auto animation_reach_target = [this]()
    {
        if(m_global_animator)
            return m_global_animator->substep_ratio() >= 1.0;
        return true;
    };

    auto convergence_check = [&](SizeT newton_iter) -> bool
    {
        if(!animation_reach_target())
            return false;

        if(m_semi_implicit_enabled)
        {
            auto k_min = m_newton_min_iter->view()[0];
            auto eps   = m_semi_implicit_beta_tol;
            if(newton_iter >= k_min)
                beta = (1.0 - alpha) * beta;

            if(beta <= eps)
                return true;
        }

        NewtonToleranceManager::ResultInfo result_info;
        result_info.frame(m_current_frame);
        result_info.newton_iter(newton_iter);
        m_newton_tolerance_manager->check(result_info);

        return result_info.converged();
    };

    auto update_diff_parm = [this]()
    {
        if(m_global_diff_sim_manager)
        {
            Timer timer{"Update Diff Parm"};
            m_global_diff_sim_manager->update();
        }
    };

    auto check_line_search_iter = [this](SizeT line_search_iter_after_loop)
    {
        if(line_search_iter_after_loop >= m_line_searcher->max_iter())
        {
            logger::warn("Line Search Exits with Max Iteration: {} (Frame={}, Newton={})",
                         m_line_searcher->max_iter(),
                         m_current_frame,
                         m_newton_iter);

            if(m_strict_mode->view()[0])
            {
                throw SimEngineException("StrictMode: Line Search Exits with Max Iteration");
            }
        }
    };

    auto check_newton_iter = [this](IndexT newton_iter_after_loop)
    {
        auto newton_max = m_newton_max_iter->view()[0];
        if(newton_iter_after_loop >= newton_max)
        {
            logger::warn("Newton Iteration Exits with Max Iteration: {} (Frame={})",
                         newton_max,
                         m_current_frame);

            if(m_strict_mode->view()[0])
            {
                throw SimEngineException("StrictMode: Newton Iteration Exits with Max Iteration");
            }
        }
        else
        {
            logger::info("Newton Iteration Converged with Iteration Count: {}, Bound: [{}, {}]",
                         newton_iter_after_loop,
                         m_newton_min_iter->view()[0],
                         newton_max);
        }
    };

    constexpr bool AbortOnException = uipc::RUNTIME_CHECK;

    auto pipeline = [&]() noexcept(AbortOnException)
    {
        Timer timer{"Pipeline"};

        ++m_current_frame;

        logger::info(R"(>>> Begin Frame: {})", m_current_frame);

        {
            Timer timer{"Rebuild Scene"};
            m_state = SimEngineState::RebuildScene;
            {
                event_rebuild_scene();
            }

            world().scene().solve_pending();
            update_diff_parm();
        }

        {
            Timer timer{"Simulation"};

            m_global_vertex_manager->update_attributes();
            [[maybe_unused]] AABB bbox =
                m_global_vertex_manager->compute_vertex_bounding_box();

            m_global_vertex_manager->record_prev_positions();

            m_state = SimEngineState::PredictMotion;
            step_animation();
            m_time_integrator_manager->predict_dof();

            m_newton_tolerance_manager->pre_newton(m_current_frame);

            auto   newton_max_iter = m_newton_max_iter->view()[0];
            auto   newton_min_iter = m_newton_min_iter->view()[0];
            beta                   = 1.0;
            IndexT newton_iter     = 0;
            for(; newton_iter < newton_max_iter; ++newton_iter)
            {
                Timer timer{"Newton Iteration"};
                m_newton_iter = newton_iter;

                compute_animation_substep_ratio(newton_iter);

                m_state = SimEngineState::ComputeDyTopoEffect;
                compute_dytopo_effect();

                m_state = SimEngineState::SolveGlobalLinearSystem;
                {
                    Timer timer{"Solve Global Linear System"};
                    m_global_linear_system->solve();
                }

                m_global_vertex_manager->collect_vertex_displacements();

                m_state = SimEngineState::LineSearch;
                {
                    Timer timer{"Line Search"};

                    alpha = 1.0;

                    m_line_searcher->record_start_point();
                    m_global_vertex_manager->record_start_point();

                    if(m_dump_surface->view()[0])
                    {
                        dump_global_surface_pre_ccd(newton_iter);
                    }

                    bool  converged        = convergence_check(newton_iter);
                    SizeT line_search_iter = 0;

                    bool line_search_enabled =
                        !m_line_search_enable || m_line_search_enable->view()[0] != 0;
                    if(line_search_enabled)
                    {
                        Float E0 = m_line_searcher->compute_energy(true);

                        for(; line_search_iter < m_line_searcher->max_iter();
                            ++line_search_iter)
                        {
                            Timer timer{"Line Search Iteration"};
                            m_line_search_iter = line_search_iter;

                            Float E = compute_energy(alpha);

                            if(converged)
                                break;

                            bool energy_decrease = (E <= E0);
                            if(energy_decrease)
                                break;

                            alpha /= 2;
                        }

                        check_line_search_iter(line_search_iter);
                    }
                    else
                    {
                        m_line_search_iter = 0;
                        m_global_vertex_manager->step_forward(alpha);
                        m_line_searcher->step_forward(alpha);
                    }

                    bool terminated = converged && (newton_iter >= newton_min_iter);
                    if(terminated)
                        break;
                }
            }

            if(m_global_twp)
            {
                m_state = SimEngineState::TWP;
                m_global_twp->project();
            }

            m_state = SimEngineState::UpdateVelocity;
            {
                Timer timer{"Update Velocity"};
                m_time_integrator_manager->update_state();
            }

            check_newton_iter(newton_iter);
        }

        logger::info("<<< End Frame: {}", m_current_frame);
    };

    try
    {
        if(std::getenv("UIPC_ENABLE_TIMING"))
        {
            Timer::enable_all();
            pipeline();
            Timer::report(std::cout);
            Timer::disable_all();
        }
        else
        {
            Timer::disable_all();
            pipeline();
        }
    }
    catch(const SimEngineException& e)
    {
        logger::error("Engine Advance Error: {}", e.what());
        status().push_back(core::EngineStatus::error(e.what()));
    }
    catch(const std::exception& e)
    {
        UIPC_ASSERT(false, "Unexpected Exception: {}", e.what());
    }
}
}  // namespace uipc::backend::cuda
