#include <app/app.h>
#include <uipc/uipc.h>
#include <uipc/constitution/neo_hookean_shell.h>
#include <uipc/common/timer.h>
#include <chrono>
#include <cstdlib>
#include <fstream>

// Test: Cloth (2D triangle mesh) with MAS preconditioner.
// A sheet of cloth fixed at two corners, sagging under gravity.
TEST_CASE("60_fem_mas_cloth", "[fem][mas]")
{
    using namespace uipc;
    using namespace uipc::core;
    using namespace uipc::geometry;
    using namespace uipc::constitution;

    auto output_path = AssetDir::output_path(UIPC_RELATIVE_SOURCE_FILE);

    Engine engine{"cuda", output_path};
    World  world{engine};

    auto read_env_int = [](const char* name, int fallback)
    {
        if(const char* value = std::getenv(name))
            return std::atoi(value);
        return fallback;
    };

    auto read_env_string = [](const char* name, std::string fallback)
    {
        if(const char* value = std::getenv(name))
            return std::string{value};
        return fallback;
    };

    const int         N = read_env_int("UIPC_60_N", 10);
    const int         ProfileFrames = read_env_int("UIPC_60_FRAMES", 20);
    const int         contact_enable = read_env_int("UIPC_60_CONTACT_ENABLE", 1);
    const std::string contact_constitution =
        read_env_string("UIPC_60_CONTACT_CONSTITUTION", "ipc");

    auto config                             = test::Scene::default_config();
    config["gravity"]                       = Vector3{0, -9.8, 0};
    config["contact"]["enable"]             = contact_enable;
    config["contact"]["constitution"]       = contact_constitution;
    config["contact"]["friction"]["enable"] = false;
    config["linear_system"]["tol_rate"]     = 1e-3;
    test::Scene::dump_config(config, output_path);

    Scene scene{config};
    {
        NeoHookeanShell nhs;
        auto default_contact = scene.contact_tabular().default_element();

        auto object = scene.objects().create("cloth");

        // Build a grid cloth mesh (N x N quads -> 2*N*N triangles)
        constexpr Float cloth_size = 0.5;
        const Float spacing = cloth_size / N;

        vector<Vector3> Vs;
        vector<Vector3i> Fs;

        for(int i = 0; i <= N; i++)
            for(int j = 0; j <= N; j++)
                Vs.push_back(Vector3{i * spacing, 0.5, j * spacing});

        for(int i = 0; i < N; i++)
        {
            for(int j = 0; j < N; j++)
            {
                int v00 = i * (N + 1) + j;
                int v10 = (i + 1) * (N + 1) + j;
                int v01 = i * (N + 1) + (j + 1);
                int v11 = (i + 1) * (N + 1) + (j + 1);
                Fs.push_back(Vector3i{v00, v10, v11});
                Fs.push_back(Vector3i{v00, v11, v01});
            }
        }

        auto mesh = trimesh(Vs, Fs);
        label_surface(mesh);

        // Partition for MAS
        mesh_partition(mesh, 16);

        auto parm = ElasticModuli2D::youngs_poisson(1.0_MPa, 0.49);
        nhs.apply_to(mesh, parm);
        default_contact.apply_to(mesh);

        // Fix two corners
        auto is_fixed      = mesh.vertices().find<IndexT>(builtin::is_fixed);
        auto is_fixed_view = view(*is_fixed);
        is_fixed_view[0]         = 1;  // corner (0,0)
        is_fixed_view[N]         = 1;  // corner (0,N)

        object->geometries().create(mesh);
    }

    world.init(scene);
    REQUIRE(world.is_valid());

    SceneIO sio{scene};
    sio.write_surface(fmt::format("{}scene_surface{}.obj", output_path, 0));

    double simulation_seconds = 0.0;

    Timer::enable_all();
    GlobalTimer profile_timer{"60_fem_mas_cloth"};
    profile_timer.set_as_current();

    while(world.frame() < ProfileFrames)
    {
        auto frame_begin = std::chrono::high_resolution_clock::now();
        world.advance();
        REQUIRE(world.is_valid());
        world.retrieve();
        auto frame_end = std::chrono::high_resolution_clock::now();

        simulation_seconds +=
            std::chrono::duration<double>(frame_end - frame_begin).count();

        sio.write_surface(
            fmt::format("{}scene_surface{}.obj", output_path, world.frame()));
    }

    Timer::disable_all();

    const double dt                = config["dt"].get<double>();
    const double simulated_seconds = ProfileFrames * dt;
    const double step_fps          = ProfileFrames / simulation_seconds;
    const double realtime_factor   = simulated_seconds / simulation_seconds;

    fmt::println("60_fem_mas_cloth N: {}", N);
    fmt::println("60_fem_mas_cloth contact constitution: {}", contact_constitution);
    fmt::println("60_fem_mas_cloth contact enable: {}", contact_enable);
    fmt::println("Simulation step FPS: {:.6f} ({} steps / {:.6f} s)",
                 step_fps,
                 ProfileFrames,
                 simulation_seconds);
    fmt::println("Real-time factor: {:.6f}x ({:.6f} simulated s / {:.6f} wall s)",
                 realtime_factor,
                 simulated_seconds,
                 simulation_seconds);

    profile_timer.print_merged_timings();

    const auto timer_path = fmt::format("{}timer_60.json", output_path);
    std::ofstream ofs{timer_path};
    ofs << profile_timer.report_merged_as_json().dump(2);
    logger::info("Timer profile saved to {}", timer_path);
}
