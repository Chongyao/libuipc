#include <app/app.h>
#include <uipc/uipc.h>
#include <uipc/constitution/neo_hookean_shell.h>
#include <chrono>

// Test: Read a shirt cloth mesh from OBJ and let it fall onto an implicit ground.
TEST_CASE("94_fem_shirt_ground", "[fem][cloth][ground]")
{
    using namespace uipc;
    using namespace uipc::core;
    using namespace uipc::geometry;
    using namespace uipc::constitution;
    namespace fs = std::filesystem;

    auto output_path = AssetDir::output_path(UIPC_RELATIVE_SOURCE_FILE);

    Engine engine{"cuda", output_path};
    World  world{engine};

    auto config                             = test::Scene::default_config();
    config["gravity"]                       = Vector3{0, -9.8, 0};
    config["contact"]["enable"]             = true;
    config["contact"]["friction"]["enable"] = false;
    config["contact"]["d_hat"]              = 0.001;
    config["line_search"]["max_iter"]       = 8;
    config["linear_system"]["tol_rate"]     = 1e-3;
    test::Scene::dump_config(config, output_path);

    Scene scene{config};
    {
        const fs::path shirt_path = fs::path{AssetDir::output_path()} / "unisex_shirt.obj";
        INFO(fmt::format("Expected converted shirt OBJ at {}", shirt_path.string()));
        REQUIRE(fs::exists(shirt_path));

        NeoHookeanShell nhs;
        scene.contact_tabular().default_model(0.0, 1.0_GPa);
        auto default_contact = scene.contact_tabular().default_element();

        Transform pre_transform = Transform::Identity();
        pre_transform.scale(0.01);

        SimplicialComplexIO io{pre_transform};
        auto                shirt_mesh = io.read(shirt_path.string());
        label_surface(shirt_mesh);
        mesh_partition(shirt_mesh, 16);

        auto moduli = ElasticModuli2D::youngs_poisson(100.0_MPa, 0.49);
        nhs.apply_to(shirt_mesh, moduli, 2e2, 0.0002_m);
        default_contact.apply_to(shirt_mesh);

        // Keep cloth-ground contact, but avoid expensive cloth self-collision for this demo.
        auto self_collision = shirt_mesh.meta().find<IndexT>(builtin::self_collision);
        REQUIRE(self_collision);
        view(*self_collision)[0] = 0;

        auto shirt_object = scene.objects().create("shirt");
        shirt_object->geometries().create(shirt_mesh);

        auto ground_mesh = ground(0.0);
        default_contact.apply_to(ground_mesh);

        auto ground_object = scene.objects().create("ground");
        ground_object->geometries().create(ground_mesh);
    }

    world.init(scene);
    REQUIRE(world.is_valid());

    SceneIO sio{scene};
    sio.write_surface(fmt::format("{}scene_surface{}.obj", output_path, 0));

    constexpr int ProfileFrames = 80;
    double        simulation_seconds = 0.0;

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

    const double dt                = config["dt"].get<double>();
    const double simulated_seconds = ProfileFrames * dt;
    const double step_fps          = ProfileFrames / simulation_seconds;
    const double realtime_factor   = simulated_seconds / simulation_seconds;

    fmt::println("Simulation step FPS: {:.6f} ({} steps / {:.6f} s)",
                 step_fps,
                 ProfileFrames,
                 simulation_seconds);
    fmt::println("Real-time factor: {:.6f}x ({:.6f} simulated s / {:.6f} wall s)",
                 realtime_factor,
                 simulated_seconds,
                 simulation_seconds);
}
