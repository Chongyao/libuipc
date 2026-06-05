"""Interactive Polyscope version of C++ test 94_fem_shirt_ground.

Run after building/installing pyuipc and polyscope:
    python python/examples/shirt_ground_gui_demo.py

Controls:
    - run / pause: toggle continuous stepping
    - step: advance one simulation step
"""

import argparse
import pathlib
import time

import numpy as np

try:
    import polyscope as ps
    import polyscope.imgui as psim
except ModuleNotFoundError as exc:
    raise SystemExit("This demo requires polyscope. Install it with `pip install polyscope`.") from exc

try:
    from uipc import Engine, Logger, Matrix4x4, Scene, SceneIO, World
    from uipc.geometry import SimplicialComplexIO, ground, label_surface, mesh_partition
    from uipc.constitution import ElasticModuli2D, NeoHookeanShell
except ImportError as exc:
    raise SystemExit(
        "This demo requires the libuipc Python bindings. Build/install pyuipc first."
    ) from exc


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_SHIRT = REPO_ROOT / "output" / "unisex_shirt.obj"
DEFAULT_WORKSPACE = REPO_ROOT / "output" / "python" / "shirt_ground_gui_demo"


def build_scene(args: argparse.Namespace):
    Logger.set_level(Logger.Level.Warn)

    workspace = pathlib.Path(args.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    engine = Engine("cuda", str(workspace))
    world = World(engine)

    config = Scene.default_config()
    config["gravity"] = [[0.0], [-9.8], [0.0]]
    config["contact"]["enable"] = True
    config["contact"]["constitution"] = args.contact
    config["contact"]["friction"]["enable"] = False
    config["contact"]["d_hat"] = args.d_hat
    config["line_search"]["max_iter"] = args.line_search_max_iter
    config["newton"]["max_iter"] = args.newton_max_iter
    config["newton"]["min_iter"] = args.newton_min_iter
    config["newton"]["velocity_tol"] = args.newton_velocity_tol
    config["linear_system"]["tol_rate"] = args.linear_tol
    config["linear_system"]["block_diagonal_scaling"]["enable"] = int(args.block_diagonal_scaling)

    scene = Scene(config)
    scene.contact_tabular().default_model(0.0, 1.0e9)
    default_contact = scene.contact_tabular().default_element()

    shirt_path = pathlib.Path(args.shirt).resolve()
    if not shirt_path.exists():
        raise FileNotFoundError(
            f"Cannot find shirt OBJ: {shirt_path}\n"
            "Convert the USD first, or pass --shirt /path/to/unisex_shirt.obj."
        )

    pre_transform = Matrix4x4.Identity()
    pre_transform[0, 0] = args.scale
    pre_transform[1, 1] = args.scale
    pre_transform[2, 2] = args.scale

    io = SimplicialComplexIO(pre_transform)
    shirt_mesh = io.read(str(shirt_path))
    label_surface(shirt_mesh)
    mesh_partition(shirt_mesh, args.part_size)

    shell = NeoHookeanShell()
    moduli = ElasticModuli2D.youngs_poisson(args.young, args.poisson)
    shell.apply_to(shirt_mesh, moduli, args.density, args.thickness)
    default_contact.apply_to(shirt_mesh)

    shirt_object = scene.objects().create("shirt")
    shirt_object.geometries().create(shirt_mesh)

    ground_mesh = ground(0.0)
    default_contact.apply_to(ground_mesh)
    ground_object = scene.objects().create("ground")
    ground_object.geometries().create(ground_mesh)

    world.init(scene)
    if not world.is_valid():
        raise RuntimeError("World failed sanity check; scene is not valid.")

    return engine, world, scene


def run_gui(args: argparse.Namespace):
    engine, world, scene = build_scene(args)
    scene_io = SceneIO(scene)

    ps.init()
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("tile_reflection")
    ps.set_ground_plane_height(0.0)

    surface = scene_io.simplicial_surface()
    vertices = surface.positions().view().reshape(-1, 3)
    triangles = surface.triangles().topo().view().reshape(-1, 3)
    mesh = ps.register_surface_mesh("94_fem_shirt_ground", vertices, triangles)
    mesh.set_edge_width(args.edge_width)

    state = {
        "run": args.run,
        "engine": engine,  # keep native engine alive during the callback
        "last_wall_time": time.perf_counter(),
        "fps": 0.0,
        "step_ms": 0.0,
    }

    def update_mesh():
        surf = scene_io.simplicial_surface()
        mesh.update_vertex_positions(surf.positions().view().reshape(-1, 3))

    def advance_one():
        t0 = time.perf_counter()
        world.advance()
        if not world.is_valid():
            state["run"] = False
            return
        world.retrieve()
        update_mesh()
        t1 = time.perf_counter()
        state["step_ms"] = 1000.0 * (t1 - t0)
        state["fps"] = 1.0 / max(t1 - t0, 1.0e-12)
        if args.max_frames > 0 and world.frame() >= args.max_frames:
            state["run"] = False

    def on_update():
        if psim.Button("run / pause"):
            state["run"] = not state["run"]

        psim.SameLine()
        if psim.Button("step"):
            advance_one()

        if state["run"]:
            advance_one()

        psim.Separator()
        psim.TextUnformatted("94_fem_shirt_ground")
        psim.TextUnformatted(f"Frame: {world.frame()}")
        psim.TextUnformatted(f"Contact: {args.contact}")
        psim.TextUnformatted(
            f"Newton iter: [{args.newton_min_iter}, {args.newton_max_iter}], "
            f"velocity tol: {args.newton_velocity_tol:.3e}"
        )
        psim.TextUnformatted(f"Step FPS: {state['fps']:.3f}")
        psim.TextUnformatted(f"Step time: {state['step_ms']:.3f} ms")

    ps.set_user_callback(on_update)
    ps.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shirt", default=str(DEFAULT_SHIRT))
    parser.add_argument("--workspace", default=str(DEFAULT_WORKSPACE))
    parser.add_argument("--contact", choices=("ipc", "al-ipc"), default="ipc")
    parser.add_argument("--run", action="store_true", help="start simulation immediately")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means no limit")
    parser.add_argument("--scale", type=float, default=0.01)
    parser.add_argument("--part-size", type=int, default=16)
    parser.add_argument("--young", type=float, default=1.0e6)
    parser.add_argument("--poisson", type=float, default=0.49)
    parser.add_argument("--density", type=float, default=2.0e2)
    parser.add_argument("--thickness", type=float, default=0.0002)
    parser.add_argument("--d-hat", type=float, default=0.001)
    parser.add_argument("--line-search-max-iter", type=int, default=8)
    parser.add_argument("--newton-max-iter", type=int, default=1024)
    parser.add_argument("--newton-min-iter", type=int, default=1)
    parser.add_argument("--newton-velocity-tol", type=float, default=0.05)
    parser.add_argument("--linear-tol", type=float, default=1.0e-6)
    parser.add_argument("--block-diagonal-scaling", action="store_true")
    parser.add_argument("--edge-width", type=float, default=0.6)
    return parser.parse_args()


if __name__ == "__main__":
    run_gui(parse_args())
