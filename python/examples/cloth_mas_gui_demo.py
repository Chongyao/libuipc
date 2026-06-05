"""Interactive Polyscope version of C++ test 60_fem_mas_cloth."""

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
    from uipc import Engine, Logger, Scene, SceneIO, World, builtin, view
    from uipc.constitution import ElasticModuli2D, NeoHookeanShell
    from uipc.geometry import label_surface, mesh_partition, trimesh
except ImportError as exc:
    raise SystemExit(
        "This demo requires the libuipc Python bindings. Build/install pyuipc first."
    ) from exc


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_WORKSPACE = REPO_ROOT / "output" / "python" / "cloth_mas_gui_demo"


def make_grid_cloth(n: int, cloth_size: float):
    spacing = cloth_size / n
    vertices = []
    faces = []

    for i in range(n + 1):
        for j in range(n + 1):
            vertices.append((i * spacing, 0.5, j * spacing))

    for i in range(n):
        for j in range(n):
            v00 = i * (n + 1) + j
            v10 = (i + 1) * (n + 1) + j
            v01 = i * (n + 1) + (j + 1)
            v11 = (i + 1) * (n + 1) + (j + 1)
            faces.append((v00, v10, v11))
            faces.append((v00, v11, v01))

    mesh = trimesh(
        np.asarray(vertices, dtype=np.float64),
        np.asarray(faces, dtype=np.int32),
    )
    label_surface(mesh)
    return mesh


def build_scene(args: argparse.Namespace):
    Logger.set_level(Logger.Level.Warn)

    workspace = pathlib.Path(args.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    engine = Engine("cuda", str(workspace))
    world = World(engine)

    config = Scene.default_config()
    config["gravity"] = [[0.0], [-9.8], [0.0]]
    config["contact"]["enable"] = int(args.contact_enable)
    config["contact"]["constitution"] = args.contact
    config["contact"]["friction"]["enable"] = False
    config["newton"]["max_iter"] = args.newton_max_iter
    config["newton"]["min_iter"] = args.newton_min_iter
    config["newton"]["velocity_tol"] = args.newton_velocity_tol
    config["linear_system"]["tol_rate"] = args.linear_tol
    config["linear_system"]["block_diagonal_scaling"]["enable"] = int(args.block_diagonal_scaling)

    scene = Scene(config)
    default_contact = scene.contact_tabular().default_element()

    mesh = make_grid_cloth(args.n, args.cloth_size)
    if args.mas:
        mesh_partition(mesh, args.part_size)

    shell = NeoHookeanShell()
    moduli = ElasticModuli2D.youngs_poisson(args.young, args.poisson)
    shell.apply_to(mesh, moduli)
    default_contact.apply_to(mesh)

    is_fixed = view(mesh.vertices().find(builtin.is_fixed))
    is_fixed[:] = 0
    is_fixed[0] = 1
    is_fixed[args.n] = 1

    obj = scene.objects().create("cloth")
    obj.geometries().create(mesh)

    world.init(scene)
    if not world.is_valid():
        raise RuntimeError("World failed sanity check; scene is not valid.")

    return engine, world, scene


def run_gui(args: argparse.Namespace):
    engine, world, scene = build_scene(args)
    scene_io = SceneIO(scene)

    ps.init()
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("none")

    surface = scene_io.simplicial_surface()
    vertices = surface.positions().view().reshape(-1, 3)
    triangles = surface.triangles().topo().view().reshape(-1, 3)
    mesh = ps.register_surface_mesh("60_fem_mas_cloth", vertices, triangles)
    mesh.set_edge_width(args.edge_width)

    state = {
        "run": args.run,
        "engine": engine,
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
        dt = t1 - t0
        state["step_ms"] = 1000.0 * dt
        state["fps"] = 1.0 / max(dt, 1.0e-12)
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
        psim.TextUnformatted("60_fem_mas_cloth")
        psim.TextUnformatted(f"Frame: {world.frame()}")
        psim.TextUnformatted(f"N: {args.n}")
        psim.TextUnformatted(f"Contact: {args.contact if args.contact_enable else 'off'}")
        psim.TextUnformatted(f"MAS: {'on' if args.mas else 'off'}")
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
    parser.add_argument("--workspace", default=str(DEFAULT_WORKSPACE))
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--cloth-size", type=float, default=0.5)
    parser.add_argument("--contact", choices=("ipc", "al-ipc", "twp"), default="ipc")
    parser.add_argument("--contact-enable", action="store_true")
    parser.add_argument("--mas", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--part-size", type=int, default=16)
    parser.add_argument("--young", type=float, default=1.0e6)
    parser.add_argument("--poisson", type=float, default=0.49)
    parser.add_argument("--newton-max-iter", type=int, default=1024)
    parser.add_argument("--newton-min-iter", type=int, default=1)
    parser.add_argument("--newton-velocity-tol", type=float, default=0.05)
    parser.add_argument("--linear-tol", type=float, default=1.0e-3)
    parser.add_argument("--block-diagonal-scaling", action="store_true")
    parser.add_argument("--run", action="store_true", help="start simulation immediately")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means no limit")
    parser.add_argument("--edge-width", type=float, default=0.6)
    return parser.parse_args()


if __name__ == "__main__":
    run_gui(parse_args())
