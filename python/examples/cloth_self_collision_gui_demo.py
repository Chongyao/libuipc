"""Minimal Polyscope scene for debugging cloth-cloth contact with TWP."""

import argparse
import os
import pathlib
import sys
import time

import numpy as np

try:
    import polyscope as ps
    import polyscope.imgui as psim
except ModuleNotFoundError as exc:
    raise SystemExit("This demo requires polyscope. Install it with `pip install polyscope`.") from exc

try:
    from uipc import Engine, Logger, Scene, SceneIO, World, builtin, view
    from uipc.constitution import DiscreteShellBending, ElasticModuli2D, NeoHookeanShell
    from uipc.geometry import label_surface, mesh_partition, trimesh
except ImportError as exc:
    raise SystemExit(
        "This demo requires the libuipc Python bindings. Build/install pyuipc first."
    ) from exc


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_WORKSPACE = REPO_ROOT / "output" / "python" / "cloth_self_collision_gui_demo"
DEFAULT_TWP_CONVERGENCE_MAX_ITER = 100000
_LOG_FILE_HANDLE = None


def make_flat_box_cloth(
    n: int,
    cloth_size: float,
    lower_y: float,
    upper_y: float,
    upper_x_offset: float = 0.0,
    side_layers: int = 4,
):
    side_layers = max(1, side_layers)
    spacing = cloth_size / n
    half = 0.5 * cloth_size
    vertices = []
    faces = []

    def lower_id(i: int, j: int) -> int:
        return i * (n + 1) + j

    def upper_id(i: int, j: int) -> int:
        return (n + 1) * (n + 1) + i * (n + 1) + j

    lower_count = (n + 1) * (n + 1)
    upper_count = lower_count
    side_start = lower_count + upper_count

    def side_vertex(layer: int, side: int, i: int) -> int:
        j = 0 if side == 0 else n
        if layer == 0:
            return lower_id(i, j)
        if layer == side_layers:
            return upper_id(i, j)
        return side_start + (layer - 1) * 2 * (n + 1) + side * (n + 1) + i

    def side_position(side: int, i: int, y: float, x_offset: float) -> tuple[float, float, float]:
        z = -half if side == 0 else half
        x = x_offset + i * spacing - half
        return (x, y, z)

    for i in range(n + 1):
        for j in range(n + 1):
            x = i * spacing - half
            z = j * spacing - half
            vertices.append((x, lower_y, z))

    for i in range(n + 1):
        for j in range(n + 1):
            x = upper_x_offset + i * spacing - half
            z = j * spacing - half
            vertices.append((x, upper_y, z))

    for layer in range(1, side_layers):
        t = layer / side_layers
        y = (1.0 - t) * lower_y + t * upper_y
        x_offset = t * upper_x_offset
        for side in range(2):
            for i in range(n + 1):
                vertices.append(side_position(side, i, y, x_offset))

    for i in range(n):
        for j in range(n):
            l00 = lower_id(i, j)
            l10 = lower_id(i + 1, j)
            l01 = lower_id(i, j + 1)
            l11 = lower_id(i + 1, j + 1)
            u00 = upper_id(i, j)
            u10 = upper_id(i + 1, j)
            u01 = upper_id(i, j + 1)
            u11 = upper_id(i + 1, j + 1)
            faces.append((l00, l11, l10))
            faces.append((l00, l01, l11))
            faces.append((u00, u10, u11))
            faces.append((u00, u11, u01))

    for layer in range(side_layers):
        for side in range(2):
            for i in range(n):
                a = side_vertex(layer, side, i)
                b = side_vertex(layer, side, i + 1)
                c = side_vertex(layer + 1, side, i)
                d = side_vertex(layer + 1, side, i + 1)
                if side == 0:
                    faces.append((a, b, d))
                    faces.append((a, d, c))
                else:
                    faces.append((a, d, b))
                    faces.append((a, c, d))

    mesh = trimesh(
        np.asarray(vertices, dtype=np.float64),
        np.asarray(faces, dtype=np.int32),
    )
    label_surface(mesh)
    return mesh


def make_bottom_cloth(n: int, cloth_size: float, y: float):
    spacing = cloth_size / n
    half = 0.5 * cloth_size
    vertices = []
    faces = []

    def vertex_id(i: int, j: int) -> int:
        return i * (n + 1) + j

    for i in range(n + 1):
        for j in range(n + 1):
            x = i * spacing - half
            z = j * spacing - half
            vertices.append((x, y, z))

    for i in range(n):
        for j in range(n):
            v00 = vertex_id(i, j)
            v10 = vertex_id(i + 1, j)
            v01 = vertex_id(i, j + 1)
            v11 = vertex_id(i + 1, j + 1)
            faces.append((v00, v11, v10))
            faces.append((v00, v01, v11))

    mesh = trimesh(
        np.asarray(vertices, dtype=np.float64),
        np.asarray(faces, dtype=np.int32),
    )
    label_surface(mesh)
    return mesh


def make_fixed_ground_cloth(size: float, y: float):
    half = 0.5 * size
    vertices = np.asarray(
        [
            [-half, y, -half],
            [half, y, -half],
            [half, y, half],
            [-half, y, half],
        ],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    mesh = trimesh(vertices, faces)
    label_surface(mesh)
    return mesh


def redirect_native_logs(args: argparse.Namespace):
    if not args.twp_debug:
        return

    global _LOG_FILE_HANDLE

    workspace = pathlib.Path(args.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    log_path = pathlib.Path(args.log_file).resolve() if args.log_file else workspace / "twp_debug.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing TWP debug log to {log_path}", file=sys.stderr, flush=True)
    _LOG_FILE_HANDLE = open(log_path, "w", buffering=1)
    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(_LOG_FILE_HANDLE.fileno(), sys.stdout.fileno())
    os.dup2(_LOG_FILE_HANDLE.fileno(), sys.stderr.fileno())


def apply_cloth_model(mesh, args: argparse.Namespace, default_contact):
    shell = NeoHookeanShell()
    bending = DiscreteShellBending()
    moduli = ElasticModuli2D.youngs_poisson(args.young, args.poisson)
    shell.apply_to(mesh, moduli, args.density, args.thickness)
    bending.apply_to(mesh, args.bending_stiffness)
    default_contact.apply_to(mesh)


def build_scene(args: argparse.Namespace):
    Logger.set_level(Logger.Level.Warn)

    workspace = pathlib.Path(args.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    engine = Engine("cuda", str(workspace))
    world = World(engine)

    config = Scene.default_config()
    config["gravity"] = [[0.0], [args.gravity], [0.0]]
    config["contact"]["enable"] = True
    config["contact"]["constitution"] = args.contact
    config["contact"]["friction"]["enable"] = False
    config["contact"]["d_hat"] = args.d_hat
    config["contact"]["twp"]["debug"] = int(args.twp_debug)
    config["contact"]["twp"]["max_iter"] = (
        args.twp_max_iter
        if args.twp_max_iter is not None
        else DEFAULT_TWP_CONVERGENCE_MAX_ITER
    )
    config["contact"]["twp"]["edge_sigma"] = args.twp_edge_sigma
    config["contact"]["twp"]["backward_max_iter"] = args.twp_backward_max_iter
    config["contact"]["twp"]["self_collision_enable"] = int(not args.disable_twp_self_collision)
    config["line_search"]["enable"] = int(not args.disable_line_search)
    config["line_search"]["max_iter"] = args.line_search_max_iter
    config["newton"]["max_iter"] = args.newton_max_iter
    config["newton"]["min_iter"] = args.newton_min_iter
    config["newton"]["velocity_tol"] = args.newton_velocity_tol
    config["linear_system"]["tol_rate"] = args.linear_tol
    config["linear_system"]["block_diagonal_scaling"]["enable"] = int(args.block_diagonal_scaling)

    scene = Scene(config)
    scene.contact_tabular().default_model(0.0, 1.0e9)
    default_contact = scene.contact_tabular().default_element()

    if args.bottom_only:
        cloth = make_bottom_cloth(args.n, args.cloth_size, args.lower_y)
        cloth_name = "bottom_cloth"
    else:
        cloth = make_flat_box_cloth(
            args.n,
            args.cloth_size,
            args.lower_y,
            args.upper_y,
            args.upper_x_offset,
            args.side_layers,
        )
        cloth_name = "flat_box_cloth"

    if args.mas:
        mesh_partition(cloth, args.part_size)

    apply_cloth_model(cloth, args, default_contact)

    fixed = view(cloth.vertices().find(builtin.is_fixed))
    fixed[:] = 0

    cloth_obj = scene.objects().create(cloth_name)
    cloth_obj.geometries().create(cloth)

    ground_mesh = make_fixed_ground_cloth(args.ground_size, args.ground_y)
    apply_cloth_model(ground_mesh, args, default_contact)
    ground_fixed = view(ground_mesh.vertices().find(builtin.is_fixed))
    ground_fixed[:] = 1
    ground_obj = scene.objects().create("ground")
    ground_obj.geometries().create(ground_mesh)

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
    ps.set_ground_plane_height(args.ground_y)

    surface = scene_io.simplicial_surface()
    vertices = surface.positions().view().reshape(-1, 3)
    triangles = surface.triangles().topo().view().reshape(-1, 3)
    mesh = ps.register_surface_mesh("cloth_self_collision", vertices, triangles)
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
        scene_label = (
            "cloth_self_collision_bottom_only"
            if args.bottom_only
            else "cloth_self_collision_flat_box"
        )
        psim.TextUnformatted(scene_label)
        psim.TextUnformatted(f"Frame: {world.frame()}")
        psim.TextUnformatted(f"N: {args.n}")
        psim.TextUnformatted(f"Contact: {args.contact}")
        psim.TextUnformatted(f"TWP self collision: {'off' if args.disable_twp_self_collision else 'on'}")
        psim.TextUnformatted(f"Step FPS: {state['fps']:.3f}")
        psim.TextUnformatted(f"Step time: {state['step_ms']:.3f} ms")

    ps.set_user_callback(on_update)
    ps.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=str(DEFAULT_WORKSPACE))
    parser.add_argument("--contact", choices=("ipc", "al-ipc", "twp"), default="twp")
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--cloth-size", type=float, default=0.5)
    parser.add_argument("--lower-y", type=float, default=0.12)
    parser.add_argument("--upper-y", type=float, default=0.20)
    parser.add_argument("--upper-x-offset", type=float, default=0.0)
    parser.add_argument("--side-layers", type=int, default=4)
    parser.add_argument("--bottom-only", action="store_true")
    parser.add_argument("--ground-y", type=float, default=0.0)
    parser.add_argument("--ground-size", type=float, default=10.0)
    parser.add_argument("--young", type=float, default=1.0e6)
    parser.add_argument("--poisson", type=float, default=0.49)
    parser.add_argument("--density", type=float, default=2.0e2)
    parser.add_argument("--thickness", type=float, default=0.0002)
    parser.add_argument("--bending-stiffness", type=float, default=5.0e3)
    parser.add_argument("--gravity", type=float, default=-9.8)
    parser.add_argument("--d-hat", type=float, default=0.001)
    parser.add_argument("--twp-debug", action="store_true")
    parser.add_argument("--twp-max-iter", type=int, default=None)
    parser.add_argument("--twp-edge-sigma", type=float, default=1.1)
    parser.add_argument("--twp-backward-max-iter", type=int, default=32)
    parser.add_argument("--disable-twp-self-collision", action="store_true")
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--line-search-max-iter", type=int, default=8)
    parser.add_argument("--disable-line-search", action="store_true")
    parser.add_argument("--newton-max-iter", type=int, default=1024)
    parser.add_argument("--newton-min-iter", type=int, default=1)
    parser.add_argument("--newton-velocity-tol", type=float, default=0.05)
    parser.add_argument("--linear-tol", type=float, default=1.0e-6)
    parser.add_argument("--block-diagonal-scaling", action="store_true")
    parser.add_argument("--mas", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--part-size", type=int, default=16)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--edge-width", type=float, default=0.6)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    redirect_native_logs(args)
    run_gui(args)
