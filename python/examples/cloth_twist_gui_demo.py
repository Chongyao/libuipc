"""Interactive Polyscope cloth twist demo based on Newton's cloth_twist setup."""

import argparse
import math
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
    from uipc.constitution import ElasticModuli2D, NeoHookeanShell, SoftPositionConstraint
    from uipc.geometry import label_surface, mesh_partition, trimesh
except ImportError as exc:
    raise SystemExit(
        "This demo requires the libuipc Python bindings. Build/install pyuipc first."
    ) from exc


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_WORKSPACE = REPO_ROOT / "output" / "python" / "cloth_twist_gui_demo"


def default_square_cloth_usd() -> pathlib.Path | None:
    try:
        import warp.examples
    except ModuleNotFoundError:
        return None
    return pathlib.Path(warp.examples.get_asset_directory()) / "square_cloth.usd"


def rodrigues_rotate(v: np.ndarray, axis: np.ndarray, theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1.0 - c)


def transform_newton_cloth_vertices(vertices: np.ndarray, scale: float) -> np.ndarray:
    # Newton uses rot=quat_from_axis_angle((0,0,1), pi/2), then scale=0.01.
    rot_z_90 = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return scale * (vertices @ rot_z_90.T)


def read_newton_square_cloth(cloth_usd: pathlib.Path, scale: float):
    try:
        from pxr import Usd
    except ModuleNotFoundError as exc:
        raise RuntimeError("Reading Newton's square_cloth.usd requires the pxr Python package.") from exc

    stage = Usd.Stage.Open(str(cloth_usd))
    if stage is None:
        raise FileNotFoundError(f"Cannot open cloth USD: {cloth_usd}")

    prim = stage.GetPrimAtPath("/root/cloth/cloth")
    if not prim.IsValid():
        raise RuntimeError(f"Cannot find /root/cloth/cloth in {cloth_usd}")

    points = np.asarray(prim.GetAttribute("points").Get(), dtype=np.float64)
    indices = np.asarray(prim.GetAttribute("faceVertexIndices").Get(), dtype=np.int32)
    counts = np.asarray(prim.GetAttribute("faceVertexCounts").Get(), dtype=np.int32)
    if not np.all(counts == 3):
        raise RuntimeError("Only triangular square_cloth.usd faces are supported.")

    vertices = np.array(transform_newton_cloth_vertices(points, scale), dtype=np.float64, copy=True)
    faces = np.array(indices.reshape(-1, 3), dtype=np.int32, copy=True)
    n = int(round(math.sqrt(vertices.shape[0])))
    if n * n != vertices.shape[0]:
        raise RuntimeError(f"Expected a square cloth vertex count, got {vertices.shape[0]}")

    mesh = trimesh(vertices, faces)
    label_surface(mesh)
    return mesh, vertices, faces, n


def make_generated_twist_cloth(n: int, scale: float):
    vertices = []
    faces = []

    offset = 0.5 * (n - 1)
    for i in range(n):
        for j in range(n):
            x = (j - offset) * scale
            z = (i - offset) * scale
            vertices.append((x, 0.0, z))

    for i in range(n - 1):
        for j in range(n - 1):
            v00 = i * n + j
            v10 = (i + 1) * n + j
            v01 = i * n + (j + 1)
            v11 = (i + 1) * n + (j + 1)
            faces.append((v00, v10, v11))
            faces.append((v00, v11, v01))

    vertices = transform_newton_cloth_vertices(np.asarray(vertices, dtype=np.float64), 1.0)
    faces = np.asarray(faces, dtype=np.int32)
    mesh = trimesh(vertices, faces)
    label_surface(mesh)
    return mesh, vertices, faces, n


def make_twist_cloth(args: argparse.Namespace):
    cloth_usd = pathlib.Path(args.cloth_usd).expanduser() if args.cloth_usd else None
    if cloth_usd is None:
        cloth_usd = default_square_cloth_usd()

    if cloth_usd is not None and cloth_usd.exists() and not args.generated_grid:
        return read_newton_square_cloth(cloth_usd, args.scale)

    return make_generated_twist_cloth(args.n, args.scale)


def boundary_indices(n: int):
    right_side = [i * n for i in range(n)]
    left_side = [n - 1 + i * n for i in range(n)]
    return right_side, left_side, right_side + left_side


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
    config["contact"]["friction"]["enable"] = args.friction
    config["contact"]["d_hat"] = args.d_hat
    config["line_search"]["max_iter"] = args.line_search_max_iter
    config["newton"]["max_iter"] = args.newton_max_iter
    config["newton"]["min_iter"] = args.newton_min_iter
    config["newton"]["velocity_tol"] = args.newton_velocity_tol
    config["linear_system"]["tol_rate"] = args.linear_tol
    config["linear_system"]["block_diagonal_scaling"]["enable"] = int(args.block_diagonal_scaling)

    scene = Scene(config)
    scene.animator().substep(args.animator_substep)
    scene.contact_tabular().default_model(0.0, 1.0e9)
    default_contact = scene.contact_tabular().default_element()

    mesh, rest_positions, faces, cloth_n = make_twist_cloth(args)
    if args.mas:
        mesh_partition(mesh, args.part_size)

    shell = NeoHookeanShell()
    soft_position = SoftPositionConstraint()
    moduli = ElasticModuli2D.youngs_poisson(args.young, args.poisson)
    shell.apply_to(mesh, moduli, args.density, args.thickness)
    soft_position.apply_to(mesh, args.constraint_strength)
    default_contact.apply_to(mesh)

    right_side, left_side, constrained_ids = boundary_indices(cloth_n)

    is_fixed = view(mesh.vertices().find(builtin.is_fixed))
    is_fixed[:] = 0

    is_constrained = view(mesh.vertices().find(builtin.is_constrained))
    aim_position = view(mesh.vertices().find(builtin.aim_position))
    is_constrained[:] = 0
    aim_position[:] = rest_positions.reshape(-1, 3, 1)
    is_constrained[constrained_ids] = 1

    axes = np.zeros((len(constrained_ids), 3), dtype=np.float64)
    axes[: len(right_side)] = np.array([0.0, 1.0, 0.0])
    axes[len(right_side) :] = np.array([0.0, -1.0, 0.0])
    constrained_rest_positions = rest_positions[constrained_ids]
    roots = np.einsum("ij,ij->i", constrained_rest_positions, axes)[:, None] * axes
    roots_to_points = constrained_rest_positions - roots

    cloth_object = scene.objects().create("twist_cloth")
    cloth_object.geometries().create(mesh)

    def animate_twist(info):
        geo = info.geo_slots()[0].geometry()
        constrained = view(geo.vertices().find(builtin.is_constrained))
        target = view(geo.vertices().find(builtin.aim_position))

        constrained[:] = 0
        constrained[constrained_ids] = 1

        t = min(max(info.frame() - 1, 0) * info.dt(), args.rotation_end_time)
        theta = args.angular_velocity * t
        for local_id, vertex_id in enumerate(constrained_ids):
            p = roots[local_id] + rodrigues_rotate(roots_to_points[local_id], axes[local_id], theta)
            target[vertex_id] = p.reshape(3, 1)

    scene.animator().insert(cloth_object, animate_twist)

    world.init(scene)
    if not world.is_valid():
        raise RuntimeError("World failed sanity check; scene is not valid.")

    return {
        "engine": engine,
        "world": world,
        "scene": scene,
        "scene_io": SceneIO(scene),
        "rest_positions": rest_positions,
        "faces": faces,
        "cloth_n": cloth_n,
        "constrained_ids": constrained_ids,
    }


def run_gui(args: argparse.Namespace):
    state = build_scene(args)
    world = state["world"]
    scene_io = state["scene_io"]
    cloth_n = state["cloth_n"]

    ps.init()
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("none")

    surface = scene_io.simplicial_surface()
    vertices = surface.positions().view().reshape(-1, 3)
    triangles = surface.triangles().topo().view().reshape(-1, 3)
    mesh = ps.register_surface_mesh("cloth_twist", vertices, triangles)
    mesh.set_edge_width(args.edge_width)

    ui = {
        "run": args.run,
        "fps": 0.0,
        "step_ms": 0.0,
        "engine": state["engine"],
    }

    def update_mesh():
        surf = scene_io.simplicial_surface()
        mesh.update_vertex_positions(surf.positions().view().reshape(-1, 3))

    def advance_one():
        t0 = time.perf_counter()
        world.advance()
        if not world.is_valid():
            ui["run"] = False
            return
        world.retrieve()
        update_mesh()
        t1 = time.perf_counter()
        dt = t1 - t0
        ui["step_ms"] = 1000.0 * dt
        ui["fps"] = 1.0 / max(dt, 1.0e-12)
        if args.max_frames > 0 and world.frame() >= args.max_frames:
            ui["run"] = False

    def on_update():
        if psim.Button("run / pause"):
            ui["run"] = not ui["run"]

        psim.SameLine()
        if psim.Button("step"):
            advance_one()

        if ui["run"]:
            advance_one()

        psim.Separator()
        psim.TextUnformatted("cloth_twist")
        psim.TextUnformatted(f"Frame: {world.frame()}")
        psim.TextUnformatted(f"Contact: {args.contact}")
        psim.TextUnformatted(f"N: {cloth_n} x {cloth_n}")
        psim.TextUnformatted(f"Constrained vertices: {2 * cloth_n}")
        psim.TextUnformatted(f"Constraint strength: {args.constraint_strength:.3e}")
        psim.TextUnformatted(f"Angular velocity: {args.angular_velocity:.4f} rad/s")
        psim.TextUnformatted(
            f"Newton iter: [{args.newton_min_iter}, {args.newton_max_iter}], "
            f"velocity tol: {args.newton_velocity_tol:.3e}"
        )
        psim.TextUnformatted(f"Step FPS: {ui['fps']:.3f}")
        psim.TextUnformatted(f"Step time: {ui['step_ms']:.3f} ms")

    ps.set_user_callback(on_update)
    ps.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=str(DEFAULT_WORKSPACE))
    parser.add_argument("--contact", choices=("ipc", "al-ipc"), default="ipc")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--cloth-usd", default="", help="path to Newton/Warp square_cloth.usd")
    parser.add_argument("--generated-grid", action="store_true", help="use generated fallback grid")
    parser.add_argument("--scale", type=float, default=0.01)
    parser.add_argument("--young", type=float, default=1.0e3)
    parser.add_argument("--poisson", type=float, default=0.49)
    parser.add_argument("--density", type=float, default=0.2)
    parser.add_argument("--thickness", type=float, default=0.0002)
    parser.add_argument("--gravity", type=float, default=0.0)
    parser.add_argument("--angular-velocity", type=float, default=math.pi / 3.0)
    parser.add_argument("--rotation-end-time", type=float, default=10.0)
    parser.add_argument("--constraint-strength", type=float, default=1.0e6)
    parser.add_argument("--d-hat", type=float, default=0.001)
    parser.add_argument("--line-search-max-iter", type=int, default=16)
    parser.add_argument("--newton-max-iter", type=int, default=1024)
    parser.add_argument("--newton-min-iter", type=int, default=1)
    parser.add_argument("--newton-velocity-tol", type=float, default=0.05)
    parser.add_argument("--linear-tol", type=float, default=1.0e-3)
    parser.add_argument("--animator-substep", type=int, default=10)
    parser.add_argument("--mas", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--part-size", type=int, default=16)
    parser.add_argument("--friction", action="store_true")
    parser.add_argument("--block-diagonal-scaling", action="store_true")
    parser.add_argument("--run", action="store_true", help="start simulation immediately")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means no limit")
    parser.add_argument("--edge-width", type=float, default=0.6)
    return parser.parse_args()


if __name__ == "__main__":
    run_gui(parse_args())
