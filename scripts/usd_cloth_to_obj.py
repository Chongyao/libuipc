#!/usr/bin/env python3
"""Export a USD mesh, such as the Newton shirt asset, to OBJ.

This uses OpenUSD's pxr Python API directly. It exports the authored mesh cage,
applies the USD world transform by default, and triangulates non-triangle faces
with a simple fan.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from pxr import Gf, Usd, UsdGeom


DEFAULT_INPUT = (
    "/home/CNF2026202838/workspace/newton/newton/examples/assets/unisex_shirt.usd"
)


def _time_code(value: float | None) -> Usd.TimeCode:
    if value is None:
        return Usd.TimeCode.Default()
    return Usd.TimeCode(value)


def _mesh_prims(stage: Usd.Stage) -> list[Usd.Prim]:
    return [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]


def _mesh_stats(prim: Usd.Prim, time: Usd.TimeCode) -> tuple[int, int, int]:
    mesh = UsdGeom.Mesh(prim)
    points = mesh.GetPointsAttr().Get(time) or []
    counts = mesh.GetFaceVertexCountsAttr().Get(time) or []
    indices = mesh.GetFaceVertexIndicesAttr().Get(time) or []
    return len(points), len(counts), len(indices)


def _pick_mesh(
    stage: Usd.Stage, prim_path: str | None, time: Usd.TimeCode
) -> Usd.Prim:
    if prim_path:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"Prim not found: {prim_path}")
        if not prim.IsA(UsdGeom.Mesh):
            raise RuntimeError(f"Prim is not a UsdGeomMesh: {prim_path}")
        return prim

    meshes = _mesh_prims(stage)
    if not meshes:
        raise RuntimeError("No UsdGeomMesh prims found in the input USD.")

    return max(meshes, key=lambda prim: _mesh_stats(prim, time)[0])


def _triangulate(
    counts: Iterable[int], indices: Iterable[int]
) -> list[tuple[int, int, int]]:
    index_list = list(indices)
    triangles: list[tuple[int, int, int]] = []
    offset = 0

    for count in counts:
        face = index_list[offset : offset + count]
        offset += count

        if count < 3:
            continue
        if count == 3:
            triangles.append((face[0], face[1], face[2]))
            continue

        for i in range(1, count - 1):
            triangles.append((face[0], face[i], face[i + 1]))

    if offset != len(index_list):
        raise RuntimeError(
            f"Face index data mismatch: consumed {offset}, total {len(index_list)}"
        )

    return triangles


def export_mesh_to_obj(
    input_path: Path,
    output_path: Path,
    prim_path: str | None,
    time: Usd.TimeCode,
    apply_world_transform: bool,
) -> None:
    stage = Usd.Stage.Open(str(input_path))
    if stage is None:
        raise RuntimeError(f"Failed to open USD file: {input_path}")

    prim = _pick_mesh(stage, prim_path, time)
    mesh = UsdGeom.Mesh(prim)

    points = mesh.GetPointsAttr().Get(time) or []
    counts = mesh.GetFaceVertexCountsAttr().Get(time) or []
    indices = mesh.GetFaceVertexIndicesAttr().Get(time) or []
    triangles = _triangulate(counts, indices)

    if apply_world_transform:
        xform_cache = UsdGeom.XformCache(time)
        matrix = xform_cache.GetLocalToWorldTransform(prim)
        points = [matrix.Transform(Gf.Vec3d(p)) for p in points]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        out.write(f"# Exported from {input_path}\n")
        out.write(f"# USD prim: {prim.GetPath()}\n")
        out.write(f"# vertices: {len(points)}, triangles: {len(triangles)}\n")
        for point in points:
            out.write(f"v {point[0]:.17g} {point[1]:.17g} {point[2]:.17g}\n")

        for a, b, c in triangles:
            out.write(f"f {a + 1} {b + 1} {c + 1}\n")

    subdivision = mesh.GetSubdivisionSchemeAttr().Get(time)
    print(f"Exported prim: {prim.GetPath()}")
    print(f"Subdivision scheme: {subdivision}")
    print(f"Vertices: {len(points)}")
    print(f"Triangles: {len(triangles)}")
    print(f"OBJ: {output_path}")


def list_meshes(input_path: Path, time: Usd.TimeCode) -> None:
    stage = Usd.Stage.Open(str(input_path))
    if stage is None:
        raise RuntimeError(f"Failed to open USD file: {input_path}")

    meshes = _mesh_prims(stage)
    if not meshes:
        print("No UsdGeomMesh prims found.")
        return

    for prim in meshes:
        point_count, face_count, index_count = _mesh_stats(prim, time)
        subdivision = UsdGeom.Mesh(prim).GetSubdivisionSchemeAttr().Get(time)
        print(
            f"{prim.GetPath()} points={point_count} "
            f"faces={face_count} indices={index_count} subdivision={subdivision}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a USD cloth mesh to OBJ using OpenUSD."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=DEFAULT_INPUT,
        help=f"Input USD file. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output OBJ file. Default: <input basename>.obj next to the input.",
    )
    parser.add_argument(
        "--prim",
        default=None,
        help="UsdGeomMesh prim path to export. Default: largest mesh in the stage.",
    )
    parser.add_argument(
        "--time",
        type=float,
        default=None,
        help="USD time sample to read. Default: Usd.TimeCode.Default().",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List mesh prims and exit.",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Do not apply the mesh prim's local-to-world transform.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else input_path.with_suffix(".obj")
    )
    time = _time_code(args.time)

    if args.list:
        list_meshes(input_path, time)
        return

    export_mesh_to_obj(
        input_path=input_path,
        output_path=output_path,
        prim_path=args.prim,
        time=time,
        apply_world_transform=not args.local,
    )


if __name__ == "__main__":
    main()
