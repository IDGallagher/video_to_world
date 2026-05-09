#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image


WINDOWS_DRIVE_RE = re.compile(r"^(?P<drive>[a-zA-Z]):[\\/](?P<rest>.*)$")


def _windows_to_wsl_path(path_text: str) -> str:
    match = WINDOWS_DRIVE_RE.match(path_text)
    if not match:
        return path_text
    drive = match.group("drive").lower()
    rest = match.group("rest").replace("\\", "/")
    return f"/mnt/{drive}/{rest}"


def _as_homogeneous44(extrinsics: np.ndarray) -> np.ndarray:
    ext = np.asarray(extrinsics, dtype=np.float32)
    if ext.shape == (4, 4):
        return ext.copy()
    if ext.shape == (3, 4):
        out = np.eye(4, dtype=np.float32)
        out[:3, :4] = ext
        return out
    raise ValueError(f"Expected extrinsics shape (3, 4) or (4, 4), got {ext.shape}")


def _load_selected_frame_paths(scene_root: Path) -> list[Path]:
    meta_path = scene_root / "preprocess_frames.json"
    if not meta_path.exists():
        return []

    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    frame_paths = payload.get("selected_frame_paths")
    if not isinstance(frame_paths, list):
        return []

    resolved: list[Path] = []
    for raw_path in frame_paths:
        path_text = str(raw_path or "").strip()
        if not path_text:
            continue
        path = Path(_windows_to_wsl_path(path_text))
        resolved.append(path)
    return resolved


def _load_frame_colors(frame_path: Path | None, *, width: int, height: int, fallback_rgb: np.ndarray) -> np.ndarray:
    if frame_path is None or not frame_path.exists():
        return np.broadcast_to(fallback_rgb.reshape(1, 1, 3), (height, width, 3)).copy()

    try:
        resample = Image.Resampling.BILINEAR
    except AttributeError:  # pragma: no cover - Pillow < 9.
        resample = Image.BILINEAR

    with Image.open(frame_path) as image:
        image = image.convert("RGB").resize((width, height), resample=resample)
        return np.asarray(image, dtype=np.uint8)


def _fallback_color(frame_number: int) -> np.ndarray:
    hue = (float(frame_number) * 0.61803398875) % 1.0
    chroma = 0.75
    value = 0.95
    h6 = hue * 6.0
    x = chroma * (1.0 - abs((h6 % 2.0) - 1.0))
    if h6 < 1.0:
        rgb = (chroma, x, 0.0)
    elif h6 < 2.0:
        rgb = (x, chroma, 0.0)
    elif h6 < 3.0:
        rgb = (0.0, chroma, x)
    elif h6 < 4.0:
        rgb = (0.0, x, chroma)
    elif h6 < 5.0:
        rgb = (x, 0.0, chroma)
    else:
        rgb = (chroma, 0.0, x)
    m = value - chroma
    return np.asarray([(channel + m) * 255.0 for channel in rgb], dtype=np.uint8)


def _frame_points_world(depth: np.ndarray, intrinsics: np.ndarray, extrinsics: np.ndarray, mask: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    yy, xx = np.nonzero(mask)
    if yy.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    pixels = np.stack(
        [
            xx.astype(np.float32, copy=False),
            yy.astype(np.float32, copy=False),
            np.ones_like(xx, dtype=np.float32),
        ],
        axis=0,
    )
    inv_intrinsics = np.linalg.inv(np.asarray(intrinsics, dtype=np.float32))
    rays = (inv_intrinsics @ pixels).T
    camera_points = rays * depth[yy, xx].astype(np.float32, copy=False)[:, None]
    camera_points_h = np.concatenate(
        [camera_points, np.ones((camera_points.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    c2w = np.linalg.inv(_as_homogeneous44(extrinsics))
    world_points = camera_points_h @ c2w.T
    return world_points[:, :3].astype(np.float32, copy=False)


def export_streaming_guide_ply(
    *,
    scene_root: Path,
    output_path: Path,
    guide_npz_path: Path | None = None,
) -> Path:
    if guide_npz_path is None:
        guide_npz_path = scene_root / "exports" / "npz" / "streaming_guide.npz"
    if not guide_npz_path.exists():
        raise FileNotFoundError(f"Streaming guide NPZ was not found: {guide_npz_path}")

    selected_frame_paths = _load_selected_frame_paths(scene_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with np.load(guide_npz_path) as guide:
        required = {"guide_indices", "depth", "extrinsics", "intrinsics"}
        missing = sorted(required.difference(guide.files))
        if missing:
            raise ValueError(f"Streaming guide NPZ is missing required arrays: {', '.join(missing)}")

        guide_indices = np.asarray(guide["guide_indices"], dtype=np.int64)
        depths = np.asarray(guide["depth"], dtype=np.float32)
        extrinsics = np.asarray(guide["extrinsics"], dtype=np.float32)
        intrinsics = np.asarray(guide["intrinsics"], dtype=np.float32)

        if depths.ndim != 3:
            raise ValueError(f"Expected guide depth shape (N, H, W), got {depths.shape}")
        if not (len(guide_indices) == depths.shape[0] == extrinsics.shape[0] == intrinsics.shape[0]):
            raise ValueError("Guide NPZ arrays have inconsistent frame counts.")

        masks = np.isfinite(depths) & (depths > 0.0)
        frame_counts = masks.reshape(masks.shape[0], -1).sum(axis=1).astype(np.int64)
        total_points = int(frame_counts.sum())
        if total_points <= 0:
            raise ValueError("Streaming guide contains no valid depth points.")

        vertex_dtype = np.dtype(
            [
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ]
        )
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {total_points}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )

        with output_path.open("wb") as handle:
            handle.write(header.encode("ascii"))
            for local_idx, guide_frame_idx in enumerate(guide_indices.tolist()):
                mask = masks[local_idx]
                count = int(frame_counts[local_idx])
                if count <= 0:
                    continue

                height, width = depths[local_idx].shape
                source_path = (
                    selected_frame_paths[int(guide_frame_idx)]
                    if 0 <= int(guide_frame_idx) < len(selected_frame_paths)
                    else None
                )
                colors_image = _load_frame_colors(
                    source_path,
                    width=width,
                    height=height,
                    fallback_rgb=_fallback_color(int(guide_frame_idx)),
                )
                points = _frame_points_world(
                    depths[local_idx],
                    intrinsics[local_idx],
                    extrinsics[local_idx],
                    mask,
                )
                colors = colors_image[mask]

                vertices = np.empty(count, dtype=vertex_dtype)
                vertices["x"] = points[:, 0]
                vertices["y"] = points[:, 1]
                vertices["z"] = points[:, 2]
                vertices["red"] = colors[:, 0]
                vertices["green"] = colors[:, 1]
                vertices["blue"] = colors[:, 2]
                vertices.tofile(handle)

    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export DA3 streaming global-guide NPZ as a colored PLY.")
    parser.add_argument("scene_root", help="Scene root containing exports/npz/streaming_guide.npz.")
    parser.add_argument("--output", required=True, help="Output PLY path.")
    parser.add_argument("--guide-npz", default="", help="Optional explicit streaming_guide.npz path.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    output_path = export_streaming_guide_ply(
        scene_root=Path(args.scene_root).resolve(),
        output_path=Path(args.output).resolve(),
        guide_npz_path=(Path(args.guide_npz).resolve() if str(args.guide_npz or "").strip() else None),
    )
    size_bytes = output_path.stat().st_size
    print(f"Exported guide PLY: {output_path}")
    print(f"Size: {size_bytes} bytes ({size_bytes / (1024.0 * 1024.0):.2f} MiB)")


if __name__ == "__main__":
    main()
