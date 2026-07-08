"""Render VACE guide, mask, known depth, and authored cameras.

This is the v0.3 front half for extending a static DepthImageVolume:

  existing volume -> authored camera path -> guide video + inpaint mask

The renderer intentionally uses the DepthImageVolume itself as the renderable
cache. It unprojects the source volume's posed RGB-D images into asset-space
points, then z-buffers those points into a small authored camera move. Empty
pixels are grey in the guide and white in the VACE mask.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _frame_intrinsics(cameras: dict, frame: dict) -> tuple[float, float, float, float]:
    return (
        float(frame.get("fl_x", cameras["fl_x"])),
        float(frame.get("fl_y", cameras["fl_y"])),
        float(frame.get("cx", cameras["cx"])),
        float(frame.get("cy", cameras["cy"])),
    )


def _scaled_intrinsics(
    cameras: dict,
    frame: dict,
    *,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    src_w = int(cameras["w"])
    src_h = int(cameras["h"])
    sx = float(width) / float(src_w)
    sy = float(height) / float(src_h)
    fx, fy, cx, cy = _frame_intrinsics(cameras, frame)
    return fx * sx, fy * sy, cx * sx, cy * sy


def _decode_depth_u16(path: Path, *, near: float, far: float) -> np.ndarray:
    code = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if code is None:
        raise FileNotFoundError(path)
    if code.dtype != np.uint16:
        raise ValueError(f"Expected uint16 depth PNG, got {code.dtype}: {path}")
    depth = near + ((code.astype(np.float32) - 1.0) / 65534.0) * (far - near)
    depth[code == 0] = 0.0
    return depth


def _encode_depth_u16(depth: np.ndarray, *, near: float, far: float) -> np.ndarray:
    encoded = np.zeros(depth.shape, dtype=np.uint16)
    valid = np.isfinite(depth) & (depth > 0.0)
    if not np.any(valid):
        return encoded
    norm = (depth[valid] - near) / max(far - near, 1.0e-6)
    code = np.rint(np.clip(norm, 0.0, 1.0) * 65534.0 + 1.0)
    encoded[valid] = code.astype(np.uint16)
    return encoded


def _unproject_volume(
    volume_dir: Path,
    cameras: dict,
    *,
    max_points_per_frame: int,
    rng_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(rng_seed)
    all_points: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    for frame_idx, frame in enumerate(cameras["frames"]):
        depth_path = volume_dir / os.path.basename(frame["file_path"])
        color_path = volume_dir / os.path.basename(frame["color_path"])
        depth = _decode_depth_u16(depth_path, near=float(frame["near"]), far=float(frame["far"]))
        color = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
        if color is None:
            raise FileNotFoundError(color_path)
        if color.shape[:2] != depth.shape:
            color = cv2.resize(color, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_AREA)

        v, u = np.nonzero(depth > 0.0)
        if v.size == 0:
            continue
        if v.size > max_points_per_frame:
            sel = rng.choice(v.size, size=max_points_per_frame, replace=False)
            v = v[sel]
            u = u[sel]

        fx, fy, cx, cy = _frame_intrinsics(cameras, frame)
        z = depth[v, u].astype(np.float32)
        x = ((u.astype(np.float32) - cx) / fx) * z
        y = ((v.astype(np.float32) - cy) / fy) * z
        cam_points = np.stack([x, y, z, np.ones_like(z)], axis=1)
        c2w = np.asarray(frame["transform_matrix"], dtype=np.float32)
        world = (c2w @ cam_points.T).T[:, :3]

        all_points.append(world.astype(np.float32, copy=False))
        all_colors.append(color[v, u].astype(np.uint8, copy=False))
        print(f"loaded frame {frame_idx:04d}: {v.size} points")

    if not all_points:
        raise RuntimeError(f"No valid points found in {volume_dir}")
    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    print(f"source cache: {points.shape[0]} points")
    return points, colors


def _rotation_about_axis(axis: np.ndarray, angle_radians: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / max(np.linalg.norm(axis), 1.0e-12)
    x, y, z = axis
    c = math.cos(angle_radians)
    s = math.sin(angle_radians)
    C = 1.0 - c
    return np.array(
        [
            [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
        ],
        dtype=np.float64,
    )


def _look_at_c2w(
    camera_center: np.ndarray,
    target: np.ndarray,
    *,
    down_hint: np.ndarray,
) -> np.ndarray:
    forward = target - camera_center
    forward = forward / max(np.linalg.norm(forward), 1.0e-12)

    right = np.cross(down_hint, forward)
    if np.linalg.norm(right) < 1.0e-6:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    right = right / max(np.linalg.norm(right), 1.0e-12)

    down = np.cross(forward, right)
    down = down / max(np.linalg.norm(down), 1.0e-12)

    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = right
    c2w[:3, 1] = down
    c2w[:3, 2] = forward
    c2w[:3, 3] = camera_center
    return c2w


def _source_target_point(volume_dir: Path, cameras: dict, source_frame: dict) -> tuple[np.ndarray, float]:
    depth = _decode_depth_u16(
        volume_dir / os.path.basename(source_frame["file_path"]),
        near=float(source_frame["near"]),
        far=float(source_frame["far"]),
    )
    fx, fy, cx, cy = _frame_intrinsics(cameras, source_frame)
    h, w = depth.shape
    x0 = max(0, int(round(cx - w * 0.10)))
    x1 = min(w, int(round(cx + w * 0.10)))
    y0 = max(0, int(round(cy - h * 0.10)))
    y1 = min(h, int(round(cy + h * 0.10)))
    crop = depth[y0:y1, x0:x1]
    valid = crop[crop > 0.0]
    if valid.size < 100:
        valid = depth[depth > 0.0]
    if valid.size == 0:
        raise RuntimeError("Source view has no valid depth")
    z = float(np.median(valid))
    cam_point = np.array([0.0, 0.0, z, 1.0], dtype=np.float64)
    c2w = np.asarray(source_frame["transform_matrix"], dtype=np.float64)
    target = (c2w @ cam_point)[:3]
    return target, z


def _source_silhouette_meta(volume_dir: Path, cameras: dict, source_frame: dict) -> dict:
    depth = _decode_depth_u16(
        volume_dir / os.path.basename(source_frame["file_path"]),
        near=float(source_frame["near"]),
        far=float(source_frame["far"]),
    )
    valid = depth > 0.0
    v, u = np.nonzero(valid)
    if v.size == 0:
        raise RuntimeError("Source view has no valid depth")
    height, width = depth.shape
    u0 = int(u.min())
    u1 = int(u.max())
    v0 = int(v.min())
    v1 = int(v.max())
    return {
        "bbox_px": [u0, v0, u1, v1],
        "width_fraction": float((u1 - u0 + 1) / float(width)),
        "height_fraction": float((v1 - v0 + 1) / float(height)),
        "margins_px": [u0, v0, width - 1 - u1, height - 1 - v1],
    }


def _author_camera_path(
    cameras: dict,
    volume_dir: Path,
    *,
    source_view: int,
    num_frames: int,
    arc_degrees: float,
    slide_cm: float,
    start_fill_width: float,
    min_start_distance_scale: float,
    width: int,
    height: int,
) -> tuple[list[dict], dict]:
    if source_view < 0 or source_view >= len(cameras["frames"]):
        raise IndexError(f"source_view {source_view} outside 0..{len(cameras['frames']) - 1}")

    source = cameras["frames"][source_view]
    source_c2w = np.asarray(source["transform_matrix"], dtype=np.float64)
    source_center = source_c2w[:3, 3]
    source_right = source_c2w[:3, 0]
    source_down = source_c2w[:3, 1]
    target, target_distance = _source_target_point(volume_dir, cameras, source)
    silhouette = _source_silhouette_meta(volume_dir, cameras, source)
    if start_fill_width > 0.0:
        start_distance_scale = silhouette["width_fraction"] / start_fill_width
    else:
        start_distance_scale = 1.0
    start_distance_scale = float(np.clip(start_distance_scale, min_start_distance_scale, 1.0))

    frames: list[dict] = []
    denom = max(1, num_frames - 1)
    initial_offset = source_center - target
    for idx in range(num_frames):
        t = float(idx) / float(denom)
        ease_t = t * t * (3.0 - 2.0 * t)
        distance_scale = start_distance_scale + (1.0 - start_distance_scale) * ease_t
        angle = math.radians(arc_degrees) * t
        rotated_offset = _rotation_about_axis(source_down, angle) @ (initial_offset * distance_scale)
        center = target + rotated_offset + source_right * (slide_cm * t)
        look_target = target + source_right * (slide_cm * t)
        c2w = _look_at_c2w(center, look_target, down_hint=source_down)
        fx, fy, cx, cy = _scaled_intrinsics(cameras, source, width=width, height=height)
        frames.append(
            {
                "file_path": f"known_depth_png/depth_{idx:04d}.png",
                "color_path": f"guide_frames/{idx:06d}.png",
                "mask_path": f"mask_frames/{idx:06d}.png",
                "known_depth_npy": f"known_depth/{idx:06d}.npy",
                "transform_matrix": c2w.tolist(),
                "fl_x": float(fx),
                "fl_y": float(fy),
                "cx": float(cx),
                "cy": float(cy),
            }
        )

    meta = {
        "source_view": int(source_view),
        "arc_degrees": float(arc_degrees),
        "slide_cm": float(slide_cm),
        "start_fill_width": float(start_fill_width),
        "min_start_distance_scale": float(min_start_distance_scale),
        "start_distance_scale": float(start_distance_scale),
        "source_silhouette": silhouette,
        "target_distance_cm": float(target_distance),
        "target_world_cm": target.tolist(),
        "source_camera_center_cm": source_center.tolist(),
    }
    return frames, meta


def _render_points(
    points_world: np.ndarray,
    colors_bgr: np.ndarray,
    frame: dict,
    *,
    width: int,
    height: int,
    background_bgr: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c2w = np.asarray(frame["transform_matrix"], dtype=np.float64)
    w2c = np.linalg.inv(c2w)
    homog = np.concatenate(
        [points_world.astype(np.float64), np.ones((points_world.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    cam = (w2c @ homog.T).T[:, :3]
    z = cam[:, 2]
    valid = np.isfinite(cam).all(axis=1) & (z > 0.0)
    if not np.any(valid):
        image = np.full((height, width, 3), background_bgr, dtype=np.uint8)
        return image, np.zeros((height, width), dtype=np.float32), np.zeros((height, width), dtype=bool)

    cam = cam[valid]
    z = z[valid]
    col = colors_bgr[valid]
    fx = float(frame["fl_x"])
    fy = float(frame["fl_y"])
    cx = float(frame["cx"])
    cy = float(frame["cy"])
    u = np.rint(fx * (cam[:, 0] / z) + cx).astype(np.int64)
    v = np.rint(fy * (cam[:, 1] / z) + cy).astype(np.int64)
    in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)

    image = np.full((height, width, 3), background_bgr, dtype=np.uint8)
    depth = np.zeros((height, width), dtype=np.float32)
    known = np.zeros((height, width), dtype=bool)
    if not np.any(in_bounds):
        return image, depth, known

    u = u[in_bounds]
    v = v[in_bounds]
    z = z[in_bounds]
    col = col[in_bounds]
    linear = v * width + u
    order = np.lexsort((z, linear))
    linear_sorted = linear[order]
    keep_sorted = np.ones(linear_sorted.shape[0], dtype=bool)
    if linear_sorted.shape[0] > 1:
        keep_sorted[1:] = linear_sorted[1:] != linear_sorted[:-1]
    keep = order[keep_sorted]
    kept_linear = linear[keep]

    image_flat = image.reshape(-1, 3)
    depth_flat = depth.reshape(-1)
    known_flat = known.reshape(-1)
    image_flat[kept_linear] = col[keep]
    depth_flat[kept_linear] = z[keep].astype(np.float32, copy=False)
    known_flat[kept_linear] = True
    return image, depth, known


def _mask_from_known(known: np.ndarray, *, feather_px: int) -> np.ndarray:
    missing = (~known).astype(np.uint8)
    if feather_px <= 0:
        return (missing * 255).astype(np.uint8)
    dist = cv2.distanceTransform(missing, cv2.DIST_L2, 3)
    alpha = np.clip(dist / float(feather_px), 0.0, 1.0)
    alpha[known] = 0.0
    return np.rint(alpha * 255.0).astype(np.uint8)


def _open_video(path: Path, *, width: int, height: int, fps: float, is_color: bool) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height), isColor=is_color)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {path}")
    return writer


def main() -> None:
    ap = argparse.ArgumentParser(description="Render VACE conditioning assets from a DepthImageVolume.")
    ap.add_argument("--volume_dir", default=r"D:\archive\DeepRock1\DeepRock1_depth_volume")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--source_view", type=int, default=0)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--frames", type=int, default=49)
    ap.add_argument("--fps", type=float, default=16.0)
    ap.add_argument("--arc_degrees", type=float, default=20.0)
    ap.add_argument("--slide_cm", type=float, default=0.0)
    ap.add_argument("--start_fill_width", type=float, default=1.08)
    ap.add_argument("--min_start_distance_scale", type=float, default=0.35)
    ap.add_argument("--max_points_per_frame", type=int, default=220_000)
    ap.add_argument("--mask_feather_px", type=int, default=8)
    ap.add_argument("--background_rgb", default="127,127,127")
    ap.add_argument("--rng_seed", type=int, default=0)
    args = ap.parse_args()

    volume_dir = Path(args.volume_dir)
    out_dir = Path(args.out_dir)
    cameras = _load_json(volume_dir / "cameras.json")
    background_rgb = tuple(int(x) for x in args.background_rgb.split(","))
    if len(background_rgb) != 3:
        raise ValueError("--background_rgb must be R,G,B")
    background_bgr = (background_rgb[2], background_rgb[1], background_rgb[0])

    out_dir.mkdir(parents=True, exist_ok=True)
    guide_dir = out_dir / "guide_frames"
    mask_dir = out_dir / "mask_frames"
    depth_dir = out_dir / "known_depth"
    depth_png_dir = out_dir / "known_depth_png"
    for path in (guide_dir, mask_dir, depth_dir, depth_png_dir):
        path.mkdir(parents=True, exist_ok=True)

    points, colors_bgr = _unproject_volume(
        volume_dir,
        cameras,
        max_points_per_frame=int(args.max_points_per_frame),
        rng_seed=int(args.rng_seed),
    )
    authored_frames, path_meta = _author_camera_path(
        cameras,
        volume_dir,
        source_view=int(args.source_view),
        num_frames=int(args.frames),
        arc_degrees=float(args.arc_degrees),
        slide_cm=float(args.slide_cm),
        start_fill_width=float(args.start_fill_width),
        min_start_distance_scale=float(args.min_start_distance_scale),
        width=int(args.width),
        height=int(args.height),
    )

    guide_video = _open_video(
        out_dir / "video_guide.mp4",
        width=int(args.width),
        height=int(args.height),
        fps=float(args.fps),
        is_color=True,
    )
    mask_video = _open_video(
        out_dir / "video_mask.mp4",
        width=int(args.width),
        height=int(args.height),
        fps=float(args.fps),
        is_color=False,
    )

    known_counts: list[int] = []
    mask_means: list[float] = []
    for idx, frame in enumerate(authored_frames):
        image, depth, known = _render_points(
            points,
            colors_bgr,
            frame,
            width=int(args.width),
            height=int(args.height),
            background_bgr=background_bgr,
        )
        mask = _mask_from_known(known, feather_px=int(args.mask_feather_px))
        valid_depth = depth[depth > 0.0]
        if valid_depth.size:
            near = float(valid_depth.min())
            far = float(valid_depth.max())
            if far - near < 1.0e-4:
                far = near + 1.0
        else:
            near = 1.0
            far = 2.0

        frame["near"] = near
        frame["far"] = far
        frame["num_known_pixels"] = int(known.sum())
        frame["mask_mean"] = float(mask.mean())

        cv2.imwrite(str(guide_dir / f"{idx:06d}.png"), image)
        cv2.imwrite(str(mask_dir / f"{idx:06d}.png"), mask)
        cv2.imwrite(str(depth_png_dir / f"depth_{idx:04d}.png"), _encode_depth_u16(depth, near=near, far=far))
        np.save(depth_dir / f"{idx:06d}.npy", depth.astype(np.float32, copy=False))
        guide_video.write(image)
        mask_video.write(mask)
        known_counts.append(int(known.sum()))
        mask_means.append(float(mask.mean()))
        print(f"rendered {idx:04d}: known={known_counts[-1]} mask_mean={mask_means[-1]:.1f}")

    guide_video.release()
    mask_video.release()

    authored_cameras = {
        k: v
        for k, v in cameras.items()
        if k not in {"frames", "w", "h", "fl_x", "fl_y", "cx", "cy"}
    }
    first = authored_frames[0]
    authored_cameras.update(
        {
            "w": int(args.width),
            "h": int(args.height),
            "fl_x": float(first["fl_x"]),
            "fl_y": float(first["fl_y"]),
            "cx": float(first["cx"]),
            "cy": float(first["cy"]),
            "coordinate_units": "centimeters",
            "transform_translation_units": "centimeters",
            "source_depth_volume": str(volume_dir.resolve()),
            "conditioning": {
                "guide_video": "video_guide.mp4",
                "mask_video": "video_mask.mp4",
                "mask_semantics": "white=generate, black=keep",
                "mask_feather_px": int(args.mask_feather_px),
                **path_meta,
            },
            "frames": authored_frames,
        }
    )
    _write_json(out_dir / "cameras.json", authored_cameras)
    _write_json(
        out_dir / "conditioning_meta.json",
        {
            "volume_dir": str(volume_dir.resolve()),
            "out_dir": str(out_dir.resolve()),
            "guide_video": str((out_dir / "video_guide.mp4").resolve()),
            "mask_video": str((out_dir / "video_mask.mp4").resolve()),
            "width": int(args.width),
            "height": int(args.height),
            "frames": int(args.frames),
            "fps": float(args.fps),
            "known_pixels_min": int(min(known_counts)),
            "known_pixels_max": int(max(known_counts)),
            "known_pixels_mean": float(np.mean(known_counts)),
            "mask_mean_min": float(min(mask_means)),
            "mask_mean_max": float(max(mask_means)),
            "mask_mean_mean": float(np.mean(mask_means)),
            **path_meta,
        },
    )
    print("wrote conditioning:", out_dir)


if __name__ == "__main__":
    main()
