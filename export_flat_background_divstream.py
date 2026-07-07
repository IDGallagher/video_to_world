#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent


def _ensure_even(value: int) -> int:
    return value if value % 2 == 0 else value + 1


def _validate_rgb_range(background_min_rgb: float, background_max_rgb: float) -> tuple[float, float]:
    min_rgb = float(background_min_rgb)
    max_rgb = float(background_max_rgb)
    if not math.isfinite(min_rgb) or not math.isfinite(max_rgb):
        raise ValueError("Background RGB range must be finite.")
    if min_rgb < 0.0 or min_rgb > 255.0 or max_rgb < 0.0 or max_rgb > 255.0:
        raise ValueError("Background RGB range must be between 0 and 255.")
    if min_rgb > max_rgb:
        raise ValueError("Background RGB Min must be less than or equal to Background RGB Max.")
    return min_rgb, max_rgb


def _validate_common_settings(
    *,
    stride: int,
    max_frames: int,
    max_res: int,
    flat_depth_meters: float,
    fixed_camera_fov_degrees: float,
    background_min_rgb: float,
    background_max_rgb: float,
    background_grow_px: int,
) -> tuple[float, float]:
    if int(stride) < 1:
        raise ValueError("Input Stride must be at least 1.")
    if int(max_frames) == 0 or int(max_frames) < -1:
        raise ValueError("Max Output Frames must be -1 or a positive integer.")
    if int(max_res) == 0 or int(max_res) < -1:
        raise ValueError("Video Max Resolution must be -1 or a positive integer.")
    if not math.isfinite(float(flat_depth_meters)) or float(flat_depth_meters) <= 0.0:
        raise ValueError("Flat Depth Metres must be greater than 0.")
    if not math.isfinite(float(fixed_camera_fov_degrees)) or not (1.0 < float(fixed_camera_fov_degrees) < 179.0):
        raise ValueError("Fixed Camera HFOV must be between 1 and 179 degrees.")
    if int(background_grow_px) < 0:
        raise ValueError("Background Grow Pixels must be non-negative.")
    return _validate_rgb_range(background_min_rgb, background_max_rgb)


def _resize_frame_rgb(frame_rgb: np.ndarray, *, max_res: int) -> np.ndarray:
    height, width = frame_rgb.shape[:2]
    if max_res > 0 and max(width, height) > max_res:
        scale = float(max_res) / float(max(width, height))
        output_width = _ensure_even(max(2, round(width * scale)))
        output_height = _ensure_even(max(2, round(height * scale)))
        return cv2.resize(frame_rgb, (output_width, output_height), interpolation=cv2.INTER_AREA)
    return frame_rgb


def _read_video_frames(
    video_path: Path,
    *,
    stride: int,
    max_frames: int,
    max_res: int,
) -> tuple[np.ndarray, float, list[int]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open input video: {video_path}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not math.isfinite(source_fps) or source_fps <= 0.0:
        cap.release()
        raise ValueError(f"Could not read a valid FPS from {video_path}")

    frames: list[np.ndarray] = []
    selected_indices: list[int] = []
    frame_idx = 0
    stride = max(1, int(stride))
    max_frames = int(max_frames)
    max_res = int(max_res)

    while cap.isOpened():
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if frame_idx % stride == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb = _resize_frame_rgb(frame_rgb, max_res=max_res)
            frames.append(frame_rgb.astype(np.uint8, copy=False))
            selected_indices.append(frame_idx)
            if max_frames > 0 and len(frames) >= max_frames:
                break
        frame_idx += 1

    cap.release()
    if not frames:
        raise ValueError("No frames were selected from the input video.")

    output_fps = source_fps / float(stride)
    return np.stack(frames, axis=0), output_fps, selected_indices


def _read_video_frame(
    video_path: Path,
    *,
    frame_index: int,
    max_res: int,
) -> tuple[np.ndarray, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open input video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if int(frame_index) < 0:
        cap.release()
        raise ValueError("Preview Frame Index must be non-negative.")
    if total_frames > 0 and int(frame_index) >= total_frames:
        cap.release()
        raise ValueError(f"Preview Frame Index {int(frame_index)} is outside the video frame count ({total_frames}).")

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Could not read frame {int(frame_index)} from {video_path}.")

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb = _resize_frame_rgb(frame_rgb, max_res=int(max_res))
    return frame_rgb.astype(np.uint8, copy=False), int(frame_index), total_frames


def _background_mask_for_frame(
    frame_rgb: np.ndarray,
    *,
    background_min_rgb: float,
    background_max_rgb: float,
    background_grow_px: int,
) -> np.ndarray:
    min_rgb, max_rgb = _validate_rgb_range(background_min_rgb, background_max_rgb)
    pixels = frame_rgb.astype(np.float32, copy=False)
    background = np.all((pixels >= min_rgb) & (pixels <= max_rgb), axis=-1)
    grow_px = int(background_grow_px)
    if grow_px > 0:
        kernel_size = grow_px * 2 + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        background = cv2.dilate(background.astype(np.uint8), kernel, iterations=1).astype(bool)
    return background


def _background_mask_for_frames(
    frames_rgb: np.ndarray,
    *,
    background_min_rgb: float,
    background_max_rgb: float,
    background_grow_px: int,
) -> np.ndarray:
    masks = [
        _background_mask_for_frame(
            frame,
            background_min_rgb=background_min_rgb,
            background_max_rgb=background_max_rgb,
            background_grow_px=background_grow_px,
        )
        for frame in frames_rgb
    ]
    return np.stack(masks, axis=0)


def _checkerboard(height: int, width: int, *, tile: int = 16) -> np.ndarray:
    yy, xx = np.indices((height, width))
    pattern = ((yy // tile) + (xx // tile)) % 2
    light = np.array([232, 232, 232], dtype=np.uint8)
    dark = np.array([176, 176, 176], dtype=np.uint8)
    return np.where(pattern[..., None] == 0, light, dark).astype(np.uint8)


def build_background_removal_preview(
    video_path: str | Path,
    *,
    frame_index: int = 0,
    max_res: int = -1,
    background_min_rgb: float = 0.0,
    background_max_rgb: float = 32.0,
    background_grow_px: int = 1,
) -> tuple[np.ndarray, np.ndarray, str]:
    frame_rgb, actual_frame_index, total_frames = _read_video_frame(
        Path(video_path).resolve(),
        frame_index=int(frame_index),
        max_res=int(max_res),
    )
    background = _background_mask_for_frame(
        frame_rgb,
        background_min_rgb=background_min_rgb,
        background_max_rgb=background_max_rgb,
        background_grow_px=int(background_grow_px),
    )
    foreground = ~background
    preview = _checkerboard(frame_rgb.shape[0], frame_rgb.shape[1])
    preview[foreground] = frame_rgb[foreground]

    kept_pixels = int(np.count_nonzero(foreground))
    total_pixels = int(foreground.size)
    removed_pixels = total_pixels - kept_pixels
    total_note = f" / {total_frames:,}" if total_frames > 0 else ""
    status = (
        f"**Preview frame**: `{actual_frame_index:,}{total_note}`\n\n"
        f"**Frame size**: `{frame_rgb.shape[1]}x{frame_rgb.shape[0]}`\n\n"
        f"**Kept**: `{kept_pixels:,}` pixels (`{kept_pixels * 100.0 / total_pixels:.2f}%`)\n\n"
        f"**Removed**: `{removed_pixels:,}` pixels (`{removed_pixels * 100.0 / total_pixels:.2f}%`)"
    )
    return frame_rgb, preview, status


def _estimate_fixed_intrinsics(num_frames: int, width: int, height: int, hfov_degrees: float) -> np.ndarray:
    hfov = float(hfov_degrees)
    focal = (0.5 * float(width)) / math.tan(0.5 * math.radians(hfov))
    intrinsics = np.array(
        [
            [focal, 0.0, (float(width) - 1.0) * 0.5],
            [0.0, focal, (float(height) - 1.0) * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return np.repeat(intrinsics[None, :, :], num_frames, axis=0)


def _identity_extrinsics(num_frames: int) -> np.ndarray:
    identity = np.eye(4, dtype=np.float32)[:3, :4]
    return np.repeat(identity[None, :, :], num_frames, axis=0)


def _write_results_npz(
    *,
    scene_root: Path,
    images: np.ndarray,
    depth: np.ndarray,
    valid_mask: np.ndarray,
    fps: float,
    selected_indices: list[int],
    input_video: Path,
    stride: int,
    max_res: int,
    flat_depth_meters: float,
    fixed_camera_fov_degrees: float,
    background_min_rgb: float,
    background_max_rgb: float,
    background_grow_px: int,
) -> Path:
    num_frames, height, width, _ = images.shape
    intrinsics = _estimate_fixed_intrinsics(num_frames, width, height, fixed_camera_fov_degrees)
    extrinsics = _identity_extrinsics(num_frames)

    npz_dir = scene_root / "exports" / "npz"
    npz_dir.mkdir(parents=True, exist_ok=True)
    npz_path = npz_dir / "results.npz"
    np.savez_compressed(
        npz_path,
        image=images.astype(np.uint8, copy=False),
        depth=depth.astype(np.float32, copy=False),
        conf=valid_mask.astype(np.float32, copy=False),
        extrinsics=extrinsics,
        intrinsics=intrinsics,
    )

    meta_path = scene_root / "preprocess_frames.json"
    meta_path.write_text(
        json.dumps(
            {
                "source": "flat_background_removal",
                "source_input_path": str(input_video.resolve()),
                "runtime_export_fps": float(fps),
                "num_frames_used": int(num_frames),
                "selected_frame_indices": [int(idx) for idx in selected_indices],
                "actual_stride": int(stride),
                "max_res": int(max_res),
                "flat_depth_meters": float(flat_depth_meters),
                "fixed_camera": True,
                "fixed_camera_fov_degrees": float(fixed_camera_fov_degrees),
                "background_min_rgb": float(background_min_rgb),
                "background_max_rgb": float(background_max_rgb),
                "background_grow_px": int(background_grow_px),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return npz_path


def export_flat_background_divstream(
    *,
    input_video: str | Path,
    scene_root: str | Path,
    output: str | Path,
    stride: int = 1,
    max_frames: int = -1,
    max_res: int = -1,
    flat_depth_meters: float = 1.0,
    fixed_camera_fov_degrees: float = 60.0,
    background_min_rgb: float = 0.0,
    background_max_rgb: float = 32.0,
    background_grow_px: int = 1,
    compression_level: int = 9,
    workers: int = 0,
) -> str:
    _validate_common_settings(
        stride=int(stride),
        max_frames=int(max_frames),
        max_res=int(max_res),
        flat_depth_meters=float(flat_depth_meters),
        fixed_camera_fov_degrees=float(fixed_camera_fov_degrees),
        background_min_rgb=float(background_min_rgb),
        background_max_rgb=float(background_max_rgb),
        background_grow_px=int(background_grow_px),
    )
    if int(compression_level) < 1 or int(compression_level) > 12:
        raise ValueError("Compression Level must be between 1 and 12.")
    if int(workers) < 0:
        raise ValueError("Compression Workers must be 0 or greater.")

    from export_depth_image_stream_bc7 import export_depth_image_stream_bc7

    input_video_path = Path(input_video).resolve()
    scene_root_path = Path(scene_root).resolve()
    output_path = Path(output).resolve()
    scene_root_path.mkdir(parents=True, exist_ok=True)

    frames, fps, selected_indices = _read_video_frames(
        input_video_path,
        stride=int(stride),
        max_frames=int(max_frames),
        max_res=int(max_res),
    )
    print(
        "Flat background export selected frames: "
        f"N={frames.shape[0]}, shape={frames.shape[2]}x{frames.shape[1]}, "
        f"stride={int(stride)}, fps={fps:.6g}"
    )

    background = _background_mask_for_frames(
        frames,
        background_min_rgb=float(background_min_rgb),
        background_max_rgb=float(background_max_rgb),
        background_grow_px=int(background_grow_px),
    )
    valid_mask = ~background
    kept_pixels = int(np.count_nonzero(valid_mask))
    total_pixels = int(valid_mask.size)
    if kept_pixels <= 0:
        raise ValueError(
            "Background removal removed every pixel. Widen the kept foreground by lowering the background range."
        )

    masked_images = frames.copy()
    masked_images[~valid_mask] = 0
    depth = np.zeros(valid_mask.shape, dtype=np.float32)
    depth[valid_mask] = float(flat_depth_meters)
    print(
        "Flat background mask: "
        f"kept={kept_pixels:,}/{total_pixels:,} pixels ({kept_pixels * 100.0 / total_pixels:.2f}%), "
        f"removed={total_pixels - kept_pixels:,}, "
        f"background_rgb_range=[{float(background_min_rgb):.3g}, {float(background_max_rgb):.3g}], "
        f"grow_px={int(background_grow_px)}, flat_depth_m={float(flat_depth_meters):.6g}"
    )

    npz_path = _write_results_npz(
        scene_root=scene_root_path,
        images=masked_images,
        depth=depth,
        valid_mask=valid_mask,
        fps=float(fps),
        selected_indices=selected_indices,
        input_video=input_video_path,
        stride=int(stride),
        max_res=int(max_res),
        flat_depth_meters=float(flat_depth_meters),
        fixed_camera_fov_degrees=float(fixed_camera_fov_degrees),
        background_min_rgb=float(background_min_rgb),
        background_max_rgb=float(background_max_rgb),
        background_grow_px=int(background_grow_px),
    )
    print(f"Flat background results NPZ: {npz_path}")

    exported = export_depth_image_stream_bc7(
        scene_root=str(scene_root_path),
        output_path=str(output_path),
        fps=float(fps),
        compression_level=int(compression_level),
        overwrite=True,
        apply_stage1_filters=False,
        require_stage1_filters=False,
        max_workers=int(workers),
        fixed_camera=True,
    )
    size_bytes = os.path.getsize(exported)
    print(f"Exported flat background divstream: {exported}")
    print(f"Size: {size_bytes} bytes ({size_bytes / (1024.0 * 1024.0):.2f} MiB)")
    return str(exported)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a flat-depth divstream with simple RGB-range background removal."
    )
    parser.add_argument("--input-video", required=True)
    parser.add_argument("--scene-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--max-res", type=int, default=-1)
    parser.add_argument("--flat-depth-meters", type=float, default=1.0)
    parser.add_argument("--fixed-camera-fov-degrees", type=float, default=60.0)
    parser.add_argument("--background-min-rgb", type=float, default=0.0)
    parser.add_argument("--background-max-rgb", type=float, default=32.0)
    parser.add_argument("--background-grow-px", type=int, default=1)
    parser.add_argument("--compression-level", type=int, default=9)
    parser.add_argument("--workers", type=int, default=0)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    export_flat_background_divstream(
        input_video=args.input_video,
        scene_root=args.scene_root,
        output=args.output,
        stride=int(args.stride),
        max_frames=int(args.max_frames),
        max_res=int(args.max_res),
        flat_depth_meters=float(args.flat_depth_meters),
        fixed_camera_fov_degrees=float(args.fixed_camera_fov_degrees),
        background_min_rgb=float(args.background_min_rgb),
        background_max_rgb=float(args.background_max_rgb),
        background_grow_px=int(args.background_grow_px),
        compression_level=int(args.compression_level),
        workers=int(args.workers),
    )


if __name__ == "__main__":
    main()
