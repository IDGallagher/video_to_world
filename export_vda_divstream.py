#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parent
VDA_ROOT = PROJECT_ROOT / "third_party" / "Video-Depth-Anything"
VDA_CHECKPOINT_ROOT = VDA_ROOT / "checkpoints"

MODEL_CONFIGS: dict[str, dict[str, object]] = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}

CHECKPOINT_URLS: dict[tuple[bool, str], str] = {
    (False, "vits"): "https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth",
    (False, "vitb"): "https://huggingface.co/depth-anything/Video-Depth-Anything-Base/resolve/main/video_depth_anything_vitb.pth",
    (False, "vitl"): "https://huggingface.co/depth-anything/Video-Depth-Anything-Large/resolve/main/video_depth_anything_vitl.pth",
    (True, "vits"): "https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Small/resolve/main/metric_video_depth_anything_vits.pth",
    (True, "vitb"): "https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Base/resolve/main/metric_video_depth_anything_vitb.pth",
    (True, "vitl"): "https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Large/resolve/main/metric_video_depth_anything_vitl.pth",
}


def _parse_optional_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    lowered = str(raw).strip().lower()
    if lowered in {"", "none", "null"}:
        return None
    return float(raw)


def _checkpoint_name(*, metric: bool, encoder: str) -> str:
    prefix = "metric_video_depth_anything" if metric else "video_depth_anything"
    return f"{prefix}_{encoder}.pth"


def _download_checkpoint(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    print(f"Downloading VDA checkpoint: {url}")
    started_at = time.time()
    last_print = 0.0

    def report(count: int, block_size: int, total_size: int) -> None:
        nonlocal last_print
        now = time.time()
        if now - last_print < 2.0:
            return
        downloaded = count * block_size
        if total_size > 0:
            pct = min(100.0, downloaded * 100.0 / total_size)
            print(f"  downloaded {downloaded / (1024.0 * 1024.0):.1f}/{total_size / (1024.0 * 1024.0):.1f} MiB ({pct:.1f}%)")
        else:
            print(f"  downloaded {downloaded / (1024.0 * 1024.0):.1f} MiB")
        last_print = now

    urllib.request.urlretrieve(url, tmp_path, reporthook=report)
    tmp_path.replace(output_path)
    print(f"Checkpoint saved to {output_path} in {time.time() - started_at:.1f}s")


def _resolve_checkpoint(*, metric: bool, encoder: str, checkpoint_dir: Path, download: bool) -> Path:
    checkpoint_path = checkpoint_dir / _checkpoint_name(metric=metric, encoder=encoder)
    if checkpoint_path.exists():
        print(f"Using existing VDA checkpoint: {checkpoint_path}")
        return checkpoint_path
    if not download:
        raise FileNotFoundError(
            f"Missing VDA checkpoint: {checkpoint_path}. Enable checkpoint download or place the file there."
        )
    url = CHECKPOINT_URLS[(metric, encoder)]
    _download_checkpoint(url, checkpoint_path)
    return checkpoint_path


def _ensure_even(value: int) -> int:
    return value if value % 2 == 0 else value + 1


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
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    if not math.isfinite(source_fps) or source_fps <= 0.0:
        raise ValueError(f"Could not read a valid FPS from {video_path}")
    if original_width <= 0 or original_height <= 0:
        raise ValueError(f"Could not read a valid frame size from {video_path}")

    output_width = original_width
    output_height = original_height
    if max_res > 0 and max(original_width, original_height) > max_res:
        scale = float(max_res) / float(max(original_width, original_height))
        output_width = _ensure_even(max(2, round(original_width * scale)))
        output_height = _ensure_even(max(2, round(original_height * scale)))

    frames: list[np.ndarray] = []
    selected_indices: list[int] = []
    frame_idx = 0
    stride = max(1, int(stride))
    max_frames = int(max_frames)

    while cap.isOpened():
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if frame_idx % stride == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if output_width != original_width or output_height != original_height:
                frame_rgb = cv2.resize(frame_rgb, (output_width, output_height), interpolation=cv2.INTER_AREA)
            frames.append(frame_rgb)
            selected_indices.append(frame_idx)
            if max_frames > 0 and len(frames) >= max_frames:
                break
        frame_idx += 1

    cap.release()
    if not frames:
        raise ValueError("No frames were selected from the input video.")

    output_fps = source_fps / float(stride)
    return np.stack(frames, axis=0).astype(np.uint8, copy=False), output_fps, selected_indices


def _estimate_fixed_intrinsics(num_frames: int, width: int, height: int, hfov_degrees: float) -> np.ndarray:
    hfov = float(hfov_degrees)
    if not math.isfinite(hfov) or hfov <= 1.0 or hfov >= 179.0:
        raise ValueError("Fixed camera HFOV must be between 1 and 179 degrees.")
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


def _compute_depth_edge_mask(
    depth: np.ndarray,
    *,
    edge_rtol: float | None,
    edge_atol: float | None,
    edge_kernel_size: int,
    valid_mask: np.ndarray,
) -> np.ndarray:
    if edge_atol is not None and edge_atol <= 0.0:
        edge_atol = None
    if edge_rtol is not None and edge_rtol <= 0.0:
        edge_rtol = None
    if edge_rtol is None and edge_atol is None:
        return np.zeros_like(depth, dtype=bool)
    if int(edge_kernel_size) <= 0 or int(edge_kernel_size) % 2 == 0:
        raise ValueError("Depth Edge Kernel must be a positive odd integer.")

    depth_t = torch.from_numpy(depth.astype(np.float32, copy=False)).reshape(-1, 1, *depth.shape[-2:])
    mask_t = torch.from_numpy(valid_mask.astype(bool, copy=False)).reshape(-1, 1, *depth.shape[-2:])
    neg_inf = torch.full_like(depth_t, float("-inf"))
    diff = F.max_pool2d(
        torch.where(mask_t, depth_t, neg_inf),
        int(edge_kernel_size),
        stride=1,
        padding=int(edge_kernel_size) // 2,
    ) + F.max_pool2d(
        torch.where(mask_t, -depth_t, neg_inf),
        int(edge_kernel_size),
        stride=1,
        padding=int(edge_kernel_size) // 2,
    )

    edge = torch.zeros_like(depth_t, dtype=torch.bool)
    if edge_atol is not None:
        edge |= diff > float(edge_atol)
    if edge_rtol is not None:
        edge |= (diff / depth_t.abs().clamp_min(1e-12)).nan_to_num_() > float(edge_rtol)
    edge &= mask_t
    return edge.reshape(depth.shape).cpu().numpy()


def _apply_depth_filters(
    depth: np.ndarray,
    *,
    mask_min_depth_range_percent: bool,
    min_depth_range_percent: float,
    mask_max_depth_range_percent: bool,
    max_depth_range_percent: float,
    mask_min_depth_range_meters: bool,
    min_depth_range_meters: float,
    mask_depth_edges: bool,
    edge_rtol: float | None,
    edge_atol: float | None,
    edge_kernel_size: int,
    mask_max_depth: bool,
    max_depth_rtol: float | None,
    max_depth_atol: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    filtered = np.asarray(depth, dtype=np.float32).copy()
    valid = np.isfinite(filtered) & (filtered > 0.0)

    if mask_min_depth_range_percent and (min_depth_range_percent < 0.0 or min_depth_range_percent > 100.0):
        raise ValueError("Min Depth Range Percent must be between 0 and 100.")
    if mask_max_depth_range_percent and (max_depth_range_percent < 0.0 or max_depth_range_percent > 100.0):
        raise ValueError("Max Depth Range Percent must be between 0 and 100.")
    if mask_min_depth_range_meters and min_depth_range_meters < 0.0:
        raise ValueError("Min Depth Range Metres must be non-negative.")

    removed_min_range = 0
    if mask_min_depth_range_percent or mask_min_depth_range_meters:
        for frame_idx in range(filtered.shape[0]):
            vm = valid[frame_idx]
            if not np.any(vm):
                continue
            frame_depth = filtered[frame_idx]
            frame_min = float(frame_depth[vm].min())
            keep_limit: float | None = None
            if mask_min_depth_range_percent:
                frame_max = float(frame_depth[vm].max())
                keep_limit = frame_min + (float(min_depth_range_percent) / 100.0) * (frame_max - frame_min)
            if mask_min_depth_range_meters:
                meters_limit = frame_min + float(min_depth_range_meters)
                keep_limit = meters_limit if keep_limit is None else min(keep_limit, meters_limit)
            if keep_limit is not None:
                remove = vm & (frame_depth > keep_limit)
                removed_min_range += int(np.count_nonzero(remove))
                valid[frame_idx][remove] = False
    if removed_min_range:
        print(f"VDA min-depth-range filter removed {removed_min_range:,} pixels.")

    removed_max_range = 0
    if mask_max_depth_range_percent:
        for frame_idx in range(filtered.shape[0]):
            vm = valid[frame_idx]
            if not np.any(vm):
                continue
            frame_depth = filtered[frame_idx]
            frame_min = float(frame_depth[vm].min())
            frame_max = float(frame_depth[vm].max())
            keep_limit = frame_max - (float(max_depth_range_percent) / 100.0) * (frame_max - frame_min)
            remove = vm & (frame_depth < keep_limit)
            removed_max_range += int(np.count_nonzero(remove))
            valid[frame_idx][remove] = False
    if removed_max_range:
        print(f"VDA max-depth-range filter removed {removed_max_range:,} pixels.")

    removed_max_depth = 0
    if mask_max_depth:
        if max_depth_rtol is None and max_depth_atol is None:
            raise ValueError("Max-depth suppression needs a relative or absolute threshold.")
        rtol = float(max_depth_rtol) if max_depth_rtol is not None else 0.0
        atol = float(max_depth_atol) if max_depth_atol is not None else 0.0
        for frame_idx in range(filtered.shape[0]):
            vm = valid[frame_idx]
            if not np.any(vm):
                continue
            frame_depth = filtered[frame_idx]
            frame_max = float(frame_depth[vm].max())
            tolerance = max(atol, abs(frame_max) * rtol)
            remove = vm & (frame_depth >= (frame_max - tolerance))
            removed_max_depth += int(np.count_nonzero(remove))
            valid[frame_idx][remove] = False
    if removed_max_depth:
        print(f"VDA max-depth plateau filter removed {removed_max_depth:,} pixels.")

    if mask_depth_edges:
        edge_mask = _compute_depth_edge_mask(
            filtered,
            edge_rtol=edge_rtol,
            edge_atol=edge_atol,
            edge_kernel_size=edge_kernel_size,
            valid_mask=valid,
        )
        removed_edges = int(np.count_nonzero(edge_mask))
        valid &= ~edge_mask
        print(f"VDA depth-edge filter removed {removed_edges:,} pixels.")

    filtered[~valid] = 0.0
    return filtered.astype(np.float32, copy=False), valid


def _convert_relative_depth_to_pseudo_depth(relative_depth: np.ndarray) -> np.ndarray:
    """Convert VDA non-metric relative depth into divstream-style positive depth.

    VDA's non-metric model is aligned internally with scale * depth + shift,
    so the raw output should be treated as an affine relative signal, not as a
    metric inverse depth value. Depth Anything relative models conventionally
    produce larger values for closer surfaces. The divstream writer expects
    larger values to be farther away, so normalize the global raw range to 0..1
    and reverse it with 1 - normalized_depth. This keeps the exported pseudo
    depth bounded instead of stretching tiny raw values with a reciprocal.
    """

    rel = np.asarray(relative_depth, dtype=np.float32)
    valid = np.isfinite(rel) & (rel > 0.0)
    if not np.any(valid):
        return np.zeros_like(rel, dtype=np.float32)

    rel_valid = rel[valid]
    low = float(rel_valid.min())
    high = float(rel_valid.max())
    if high <= low:
        high = low + max(abs(low) * 1.0e-6, 1.0e-6)

    pseudo = np.zeros_like(rel, dtype=np.float32)
    denom = max(high - low, max(abs(high), abs(low), 1.0) * 1.0e-6)
    normalized = np.clip((rel_valid - low) / denom, 0.0, 1.0)
    pseudo[valid] = np.clip(1.0 - normalized, 1.0e-4, 1.0)

    print(
        "Converted non-metric VDA relative output to bounded pseudo-depth "
        f"(raw_range={float(rel_valid.min()):.6g}..{float(rel_valid.max()):.6g}, "
        f"converted_range={float(pseudo[valid].min()):.6g}..{float(pseudo[valid].max()):.6g}, "
        "conversion=1-normalized before depth_scale)."
    )
    return pseudo


def _write_results_npz(
    *,
    scene_root: Path,
    images: np.ndarray,
    depth: np.ndarray,
    valid_mask: np.ndarray,
    fps: float,
    selected_indices: list[int],
    input_video: Path,
    encoder: str,
    metric: bool,
    input_size: int,
    max_res: int,
    stride: int,
    fixed_camera_fov_degrees: float,
    relative_depth_inverse: bool,
) -> Path:
    num_frames, height, width, _ = images.shape
    intrinsics = _estimate_fixed_intrinsics(num_frames, width, height, fixed_camera_fov_degrees)
    extrinsics = _identity_extrinsics(num_frames)
    conf = valid_mask.astype(np.float32, copy=False)

    npz_dir = scene_root / "exports" / "npz"
    npz_dir.mkdir(parents=True, exist_ok=True)
    npz_path = npz_dir / "results.npz"
    np.savez_compressed(
        npz_path,
        image=images.astype(np.uint8, copy=False),
        depth=depth.astype(np.float32, copy=False),
        conf=conf,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
    )

    meta_path = scene_root / "preprocess_frames.json"
    meta_path.write_text(
        json.dumps(
            {
                "source": "video_depth_anything",
                "source_input_path": str(input_video.resolve()),
                "model_name": "Video Depth Anything",
                "encoder": encoder,
                "metric": bool(metric),
                "relative_depth_inverse": bool(relative_depth_inverse),
                "input_size": int(input_size),
                "max_res": int(max_res),
                "actual_stride": int(stride),
                "runtime_export_fps": float(fps),
                "num_frames_used": int(num_frames),
                "selected_frame_indices": [int(idx) for idx in selected_indices],
                "fixed_camera": True,
                "fixed_camera_fov_degrees": float(fixed_camera_fov_degrees),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return npz_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a Video Depth Anything divstream.")
    parser.add_argument("--input-video", required=True)
    parser.add_argument("--scene-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--encoder", choices=sorted(MODEL_CONFIGS.keys()), default="vits")
    parser.add_argument("--metric", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--max-res", type=int, default=1280)
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--checkpoint-dir", default=str(VDA_CHECKPOINT_ROOT))
    parser.add_argument("--download-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fixed-camera-fov-degrees", type=float, default=60.0)
    parser.add_argument("--depth-scale", type=float, default=1.0)
    parser.add_argument("--depth-offset", type=float, default=0.0)
    parser.add_argument(
        "--relative-depth-inverse",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For non-metric checkpoints, treat larger VDA values as closer and reverse the normalized relative depth before export.",
    )
    parser.add_argument("--mask-min-depth-range-percent", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--min-depth-range-percent", type=float, default=50.0)
    parser.add_argument("--mask-max-depth-range-percent", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-depth-range-percent", type=float, default=50.0)
    parser.add_argument("--mask-min-depth-range-meters", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--min-depth-range-meters", type=float, default=3.0)
    parser.add_argument("--mask-depth-edges", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--edge-rtol", type=_parse_optional_float, default=0.1)
    parser.add_argument("--edge-atol", type=_parse_optional_float, default=0.0)
    parser.add_argument("--edge-kernel-size", type=int, default=3)
    parser.add_argument("--mask-max-depth", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-depth-rtol", type=_parse_optional_float, default=0.001)
    parser.add_argument("--max-depth-atol", type=_parse_optional_float, default=None)
    parser.add_argument("--compression-level", type=int, default=9)
    parser.add_argument("--workers", type=int, default=0)
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if not VDA_ROOT.is_dir():
        raise FileNotFoundError(f"Video Depth Anything checkout was not found: {VDA_ROOT}")

    sys.path.insert(0, str(VDA_ROOT))
    from video_depth_anything.video_depth import VideoDepthAnything
    from export_depth_image_stream_bc7 import export_depth_image_stream_bc7

    input_video = Path(args.input_video).resolve()
    scene_root = Path(args.scene_root).resolve()
    output_path = Path(args.output).resolve()
    scene_root.mkdir(parents=True, exist_ok=True)

    stride = max(1, int(args.stride))
    frames, output_fps, selected_indices = _read_video_frames(
        input_video,
        stride=stride,
        max_frames=int(args.max_frames),
        max_res=int(args.max_res),
    )
    print(
        "VDA selected frames: "
        f"N={frames.shape[0]}, shape={frames.shape[2]}x{frames.shape[1]}, stride={stride}, fps={output_fps:.6g}"
    )

    checkpoint_path = _resolve_checkpoint(
        metric=bool(args.metric),
        encoder=str(args.encoder),
        checkpoint_dir=Path(args.checkpoint_dir).resolve(),
        download=bool(args.download_checkpoint),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        "VDA inference settings: "
        f"encoder={args.encoder}, metric={bool(args.metric)}, input_size={int(args.input_size)}, "
        f"max_res={int(args.max_res)}, fp32={bool(args.fp32)}, device={device}, checkpoint={checkpoint_path}"
    )
    if args.fp32:
        print(
            "VDA FP32 note: xFormers may not provide a float32 attention kernel on this GPU. "
            "The vendored model will fall back to PyTorch attention if needed, which is slower and uses more VRAM. "
            "For most exports, leave Use FP32 disabled."
        )

    model = VideoDepthAnything(**MODEL_CONFIGS[str(args.encoder)], metric=bool(args.metric))
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device=device).eval()

    try:
        depths, model_fps = model.infer_video_depth(
            frames,
            output_fps,
            input_size=int(args.input_size),
            device=device,
            fp32=bool(args.fp32),
        )
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise RuntimeError(
                "VDA CUDA out of memory during inference. "
                f"Failed settings: encoder={args.encoder}, metric={bool(args.metric)}, "
                f"input_size={int(args.input_size)}, max_res={int(args.max_res)}, "
                f"selected_frames={frames.shape[0]}, frame_shape={frames.shape[2]}x{frames.shape[1]}. "
                "Reduce VDA Input Size first (try 384 or 518 instead of 768), reduce Video Max Resolution, "
                "or use vitb/vits. Large/vitl at input size 768 is too large for this GPU in a 32-frame VDA chunk."
            ) from exc
        raise
    depths = np.asarray(depths, dtype=np.float32)
    if not bool(args.metric) and bool(args.relative_depth_inverse):
        depths = _convert_relative_depth_to_pseudo_depth(depths)
    depths = depths * float(args.depth_scale) + float(args.depth_offset)
    depths[~np.isfinite(depths) | (depths <= 0.0)] = 0.0
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    depths, valid_mask = _apply_depth_filters(
        depths,
        mask_min_depth_range_percent=bool(args.mask_min_depth_range_percent),
        min_depth_range_percent=float(args.min_depth_range_percent),
        mask_max_depth_range_percent=bool(args.mask_max_depth_range_percent),
        max_depth_range_percent=float(args.max_depth_range_percent),
        mask_min_depth_range_meters=bool(args.mask_min_depth_range_meters),
        min_depth_range_meters=float(args.min_depth_range_meters),
        mask_depth_edges=bool(args.mask_depth_edges),
        edge_rtol=args.edge_rtol,
        edge_atol=args.edge_atol,
        edge_kernel_size=int(args.edge_kernel_size),
        mask_max_depth=bool(args.mask_max_depth),
        max_depth_rtol=args.max_depth_rtol,
        max_depth_atol=args.max_depth_atol,
    )

    npz_path = _write_results_npz(
        scene_root=scene_root,
        images=frames,
        depth=depths,
        valid_mask=valid_mask,
        fps=float(model_fps),
        selected_indices=selected_indices,
        input_video=input_video,
        encoder=str(args.encoder),
        metric=bool(args.metric),
        input_size=int(args.input_size),
        max_res=int(args.max_res),
        stride=stride,
        fixed_camera_fov_degrees=float(args.fixed_camera_fov_degrees),
        relative_depth_inverse=bool(args.relative_depth_inverse),
    )
    print(f"VDA results NPZ: {npz_path}")

    exported = export_depth_image_stream_bc7(
        scene_root=str(scene_root),
        output_path=str(output_path),
        fps=float(model_fps),
        compression_level=int(args.compression_level),
        overwrite=True,
        apply_stage1_filters=False,
        require_stage1_filters=False,
        max_workers=int(args.workers),
        fixed_camera=True,
    )
    size_bytes = os.path.getsize(exported)
    print(f"Exported VDA divstream: {exported}")
    print(f"Size: {size_bytes} bytes ({size_bytes / (1024.0 * 1024.0):.2f} MiB)")


if __name__ == "__main__":
    main()
