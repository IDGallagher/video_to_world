"""Assemble a VACE masked generation into a sibling DepthImageVolume.

The generated clip supplies color. DA3 supplies relative depth. The rendered
conditioning pass supplies known metric depth in the authored camera poses.
For each frame we fit DA3 depth to the known depth on kept pixels, then write
only the generated mask region into a new static DepthImageVolume.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _encode_depth_u16(depth: np.ndarray, *, near: float, far: float) -> np.ndarray:
    encoded = np.zeros(depth.shape, dtype=np.uint16)
    valid = np.isfinite(depth) & (depth > 0.0)
    if not np.any(valid):
        return encoded
    norm = (depth[valid] - near) / max(far - near, 1.0e-6)
    code = np.rint(np.clip(norm, 0.0, 1.0) * 65534.0 + 1.0)
    encoded[valid] = code.astype(np.uint16)
    return encoded


def _read_video_frames(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from {path}")
    return frames


def _read_frame_dir(path: Path, *, image_ext: str) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for frame_path in sorted(path.glob(f"*.{image_ext}")):
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(frame_path)
        frames.append(frame)
    if not frames:
        raise RuntimeError(f"No '*.{image_ext}' frames found in {path}")
    return frames


def _run_da3(
    *,
    generated_video: Path | None,
    generated_frames_dir: Path | None,
    scene_root: Path,
    image_ext: str,
    max_frames: int,
    process_res: int,
    python_exe: str,
    overwrite: bool,
) -> Path:
    cmd = [
        python_exe,
        "preprocess_video.py",
        "--scene_root",
        str(scene_root),
        "--max_frames",
        str(max_frames),
        "--max_stride",
        "1",
        "--process_res",
        str(process_res),
        "--process_res_method",
        "upper_bound_resize",
        "--ref_view_strategy",
        "saddle_balanced",
        "--runtime_export_format",
        "none",
    ]
    if generated_video is not None:
        cmd.extend(["--input_video", str(generated_video)])
    elif generated_frames_dir is not None:
        cmd.extend(["--frames_dir", str(generated_frames_dir), "--image_ext", image_ext])
    else:
        raise ValueError("Need generated_video or generated_frames_dir")
    if overwrite:
        cmd.append("--overwrite")

    print("running DA3:", " ".join(cmd))
    subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], check=True)
    npz_path = scene_root / "exports" / "npz" / "results.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)
    return npz_path


def _resize_depth(depth: np.ndarray, *, width: int, height: int) -> np.ndarray:
    if depth.shape[:2] == (height, width):
        return depth.astype(np.float32, copy=False)
    return cv2.resize(depth.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)


def _resize_mask(mask: np.ndarray, *, width: int, height: int) -> np.ndarray:
    if mask.shape[:2] == (height, width):
        return mask.astype(np.uint8, copy=False)
    return cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_LINEAR)


def _resize_color(color: np.ndarray, *, width: int, height: int) -> np.ndarray:
    if color.shape[:2] == (height, width):
        return color
    return cv2.resize(color, (width, height), interpolation=cv2.INTER_AREA)


def _fit_scale_shift(
    da3_depth: np.ndarray,
    known_depth: np.ndarray,
    keep_mask: np.ndarray,
    *,
    min_pixels: int,
) -> tuple[float, float, int, float]:
    valid = (
        keep_mask
        & np.isfinite(da3_depth)
        & np.isfinite(known_depth)
        & (da3_depth > 0.0)
        & (known_depth > 0.0)
    )
    n = int(valid.sum())
    if n < min_pixels:
        raise RuntimeError(f"Only {n} kept pixels available for depth fit; need {min_pixels}")
    x = da3_depth[valid].astype(np.float64)
    y = known_depth[valid].astype(np.float64)

    # Trim extreme ratios before the least-squares fit; DA3 depth can have a few
    # wild pixels near holes and mask boundaries.
    ratio = y / np.maximum(x, 1.0e-9)
    lo, hi = np.percentile(ratio, [2.0, 98.0])
    robust = valid.copy()
    robust[valid] = (ratio >= lo) & (ratio <= hi)
    x = da3_depth[robust].astype(np.float64)
    y = known_depth[robust].astype(np.float64)
    A = np.stack([x, np.ones_like(x)], axis=1)
    scale, shift = np.linalg.lstsq(A, y, rcond=None)[0]
    residual = (scale * x + shift) - y
    rmse = float(np.sqrt(np.mean(residual * residual)))
    return float(scale), float(shift), int(robust.sum()), rmse


def _default_normal_map(valid: np.ndarray) -> np.ndarray:
    normal = np.zeros((valid.shape[0], valid.shape[1], 3), dtype=np.uint8)
    normal[valid] = [128, 128, 255]
    return normal


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert masked VACE output into a sibling DepthImageVolume.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--generated_video", default=None)
    src.add_argument("--generated_frames_dir", default=None)
    ap.add_argument("--conditioning_dir", required=True)
    ap.add_argument("--out_div", required=True)
    ap.add_argument("--da3_npz", default=None)
    ap.add_argument("--da3_scene_root", default=None)
    ap.add_argument("--run_da3", action="store_true")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--image_ext", default="png")
    ap.add_argument("--process_res", type=int, default=512)
    ap.add_argument("--keep_mask_max", type=int, default=16)
    ap.add_argument("--write_mask_min", type=int, default=1)
    ap.add_argument("--min_fit_pixels", type=int, default=1000)
    ap.add_argument("--overwrite_da3", action="store_true")
    args = ap.parse_args()

    conditioning_dir = Path(args.conditioning_dir)
    out_div = Path(args.out_div)
    authored = _load_json(conditioning_dir / "cameras.json")
    width = int(authored["w"])
    height = int(authored["h"])
    generated_video = Path(args.generated_video) if args.generated_video else None
    generated_frames_dir = Path(args.generated_frames_dir) if args.generated_frames_dir else None

    if args.da3_npz:
        da3_npz = Path(args.da3_npz)
    elif args.run_da3:
        scene_root = Path(args.da3_scene_root) if args.da3_scene_root else out_div.parent / "da3_generated"
        da3_npz = _run_da3(
            generated_video=generated_video,
            generated_frames_dir=generated_frames_dir,
            scene_root=scene_root,
            image_ext=str(args.image_ext),
            max_frames=len(authored["frames"]),
            process_res=int(args.process_res),
            python_exe=str(args.python),
            overwrite=bool(args.overwrite_da3),
        )
    else:
        raise ValueError("Provide --da3_npz or pass --run_da3")

    if generated_video is not None:
        color_frames = _read_video_frames(generated_video)
    else:
        color_frames = _read_frame_dir(generated_frames_dir, image_ext=str(args.image_ext))  # type: ignore[arg-type]

    da3 = np.load(da3_npz)
    da3_depths = da3["depth"]
    da3_images = da3["image"] if "image" in da3.files else None
    frame_count = min(len(authored["frames"]), int(da3_depths.shape[0]), len(color_frames))
    if frame_count < len(authored["frames"]):
        print(f"warning: only assembling {frame_count} of {len(authored['frames'])} authored frames")

    out_div.mkdir(parents=True, exist_ok=True)
    out_frames: list[dict] = []
    stats: list[dict] = []
    for idx in range(frame_count):
        frame = authored["frames"][idx]
        known_depth = np.load(conditioning_dir / frame["known_depth_npy"])
        mask = cv2.imread(str(conditioning_dir / frame["mask_path"]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(conditioning_dir / frame["mask_path"])

        da3_depth = _resize_depth(da3_depths[idx], width=width, height=height)
        known_depth = _resize_depth(known_depth, width=width, height=height)
        mask = _resize_mask(mask, width=width, height=height)
        keep_mask = mask <= int(args.keep_mask_max)
        write_mask = mask >= int(args.write_mask_min)
        scale, shift, fit_pixels, fit_rmse = _fit_scale_shift(
            da3_depth,
            known_depth,
            keep_mask,
            min_pixels=int(args.min_fit_pixels),
        )
        metric_depth = (scale * da3_depth + shift).astype(np.float32)
        valid = write_mask & np.isfinite(metric_depth) & (metric_depth > 0.0)
        if not np.any(valid):
            print(f"skipping frame {idx:04d}: no valid generated pixels")
            continue

        color = _resize_color(color_frames[idx], width=width, height=height)
        if da3_images is not None and color.shape[:2] != (height, width):
            color = cv2.cvtColor(_resize_color(da3_images[idx], width=width, height=height), cv2.COLOR_RGB2BGR)

        depth_out = np.zeros((height, width), dtype=np.float32)
        depth_out[valid] = metric_depth[valid]
        near = float(depth_out[valid].min())
        far = float(depth_out[valid].max())
        if far - near < 1.0e-4:
            far = near + 1.0

        color_out = np.zeros((height, width, 3), dtype=np.uint8)
        color_out[valid] = color[valid]
        normal_out = _default_normal_map(valid)

        depth_name = f"depth_{idx:04d}.png"
        color_name = f"color_{idx:04d}.png"
        normal_name = f"normal_{idx:04d}.png"
        cv2.imwrite(str(out_div / depth_name), _encode_depth_u16(depth_out, near=near, far=far))
        cv2.imwrite(str(out_div / color_name), color_out)
        cv2.imwrite(str(out_div / normal_name), normal_out)

        out_frame = {
            k: v
            for k, v in frame.items()
            if k not in {"known_depth_npy", "mask_path", "num_known_pixels", "mask_mean"}
        }
        out_frame.update(
            {
                "file_path": depth_name,
                "color_path": color_name,
                "normal_path": normal_name,
                "near": near,
                "far": far,
                "num_pixels_written": int(valid.sum()),
                "depth_fit_scale": scale,
                "depth_fit_shift": shift,
                "depth_fit_pixels": fit_pixels,
                "depth_fit_rmse_cm": fit_rmse,
            }
        )
        out_frames.append(out_frame)
        stats.append(
            {
                "frame": idx,
                "pixels": int(valid.sum()),
                "scale": scale,
                "shift": shift,
                "fit_pixels": fit_pixels,
                "fit_rmse_cm": fit_rmse,
                "near": near,
                "far": far,
            }
        )
        print(
            f"frame {idx:04d}: wrote={int(valid.sum())} "
            f"fit={fit_pixels} scale={scale:.4f} shift={shift:.2f} rmse={fit_rmse:.2f}cm"
        )

    output_cameras = {
        k: v
        for k, v in authored.items()
        if k not in {"frames", "conditioning"}
    }
    output_cameras.update(
        {
            "coordinate_units": "centimeters",
            "transform_translation_units": "centimeters",
            "source_conditioning_dir": str(conditioning_dir.resolve()),
            "source_generated_video": str(generated_video.resolve()) if generated_video else None,
            "source_generated_frames_dir": str(generated_frames_dir.resolve()) if generated_frames_dir else None,
            "source_da3_npz": str(da3_npz.resolve()),
            "assembly": {
                "method": "masked_vace_da3_affine_depth_fit",
                "keep_mask_max": int(args.keep_mask_max),
                "write_mask_min": int(args.write_mask_min),
                "min_fit_pixels": int(args.min_fit_pixels),
                "frames_written": len(out_frames),
            },
            "depth_encoding": {
                "dtype": "uint16",
                "invalid_code": 0,
                "valid_code_min": 1,
                "valid_code_max": 65535,
                "units": "centimeters",
                "decode_formula_export_units": "near + ((code - 1) / 65534) * (far - near)",
            },
            "frames": out_frames,
        }
    )
    _write_json(out_div / "cameras.json", output_cameras)
    _write_json(out_div / "assembly_stats.json", {"frames": stats})
    print("wrote volume:", out_div)


if __name__ == "__main__":
    main()
