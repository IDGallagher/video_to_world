#!/usr/bin/env python3
import argparse
import glob
import os
import shutil
import subprocess
import json
import sys
import time
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from configs.common import AlignmentDataConfig
from utils.third_party_bootstrap import prepend_local_third_party_paths
from utils.stage1_preparation import prepare_stage1_inputs


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _parse_optional_float(raw: str) -> Optional[float]:
    lowered = str(raw).strip().lower()
    if lowered in {"", "none", "null"}:
        return None
    return float(raw)


def _resolve_executable(name: str) -> str:
    resolved = shutil.which(name)
    if resolved:
        return resolved

    python_bin_dir = Path(sys.executable).resolve().parent
    for candidate in (python_bin_dir / name, python_bin_dir / f"{name}.exe"):
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        f"Required executable '{name}' was not found on PATH and was not found next to "
        f"the active Python interpreter at '{python_bin_dir}'."
    )


def _wait_for_readable_npz(
    npz_path: str,
    *,
    timeout_s: float = 60.0,
    poll_interval_s: float = 0.25,
) -> None:
    """Wait for DA3's async NPZ export thread to finish writing a readable file."""
    deadline = time.time() + timeout_s
    last_error: Exception | None = None

    while time.time() < deadline:
        if os.path.exists(npz_path):
            try:
                predictions = np.load(npz_path)
                try:
                    _ = predictions.files
                finally:
                    if hasattr(predictions, "close"):
                        predictions.close()
                return
            except (FileNotFoundError, EOFError, OSError, ValueError, zipfile.BadZipFile) as exc:
                last_error = exc
        time.sleep(poll_interval_s)

    base_msg = f"Timed out waiting for readable DA3 results at '{npz_path}'."
    if last_error is not None:
        raise TimeoutError(f"{base_msg} Last error: {last_error}") from last_error
    raise TimeoutError(base_msg)


def subsample_frames(
    images: list[str],
    max_frames: Optional[int] = None,
    max_stride: Optional[int] = None,
) -> tuple[list[str], int]:
    total_frames = len(images)

    # If we have fewer or equal frames than requested, just use all with stride 1.
    # This also respects the "max_stride is an upper bound" semantics.
    if max_frames is not None and total_frames <= max_frames:
        return images, 1

    # If no constraints, return everything
    if max_frames is None and max_stride is None:
        return images, 1

    # If max_frames is not specified, treat it as "use everything" and ignore max_stride.
    # (Current CLI always passes max_frames when subsampling is desired.)
    if max_frames is None:
        return images, 1

    # Normalise max_stride: None or <1 means "no effective upper bound"
    if max_stride is None or max_stride < 1:
        max_stride = total_frames

    # Ideal average stride to hit max_frames over total_frames
    ideal_stride = total_frames / float(max_frames)

    if ideal_stride <= max_stride:
        # We can (approximately) span the whole sequence.
        # Pick the smallest integer stride that still gives us at least max_frames,
        # i.e. floor(ideal_stride), but at least 1.
        stride = max(1, total_frames // max_frames)
    else:
        # Hard max_stride constraint prevents covering the full range uniformly.
        # Enforce max_stride strictly and truncate coverage.
        stride = max_stride

    indices = list(range(0, total_frames, stride))
    if len(indices) > max_frames:
        indices = indices[:max_frames]

    return [images[i] for i in indices], stride


def extract_frames(
    input_video: str,
    frames_dir: str,
    image_ext: str = "png",
) -> None:
    os.makedirs(frames_dir, exist_ok=True)

    # Clear destination if it already has files, to avoid mixing runs.
    existing = glob.glob(os.path.join(frames_dir, f"*.{image_ext}"))
    if existing:
        for p in existing:
            os.remove(p)

    out_pattern = os.path.join(frames_dir, f"%06d.{image_ext}")
    ffmpeg = _resolve_executable("ffmpeg")
    cmd = [ffmpeg, "-y", "-i", input_video]
    cmd += ["-vsync", "0", out_pattern]
    _run(cmd)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Preprocess a video (or frame folder) with Depth Anything 3 (DA3).\n"
            "Outputs are written into <scene_root>/exports/npz/results.npz and, when enabled, "
            "<scene_root>/gs_video/."
        )
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--input_video", type=str, default=None, help="Path to input video.")
    g.add_argument("--frames_dir", type=str, default=None, help="Path to an existing frames folder.")

    p.add_argument(
        "--scene_root",
        type=str,
        default=None,
        help=(
            "Scene output directory root. If omitted:\n"
            "  - with --input_video: defaults to the input video path without extension "
            "(e.g. /path/to/video.mp4 -> /path/to/video)\n"
            "  - with --frames_dir: defaults to <frames_dir>_preprocessed "
            "(e.g. /path/to/frames -> /path/to/frames_preprocessed)"
        ),
    )
    p.add_argument(
        "--model_name",
        type=str,
        default="depth-anything/DA3NESTED-GIANT-LARGE",
        help="DA3 model name/path (HuggingFace repo or local).",
    )
    p.add_argument("--image_ext", type=str, default="png", help="Frame file extension.")
    p.add_argument(
        "--max_frames",
        type=int,
        default=30,
        help="Maximum number of frames to run DA3 on.",
    )
    p.add_argument(
        "--max_stride",
        type=int,
        default=8,
        help="Maximum stride between frames when subsampling.",
    )
    p.add_argument(
        "--process_res",
        type=int,
        default=768,
        help=(
            "Processing resolution for DA3 inference. The longest side of each frame is "
            "resized to this value (in pixels) before running the depth model. Higher values "
            "produce denser point clouds but require more VRAM. VRAM scales roughly as "
            "(process_res / 504)^2. Common values: 504 (low), 768 (medium), 1024 (high)."
        ),
    )
    p.add_argument(
        "--process_res_method",
        type=str,
        default="upper_bound_resize",
        help="DA3 preprocessing resize strategy (for example: upper_bound_resize or upper_bound_crop).",
    )
    p.add_argument(
        "--use_ray_pose",
        action="store_true",
        help="Use DA3 ray-based pose estimation instead of the camera decoder.",
    )
    p.add_argument(
        "--ref_view_strategy",
        type=str,
        default="first",
        help=(
            "DA3 multi-view reference-view strategy. "
            "Default is 'first' for this project."
        ),
    )
    p.add_argument(
        "--export_gs_video",
        action="store_true",
        help="Also export DA3's gs_video preview outputs. Disabled by default to keep Stage 0 faster.",
    )
    p.add_argument(
        "--export_kinect_rgbd_video",
        action="store_true",
        help=(
            "Also export a KinectStreamer-style packed RGBD video under "
            "<scene_root>/exports/kinect_rgbd_video/."
        ),
    )
    p.add_argument(
        "--kinect_rgbd_video_fps",
        type=int,
        default=30,
        help="Frame rate for the packed Kinect-layout RGBD video export.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing DA3 outputs under scene_root.",
    )
    p.add_argument(
        "--prepare_stage1_inputs",
        action="store_true",
        help="Also materialize the Stage 1 pre-ICP cache and before_non_rigid_icp.ply during Stage 0.",
    )
    p.add_argument("--prepare_num_frames", type=int, default=50, help="Pre-ICP preparation: number of DA3 frames.")
    p.add_argument("--prepare_stride", type=int, default=2, help="Pre-ICP preparation: stride over DA3 frames.")
    p.add_argument("--prepare_offset", type=int, default=0, help="Pre-ICP preparation: starting DA3 frame offset.")
    p.add_argument(
        "--prepare_conf_thresh_percentile",
        type=float,
        default=80.0,
        help="Pre-ICP preparation: confidence percentile threshold.",
    )
    p.add_argument(
        "--prepare_conf_mode",
        type=str,
        default="voxel_or",
        help="Pre-ICP preparation: confidence-filter mode.",
    )
    p.add_argument(
        "--prepare_conf_global_percentile",
        type=_parse_optional_float,
        default=10.0,
        help="Pre-ICP preparation: optional global DA3 percentile.",
    )
    p.add_argument(
        "--prepare_conf_local_percentile",
        type=_parse_optional_float,
        default=10.0,
        help="Pre-ICP preparation: optional local/per-voxel DA3 percentile.",
    )
    p.add_argument(
        "--prepare_conf_voxel_size",
        type=float,
        default=1.0,
        help="Pre-ICP preparation: voxel size for voxel-guided filtering.",
    )
    p.add_argument(
        "--prepare_conf_voxel_min_count_percentile",
        type=_parse_optional_float,
        default=50.0,
        help="Pre-ICP preparation: minimum voxel occupancy percentile.",
    )
    p.add_argument(
        "--prepare_conf_mask_sky",
        action="store_true",
        help="Pre-ICP preparation: exclude DA3 sky-mask pixels before point-cloud generation.",
    )
    p.add_argument(
        "--prepare_conf_mask_sky_depth_band",
        action="store_true",
        help="Pre-ICP preparation: expand sky suppression by the top depth band of the sky plateau.",
    )
    p.add_argument(
        "--prepare_conf_sky_depth_band_percent",
        type=float,
        default=2.0,
        help="Pre-ICP preparation: percent width of the sky-depth expansion band.",
    )
    p.add_argument(
        "--prepare_conf_mask_depth_edges",
        action="store_true",
        help="Pre-ICP preparation: suppress DA3 depth edges before point-cloud generation.",
    )
    p.add_argument(
        "--prepare_conf_edge_rtol",
        type=_parse_optional_float,
        default=0.03,
        help="Pre-ICP preparation: relative threshold for depth-edge suppression.",
    )
    p.add_argument(
        "--prepare_conf_edge_atol",
        type=_parse_optional_float,
        default=None,
        help="Pre-ICP preparation: absolute threshold for depth-edge suppression.",
    )
    p.add_argument(
        "--prepare_conf_edge_kernel_size",
        type=int,
        default=3,
        help="Pre-ICP preparation: odd kernel size for depth-edge suppression.",
    )
    p.add_argument(
        "--prepare_conf_mask_max_depth",
        action="store_true",
        help="Pre-ICP preparation: suppress the max-depth plateau before point-cloud generation.",
    )
    p.add_argument(
        "--prepare_conf_max_depth_rtol",
        type=_parse_optional_float,
        default=0.001,
        help="Pre-ICP preparation: relative threshold for max-depth suppression.",
    )
    p.add_argument(
        "--prepare_conf_max_depth_atol",
        type=_parse_optional_float,
        default=None,
        help="Pre-ICP preparation: absolute threshold for max-depth suppression.",
    )
    p.add_argument(
        "--prepare_out_path",
        type=str,
        default=None,
        help="Optional output path for the prepared Stage 1 run directory.",
    )
    p.add_argument(
        "--prepare_out_suffix",
        type=str,
        default="",
        help="Optional suffix appended to the default prepared Stage 1 run directory name.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    prepend_local_third_party_paths("depth_anything_3")

    ref_view_strategy = str(args.ref_view_strategy).strip()

    if args.scene_root is None or str(args.scene_root).strip() == "":
        if args.input_video is not None:
            scene_root = os.path.splitext(os.path.abspath(args.input_video))[0]
        else:
            # Default for an existing frames folder: sibling scene dir next to the folder.
            frames_dir_abs = os.path.abspath(args.frames_dir)
            scene_root = f"{frames_dir_abs}_preprocessed"
    else:
        scene_root = os.path.abspath(args.scene_root)

    os.makedirs(scene_root, exist_ok=True)

    source_frames_dir: str
    if args.input_video is not None:
        frames_dir = os.path.join(scene_root, "frames")
        extract_frames(
            input_video=os.path.abspath(args.input_video),
            frames_dir=frames_dir,
            image_ext=args.image_ext,
        )
        source_frames_dir = frames_dir
    else:
        frames_dir = os.path.abspath(args.frames_dir)
        if not os.path.isdir(frames_dir):
            raise ValueError(f"frames_dir '{frames_dir}' is not a directory.")
        source_frames_dir = frames_dir

    images = sorted(glob.glob(os.path.join(frames_dir, f"*.{args.image_ext}")))
    if not images:
        raise ValueError(f"No '*.{args.image_ext}' frames found in '{frames_dir}'.")

    if args.input_video is not None:
        selected_images, stride = subsample_frames(
            images,
            max_frames=args.max_frames,
            max_stride=args.max_stride,
        )
        print(f"Subsampled frames: N={len(selected_images)} (stride {stride})")
        max_frames_meta = args.max_frames
        max_stride_meta = args.max_stride
    else:
        selected_images = images
        stride = 1
        max_frames_meta = None
        max_stride_meta = None
        print(f"Using existing frames directory directly: N={len(selected_images)}")

    # Materialize the selected frames into a dedicated folder so downstream code
    # can reliably "refer back" to the exact frames DA3 was run on.
    used_frames_dir = os.path.join(scene_root, "frames_subsampled")
    os.makedirs(used_frames_dir, exist_ok=True)
    # Clear destination if it already has files, to avoid mixing runs.
    existing = glob.glob(os.path.join(used_frames_dir, f"*.{args.image_ext}"))
    if existing:
        for p in existing:
            os.remove(p)

    for i, src_path in enumerate(selected_images):
        dst_path = os.path.join(used_frames_dir, f"{i:06d}.{args.image_ext}")
        shutil.copy2(src_path, dst_path)

    images_for_da3 = sorted(glob.glob(os.path.join(used_frames_dir, f"*.{args.image_ext}")))
    if len(images_for_da3) != len(selected_images):
        raise RuntimeError(
            f"Failed to materialize subsampled frames: expected {len(selected_images)} "
            f"but found {len(images_for_da3)} in '{used_frames_dir}'."
        )

    # Record which frames were used so downstream loaders can find "original" images.
    meta_path = os.path.join(scene_root, "preprocess_frames.json")
    with open(meta_path, "w") as f:
        json.dump(
            {
                "frames_dir": used_frames_dir,
                "source_frames_dir": source_frames_dir,
                "image_ext": args.image_ext,
                "source": ("input_video" if args.input_video is not None else "frames_dir"),
                "max_frames": max_frames_meta,
                "max_stride": max_stride_meta,
                "actual_stride": stride,
                "num_frames_used": len(images_for_da3),
            },
            f,
            indent=2,
            sort_keys=True,
        )

    # Optionally clear previous outputs (but keep frames).
    if args.overwrite:
        for rel in ["exports", "gs_video", "gs_ply", "glb", "depth_vis", "feat_vis", "colmap"]:
            p = os.path.join(scene_root, rel)
            if os.path.isdir(p):
                shutil.rmtree(p)

    # Import DA3 only after env is set up.
    from depth_anything_3.api import DepthAnything3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = DepthAnything3.from_pretrained(args.model_name)
    model = model.to(device=device)
    model.eval()

    export_format = "npz-gs_video" if args.export_gs_video else "npz"
    print(
        "DA3 inference settings: "
        f"process_res={args.process_res}, "
        f"process_res_method={args.process_res_method}, "
        f"use_ray_pose={args.use_ray_pose}, "
        f"ref_view_strategy={ref_view_strategy}, "
        f"export_gs_video={args.export_gs_video}"
    )
    model.inference(
        image=images_for_da3,
        export_dir=scene_root,
        export_format=export_format,
        process_res=args.process_res,
        process_res_method=args.process_res_method,
        infer_gs=args.export_gs_video,
        use_ray_pose=args.use_ray_pose,
        ref_view_strategy=ref_view_strategy,
        align_to_input_ext_scale=False,
    )

    npz_path = os.path.join(scene_root, "exports", "npz", "results.npz")
    _wait_for_readable_npz(npz_path)
    summary_lines = [
        "DA3 preprocessing complete.",
        f"- NPZ: {npz_path}",
    ]
    if args.export_gs_video:
        summary_lines.append(f"- GS video: {os.path.join(scene_root, 'gs_video')}")
    else:
        summary_lines.append("- GS video: skipped")
    if args.export_kinect_rgbd_video:
        from export_stage0_kinect_video import export_stage0_kinect_video

        packed_video_dir = export_stage0_kinect_video(
            scene_root=scene_root,
            fps=int(args.kinect_rgbd_video_fps),
            overwrite=bool(args.overwrite),
        )
        summary_lines.append(f"- Kinect RGBD video: {packed_video_dir}")
    else:
        summary_lines.append("- Kinect RGBD video: skipped")
    if args.prepare_stage1_inputs:
        alignment = AlignmentDataConfig(
            num_frames=int(args.prepare_num_frames),
            stride=int(args.prepare_stride),
            offset=int(args.prepare_offset),
            conf_thresh_percentile=float(args.prepare_conf_thresh_percentile),
            conf_mode=str(args.prepare_conf_mode),
            conf_global_percentile=(
                None if args.prepare_conf_global_percentile is None else float(args.prepare_conf_global_percentile)
            ),
            conf_local_percentile=(
                None if args.prepare_conf_local_percentile is None else float(args.prepare_conf_local_percentile)
            ),
            conf_voxel_size=float(args.prepare_conf_voxel_size),
            conf_voxel_min_count_percentile=(
                None
                if args.prepare_conf_voxel_min_count_percentile is None
                else float(args.prepare_conf_voxel_min_count_percentile)
            ),
            conf_mask_sky=bool(args.prepare_conf_mask_sky),
            conf_mask_sky_depth_band=bool(args.prepare_conf_mask_sky_depth_band),
            conf_sky_depth_band_percent=float(args.prepare_conf_sky_depth_band_percent),
            conf_mask_depth_edges=bool(args.prepare_conf_mask_depth_edges),
            conf_edge_rtol=None if args.prepare_conf_edge_rtol is None else float(args.prepare_conf_edge_rtol),
            conf_edge_atol=None if args.prepare_conf_edge_atol is None else float(args.prepare_conf_edge_atol),
            conf_edge_kernel_size=int(args.prepare_conf_edge_kernel_size),
            conf_mask_max_depth=bool(args.prepare_conf_mask_max_depth),
            conf_max_depth_rtol=(
                None if args.prepare_conf_max_depth_rtol is None else float(args.prepare_conf_max_depth_rtol)
            ),
            conf_max_depth_atol=(
                None if args.prepare_conf_max_depth_atol is None else float(args.prepare_conf_max_depth_atol)
            ),
        )
        prep_out_path, before_non_rigid_path = prepare_stage1_inputs(
            root_path=scene_root,
            alignment=alignment,
            out_path=args.prepare_out_path,
            out_suffix=str(args.prepare_out_suffix or ""),
            device="cpu",
            overwrite_before_non_rigid=bool(args.overwrite),
        )
        summary_lines.append(f"- Stage 1 prep cache: {os.path.join(scene_root, 'exports', 'ply')}")
        summary_lines.append(f"- Pre-ICP merge: {before_non_rigid_path}")
        summary_lines.append(f"- Prepared run dir: {prep_out_path}")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()

