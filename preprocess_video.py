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


_STREAMING_ALIGN_IRLS = {
    "delta": 0.1,
    "max_iters": 5,
    "tol": "1e-9",
}
_RUNTIME_EXPORT_NONE = "none"
_RUNTIME_EXPORT_DIRECTSTORAGE = "directstorage_stream"
_RUNTIME_EXPORT_HAP = "kinect_rgbd_video"
_RUNTIME_EXPORT_SEQUENCE = "packed_frame_sequence"
_RUNTIME_EXPORT_SEQUENCE_DEPTH8 = "packed_frame_sequence_depth8"
_RUNTIME_EXPORT_CHOICES = (
    _RUNTIME_EXPORT_NONE,
    _RUNTIME_EXPORT_DIRECTSTORAGE,
    _RUNTIME_EXPORT_HAP,
    _RUNTIME_EXPORT_SEQUENCE,
    _RUNTIME_EXPORT_SEQUENCE_DEPTH8,
)


def _streaming_select_frames(images: list[str], stride: Optional[int]) -> tuple[list[str], int]:
    _ = stride
    return images, 1


def _normalize_runtime_export_format(raw: str | None, *, legacy_kinect_rgbd_video: bool = False) -> str:
    candidate = str(raw or _RUNTIME_EXPORT_DIRECTSTORAGE).strip().lower()
    if candidate in {"", "default"}:
        candidate = _RUNTIME_EXPORT_DIRECTSTORAGE
    if candidate not in _RUNTIME_EXPORT_CHOICES:
        raise ValueError(
            f"Unknown runtime export format '{raw}'. Expected one of: {', '.join(_RUNTIME_EXPORT_CHOICES)}."
        )
    if legacy_kinect_rgbd_video and candidate in {_RUNTIME_EXPORT_NONE, _RUNTIME_EXPORT_DIRECTSTORAGE}:
        return _RUNTIME_EXPORT_HAP
    return candidate


def _runtime_export_label(runtime_export_format: str) -> str:
    return {
        _RUNTIME_EXPORT_NONE: "None",
        _RUNTIME_EXPORT_DIRECTSTORAGE: "DirectStorage stream",
        _RUNTIME_EXPORT_HAP: "Kinect RGBD video (HAP Q)",
        _RUNTIME_EXPORT_SEQUENCE: "Packed frame sequence",
        _RUNTIME_EXPORT_SEQUENCE_DEPTH8: "Packed frame sequence (8-bit depth)",
    }[runtime_export_format]


def _export_stage0_runtime_format(
    *,
    scene_root: str,
    runtime_export_format: str,
    fps: int,
    overwrite: bool,
) -> Optional[str]:
    if runtime_export_format == _RUNTIME_EXPORT_NONE:
        return None
    if runtime_export_format == _RUNTIME_EXPORT_DIRECTSTORAGE:
        from export_depth_image_stream_bc7 import export_depth_image_stream_bc7

        return str(
            export_depth_image_stream_bc7(
                scene_root=scene_root,
                fps=int(fps),
                overwrite=overwrite,
            )
        )
    if runtime_export_format == _RUNTIME_EXPORT_HAP:
        from export_stage0_kinect_video import export_stage0_kinect_video

        return str(
            export_stage0_kinect_video(
                scene_root=scene_root,
                fps=int(fps),
                overwrite=overwrite,
            )
        )
    if runtime_export_format == _RUNTIME_EXPORT_SEQUENCE:
        from export_stage0_kinect_video import export_stage0_kinect_image_sequence

        return str(
            export_stage0_kinect_image_sequence(
                scene_root=scene_root,
                fps=int(fps),
                overwrite=overwrite,
            )
        )
    if runtime_export_format == _RUNTIME_EXPORT_SEQUENCE_DEPTH8:
        from export_stage0_kinect_video import export_stage0_kinect_image_sequence_depth8

        return str(
            export_stage0_kinect_image_sequence_depth8(
                scene_root=scene_root,
                fps=int(fps),
                overwrite=overwrite,
            )
        )
    raise ValueError(f"Unsupported runtime export format: {runtime_export_format}")


def _build_streaming_chunk_indices(num_frames: int, chunk_size: int, overlap: int) -> list[tuple[int, int]]:
    if chunk_size < 1:
        raise ValueError("Streaming chunk size must be at least 1.")
    if overlap < 0:
        raise ValueError("Streaming overlap must be >= 0.")
    if overlap >= chunk_size:
        raise ValueError(
            f"Streaming overlap ({overlap}) must be smaller than the chunk size ({chunk_size})."
        )
    if num_frames <= chunk_size:
        return [(0, num_frames)]

    step = chunk_size - overlap
    chunk_indices: list[tuple[int, int]] = []
    for start_idx in range(0, num_frames, step):
        end_idx = min(start_idx + chunk_size, num_frames)
        chunk_indices.append((start_idx, end_idx))
        if end_idx >= num_frames:
            break
    return chunk_indices


def _as_homogeneous44(extrinsics: np.ndarray) -> np.ndarray:
    ext = np.asarray(extrinsics, dtype=np.float32)
    if ext.shape == (4, 4):
        return ext.copy()
    if ext.shape == (3, 4):
        H = np.eye(4, dtype=np.float32)
        H[:3, :4] = ext
        return H
    raise ValueError(f"Expected extrinsics with shape (3, 4) or (4, 4), got {ext.shape}")


def _depth_to_point_cloud_vectorized(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
) -> np.ndarray:
    """Convert a depth batch to world-space point maps using DA3-style w2c extrinsics."""
    depth_batch = np.asarray(depth, dtype=np.float32)
    intrinsics_batch = np.asarray(intrinsics, dtype=np.float32)
    extrinsics_batch = np.asarray(extrinsics, dtype=np.float32)

    if depth_batch.ndim == 2:
        depth_batch = depth_batch[None]
    if intrinsics_batch.ndim == 2:
        intrinsics_batch = intrinsics_batch[None]
    if extrinsics_batch.ndim == 2:
        extrinsics_batch = extrinsics_batch[None]

    N, H, W = depth_batch.shape
    us, vs = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    ones = np.ones_like(us, dtype=np.float32)
    pixel_coords = np.stack([us, vs, ones], axis=-1)

    intrinsics_inv = np.linalg.inv(intrinsics_batch)
    camera_dirs = np.einsum("nij,hwj->nhwi", intrinsics_inv, pixel_coords, optimize=True)
    camera_coords = camera_dirs * depth_batch[..., None]
    camera_coords_h = np.concatenate(
        [camera_coords, np.ones((N, H, W, 1), dtype=np.float32)],
        axis=-1,
    )

    extrinsics_h = np.stack([_as_homogeneous44(ext) for ext in extrinsics_batch], axis=0)
    c2w = np.linalg.inv(extrinsics_h)
    world_coords_h = np.einsum("nij,nhwj->nhwi", c2w, camera_coords_h, optimize=True)
    return world_coords_h[..., :3].astype(np.float32, copy=False)


def _save_combined_npz(
    *,
    output_path: str,
    images: list[np.ndarray],
    depths: list[np.ndarray],
    confs: list[np.ndarray],
    extrinsics: list[np.ndarray],
    intrinsics: list[np.ndarray],
    skies: Optional[list[np.ndarray]],
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_dict = {
        "image": np.stack(images, axis=0),
        "depth": np.round(np.stack(depths, axis=0), 6),
        "conf": np.round(np.stack(confs, axis=0), 2),
        "extrinsics": np.stack(extrinsics, axis=0),
        "intrinsics": np.stack(intrinsics, axis=0),
    }
    if skies is not None:
        save_dict["sky"] = np.stack(skies, axis=0).astype(bool)
    np.savez_compressed(output_path, **save_dict)


def _weighted_estimate_sim3(
    source_points: np.ndarray,
    target_points: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    total_weight = float(np.sum(weights))
    if total_weight < 1e-6:
        raise ValueError("Total alignment weight is too small for meaningful estimation.")

    normalized_weights = (weights / total_weight).astype(np.float32, copy=False)
    mu_src = np.sum(normalized_weights[:, None] * source_points, axis=0)
    mu_tgt = np.sum(normalized_weights[:, None] * target_points, axis=0)

    src_centered = source_points - mu_src
    tgt_centered = target_points - mu_tgt

    scale_src = np.sqrt(np.sum(normalized_weights * np.sum(src_centered**2, axis=1)))
    scale_tgt = np.sqrt(np.sum(normalized_weights * np.sum(tgt_centered**2, axis=1)))
    s = float(scale_tgt / max(scale_src, 1e-12))

    weighted_src = (s * src_centered) * np.sqrt(normalized_weights)[:, None]
    weighted_tgt = tgt_centered * np.sqrt(normalized_weights)[:, None]
    H = weighted_src.T @ weighted_tgt

    U, _, Vt = np.linalg.svd(H.astype(np.float32))
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    t = mu_tgt - s * (R @ mu_src)
    return s, R.astype(np.float32), t.astype(np.float32)


def _weighted_estimate_scale_and_translation(
    source_points: np.ndarray,
    target_points: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, np.ndarray]:
    total_weight = float(np.sum(weights))
    if total_weight < 1e-6:
        raise ValueError("Total alignment weight is too small for meaningful estimation.")

    normalized_weights = (weights / total_weight).astype(np.float32, copy=False)
    mu_src = np.sum(normalized_weights[:, None] * source_points, axis=0)
    mu_tgt = np.sum(normalized_weights[:, None] * target_points, axis=0)

    src_centered = source_points - mu_src
    tgt_centered = target_points - mu_tgt

    scale_src = np.sqrt(np.sum(normalized_weights * np.sum(src_centered**2, axis=1)))
    scale_tgt = np.sqrt(np.sum(normalized_weights * np.sum(tgt_centered**2, axis=1)))
    s = float(scale_tgt / max(scale_src, 1e-12))
    t = mu_tgt - s * mu_src
    return s, t.astype(np.float32)


def _weighted_estimate_overlap_rotation(
    *,
    rotations_target: np.ndarray,
    rotations_source: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, float]:
    total_weight = float(np.sum(weights))
    if total_weight < 1e-6:
        raise ValueError("Total overlap rotation weight is too small for meaningful estimation.")

    covariance = np.zeros((3, 3), dtype=np.float64)
    for rot_target, rot_source, weight in zip(rotations_target, rotations_source, weights):
        covariance += float(weight) * (
            np.asarray(rot_target, dtype=np.float64) @ np.asarray(rot_source, dtype=np.float64).T
        )

    U, _, Vt = np.linalg.svd(covariance, full_matrices=True)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = U @ Vt

    angle_residuals = []
    for rot_target, rot_source in zip(rotations_target, rotations_source):
        aligned = R @ np.asarray(rot_source, dtype=np.float64)
        delta = aligned @ np.asarray(rot_target, dtype=np.float64).T
        cos_angle = float(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0))
        angle_residuals.append(np.degrees(np.arccos(cos_angle)))

    mean_angle_deg = float(np.mean(angle_residuals)) if angle_residuals else 0.0
    return R.astype(np.float32), mean_angle_deg


def _robust_weighted_estimate_sim3(
    src: np.ndarray,
    tgt: np.ndarray,
    init_weights: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    delta = float(_STREAMING_ALIGN_IRLS["delta"])
    max_iters = int(_STREAMING_ALIGN_IRLS["max_iters"])
    tol = float(_STREAMING_ALIGN_IRLS["tol"])

    s, R, t = _weighted_estimate_sim3(src, tgt, init_weights)
    prev_error = float("inf")

    for _ in range(max_iters):
        transformed = s * (src @ R.T) + t
        residuals = np.linalg.norm(tgt - transformed, axis=1)
        abs_residuals = np.abs(residuals)
        huber_weights = np.ones_like(residuals, dtype=np.float32)
        large_mask = abs_residuals > delta
        huber_weights[large_mask] = delta / abs_residuals[large_mask]

        combined_weights = init_weights * huber_weights
        weight_sum = float(np.sum(combined_weights))
        if weight_sum < 1e-6:
            break
        combined_weights /= weight_sum

        s_new, R_new, t_new = _weighted_estimate_sim3(src, tgt, combined_weights)
        param_change = abs(s_new - s) + float(np.linalg.norm(t_new - t))
        rot_trace = float(np.clip((np.trace(R_new @ R.T) - 1.0) / 2.0, -1.0, 1.0))
        rot_angle = float(np.arccos(rot_trace))
        current_error = float(np.sum(np.where(
            abs_residuals <= delta,
            0.5 * residuals**2,
            delta * (abs_residuals - 0.5 * delta),
        ) * init_weights))

        if (param_change < tol and rot_angle < np.radians(0.1)) or (
            prev_error < float("inf") and abs(prev_error - current_error) < tol * max(prev_error, 1e-12)
        ):
            s, R, t = s_new, R_new, t_new
            break

        s, R, t = s_new, R_new, t_new
        prev_error = current_error

    return s, R.astype(np.float32), t.astype(np.float32)


def _weighted_align_point_maps(
    *,
    point_map_target: np.ndarray,
    conf_target: np.ndarray,
    point_map_source: np.ndarray,
    conf_source: np.ndarray,
    conf_threshold: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    aligned_points_target: list[np.ndarray] = []
    aligned_points_source: list[np.ndarray] = []
    confidence_weights: list[np.ndarray] = []

    num_frames = min(point_map_target.shape[0], point_map_source.shape[0])
    for frame_idx in range(num_frames):
        target_conf = conf_target[frame_idx]
        source_conf = conf_source[frame_idx]
        valid_mask = (target_conf > conf_threshold) & (source_conf > conf_threshold)
        valid_mask &= np.all(np.isfinite(point_map_target[frame_idx]), axis=-1)
        valid_mask &= np.all(np.isfinite(point_map_source[frame_idx]), axis=-1)

        if not np.any(valid_mask):
            continue

        target_points = point_map_target[frame_idx][valid_mask]
        source_points = point_map_source[frame_idx][valid_mask]
        combined_conf = np.sqrt(target_conf[valid_mask] * source_conf[valid_mask]).astype(np.float32, copy=False)

        aligned_points_target.append(target_points.astype(np.float32, copy=False))
        aligned_points_source.append(source_points.astype(np.float32, copy=False))
        confidence_weights.append(combined_conf)

    if not aligned_points_target:
        raise ValueError("No matching point pairs were found.")

    all_target = np.concatenate(aligned_points_target, axis=0)
    all_source = np.concatenate(aligned_points_source, axis=0)
    all_weights = np.concatenate(confidence_weights, axis=0)

    print(f"Streaming overlap alignment using {all_target.shape[0]:,} dense correspondences.")

    s, R, t = _robust_weighted_estimate_sim3(all_source, all_target, all_weights)
    transformed = s * (all_source @ R.T) + t
    mean_error = float(np.mean(np.linalg.norm(all_target - transformed, axis=1)))
    print(f"Streaming overlap alignment mean residual: {mean_error:.6f}")
    return s, R, t


def _weighted_align_point_maps_da3_dense(
    *,
    point_map_target: np.ndarray,
    conf_target: np.ndarray,
    point_map_source: np.ndarray,
    conf_source: np.ndarray,
    conf_threshold: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    aligned_points_target: list[np.ndarray] = []
    aligned_points_source: list[np.ndarray] = []
    confidence_weights: list[np.ndarray] = []

    num_frames = min(point_map_target.shape[0], point_map_source.shape[0])
    for frame_idx in range(num_frames):
        target_conf = conf_target[frame_idx]
        source_conf = conf_source[frame_idx]
        valid_mask = (target_conf > conf_threshold) & (source_conf > conf_threshold)
        valid_mask &= np.all(np.isfinite(point_map_target[frame_idx]), axis=-1)
        valid_mask &= np.all(np.isfinite(point_map_source[frame_idx]), axis=-1)

        if not np.any(valid_mask):
            continue

        target_points = point_map_target[frame_idx][valid_mask]
        source_points = point_map_source[frame_idx][valid_mask]
        combined_conf = np.sqrt(target_conf[valid_mask] * source_conf[valid_mask]).astype(np.float32, copy=False)

        aligned_points_target.append(target_points.astype(np.float32, copy=False))
        aligned_points_source.append(source_points.astype(np.float32, copy=False))
        confidence_weights.append(combined_conf)

    if not aligned_points_target:
        raise ValueError("No matching point pairs were found.")

    all_target = np.concatenate(aligned_points_target, axis=0)
    all_source = np.concatenate(aligned_points_source, axis=0)
    all_weights = np.concatenate(confidence_weights, axis=0)

    align_backend = "numpy"
    if torch.cuda.is_available():
        try:
            from loop_utils.alignment_torch import robust_weighted_estimate_sim3_torch
        except Exception as exc:
            print(f"DA3 torch alignment import failed ({exc}). Falling back to local numpy SIM(3).")
        else:
            align_backend = "da3_torch"
            s, R, t = robust_weighted_estimate_sim3_torch(
                all_source,
                all_target,
                all_weights,
                delta=float(_STREAMING_ALIGN_IRLS["delta"]),
                max_iters=int(_STREAMING_ALIGN_IRLS["max_iters"]),
                tol=float(_STREAMING_ALIGN_IRLS["tol"]),
                align_method="sim3",
            )
            transformed = s * (all_source @ R.T) + t
            mean_error = float(np.mean(np.linalg.norm(all_target - transformed, axis=1)))
            print(
                "Streaming overlap alignment using "
                f"{all_target.shape[0]:,} dense correspondences ({align_backend})."
            )
            print(f"Streaming overlap alignment mean residual: {mean_error:.6f}")
            return float(s), np.asarray(R, dtype=np.float32), np.asarray(t, dtype=np.float32)

    s, R, t = _robust_weighted_estimate_sim3(all_source, all_target, all_weights)
    transformed = s * (all_source @ R.T) + t
    mean_error = float(np.mean(np.linalg.norm(all_target - transformed, axis=1)))
    print(
        "Streaming overlap alignment using "
        f"{all_target.shape[0]:,} dense correspondences ({align_backend})."
    )
    print(f"Streaming overlap alignment mean residual: {mean_error:.6f}")
    return s, R, t


def _estimate_overlap_pose_sim3(
    *,
    extrinsics_target: np.ndarray,
    conf_target: np.ndarray,
    extrinsics_source: np.ndarray,
    conf_source: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    target_h = np.stack([_as_homogeneous44(ext) for ext in np.asarray(extrinsics_target, dtype=np.float32)], axis=0)
    source_h = np.stack([_as_homogeneous44(ext) for ext in np.asarray(extrinsics_source, dtype=np.float32)], axis=0)
    target_c2w = np.linalg.inv(target_h)
    source_c2w = np.linalg.inv(source_h)

    target_centers: list[np.ndarray] = []
    source_centers: list[np.ndarray] = []
    target_rotations: list[np.ndarray] = []
    source_rotations: list[np.ndarray] = []
    weights: list[float] = []
    num_frames = min(target_c2w.shape[0], source_c2w.shape[0])
    for frame_idx in range(num_frames):
        frame_weight = float(
            np.sqrt(
                max(float(np.mean(conf_target[frame_idx])), 0.0)
                * max(float(np.mean(conf_source[frame_idx])), 0.0)
            )
        )
        if frame_weight <= 0.0:
            continue

        target_pose = target_c2w[frame_idx]
        source_pose = source_c2w[frame_idx]
        target_centers.append(target_pose[:3, 3].astype(np.float32, copy=False))
        source_centers.append(source_pose[:3, 3].astype(np.float32, copy=False))
        target_rotations.append(target_pose[:3, :3].astype(np.float32, copy=False))
        source_rotations.append(source_pose[:3, :3].astype(np.float32, copy=False))
        weights.append(frame_weight)

    if not target_centers:
        raise ValueError("No overlapping camera poses were available for streaming alignment.")

    target_centers_np = np.stack(target_centers, axis=0)
    source_centers_np = np.stack(source_centers, axis=0)
    target_rotations_np = np.stack(target_rotations, axis=0)
    source_rotations_np = np.stack(source_rotations, axis=0)
    weights_np = np.asarray(weights, dtype=np.float32)
    R, mean_angle_deg = _weighted_estimate_overlap_rotation(
        rotations_target=target_rotations_np,
        rotations_source=source_rotations_np,
        weights=weights_np,
    )
    s, t = _weighted_estimate_scale_and_translation(
        source_points=source_centers_np @ R.T,
        target_points=target_centers_np,
        weights=weights_np,
    )
    transformed_centers = s * (source_centers_np @ R.T) + t
    mean_center_error = float(np.mean(np.linalg.norm(target_centers_np - transformed_centers, axis=1)))
    print(
        "Streaming overlap pose alignment: "
        f"mean center residual={mean_center_error:.6f}, "
        f"mean rotation residual={mean_angle_deg:.3f} deg"
    )
    return s, R.astype(np.float32), t.astype(np.float32)


def _compose_sim3(
    base: tuple[float, np.ndarray, np.ndarray],
    step: tuple[float, np.ndarray, np.ndarray],
) -> tuple[float, np.ndarray, np.ndarray]:
    base_s, base_R, base_t = base
    step_s, step_R, step_t = step
    composed_R = base_R @ step_R
    composed_s = base_s * step_s
    composed_t = base_s * (base_R @ step_t) + base_t
    return composed_s, composed_R.astype(np.float32), composed_t.astype(np.float32)


def _transform_extrinsics_to_global(
    extrinsic_w2c: np.ndarray,
    sim3: tuple[float, np.ndarray, np.ndarray],
) -> np.ndarray:
    s, R, t = sim3
    c2w = np.linalg.inv(_as_homogeneous44(extrinsic_w2c))
    S = np.eye(4, dtype=np.float32)
    S[:3, :3] = s * R
    S[:3, 3] = t
    transformed_c2w = S @ c2w
    transformed_c2w[:3, :3] /= s
    transformed_w2c = np.linalg.inv(transformed_c2w)
    return transformed_w2c[:3, :4].astype(np.float32, copy=False)


def _streaming_chunk_save_indices(
    *,
    chunk_index: int,
    num_chunks: int,
    chunk_len: int,
    overlap: int,
) -> range:
    if num_chunks == 1:
        return range(0, chunk_len)
    if chunk_index == num_chunks - 1:
        return range(0, chunk_len)
    save_end = max(chunk_len - overlap, 0)
    return range(0, save_end)


def _rebase_extrinsics_to_frame0_origin(extrinsics: list[np.ndarray]) -> list[np.ndarray]:
    if not extrinsics:
        return extrinsics

    extrinsics_h = np.stack([_as_homogeneous44(ext) for ext in extrinsics], axis=0)
    c2w = np.linalg.inv(extrinsics_h)
    c2w0_inv = np.linalg.inv(c2w[0])
    rebased_c2w = np.einsum("ij,njk->nik", c2w0_inv, c2w, optimize=True)
    rebased_w2c = np.linalg.inv(rebased_c2w)
    return [rebased_w2c[idx, :3, :4].astype(np.float32, copy=False) for idx in range(rebased_w2c.shape[0])]


def _run_streaming_da3(
    *,
    model,
    images_for_da3: list[str],
    scene_root: str,
    chunk_size: int,
    overlap: int,
    process_res: int,
    process_res_method: str,
    use_ray_pose: bool,
    ref_view_strategy: str,
) -> dict[str, object]:
    chunk_indices = _build_streaming_chunk_indices(len(images_for_da3), chunk_size, overlap)

    print(
        "DA3 streaming settings: "
        f"chunk_size={chunk_size}, overlap={overlap}, chunks={len(chunk_indices)}, "
        f"dense_align={'da3_torch' if torch.cuda.is_available() else 'numpy'}"
    )

    stored_images: list[np.ndarray | None] = [None] * len(images_for_da3)
    stored_depths: list[np.ndarray | None] = [None] * len(images_for_da3)
    stored_confs: list[np.ndarray | None] = [None] * len(images_for_da3)
    stored_extrinsics: list[np.ndarray | None] = [None] * len(images_for_da3)
    stored_intrinsics: list[np.ndarray | None] = [None] * len(images_for_da3)
    stored_skies: Optional[list[np.ndarray | None]] = None

    chunk_sim3: list[tuple[float, np.ndarray, np.ndarray]] = []
    previous_predictions = None
    current_transform = (
        1.0,
        np.eye(3, dtype=np.float32),
        np.zeros(3, dtype=np.float32),
    )

    for chunk_index, (chunk_start, chunk_end) in enumerate(chunk_indices):
        chunk_image_paths = images_for_da3[chunk_start:chunk_end]
        print(
            f"Streaming chunk {chunk_index + 1}/{len(chunk_indices)}: "
            f"frames {chunk_start}..{chunk_end - 1} ({len(chunk_image_paths)} images)"
        )

        predictions = model.inference(
            image=chunk_image_paths,
            process_res=process_res,
            process_res_method=process_res_method,
            infer_gs=False,
            use_ray_pose=use_ray_pose,
            ref_view_strategy=ref_view_strategy,
            align_to_input_ext_scale=False,
        )

        if predictions.conf is None or predictions.extrinsics is None or predictions.intrinsics is None:
            raise RuntimeError("DA3 streaming mode requires confidence, intrinsics, and extrinsics outputs.")
        if predictions.processed_images is None:
            raise RuntimeError("DA3 streaming mode requires processed images for final NPZ export.")

        chunk_depth = np.asarray(predictions.depth, dtype=np.float32)
        chunk_conf = np.asarray(predictions.conf, dtype=np.float32)
        chunk_intrinsics = np.asarray(predictions.intrinsics, dtype=np.float32)
        chunk_extrinsics = np.asarray(predictions.extrinsics, dtype=np.float32)
        chunk_images = np.asarray(predictions.processed_images, dtype=np.uint8)
        chunk_sky = None if predictions.sky is None else np.asarray(predictions.sky, dtype=bool)

        if stored_skies is None and chunk_sky is not None:
            stored_skies = [None] * len(images_for_da3)

        if chunk_index > 0:
            if previous_predictions is None:
                raise RuntimeError("Missing previous chunk state during DA3 streaming alignment.")

            conf_prev = previous_predictions["conf"][-overlap:]
            conf_cur = chunk_conf[:overlap]

            try:
                point_map_prev = _depth_to_point_cloud_vectorized(
                    previous_predictions["depth"][-overlap:],
                    previous_predictions["intrinsics"][-overlap:],
                    previous_predictions["extrinsics"][-overlap:],
                )
                point_map_cur = _depth_to_point_cloud_vectorized(
                    chunk_depth[:overlap],
                    chunk_intrinsics[:overlap],
                    chunk_extrinsics[:overlap],
                )
                conf_threshold = min(float(np.median(conf_prev)), float(np.median(conf_cur))) * 0.1
                pair_transform = _weighted_align_point_maps_da3_dense(
                    point_map_target=point_map_prev,
                    conf_target=conf_prev,
                    point_map_source=point_map_cur,
                    conf_source=conf_cur,
                    conf_threshold=conf_threshold,
                )
            except ValueError:
                print("Dense point-map overlap alignment failed. Falling back to camera-rig overlap alignment.")
                pair_transform = _estimate_overlap_pose_sim3(
                    extrinsics_target=previous_predictions["extrinsics"][-overlap:],
                    conf_target=conf_prev,
                    extrinsics_source=chunk_extrinsics[:overlap],
                    conf_source=conf_cur,
                )

            current_transform = _compose_sim3(current_transform, pair_transform)
            chunk_sim3.append(pair_transform)

        save_indices = _streaming_chunk_save_indices(
            chunk_index=chunk_index,
            num_chunks=len(chunk_indices),
            chunk_len=chunk_end - chunk_start,
            overlap=overlap,
        )
        depth_scale = float(current_transform[0])
        for local_idx in save_indices:
            global_idx = chunk_start + local_idx
            stored_images[global_idx] = chunk_images[local_idx]
            stored_depths[global_idx] = (chunk_depth[local_idx] * depth_scale).astype(np.float32, copy=False)
            stored_confs[global_idx] = chunk_conf[local_idx]
            stored_intrinsics[global_idx] = chunk_intrinsics[local_idx]
            stored_extrinsics[global_idx] = _transform_extrinsics_to_global(
                chunk_extrinsics[local_idx],
                current_transform,
            )
            if stored_skies is not None:
                if chunk_sky is None:
                    raise RuntimeError("Chunk sky output disappeared mid-stream.")
                stored_skies[global_idx] = chunk_sky[local_idx]

        previous_predictions = {
            "depth": chunk_depth,
            "conf": chunk_conf,
            "intrinsics": chunk_intrinsics,
            "extrinsics": chunk_extrinsics,
        }
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    missing_indices = [idx for idx, value in enumerate(stored_images) if value is None]
    if missing_indices:
        raise RuntimeError(f"DA3 streaming did not fill all output frames. Missing indices: {missing_indices[:8]}")

    final_extrinsics = _rebase_extrinsics_to_frame0_origin(
        [value for value in stored_extrinsics if value is not None]
    )
    npz_path = os.path.join(scene_root, "exports", "npz", "results.npz")
    _save_combined_npz(
        output_path=npz_path,
        images=[value for value in stored_images if value is not None],
        depths=[value for value in stored_depths if value is not None],
        confs=[value for value in stored_confs if value is not None],
        extrinsics=final_extrinsics,
        intrinsics=[value for value in stored_intrinsics if value is not None],
        skies=(
            None
            if stored_skies is None
            else [value for value in stored_skies if value is not None]
        ),
    )
    return {
        "npz_path": npz_path,
        "num_chunks": len(chunk_indices),
        "chunk_indices": chunk_indices,
        "num_frames": len(images_for_da3),
        "pairwise_sim3": chunk_sim3,
    }


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
        default=20,
        help=(
            "Maximum number of frames to run DA3 on. In streaming mode, this becomes the per-chunk "
            "DA3 batch size instead of a global frame cap."
        ),
    )
    p.add_argument(
        "--max_stride",
        type=int,
        default=6,
        help=(
            "Maximum stride between frames when subsampling. This is ignored in streaming mode, "
            "which now processes the full extracted clip by default."
        ),
    )
    p.add_argument(
        "--streaming",
        action="store_true",
        help=(
            "Use DA3-Streaming-style overlapping chunks. This keeps DA3 memory bounded while covering "
            "the full selected sequence."
        ),
    )
    p.add_argument(
        "--streaming_overlap",
        type=int,
        default=10,
        help="Frame overlap between adjacent DA3 streaming chunks.",
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
        default="saddle_balanced",
        help=(
            "DA3 multi-view reference-view strategy. "
            "Default is 'saddle_balanced' for this project."
        ),
    )
    p.add_argument(
        "--export_gs_video",
        action="store_true",
        help="Also export DA3's gs_video preview outputs. Disabled by default to keep Stage 0 faster.",
    )
    p.add_argument(
        "--runtime_export_format",
        type=str,
        default=_RUNTIME_EXPORT_DIRECTSTORAGE,
        choices=_RUNTIME_EXPORT_CHOICES,
        help=(
            "Optional Stage 0 runtime export written after results.npz. "
            "Choices: none, directstorage_stream, kinect_rgbd_video, "
            "packed_frame_sequence, packed_frame_sequence_depth8."
        ),
    )
    p.add_argument(
        "--export_kinect_rgbd_video",
        action="store_true",
        help=(
            "Legacy alias for `--runtime_export_format kinect_rgbd_video`."
        ),
    )
    p.add_argument(
        "--runtime_export_fps",
        "--kinect_rgbd_video_fps",
        dest="runtime_export_fps",
        type=int,
        default=30,
        help="Frame rate metadata used by the selected Stage 0 runtime export.",
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
        "--prepare_conf_mask_min_depth_range_percent",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Pre-ICP preparation: keep only pixels within a per-frame percentage of the valid depth range "
            "measured from the frame minimum depth."
        ),
    )
    p.add_argument(
        "--prepare_conf_min_depth_range_percent",
        type=float,
        default=50.0,
        help="Pre-ICP preparation: percent of the per-frame valid depth range kept from the minimum depth.",
    )
    p.add_argument(
        "--prepare_conf_mask_min_depth_range_meters",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Pre-ICP preparation: keep only pixels within a fixed per-frame metric distance from the minimum depth."
        ),
    )
    p.add_argument(
        "--prepare_conf_min_depth_range_meters",
        type=float,
        default=3.0,
        help="Pre-ICP preparation: metric distance kept from the per-frame minimum depth.",
    )
    p.add_argument(
        "--prepare_conf_mask_depth_edges",
        action="store_true",
        help="Pre-ICP preparation: suppress DA3 depth edges before point-cloud generation.",
    )
    p.add_argument(
        "--prepare_conf_edge_rtol",
        type=_parse_optional_float,
        default=0.1,
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
    streaming_enabled = bool(args.streaming)
    streaming_chunk_size = int(args.max_frames)
    streaming_overlap = int(args.streaming_overlap)
    runtime_export_format = _normalize_runtime_export_format(
        args.runtime_export_format,
        legacy_kinect_rgbd_video=bool(args.export_kinect_rgbd_video),
    )
    runtime_export_fps = int(args.runtime_export_fps)

    if streaming_enabled and args.export_gs_video:
        raise ValueError(
            "Stage 0 DA3 streaming mode does not support `--export_gs_video` yet. Disable the preview export or "
            "run non-streaming Stage 0."
        )

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
        if streaming_enabled:
            selected_images, stride = _streaming_select_frames(images, args.max_stride)
            print(
                "Streaming-selected frames: "
                f"N={len(selected_images)} (full extracted clip, chunk_size={streaming_chunk_size}, "
                f"overlap={streaming_overlap})"
            )
        else:
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
        if streaming_enabled:
            print(
                "Using existing frames directory directly in DA3 streaming mode: "
                f"N={len(selected_images)} (chunk_size={streaming_chunk_size}, overlap={streaming_overlap})"
            )
        else:
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
                "source_input_path": (
                    os.path.abspath(args.input_video) if args.input_video is not None else os.path.abspath(args.frames_dir)
                ),
                "image_ext": args.image_ext,
                "source": ("input_video" if args.input_video is not None else "frames_dir"),
                "max_frames": max_frames_meta,
                "max_stride": max_stride_meta,
                "actual_stride": stride,
                "streaming_enabled": streaming_enabled,
                "streaming_chunk_size": (streaming_chunk_size if streaming_enabled else None),
                "streaming_overlap": (streaming_overlap if streaming_enabled else None),
                "runtime_export_format": runtime_export_format,
                "runtime_export_fps": runtime_export_fps,
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
        f"export_gs_video={args.export_gs_video}, "
        f"streaming={streaming_enabled}"
    )
    if streaming_enabled:
        streaming_result = _run_streaming_da3(
            model=model,
            images_for_da3=images_for_da3,
            scene_root=scene_root,
            chunk_size=streaming_chunk_size,
            overlap=streaming_overlap,
            process_res=args.process_res,
            process_res_method=args.process_res_method,
            use_ray_pose=args.use_ray_pose,
            ref_view_strategy=ref_view_strategy,
        )
        npz_path = str(streaming_result["npz_path"])
    else:
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
    if streaming_enabled:
        summary_lines.append(
            f"- DA3 streaming: enabled (chunk_size={streaming_chunk_size}, overlap={streaming_overlap})"
        )
        summary_lines.append(f"- Frames processed: {len(images_for_da3)}")
    else:
        summary_lines.append("- DA3 streaming: disabled")
    if args.export_gs_video:
        summary_lines.append(f"- GS video: {os.path.join(scene_root, 'gs_video')}")
    else:
        summary_lines.append("- GS video: skipped")
    runtime_export_output = _export_stage0_runtime_format(
        scene_root=scene_root,
        runtime_export_format=runtime_export_format,
        fps=runtime_export_fps,
        overwrite=bool(args.overwrite),
    )
    if runtime_export_output is None:
        summary_lines.append("- Stage 0 runtime export: skipped")
    else:
        summary_lines.append(
            f"- Stage 0 runtime export ({_runtime_export_label(runtime_export_format)}): {runtime_export_output}"
        )
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
            conf_mask_min_depth_range_percent=bool(args.prepare_conf_mask_min_depth_range_percent),
            conf_min_depth_range_percent=float(args.prepare_conf_min_depth_range_percent),
            conf_mask_min_depth_range_meters=bool(args.prepare_conf_mask_min_depth_range_meters),
            conf_min_depth_range_meters=float(args.prepare_conf_min_depth_range_meters),
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

