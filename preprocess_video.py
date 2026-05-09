#!/usr/bin/env python3
import argparse
import glob
import math
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
from PIL import Image

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
    stride = 1 if max_stride is None or int(max_stride) < 1 else int(max_stride)
    selected = images[::stride]
    if max_frames is not None and int(max_frames) > 0:
        selected = selected[: int(max_frames)]
    return selected, stride


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


def _estimate_fixed_camera_inputs(
    image_paths: list[str],
    *,
    horizontal_fov_degrees: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not image_paths:
        raise ValueError("Cannot estimate fixed-camera inputs with no images.")

    hfov = float(horizontal_fov_degrees)
    if not math.isfinite(hfov) or hfov <= 1.0 or hfov >= 179.0:
        raise ValueError("Fixed camera horizontal FOV must be between 1 and 179 degrees.")

    hfov_rad = math.radians(hfov)
    intrinsics: list[np.ndarray] = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            width, height = image.size
        focal = (0.5 * float(width)) / math.tan(0.5 * hfov_rad)
        K = np.array(
            [
                [focal, 0.0, (float(width) - 1.0) * 0.5],
                [0.0, focal, (float(height) - 1.0) * 0.5],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        intrinsics.append(K)

    extrinsics = np.repeat(
        np.eye(4, dtype=np.float32)[None, :, :],
        len(image_paths),
        axis=0,
    )
    return extrinsics, np.stack(intrinsics, axis=0)


def _run_da3_inference(
    *,
    model,
    image_paths: list[str],
    process_res: int,
    process_res_method: str,
    infer_gs: bool,
    use_ray_pose: bool,
    ref_view_strategy: str,
    export_dir: str | None = None,
    export_format: str = "npz",
    fixed_camera: bool = False,
    fixed_camera_fov_degrees: float = 60.0,
    input_extrinsics: np.ndarray | None = None,
    input_intrinsics: np.ndarray | None = None,
    input_camera_params_processed: bool = False,
    align_to_input_ext_scale: bool = False,
):
    has_camera_priors = input_extrinsics is not None or input_intrinsics is not None
    if has_camera_priors and (input_extrinsics is None or input_intrinsics is None):
        raise ValueError("DA3 camera priors require both extrinsics and intrinsics.")
    if fixed_camera and has_camera_priors:
        raise ValueError("Fixed-camera DA3 mode cannot also use external camera priors.")

    if not fixed_camera and not has_camera_priors:
        return model.inference(
            image=image_paths,
            export_dir=export_dir,
            export_format=export_format,
            process_res=process_res,
            process_res_method=process_res_method,
            infer_gs=infer_gs,
            use_ray_pose=use_ray_pose,
            ref_view_strategy=ref_view_strategy,
            align_to_input_ext_scale=False,
        )

    if has_camera_priors and not input_camera_params_processed:
        return model.inference(
            image=image_paths,
            extrinsics=np.asarray(input_extrinsics, dtype=np.float32),
            intrinsics=np.asarray(input_intrinsics, dtype=np.float32),
            export_dir=export_dir,
            export_format=export_format,
            process_res=process_res,
            process_res_method=process_res_method,
            infer_gs=infer_gs,
            use_ray_pose=use_ray_pose,
            ref_view_strategy=ref_view_strategy,
            align_to_input_ext_scale=align_to_input_ext_scale,
        )

    if has_camera_priors:
        input_ext_np = np.stack(
            [_as_homogeneous44(ext) for ext in np.asarray(input_extrinsics, dtype=np.float32)],
            axis=0,
        )
        input_intr_np = np.asarray(input_intrinsics, dtype=np.float32)
        if input_ext_np.shape[0] != len(image_paths) or input_intr_np.shape[0] != len(image_paths):
            raise ValueError("DA3 camera prior counts must match image_paths.")

        # The guide pass produces intrinsics in DA3's processed image space.
        # Running the public API would resize those intrinsics again, so use the
        # lower-level path and attach the already-processed camera priors directly.
        imgs_cpu, _, _ = model._preprocess_inputs(
            image_paths,
            None,
            None,
            process_res,
            process_res_method,
        )
        ex_cpu = torch.from_numpy(input_ext_np).float()
        in_cpu = torch.from_numpy(input_intr_np).float()
        imgs, ex_t, in_t = model._prepare_model_inputs(imgs_cpu, ex_cpu, in_cpu)
        ex_t_norm = model._normalize_extrinsics(ex_t.clone() if ex_t is not None else None)
        export_feat_layers: list[int] = []
        raw_output = model._run_model_forward(
            imgs,
            ex_t_norm,
            in_t,
            export_feat_layers,
            infer_gs,
            use_ray_pose,
            ref_view_strategy,
        )
        prediction = model._convert_to_prediction(raw_output)
        prediction = model._align_to_input_extrinsics_intrinsics(
            ex_cpu,
            in_cpu,
            prediction,
            align_to_input_ext_scale,
        )
        prediction = model._add_processed_images(prediction, imgs_cpu)

        if export_dir is not None:
            model._export_results(prediction, export_format, export_dir)

        return prediction

    fixed_extrinsics, fixed_intrinsics = _estimate_fixed_camera_inputs(
        image_paths,
        horizontal_fov_degrees=fixed_camera_fov_degrees,
    )
    print(
        "DA3 fixed-camera mode: supplying identity extrinsics and estimated pinhole "
        f"intrinsics (horizontal_fov={fixed_camera_fov_degrees:g} deg)."
    )

    # DA3's public inference path always tries to Umeyama-align predictions to
    # supplied extrinsics. A static camera path is degenerate for that alignment,
    # so run the same lower-level steps and then explicitly keep the supplied
    # fixed-camera calibration in the exported prediction.
    imgs_cpu, ex_cpu, in_cpu = model._preprocess_inputs(
        image_paths,
        fixed_extrinsics,
        fixed_intrinsics,
        process_res,
        process_res_method,
    )
    imgs, ex_t, in_t = model._prepare_model_inputs(imgs_cpu, ex_cpu, in_cpu)
    ex_t_norm = model._normalize_extrinsics(ex_t.clone() if ex_t is not None else None)
    export_feat_layers: list[int] = []
    raw_output = model._run_model_forward(
        imgs,
        ex_t_norm,
        in_t,
        export_feat_layers,
        infer_gs,
        use_ray_pose,
        ref_view_strategy,
    )
    prediction = model._convert_to_prediction(raw_output)

    ex_np = ex_cpu.detach().cpu().numpy()
    if ex_np.shape[-2:] == (4, 4):
        ex_np = ex_np[:, :3, :4]
    prediction.extrinsics = ex_np.astype(np.float32, copy=False)
    prediction.intrinsics = in_cpu.detach().cpu().numpy().astype(np.float32, copy=False)
    prediction = model._add_processed_images(prediction, imgs_cpu)

    if export_dir is not None:
        model._export_results(prediction, export_format, export_dir)

    return prediction


_STREAMING_ALIGN_IRLS = {
    "delta": 0.1,
    "max_iters": 5,
    "tol": "1e-9",
}
# Dense overlap point correspondences are the primary stitch signal. Camera
# centers are only accepted when they do not make those overlapping 3-D points
# line up worse in the shared chunk basis.
_STREAMING_CAMERA_CENTER_ALIGN_WEIGHT = 1.0
_STREAMING_CAMERA_CENTER_MAX_DENSE_RESIDUAL_RATIO = 1.0
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
    actual_stride = 1 if stride is None or int(stride) < 1 else int(stride)
    return images[::actual_stride], actual_stride


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
    fps: float,
    overwrite: bool,
    prep_run: str | None = None,
) -> Optional[str]:
    if runtime_export_format == _RUNTIME_EXPORT_NONE:
        return None
    if runtime_export_format == _RUNTIME_EXPORT_DIRECTSTORAGE:
        from export_depth_image_stream_bc7 import export_depth_image_stream_bc7

        return str(
            export_depth_image_stream_bc7(
                scene_root=scene_root,
                fps=float(fps),
                overwrite=overwrite,
                prep_run=prep_run,
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


def _transform_point_map_sim3(
    point_map: np.ndarray,
    sim3: tuple[float, np.ndarray, np.ndarray],
) -> np.ndarray:
    s, R, t = sim3
    points = np.asarray(point_map, dtype=np.float32)
    transformed = float(s) * (points @ np.asarray(R, dtype=np.float32).T) + np.asarray(t, dtype=np.float32)
    return transformed.astype(np.float32, copy=False)


def _orthonormalize_rotation(rotation: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(np.asarray(rotation, dtype=np.float32))
    R = U @ Vt
    if np.linalg.det(R) < 0.0:
        U[:, -1] *= -1.0
        R = U @ Vt
    return R.astype(np.float32, copy=False)


def _rotation_matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    R = _orthonormalize_rotation(rotation)
    trace = float(np.trace(R))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * s,
                (R[2, 1] - R[1, 2]) / s,
                (R[0, 2] - R[2, 0]) / s,
                (R[1, 0] - R[0, 1]) / s,
            ],
            dtype=np.float32,
        )
    else:
        diag = np.diag(R)
        axis = int(np.argmax(diag))
        if axis == 0:
            s = np.sqrt(max(1.0 + R[0, 0] - R[1, 1] - R[2, 2], 1.0e-12)) * 2.0
            quat = np.array(
                [
                    (R[2, 1] - R[1, 2]) / s,
                    0.25 * s,
                    (R[0, 1] + R[1, 0]) / s,
                    (R[0, 2] + R[2, 0]) / s,
                ],
                dtype=np.float32,
            )
        elif axis == 1:
            s = np.sqrt(max(1.0 + R[1, 1] - R[0, 0] - R[2, 2], 1.0e-12)) * 2.0
            quat = np.array(
                [
                    (R[0, 2] - R[2, 0]) / s,
                    (R[0, 1] + R[1, 0]) / s,
                    0.25 * s,
                    (R[1, 2] + R[2, 1]) / s,
                ],
                dtype=np.float32,
            )
        else:
            s = np.sqrt(max(1.0 + R[2, 2] - R[0, 0] - R[1, 1], 1.0e-12)) * 2.0
            quat = np.array(
                [
                    (R[1, 0] - R[0, 1]) / s,
                    (R[0, 2] + R[2, 0]) / s,
                    (R[1, 2] + R[2, 1]) / s,
                    0.25 * s,
                ],
                dtype=np.float32,
            )

    norm = float(np.linalg.norm(quat))
    if norm < 1.0e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return (quat / norm).astype(np.float32, copy=False)


def _quaternion_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float32)
    norm = float(np.linalg.norm(q))
    if norm < 1.0e-12:
        return np.eye(3, dtype=np.float32)
    w, x, y, z = q / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _slerp_rotation_matrix(
    rotation_a: np.ndarray,
    rotation_b: np.ndarray,
    alpha: float,
) -> np.ndarray:
    a = float(np.clip(alpha, 0.0, 1.0))
    quat_a = _rotation_matrix_to_quaternion(rotation_a)
    quat_b = _rotation_matrix_to_quaternion(rotation_b)
    dot = float(np.dot(quat_a, quat_b))
    if dot < 0.0:
        quat_b = -quat_b
        dot = -dot

    if dot > 0.9995:
        quat = quat_a + a * (quat_b - quat_a)
        quat /= max(float(np.linalg.norm(quat)), 1.0e-12)
        return _quaternion_to_rotation_matrix(quat)

    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * a
    sin_theta = np.sin(theta)
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    quat = (s0 * quat_a) + (s1 * quat_b)
    return _quaternion_to_rotation_matrix(quat)


def _blend_extrinsics_w2c(
    extrinsic_a_w2c: np.ndarray,
    extrinsic_b_w2c: np.ndarray,
    alpha: float,
) -> np.ndarray:
    a = float(np.clip(alpha, 0.0, 1.0))
    c2w_a = np.linalg.inv(_as_homogeneous44(extrinsic_a_w2c))
    c2w_b = np.linalg.inv(_as_homogeneous44(extrinsic_b_w2c))

    blended_c2w = np.eye(4, dtype=np.float32)
    blended_c2w[:3, :3] = _slerp_rotation_matrix(c2w_a[:3, :3], c2w_b[:3, :3], a)
    blended_c2w[:3, 3] = ((1.0 - a) * c2w_a[:3, 3] + a * c2w_b[:3, 3]).astype(np.float32)
    return np.linalg.inv(blended_c2w)[:3, :4].astype(np.float32, copy=False)


def _blend_intrinsics(
    intrinsics_a: np.ndarray,
    intrinsics_b: np.ndarray,
    alpha: float,
) -> np.ndarray:
    a = float(np.clip(alpha, 0.0, 1.0))
    return (
        (1.0 - a) * np.asarray(intrinsics_a, dtype=np.float32)
        + a * np.asarray(intrinsics_b, dtype=np.float32)
    ).astype(np.float32, copy=False)


def _identity_sim3() -> tuple[float, np.ndarray, np.ndarray]:
    return (
        1.0,
        np.eye(3, dtype=np.float32),
        np.zeros(3, dtype=np.float32),
    )


def _build_streaming_guide_indices(num_frames: int, guide_count: int) -> list[int]:
    if num_frames <= 0:
        return []
    count = min(max(1, int(guide_count)), int(num_frames))
    if count >= num_frames:
        return list(range(num_frames))
    raw = np.linspace(0, num_frames - 1, count)
    indices: list[int] = []
    seen: set[int] = set()
    for value in raw:
        idx = int(round(float(value)))
        idx = max(0, min(num_frames - 1, idx))
        if idx not in seen:
            indices.append(idx)
            seen.add(idx)
    if 0 not in seen:
        indices.insert(0, 0)
        seen.add(0)
    if (num_frames - 1) not in seen:
        indices.append(num_frames - 1)
    return sorted(indices)


def _interpolate_streaming_guide_priors(
    *,
    num_frames: int,
    guide_indices: list[int],
    guide_extrinsics: np.ndarray,
    guide_intrinsics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if not guide_indices:
        raise ValueError("Cannot interpolate streaming guide priors with no guide frames.")

    indices = np.asarray(guide_indices, dtype=np.int64)
    order = np.argsort(indices)
    indices = indices[order]
    guide_ext_h = np.stack(
        [_as_homogeneous44(ext) for ext in np.asarray(guide_extrinsics, dtype=np.float32)[order]],
        axis=0,
    )
    guide_intr = np.asarray(guide_intrinsics, dtype=np.float32)[order]
    guide_c2w = np.linalg.inv(guide_ext_h)

    all_ext = np.empty((num_frames, 4, 4), dtype=np.float32)
    all_intr = np.empty((num_frames, 3, 3), dtype=np.float32)

    for frame_idx in range(num_frames):
        right = int(np.searchsorted(indices, frame_idx, side="left"))
        if right < len(indices) and int(indices[right]) == frame_idx:
            c2w = guide_c2w[right].astype(np.float32, copy=True)
            intr = guide_intr[right]
        elif right <= 0:
            c2w = guide_c2w[0].astype(np.float32, copy=True)
            intr = guide_intr[0]
        elif right >= len(indices):
            c2w = guide_c2w[-1].astype(np.float32, copy=True)
            intr = guide_intr[-1]
        else:
            left = right - 1
            span = max(int(indices[right]) - int(indices[left]), 1)
            alpha = float(frame_idx - int(indices[left])) / float(span)
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :3] = _slerp_rotation_matrix(
                guide_c2w[left, :3, :3],
                guide_c2w[right, :3, :3],
                alpha,
            )
            c2w[:3, 3] = (
                (1.0 - alpha) * guide_c2w[left, :3, 3]
                + alpha * guide_c2w[right, :3, 3]
            ).astype(np.float32)
            intr = _blend_intrinsics(guide_intr[left], guide_intr[right], alpha)

        all_ext[frame_idx] = np.linalg.inv(c2w).astype(np.float32, copy=False)
        all_intr[frame_idx] = intr.astype(np.float32, copy=False)

    return all_ext, all_intr


def _save_streaming_guide_npz(
    *,
    scene_root: str,
    guide_indices: list[int],
    prediction,
) -> str:
    output_dir = os.path.join(scene_root, "exports", "npz")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "streaming_guide.npz")
    save_dict: dict[str, np.ndarray] = {
        "guide_indices": np.asarray(guide_indices, dtype=np.int32),
        "depth": np.asarray(prediction.depth, dtype=np.float32),
    }
    if prediction.conf is not None:
        save_dict["conf"] = np.asarray(prediction.conf, dtype=np.float32)
    if prediction.extrinsics is not None:
        save_dict["extrinsics"] = np.asarray(prediction.extrinsics, dtype=np.float32)
    if prediction.intrinsics is not None:
        save_dict["intrinsics"] = np.asarray(prediction.intrinsics, dtype=np.float32)
    np.savez_compressed(output_path, **save_dict)
    return output_path


def _estimate_chunk_depth_scale_to_guide(
    *,
    chunk_start: int,
    chunk_depth: np.ndarray,
    guide_indices: list[int],
    guide_depth: np.ndarray,
) -> tuple[float | None, int, int]:
    guide_lookup = {int(global_idx): local_idx for local_idx, global_idx in enumerate(guide_indices)}
    per_frame_scales: list[float] = []
    per_frame_counts: list[int] = []

    for local_idx in range(chunk_depth.shape[0]):
        guide_local_idx = guide_lookup.get(chunk_start + local_idx)
        if guide_local_idx is None:
            continue
        if guide_local_idx >= guide_depth.shape[0]:
            continue
        source = np.asarray(chunk_depth[local_idx], dtype=np.float32)
        target = np.asarray(guide_depth[guide_local_idx], dtype=np.float32)
        if source.shape != target.shape:
            continue

        valid = np.isfinite(source) & np.isfinite(target) & (source > 0.0) & (target > 0.0)
        if not np.any(valid):
            continue

        ratio = (target[valid] / np.maximum(source[valid], 1.0e-12)).astype(np.float32, copy=False)
        ratio = ratio[np.isfinite(ratio) & (ratio > 0.02) & (ratio < 50.0)]
        if ratio.size < 1024:
            continue
        if ratio.size > 200_000:
            step = int(np.ceil(float(ratio.size) / 200_000.0))
            ratio = ratio[::step]
        low, high = np.percentile(ratio, [10.0, 90.0])
        trimmed = ratio[(ratio >= low) & (ratio <= high)]
        if trimmed.size < 1024:
            trimmed = ratio
        per_frame_scales.append(float(np.median(trimmed)))
        per_frame_counts.append(int(trimmed.size))

    if not per_frame_scales:
        return None, 0, 0

    weights = np.asarray(per_frame_counts, dtype=np.float64)
    scales = np.asarray(per_frame_scales, dtype=np.float64)
    scale = float(np.sum(scales * weights) / max(float(np.sum(weights)), 1.0))
    if not math.isfinite(scale) or scale <= 0.0:
        return None, len(per_frame_scales), int(np.sum(per_frame_counts))
    return scale, len(per_frame_scales), int(np.sum(per_frame_counts))


def _estimate_chunk_transform_to_guide(
    *,
    chunk_start: int,
    chunk_depth: np.ndarray,
    chunk_conf: np.ndarray,
    chunk_intrinsics: np.ndarray,
    chunk_extrinsics: np.ndarray,
    guide_indices: list[int],
    guide_depth: np.ndarray,
    guide_conf: np.ndarray,
    guide_intrinsics: np.ndarray,
    guide_extrinsics: np.ndarray,
) -> tuple[tuple[float, np.ndarray, np.ndarray] | None, list[int]]:
    guide_lookup = {int(global_idx): local_idx for local_idx, global_idx in enumerate(guide_indices)}
    chunk_local_indices: list[int] = []
    guide_local_indices: list[int] = []
    matched_global_indices: list[int] = []

    for local_idx in range(chunk_depth.shape[0]):
        global_idx = chunk_start + local_idx
        guide_local_idx = guide_lookup.get(global_idx)
        if guide_local_idx is None:
            continue
        if guide_local_idx >= guide_depth.shape[0]:
            continue
        if chunk_depth[local_idx].shape != guide_depth[guide_local_idx].shape:
            continue
        chunk_local_indices.append(local_idx)
        guide_local_indices.append(guide_local_idx)
        matched_global_indices.append(global_idx)

    if not chunk_local_indices:
        return None, []

    chunk_local = np.asarray(chunk_local_indices, dtype=np.int64)
    guide_local = np.asarray(guide_local_indices, dtype=np.int64)

    chunk_depth_anchor = np.asarray(chunk_depth[chunk_local], dtype=np.float32)
    guide_depth_anchor = np.asarray(guide_depth[guide_local], dtype=np.float32)
    chunk_conf_anchor = np.asarray(chunk_conf[chunk_local], dtype=np.float32).copy()
    guide_conf_anchor = np.asarray(guide_conf[guide_local], dtype=np.float32).copy()
    chunk_conf_anchor[~(np.isfinite(chunk_depth_anchor) & (chunk_depth_anchor > 0.0))] = 0.0
    guide_conf_anchor[~(np.isfinite(guide_depth_anchor) & (guide_depth_anchor > 0.0))] = 0.0

    chunk_points = _depth_to_point_cloud_vectorized(
        chunk_depth_anchor,
        np.asarray(chunk_intrinsics[chunk_local], dtype=np.float32),
        np.asarray(chunk_extrinsics[chunk_local], dtype=np.float32),
    )
    guide_points = _depth_to_point_cloud_vectorized(
        guide_depth_anchor,
        np.asarray(guide_intrinsics[guide_local], dtype=np.float32),
        np.asarray(guide_extrinsics[guide_local], dtype=np.float32),
    )

    conf_threshold = min(float(np.median(chunk_conf_anchor)), float(np.median(guide_conf_anchor))) * 0.1
    try:
        transform = _weighted_align_point_maps_da3_dense(
            point_map_target=guide_points,
            conf_target=guide_conf_anchor,
            point_map_source=chunk_points,
            conf_source=chunk_conf_anchor,
            conf_threshold=conf_threshold,
            camera_centers_target=_camera_centers_from_extrinsics(
                np.asarray(guide_extrinsics[guide_local], dtype=np.float32)
            ),
            camera_centers_source=_camera_centers_from_extrinsics(
                np.asarray(chunk_extrinsics[chunk_local], dtype=np.float32)
            ),
        )
        return transform, matched_global_indices
    except ValueError as exc:
        print(f"Streaming guide anchor dense Sim(3) failed ({exc}); falling back to camera-center translation.")

    guide_centers = _camera_centers_from_extrinsics(np.asarray(guide_extrinsics[guide_local], dtype=np.float32))
    chunk_centers = _camera_centers_from_extrinsics(np.asarray(chunk_extrinsics[chunk_local], dtype=np.float32))
    valid = np.all(np.isfinite(guide_centers), axis=1) & np.all(np.isfinite(chunk_centers), axis=1)
    if not np.any(valid):
        return None, matched_global_indices

    translation = np.median(guide_centers[valid] - chunk_centers[valid], axis=0).astype(np.float32)
    return (1.0, np.eye(3, dtype=np.float32), translation), matched_global_indices


def _point_map_to_depth_map(
    *,
    point_map_world: np.ndarray,
    intrinsics: np.ndarray,
    extrinsic_w2c: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Encode world-space points as z-depths along an existing camera ray grid."""
    points = np.asarray(point_map_world, dtype=np.float32)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"point_map_world must be (H,W,3), got {points.shape}")

    height, width, _ = points.shape
    us, vs = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    pixel_coords = np.stack([us, vs, np.ones_like(us, dtype=np.float32)], axis=-1)
    rays = np.einsum("ij,hwj->hwi", np.linalg.inv(np.asarray(intrinsics, dtype=np.float32)), pixel_coords)

    points_h = np.concatenate([points, np.ones((height, width, 1), dtype=np.float32)], axis=-1)
    camera_points_h = np.einsum("ij,hwj->hwi", _as_homogeneous44(extrinsic_w2c), points_h)
    camera_points = camera_points_h[..., :3]

    ray_norm_sq = np.sum(rays * rays, axis=-1)
    depth = np.sum(camera_points * rays, axis=-1) / np.maximum(ray_norm_sq, 1.0e-12)
    projected_camera_points = rays * depth[..., None]
    projection_error = np.linalg.norm(camera_points - projected_camera_points, axis=-1)

    valid = np.isfinite(depth) & (depth > 0.0) & (camera_points[..., 2] > 0.0)
    valid &= np.all(np.isfinite(camera_points), axis=-1)
    if valid_mask is not None:
        valid &= np.asarray(valid_mask, dtype=bool)

    encoded_depth = np.zeros((height, width), dtype=np.float32)
    encoded_depth[valid] = depth[valid].astype(np.float32, copy=False)
    mean_projection_error = float(np.mean(projection_error[valid])) if np.any(valid) else 0.0
    return encoded_depth, mean_projection_error


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


def _camera_centers_from_extrinsics(extrinsics: np.ndarray) -> np.ndarray:
    extrinsics_h = np.stack([_as_homogeneous44(ext) for ext in np.asarray(extrinsics, dtype=np.float32)], axis=0)
    c2w = np.linalg.inv(extrinsics_h)
    return c2w[:, :3, 3].astype(np.float32, copy=False)


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
    camera_centers_target: np.ndarray | None = None,
    camera_centers_source: np.ndarray | None = None,
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

    dense_target = np.concatenate(aligned_points_target, axis=0)
    dense_source = np.concatenate(aligned_points_source, axis=0)
    dense_weights = np.concatenate(confidence_weights, axis=0)
    dense_count = int(dense_target.shape[0])
    dense_weight_sum = float(np.sum(dense_weights))

    center_target = None
    center_source = None
    center_count = 0
    if (
        camera_centers_target is not None
        and camera_centers_source is not None
        and _STREAMING_CAMERA_CENTER_ALIGN_WEIGHT > 0.0
        and dense_weight_sum > 0.0
    ):
        center_target = np.asarray(camera_centers_target, dtype=np.float32)
        center_source = np.asarray(camera_centers_source, dtype=np.float32)
        center_count = min(center_target.shape[0], center_source.shape[0])
        if center_count > 0:
            center_target = center_target[:center_count]
            center_source = center_source[:center_count]
            center_valid = np.all(np.isfinite(center_target), axis=1) & np.all(np.isfinite(center_source), axis=1)
            center_target = center_target[center_valid]
            center_source = center_source[center_valid]
            center_count = int(center_target.shape[0])
        if center_count > 0:
            center_weight = (
                dense_weight_sum
                * float(_STREAMING_CAMERA_CENTER_ALIGN_WEIGHT)
                / float(center_count)
            )
            augmented_target = np.concatenate([dense_target, center_target], axis=0)
            augmented_source = np.concatenate([dense_source, center_source], axis=0)
            augmented_weights = np.concatenate(
                [dense_weights, np.full(center_count, center_weight, dtype=np.float32)],
                axis=0,
            )
        else:
            augmented_target = dense_target
            augmented_source = dense_source
            augmented_weights = dense_weights
    else:
        augmented_target = dense_target
        augmented_source = dense_source
        augmented_weights = dense_weights

    torch_estimator = None
    align_backend = "numpy"
    if torch.cuda.is_available():
        try:
            from loop_utils.alignment_torch import robust_weighted_estimate_sim3_torch
        except Exception as exc:
            print(f"DA3 torch alignment import failed ({exc}). Falling back to local numpy SIM(3).")
        else:
            torch_estimator = robust_weighted_estimate_sim3_torch
            align_backend = "da3_torch"

    def _estimate(
        source: np.ndarray,
        target: np.ndarray,
        weights: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        if torch_estimator is not None:
            s_est, R_est, t_est = torch_estimator(
                source,
                target,
                weights,
                delta=float(_STREAMING_ALIGN_IRLS["delta"]),
                max_iters=int(_STREAMING_ALIGN_IRLS["max_iters"]),
                tol=float(_STREAMING_ALIGN_IRLS["tol"]),
                align_method="sim3",
            )
            return (
                float(s_est),
                np.asarray(R_est, dtype=np.float32),
                np.asarray(t_est, dtype=np.float32),
            )
        return _robust_weighted_estimate_sim3(source, target, weights)

    def _dense_residual_metrics(s: float, R: np.ndarray, t: np.ndarray) -> dict[str, float]:
        transformed = s * (dense_source @ R.T) + t
        residuals = np.linalg.norm(dense_target - transformed, axis=1)
        return {
            "mean": float(np.mean(residuals)),
            "weighted_mean": float(np.average(residuals, weights=dense_weights)),
            "median": float(np.median(residuals)),
            "p95": float(np.percentile(residuals, 95)),
        }

    def _center_mean_residual(s: float, R: np.ndarray, t: np.ndarray) -> float | None:
        if center_count <= 0 or center_target is None or center_source is None:
            return None
        center_transformed = s * (center_source @ R.T) + t
        return float(np.mean(np.linalg.norm(center_target - center_transformed, axis=1)))

    def _format_metrics(metrics: dict[str, float]) -> str:
        return (
            f"mean={metrics['mean']:.6f}, weighted_mean={metrics['weighted_mean']:.6f}, "
            f"median={metrics['median']:.6f}, p95={metrics['p95']:.6f}"
        )

    print(
        "Streaming overlap alignment using "
        f"{dense_count:,} dense correspondences ({align_backend}; "
        f"{center_count} camera centers, center_weight={_STREAMING_CAMERA_CENTER_ALIGN_WEIGHT:g}x)."
    )

    dense_s, dense_R, dense_t = _estimate(dense_source, dense_target, dense_weights)
    dense_metrics = _dense_residual_metrics(dense_s, dense_R, dense_t)
    selected_s, selected_R, selected_t = dense_s, dense_R, dense_t
    selected_label = "dense points"
    print(f"Streaming overlap same-basis point residual (dense only): {_format_metrics(dense_metrics)}")

    if center_count > 0:
        center_s, center_R, center_t = _estimate(augmented_source, augmented_target, augmented_weights)
        center_metrics = _dense_residual_metrics(center_s, center_R, center_t)
        print(
            "Streaming overlap same-basis point residual (with camera centers): "
            f"{_format_metrics(center_metrics)}"
        )

        dense_center_error = _center_mean_residual(dense_s, dense_R, dense_t)
        center_center_error = _center_mean_residual(center_s, center_R, center_t)
        if dense_center_error is not None and center_center_error is not None:
            print(
                "Streaming overlap camera-center mean residual: "
                f"dense_only={dense_center_error:.6f}, with_centers={center_center_error:.6f}"
            )

        max_center_mean = dense_metrics["mean"] * float(_STREAMING_CAMERA_CENTER_MAX_DENSE_RESIDUAL_RATIO)
        if center_metrics["mean"] <= max_center_mean:
            selected_s, selected_R, selected_t = center_s, center_R, center_t
            selected_label = "dense points + camera centers"
        else:
            print(
                "Rejected camera-center stitch candidate because it worsened the "
                "same-basis dense point residual."
            )

    selected_metrics = _dense_residual_metrics(selected_s, selected_R, selected_t)
    print(
        "Selected streaming overlap stitch from "
        f"{selected_label}: {_format_metrics(selected_metrics)}"
    )
    return selected_s, selected_R, selected_t


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
    """Return local chunk frames to keep when stitching overlapping DA3 chunks.

    Keep the trailing overlap from the previous chunk and skip the leading
    duplicate frames from the next chunk. This moves cuts to the far side of the
    overlap that was used for alignment instead of switching at the first
    overlapped frame.
    """
    if num_chunks == 1:
        return range(0, chunk_len)
    if chunk_index == 0:
        return range(0, chunk_len)
    return range(min(overlap, chunk_len), chunk_len)


def _streaming_overlap_blend_alpha(frame_index: int, overlap_count: int) -> float:
    if overlap_count <= 1:
        return 0.5
    return float(frame_index) / float(overlap_count - 1)


def _blend_streaming_overlap_point_maps(
    *,
    stored_depths: list[np.ndarray | None],
    stored_confs: list[np.ndarray | None],
    stored_intrinsics: list[np.ndarray | None],
    stored_extrinsics: list[np.ndarray | None],
    chunk_start: int,
    previous_depth: np.ndarray,
    previous_conf: np.ndarray,
    previous_point_map: np.ndarray,
    previous_transform: tuple[float, np.ndarray, np.ndarray],
    current_depth: np.ndarray,
    current_conf: np.ndarray,
    current_intrinsics: np.ndarray,
    current_extrinsics: np.ndarray,
    current_point_map: np.ndarray,
    current_transform: tuple[float, np.ndarray, np.ndarray],
) -> None:
    overlap_count = min(
        previous_depth.shape[0],
        previous_conf.shape[0],
        previous_point_map.shape[0],
        current_depth.shape[0],
        current_conf.shape[0],
        current_intrinsics.shape[0],
        current_extrinsics.shape[0],
        current_point_map.shape[0],
    )
    if overlap_count <= 0:
        return

    previous_points_global = _transform_point_map_sim3(previous_point_map[:overlap_count], previous_transform)
    current_points_global = _transform_point_map_sim3(current_point_map[:overlap_count], current_transform)

    encoded_pixel_counts: list[int] = []
    projection_errors: list[float] = []
    for overlap_idx in range(overlap_count):
        global_idx = chunk_start + overlap_idx
        if global_idx >= len(stored_depths):
            continue
        previous_intrinsics = stored_intrinsics[global_idx]
        previous_extrinsics = stored_extrinsics[global_idx]
        if previous_intrinsics is None or previous_extrinsics is None:
            continue

        alpha = _streaming_overlap_blend_alpha(overlap_idx, overlap_count)
        current_extrinsic_global = _transform_extrinsics_to_global(
            current_extrinsics[overlap_idx],
            current_transform,
        )
        blended_intrinsics = _blend_intrinsics(
            previous_intrinsics,
            current_intrinsics[overlap_idx],
            alpha,
        )
        blended_extrinsics = _blend_extrinsics_w2c(
            previous_extrinsics,
            current_extrinsic_global,
            alpha,
        )
        prev_valid = np.isfinite(previous_depth[overlap_idx]) & (previous_depth[overlap_idx] > 0.0)
        cur_valid = np.isfinite(current_depth[overlap_idx]) & (current_depth[overlap_idx] > 0.0)
        prev_valid &= np.all(np.isfinite(previous_points_global[overlap_idx]), axis=-1)
        cur_valid &= np.all(np.isfinite(current_points_global[overlap_idx]), axis=-1)

        any_valid = prev_valid | cur_valid
        if not np.any(any_valid):
            continue

        blended_points = np.zeros_like(previous_points_global[overlap_idx], dtype=np.float32)
        both_valid = prev_valid & cur_valid
        prev_only = prev_valid & ~cur_valid
        cur_only = cur_valid & ~prev_valid
        blended_points[both_valid] = (
            (1.0 - alpha) * previous_points_global[overlap_idx][both_valid]
            + alpha * current_points_global[overlap_idx][both_valid]
        )
        blended_points[prev_only] = previous_points_global[overlap_idx][prev_only]
        blended_points[cur_only] = current_points_global[overlap_idx][cur_only]

        blended_depth, mean_projection_error = _point_map_to_depth_map(
            point_map_world=blended_points,
            intrinsics=blended_intrinsics,
            extrinsic_w2c=blended_extrinsics,
            valid_mask=any_valid,
        )
        encoded_valid = blended_depth > 0.0
        if stored_depths[global_idx] is not None:
            blended_depth = np.where(encoded_valid, blended_depth, stored_depths[global_idx])
        stored_depths[global_idx] = blended_depth.astype(np.float32, copy=False)
        stored_intrinsics[global_idx] = blended_intrinsics
        stored_extrinsics[global_idx] = blended_extrinsics

        if stored_confs[global_idx] is not None:
            blended_conf = np.asarray(stored_confs[global_idx], dtype=np.float32).copy()
            blended_conf[both_valid] = (
                (1.0 - alpha) * previous_conf[overlap_idx][both_valid]
                + alpha * current_conf[overlap_idx][both_valid]
            )
            blended_conf[cur_only] = current_conf[overlap_idx][cur_only]
            blended_conf[prev_only] = previous_conf[overlap_idx][prev_only]
            stored_confs[global_idx] = blended_conf.astype(np.float32, copy=False)

        encoded_pixel_counts.append(int(np.count_nonzero(encoded_valid)))
        projection_errors.append(mean_projection_error)

    if encoded_pixel_counts:
        alpha_last = _streaming_overlap_blend_alpha(overlap_count - 1, overlap_count)
        print(
            "Blended streaming overlap in point space with blended camera basis: "
            f"frames {chunk_start}..{chunk_start + overlap_count - 1}, "
            f"alpha={_streaming_overlap_blend_alpha(0, overlap_count):.2f}..{alpha_last:.2f}, "
            f"encoded_pixels_avg={float(np.mean(encoded_pixel_counts)):.0f}, "
            f"ray_projection_error_mean={float(np.mean(projection_errors)):.6f}"
        )


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
    fixed_camera: bool,
    fixed_camera_fov_degrees: float,
    global_guide: bool,
) -> dict[str, object]:
    chunk_indices = _build_streaming_chunk_indices(len(images_for_da3), chunk_size, overlap)

    print(
        "DA3 streaming settings: "
        f"chunk_size={chunk_size}, overlap={overlap}, chunks={len(chunk_indices)}, "
        f"global_guide={global_guide}, "
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
    current_transform = _identity_sim3()
    guide_context: dict[str, object] | None = None

    if global_guide and len(chunk_indices) > 1:
        guide_indices = _build_streaming_guide_indices(len(images_for_da3), chunk_size)
        guide_step = (
            0.0
            if len(guide_indices) <= 1
            else float(guide_indices[-1] - guide_indices[0]) / float(len(guide_indices) - 1)
        )
        print(
            "Streaming global guide pass: "
            f"{len(guide_indices)} frames over {len(images_for_da3)} selected frames "
            f"(approx_step={guide_step:.2f})."
        )
        guide_predictions = _run_da3_inference(
            model=model,
            image_paths=[images_for_da3[idx] for idx in guide_indices],
            process_res=process_res,
            process_res_method=process_res_method,
            infer_gs=False,
            use_ray_pose=use_ray_pose,
            ref_view_strategy=ref_view_strategy,
            fixed_camera=fixed_camera,
            fixed_camera_fov_degrees=fixed_camera_fov_degrees,
        )
        if guide_predictions.conf is None or guide_predictions.extrinsics is None or guide_predictions.intrinsics is None:
            raise RuntimeError("DA3 streaming global guide requires confidence, intrinsics, and extrinsics outputs.")

        guide_npz_path = _save_streaming_guide_npz(
            scene_root=scene_root,
            guide_indices=guide_indices,
            prediction=guide_predictions,
        )
        guide_context = {
            "indices": guide_indices,
            "depth": np.asarray(guide_predictions.depth, dtype=np.float32),
            "conf": np.asarray(guide_predictions.conf, dtype=np.float32),
            "extrinsics": np.asarray(guide_predictions.extrinsics, dtype=np.float32),
            "intrinsics": np.asarray(guide_predictions.intrinsics, dtype=np.float32),
            "npz_path": guide_npz_path,
        }
        print("Streaming global guide pass: guide frames will anchor matching dense chunks after DA3 runs.")
        print(f"Streaming global guide saved to {guide_npz_path}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    elif global_guide:
        print("Streaming global guide requested but skipped because the selected sequence fits in one chunk.")

    for chunk_index, (chunk_start, chunk_end) in enumerate(chunk_indices):
        chunk_image_paths = images_for_da3[chunk_start:chunk_end]
        print(
            f"Streaming chunk {chunk_index + 1}/{len(chunk_indices)}: "
            f"frames {chunk_start}..{chunk_end - 1} ({len(chunk_image_paths)} images)"
        )
        predictions = _run_da3_inference(
            model=model,
            image_paths=chunk_image_paths,
            process_res=process_res,
            process_res_method=process_res_method,
            infer_gs=False,
            use_ray_pose=use_ray_pose,
            ref_view_strategy=ref_view_strategy,
            fixed_camera=fixed_camera,
            fixed_camera_fov_degrees=fixed_camera_fov_degrees,
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

        guide_chunk_transform: tuple[float, np.ndarray, np.ndarray] | None = None
        if guide_context is not None and not fixed_camera:
            guide_chunk_transform, guide_anchor_frames = _estimate_chunk_transform_to_guide(
                chunk_start=chunk_start,
                chunk_depth=chunk_depth,
                chunk_conf=chunk_conf,
                chunk_intrinsics=chunk_intrinsics,
                chunk_extrinsics=chunk_extrinsics,
                guide_indices=guide_context["indices"],  # type: ignore[arg-type]
                guide_depth=np.asarray(guide_context["depth"], dtype=np.float32),
                guide_conf=np.asarray(guide_context["conf"], dtype=np.float32),
                guide_intrinsics=np.asarray(guide_context["intrinsics"], dtype=np.float32),
                guide_extrinsics=np.asarray(guide_context["extrinsics"], dtype=np.float32),
            )
            if guide_chunk_transform is not None:
                guide_s, guide_R, guide_t = guide_chunk_transform
                guide_rot_angle = float(
                    np.degrees(
                        np.arccos(
                            np.clip((np.trace(guide_R) - 1.0) / 2.0, -1.0, 1.0)
                        )
                    )
                )
                print(
                    "Streaming global guide anchor: "
                    f"chunk={chunk_index + 1}, source_truth_frames={guide_anchor_frames}, "
                    f"sim3_scale={guide_s:.6f}, rotation_deg={guide_rot_angle:.3f}, "
                    f"translation_norm={float(np.linalg.norm(guide_t)):.6f}"
                )
            else:
                print(
                    "Streaming global guide anchor: "
                    f"chunk={chunk_index + 1}, no direct guide frame in this chunk."
                )
        elif guide_context is not None:
            guide_scale, guide_match_frames, guide_match_points = _estimate_chunk_depth_scale_to_guide(
                chunk_start=chunk_start,
                chunk_depth=chunk_depth,
                guide_indices=guide_context["indices"],  # type: ignore[arg-type]
                guide_depth=np.asarray(guide_context["depth"], dtype=np.float32),
            )
            if guide_scale is not None:
                chunk_depth = (chunk_depth * float(guide_scale)).astype(np.float32, copy=False)
                print(
                    "Streaming fixed-camera guide depth scale: "
                    f"chunk={chunk_index + 1}, scale={guide_scale:.6f}, "
                    f"anchor_frames={guide_match_frames}, anchor_pixels={guide_match_points:,}"
                )

        if stored_skies is None and chunk_sky is not None:
            stored_skies = [None] * len(images_for_da3)

        if chunk_index > 0:
            if previous_predictions is None:
                raise RuntimeError("Missing previous chunk state during DA3 streaming alignment.")

            conf_prev = previous_predictions["conf"][-overlap:]
            conf_cur = chunk_conf[:overlap]
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

            previous_transform = current_transform
            if guide_context is not None and fixed_camera:
                pair_transform = _identity_sim3()
                current_transform = _identity_sim3()
                print("Streaming fixed-camera guide active: keeping chunk camera basis fixed.")
            elif guide_context is not None and guide_chunk_transform is not None:
                pair_transform = guide_chunk_transform
                current_transform = guide_chunk_transform
                print(
                    "Streaming global guide active: using direct guide-frame anchor for this "
                    "whole chunk instead of pairwise overlap Sim(3)."
                )
            else:
                if guide_context is not None:
                    print(
                        "Streaming global guide active but no guide frame anchors this chunk; "
                        "falling back to pairwise overlap Sim(3)."
                    )
                try:
                    conf_threshold = min(float(np.median(conf_prev)), float(np.median(conf_cur))) * 0.1
                    pair_transform = _weighted_align_point_maps_da3_dense(
                        point_map_target=point_map_prev,
                        conf_target=conf_prev,
                        point_map_source=point_map_cur,
                        conf_source=conf_cur,
                        conf_threshold=conf_threshold,
                        camera_centers_target=_camera_centers_from_extrinsics(
                            previous_predictions["extrinsics"][-overlap:]
                        ),
                        camera_centers_source=_camera_centers_from_extrinsics(chunk_extrinsics[:overlap]),
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
            _blend_streaming_overlap_point_maps(
                stored_depths=stored_depths,
                stored_confs=stored_confs,
                stored_intrinsics=stored_intrinsics,
                stored_extrinsics=stored_extrinsics,
                chunk_start=chunk_start,
                previous_depth=previous_predictions["depth"][-overlap:],
                previous_conf=conf_prev,
                previous_point_map=point_map_prev,
                previous_transform=previous_transform,
                current_depth=chunk_depth[:overlap],
                current_conf=conf_cur,
                current_intrinsics=chunk_intrinsics[:overlap],
                current_extrinsics=chunk_extrinsics[:overlap],
                current_point_map=point_map_cur,
                current_transform=current_transform,
            )
        elif guide_context is not None and guide_chunk_transform is not None:
            current_transform = guide_chunk_transform
            chunk_sim3.append(guide_chunk_transform)
            print("Streaming global guide active: anchored first chunk to guide-frame source of truth.")

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

    if fixed_camera:
        identity_w2c = np.eye(4, dtype=np.float32)[:3, :4]
        final_extrinsics = [identity_w2c.copy() for value in stored_extrinsics if value is not None]
    else:
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
        "guide_npz_path": (None if guide_context is None else guide_context.get("npz_path")),
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
        default=1,
        help=(
            "Frame stride for input-video frame selection. In streaming mode, this is applied exactly: "
            "for example, 6 uses every sixth extracted video frame."
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
        "--streaming_global_guide",
        action="store_true",
        help=(
            "In streaming mode, first run a sparse whole-video DA3 guide pass using chunk_size frames, "
            "then anchor each dense chunk to exact matching guide frames when available."
        ),
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
        "--fixed_camera",
        action="store_true",
        help="Tell DA3 the input sequence uses a fixed camera by supplying identity extrinsics and estimated intrinsics.",
    )
    p.add_argument(
        "--fixed_camera_fov_degrees",
        type=float,
        default=60.0,
        help="Horizontal FOV used to estimate pinhole intrinsics when --fixed_camera is enabled.",
    )
    p.add_argument(
        "--ref_view_strategy",
        type=str,
        default="first",
        help=(
            "DA3 multi-view reference-view strategy. "
            "Default is 'first' for stable chunked divstream exports."
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
        type=float,
        default=30.0,
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
        help="Also materialize the Stage 1 pre-ICP filter cache and optional before_non_rigid_icp.ply during Stage 0.",
    )
    p.add_argument(
        "--prepare_skip_before_non_rigid",
        action="store_true",
        help="Materialize the Stage 1 filter cache/config but skip before_non_rigid_icp.ply.",
    )
    p.add_argument(
        "--skip_frame_materialization",
        action="store_true",
        help="Run DA3 on the selected frame paths directly instead of copying them into frames_subsampled.",
    )
    p.add_argument(
        "--prepare_skip_debug_masks",
        action="store_true",
        help="Skip Stage 1 debug mask PNGs when preparing filter caches.",
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
    runtime_export_fps = float(args.runtime_export_fps)

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
                f"N={len(selected_images)} (stride {stride}, chunk_size={streaming_chunk_size}, "
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

    source_index_by_path = {os.path.abspath(path): idx for idx, path in enumerate(images)}
    selected_source_indices = [
        source_index_by_path.get(os.path.abspath(path), idx)
        for idx, path in enumerate(selected_images)
    ]

    if args.skip_frame_materialization:
        used_frames_dir = frames_dir
        images_for_da3 = selected_images
    else:
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
                "model_name": args.model_name,
                "process_res": int(args.process_res),
                "process_res_method": args.process_res_method,
                "export_gs_video": bool(args.export_gs_video),
                "use_ray_pose": bool(args.use_ray_pose),
                "fixed_camera": bool(args.fixed_camera),
                "fixed_camera_fov_degrees": float(args.fixed_camera_fov_degrees),
                "ref_view_strategy": ref_view_strategy,
                "source": ("input_video" if args.input_video is not None else "frames_dir"),
                "max_frames": max_frames_meta,
                "max_stride": max_stride_meta,
                "actual_stride": stride,
                "selected_frame_indices": selected_source_indices,
                "selected_frame_paths": [os.path.abspath(path) for path in selected_images],
                "streaming_enabled": streaming_enabled,
                "streaming_chunk_size": (streaming_chunk_size if streaming_enabled else None),
                "streaming_overlap": (streaming_overlap if streaming_enabled else None),
                "streaming_global_guide": (bool(args.streaming_global_guide) if streaming_enabled else False),
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
        f"fixed_camera={args.fixed_camera}, "
        f"fixed_camera_fov_degrees={args.fixed_camera_fov_degrees:g}, "
        f"ref_view_strategy={ref_view_strategy}, "
        f"export_gs_video={args.export_gs_video}, "
        f"streaming={streaming_enabled}, "
        f"streaming_global_guide={bool(args.streaming_global_guide) if streaming_enabled else False}"
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
            fixed_camera=bool(args.fixed_camera),
            fixed_camera_fov_degrees=float(args.fixed_camera_fov_degrees),
            global_guide=bool(args.streaming_global_guide),
        )
        npz_path = str(streaming_result["npz_path"])
    else:
        _run_da3_inference(
            model=model,
            image_paths=images_for_da3,
            export_dir=scene_root,
            export_format=export_format,
            process_res=args.process_res,
            process_res_method=args.process_res_method,
            infer_gs=args.export_gs_video,
            use_ray_pose=args.use_ray_pose,
            ref_view_strategy=ref_view_strategy,
            fixed_camera=bool(args.fixed_camera),
            fixed_camera_fov_degrees=float(args.fixed_camera_fov_degrees),
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
        if streaming_result.get("guide_npz_path"):
            summary_lines.append(f"- DA3 streaming guide: {streaming_result['guide_npz_path']}")
        summary_lines.append(f"- Frames processed: {len(images_for_da3)}")
    else:
        summary_lines.append("- DA3 streaming: disabled")
    if args.export_gs_video:
        summary_lines.append(f"- GS video: {os.path.join(scene_root, 'gs_video')}")
    else:
        summary_lines.append("- GS video: skipped")
    prep_out_path = None
    before_non_rigid_path = None
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
            write_before_non_rigid=not bool(args.prepare_skip_before_non_rigid),
            write_debug_masks=not bool(args.prepare_skip_debug_masks),
        )
        summary_lines.append(f"- Stage 1 prep cache: {os.path.join(scene_root, 'exports', 'ply')}")
        if before_non_rigid_path is None:
            summary_lines.append("- Pre-ICP merge: skipped")
        else:
            summary_lines.append(f"- Pre-ICP merge: {before_non_rigid_path}")
        summary_lines.append(f"- Prepared run dir: {prep_out_path}")
    runtime_export_output = _export_stage0_runtime_format(
        scene_root=scene_root,
        runtime_export_format=runtime_export_format,
        fps=runtime_export_fps,
        overwrite=bool(args.overwrite),
        prep_run=prep_out_path,
    )
    if runtime_export_output is None:
        summary_lines.append("- Stage 0 runtime export: skipped")
    else:
        summary_lines.append(
            f"- Stage 0 runtime export ({_runtime_export_label(runtime_export_format)}): {runtime_export_output}"
        )
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()

