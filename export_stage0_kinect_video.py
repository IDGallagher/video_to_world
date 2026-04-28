"""
Export Stage 0 DA3 outputs into a KinectStreamer-style packed RGBD video.

Layout per frame:
    [ metadata barcode | color image | encoded depth image ]

Metadata barcode floats:
    px, py, pz, rx, ry, rz, near, far, fx, fy, cx, cy, hash

Depth encoding is an exact lossless 16-bit inverse-depth codebook packed into
RGB8 as:
    - G: high 8 bits (coarse inverse depth)
    - R/B: low 8 bits as a 16x16 serpentine lattice

This keeps the signal luma-major enough to degrade more gracefully than a pure
hue wheel if the packed frames are later transcoded with a lossy codec.

The packed PNG frames are then transcoded to HAP Q for Unreal playback.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


logger = logging.getLogger(__name__)

_METADATA_FLOAT_ORDER = ("px", "py", "pz", "rx", "ry", "rz", "near", "far", "fx", "fy", "cx", "cy", "hash")
_METADATA_WIDTH_DEFAULT = 64
_LOW_LATTICE_LEVELS = np.asarray([98 + 4 * i for i in range(16)], dtype=np.uint8)
_DEPTH_INVALID_RGB = np.asarray([0, 0, 0], dtype=np.uint8)
_OUTPUT_VIDEO_FILENAME = "kinect_rgbd_hapq.mov"
_SEQUENCE_INFO_FILENAME = "sequence_info.json"
_PACKED_SEQUENCE_ENCODING_RGB16 = "raw_rgb_depth16_inverse"
_PACKED_SEQUENCE_ENCODING_GRAY8 = "raw_rgb_depth8_inverse_gray"


def _find_latest_valid_pixel_indices_path(scene_root: str) -> str | None:
    ply_root = Path(scene_root) / "exports" / "ply"
    if not ply_root.is_dir():
        return None

    candidates = [path for path in ply_root.glob("*/valid_pixel_indices.npz") if path.is_file()]
    if not candidates:
        return None

    candidates.sort(key=lambda path: (path.stat().st_mtime_ns, path.parent.name), reverse=True)
    chosen = candidates[0]
    if len(candidates) > 1:
        logger.info(
            "Found %d Stage 0 valid-pixel caches under %s; using newest %s",
            len(candidates),
            ply_root,
            chosen,
        )
    return str(chosen)


def _apply_valid_pixel_indices_mask(
    *,
    images: np.ndarray,
    depth: np.ndarray,
    valid_indices_path: str,
) -> tuple[np.ndarray, np.ndarray]:
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"Expected images to be (N,H,W,3), got {images.shape}")
    if depth.ndim != 3:
        raise ValueError(f"Expected depth to be (N,H,W), got {depth.shape}")
    if images.shape[:3] != depth.shape:
        raise ValueError(
            "Image/depth shapes disagree: "
            f"images={images.shape} depth={depth.shape}",
        )

    num_frames, height, width = depth.shape
    num_pixels = height * width

    masked_images = np.zeros_like(images)
    masked_depth = np.zeros_like(depth)
    src_images_flat = images.reshape(num_frames, num_pixels, 3)
    src_depth_flat = depth.reshape(num_frames, num_pixels)
    masked_images_flat = masked_images.reshape(num_frames, num_pixels, 3)
    masked_depth_flat = masked_depth.reshape(num_frames, num_pixels)

    kept_total = 0
    with np.load(valid_indices_path) as valid_indices_npz:
        for frame_idx in range(num_frames):
            key = f"frame_{frame_idx:05d}"
            if key not in valid_indices_npz:
                raise KeyError(f"Missing key '{key}' in valid-pixel cache '{valid_indices_path}'.")
            flat_indices = np.asarray(valid_indices_npz[key], dtype=np.int64).reshape(-1)
            if flat_indices.size == 0:
                continue
            if np.any(flat_indices < 0) or np.any(flat_indices >= num_pixels):
                raise ValueError(
                    f"Out-of-range flat indices in '{valid_indices_path}' for {key} "
                    f"(expected [0, {num_pixels})).",
                )
            masked_images_flat[frame_idx, flat_indices] = src_images_flat[frame_idx, flat_indices]
            masked_depth_flat[frame_idx, flat_indices] = src_depth_flat[frame_idx, flat_indices]
            kept_total += int(np.unique(flat_indices).size)

    logger.info(
        "Applied Stage 0 valid-pixel mask from %s; kept %d / %d pixels across %d frames",
        valid_indices_path,
        kept_total,
        num_frames * num_pixels,
        num_frames,
    )
    return masked_images, masked_depth


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
        f"the active Python interpreter at '{python_bin_dir}'.",
    )


def _has_hap_encoder(ffmpeg_path: str) -> bool:
    try:
        result = subprocess.run(
            [ffmpeg_path, "-hide_banner", "-encoders"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return " hap " in result.stdout


def _resolve_hap_ffmpeg() -> str:
    ffmpeg = _resolve_executable("ffmpeg")
    if _has_hap_encoder(ffmpeg):
        return ffmpeg

    try:
        ffmpeg_exe = _resolve_executable("ffmpeg.exe")
    except FileNotFoundError as exc:
        raise RuntimeError(
            "No HAP-capable ffmpeg was found. Install an ffmpeg build with the Hap encoder "
            "or expose ffmpeg.exe on PATH when running from WSL."
        ) from exc

    if _has_hap_encoder(ffmpeg_exe):
        return ffmpeg_exe

    raise RuntimeError(
        "Neither the local ffmpeg nor ffmpeg.exe exposes the Hap encoder. "
        "Install a full ffmpeg build with Hap support.",
    )


def _normalize_path_for_executable(path: str, executable: str) -> str:
    if os.name != "nt" and executable.lower().endswith(".exe"):
        converted = subprocess.run(
            ["wslpath", "-w", path],
            check=True,
            capture_output=True,
            text=True,
        )
        return converted.stdout.strip()
    return path


def _rotation_matrix_to_wxyz(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to a quaternion in (w, x, y, z) order."""
    m00, m01, m02 = float(rotation[0, 0]), float(rotation[0, 1]), float(rotation[0, 2])
    m10, m11, m12 = float(rotation[1, 0]), float(rotation[1, 1]), float(rotation[1, 2])
    m20, m21, m22 = float(rotation[2, 0]), float(rotation[2, 1]), float(rotation[2, 2])
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    return w, x, y, z


def _encode_relative_metadata(
    relative_c2w: np.ndarray,
    intrinsics: np.ndarray,
    *,
    near_m: float,
    far_m: float,
) -> np.ndarray:
    """Encode packed-video metadata floats as float32."""
    w, x, y, z = _rotation_matrix_to_wxyz(relative_c2w[:3, :3])
    if w < 0.0:
        x, y, z = -x, -y, -z

    translation = relative_c2w[:3, 3]
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    values = np.asarray(
        [
            translation[0],
            translation[1],
            translation[2],
            x,
            y,
            z,
            near_m,
            far_m,
            fx,
            fy,
            cx,
            cy,
            0.0,
        ],
        dtype=np.float32,
    )
    values[-1] = np.float32(values[:-1].sum(dtype=np.float32))
    return values


def _compute_centered_padding(size: int, multiple: int) -> tuple[int, int]:
    padded_size = ((size + multiple - 1) // multiple) * multiple
    total_pad = padded_size - size
    pad_before = total_pad // 2
    pad_after = total_pad - pad_before
    return pad_before, pad_after


def _pad_rgb_image_for_hap(image: np.ndarray) -> tuple[np.ndarray, int, int]:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must be (H,W,3), got {image.shape}")

    height, width = image.shape[:2]
    pad_left, pad_right = _compute_centered_padding(width, 2)
    pad_top, pad_bottom = _compute_centered_padding(height, 4)
    if pad_left == 0 and pad_right == 0 and pad_top == 0 and pad_bottom == 0:
        return image, 0, 0

    padded = np.pad(
        image,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    return padded, pad_left, pad_top


def _encode_metadata_barcode(metadata: np.ndarray, *, height: int, metadata_width: int) -> np.ndarray:
    """Replicate KinectStreamer's stretched barcode strip."""
    if metadata.ndim != 1 or metadata.shape[0] < 1:
        raise ValueError(f"metadata must be a non-empty 1D float array, got {metadata.shape}")

    bits = metadata.view(np.uint32)
    value_count = metadata.shape[0]
    value_indices = (np.arange(metadata_width, dtype=np.int32) * value_count) // metadata_width
    bit_indices = (np.arange(height, dtype=np.int32) * 32) // max(height, 1)
    bit_plane = ((bits[value_indices][None, :] >> bit_indices[:, None]) & 1).astype(np.uint8) * 255
    barcode = np.repeat(bit_plane[:, :, None], 3, axis=2)
    return barcode


def _depth_to_inverse_norm(
    depth_map: np.ndarray,
    *,
    near_m: float,
    far_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(depth_map) & (depth_map > 0.0)
    norm = np.zeros(depth_map.shape, dtype=np.float32)
    if not np.any(valid):
        return norm, valid

    inv_depth = np.zeros(depth_map.shape, dtype=np.float32)
    inv_depth[valid] = 1.0 / depth_map[valid]
    inv_near = 1.0 / float(near_m)
    inv_far = 1.0 / float(far_m)
    denom = inv_near - inv_far
    if abs(denom) < 1e-12:
        norm[valid] = 1.0
    else:
        norm[valid] = (inv_depth[valid] - inv_far) / denom
        norm[valid] = np.clip(norm[valid], 0.0, 1.0)
    return norm, valid


def _encode_depth_rgb16(
    depth_map: np.ndarray,
    *,
    near_m: float,
    far_m: float,
) -> np.ndarray:
    """Encode metric depth into an exact 16-bit RGB codebook."""
    norm, valid = _depth_to_inverse_norm(depth_map, near_m=near_m, far_m=far_m)
    encoded = np.zeros(depth_map.shape + (3,), dtype=np.uint8)
    if not np.any(valid):
        return encoded

    code = np.rint(norm[valid] * 65535.0).astype(np.uint16)
    hi = (code >> 8).astype(np.uint8)
    lo = (code & 255).astype(np.uint8)

    row = (lo >> 4).astype(np.uint8)
    col = (lo & 15).astype(np.uint8)
    serp_col = np.where((row & 1) == 1, 15 - col, col).astype(np.uint8)

    encoded_valid = np.empty((code.shape[0], 3), dtype=np.uint8)
    encoded_valid[:, 0] = _LOW_LATTICE_LEVELS[serp_col]
    encoded_valid[:, 1] = hi
    encoded_valid[:, 2] = _LOW_LATTICE_LEVELS[row]
    encoded[valid] = encoded_valid
    encoded[~valid] = _DEPTH_INVALID_RGB
    return encoded


def _encode_depth_gray8(
    depth_map: np.ndarray,
    *,
    near_m: float,
    far_m: float,
) -> np.ndarray:
    """Encode metric depth into an 8-bit inverse-depth grayscale codebook."""
    norm, valid = _depth_to_inverse_norm(depth_map, near_m=near_m, far_m=far_m)
    encoded = np.zeros(depth_map.shape + (3,), dtype=np.uint8)
    if not np.any(valid):
        return encoded

    # Reserve code 0 for invalid pixels, map valid samples onto [1, 255].
    code = np.rint(norm[valid] * 254.0).astype(np.uint8) + 1
    encoded_valid = np.repeat(code[:, None], 3, axis=1)
    encoded[valid] = encoded_valid
    encoded[~valid] = _DEPTH_INVALID_RGB
    return encoded


def _decode_depth_rgb16_lossless(
    encoded_rgb: np.ndarray,
    *,
    near_m: float,
    far_m: float,
) -> np.ndarray:
    """Exact inverse of `_encode_depth_rgb16` for verification/tests."""
    if encoded_rgb.ndim != 3 or encoded_rgb.shape[-1] != 3:
        raise ValueError(f"encoded_rgb must be (H,W,3), got {encoded_rgb.shape}")

    decoded = np.zeros(encoded_rgb.shape[:2], dtype=np.float32)
    valid = np.any(encoded_rgb != 0, axis=-1)
    if not np.any(valid):
        return decoded

    r = encoded_rgb[..., 0].astype(np.int32)
    g = encoded_rgb[..., 1].astype(np.uint16)
    b = encoded_rgb[..., 2].astype(np.int32)

    r_idx = np.clip(np.rint((r - int(_LOW_LATTICE_LEVELS[0])) / 4.0), 0, 15).astype(np.int32)
    b_idx = np.clip(np.rint((b - int(_LOW_LATTICE_LEVELS[0])) / 4.0), 0, 15).astype(np.int32)
    lo = np.where((b_idx & 1) == 1, b_idx * 16 + (15 - r_idx), b_idx * 16 + r_idx).astype(np.uint16)
    code = (g << 8) | lo

    inv_near = 1.0 / float(near_m)
    inv_far = 1.0 / float(far_m)
    norm = code.astype(np.float32) / 65535.0
    inv_depth = inv_far + norm * (inv_near - inv_far)
    decoded_valid = np.divide(1.0, inv_depth, out=np.zeros_like(inv_depth), where=inv_depth > 0.0)
    decoded[valid] = decoded_valid[valid]
    return decoded


def _load_stage0_results(scene_root: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    npz_path = os.path.join(scene_root, "exports", "npz", "results.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Stage 0 results not found: {npz_path}")
    predictions = np.load(npz_path)
    try:
        images = predictions["image"]
        depth = predictions["depth"]
        extrinsics = predictions["extrinsics"]
        intrinsics = predictions["intrinsics"]
    finally:
        if hasattr(predictions, "close"):
            predictions.close()
    valid_indices_path = _find_latest_valid_pixel_indices_path(scene_root)
    if valid_indices_path is not None:
        images, depth = _apply_valid_pixel_indices_mask(
            images=images,
            depth=depth,
            valid_indices_path=valid_indices_path,
        )
    else:
        logger.info("No Stage 0 valid-pixel mask found under %s; exporting raw Stage 0 pixels.", scene_root)
    return images, depth, extrinsics, intrinsics


def _relative_c2w_from_extrinsics(extrinsics: np.ndarray) -> np.ndarray:
    if extrinsics.ndim != 3 or extrinsics.shape[1:] != (3, 4):
        raise ValueError(f"extrinsics must be (N,3,4), got {extrinsics.shape}")
    n_frames = extrinsics.shape[0]
    extrinsics_h = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], n_frames, axis=0)
    extrinsics_h[:, :3, :] = extrinsics
    c2w = np.linalg.inv(extrinsics_h)
    c2w0_inv = np.linalg.inv(c2w[0])
    return np.matmul(c2w0_inv[None, :, :], c2w)


def _assemble_kinect_frame(
    *,
    color_rgb: np.ndarray,
    depth_rgb: np.ndarray,
    metadata: np.ndarray,
    metadata_width: int,
) -> np.ndarray:
    height, width = color_rgb.shape[:2]
    barcode = _encode_metadata_barcode(metadata, height=height, metadata_width=metadata_width)
    frame = np.concatenate([barcode, color_rgb, depth_rgb], axis=1)
    return frame


def _build_video_from_frames(
    *,
    frames_dir: str,
    output_video_path: str,
    fps: int,
) -> None:
    ffmpeg = _resolve_hap_ffmpeg()
    input_pattern = _normalize_path_for_executable(os.path.join(frames_dir, "%06d.png"), ffmpeg)
    output_video_path = _normalize_path_for_executable(output_video_path, ffmpeg)
    cmd = [
        ffmpeg,
        "-y",
        "-framerate",
        str(int(fps)),
        "-i",
        input_pattern,
        "-an",
        "-c:v",
        "hap",
        "-format",
        "hap_q",
        "-compressor",
        "snappy",
        "-pix_fmt",
        "rgba",
        output_video_path,
    ]
    subprocess.run(cmd, check=True)


def _write_sequence_info(
    *,
    output_dir: str,
    frames_dir: str,
    fps: int,
    encoding: str,
    metadata_width: int,
    num_frames: int,
    packed_width: int,
    packed_height: int,
) -> str:
    sequence_info_path = os.path.join(output_dir, _SEQUENCE_INFO_FILENAME)
    payload = {
        "format": "depth_image_volume_packed_sequence_v1",
        "encoding": str(encoding),
        "frame_rate": int(fps),
        "metadata_width": int(metadata_width),
        "num_frames": int(num_frames),
        "packed_width": int(packed_width),
        "packed_height": int(packed_height),
        "frame_pattern": "%06d.png",
        "frames_dir": os.path.basename(frames_dir),
    }
    with open(sequence_info_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return sequence_info_path


def _export_packed_kinect_frames(
    *,
    scene_root: str,
    output_dir: str,
    fps: int = 30,
    metadata_width: int = _METADATA_WIDTH_DEFAULT,
    depth_encoding: str = _PACKED_SEQUENCE_ENCODING_RGB16,
    overwrite: bool = False,
) -> dict[str, object]:
    scene_root = os.path.abspath(scene_root)
    output_dir = os.path.abspath(output_dir)
    frames_dir = os.path.join(output_dir, "frames")

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(frames_dir, exist_ok=True)

    images, depth_maps, extrinsics, intrinsics = _load_stage0_results(scene_root)
    if images.shape[0] == 0:
        raise ValueError("results.npz contains zero frames.")
    if images.shape[0] != depth_maps.shape[0] or depth_maps.shape[0] != extrinsics.shape[0]:
        raise ValueError(
            "Stage 0 arrays disagree on frame count: "
            f"image={images.shape[0]} depth={depth_maps.shape[0]} extrinsics={extrinsics.shape[0]}",
        )

    relative_c2w = _relative_c2w_from_extrinsics(extrinsics)
    valid_depth = np.isfinite(depth_maps) & (depth_maps > 0.0)
    if not np.any(valid_depth):
        raise ValueError("No valid positive depths found in results.npz.")

    near_m = float(depth_maps[valid_depth].min())
    far_m = float(depth_maps[valid_depth].max())
    if far_m <= near_m:
        far_m = near_m + 1e-3

    packed_width = 0
    packed_height = 0
    for frame_idx in range(images.shape[0]):
        color_rgb = images[frame_idx]
        if color_rgb.dtype != np.uint8:
            color_rgb = np.clip(color_rgb, 0, 255).astype(np.uint8)
        if depth_encoding == _PACKED_SEQUENCE_ENCODING_RGB16:
            depth_rgb = _encode_depth_rgb16(depth_maps[frame_idx], near_m=near_m, far_m=far_m)
        elif depth_encoding == _PACKED_SEQUENCE_ENCODING_GRAY8:
            depth_rgb = _encode_depth_gray8(depth_maps[frame_idx], near_m=near_m, far_m=far_m)
        else:
            raise ValueError(f"Unknown packed depth encoding: {depth_encoding}")
        color_rgb, pad_left, pad_top = _pad_rgb_image_for_hap(color_rgb)
        depth_rgb, _, _ = _pad_rgb_image_for_hap(depth_rgb)
        padded_intrinsics = np.asarray(intrinsics[frame_idx], dtype=np.float32).copy()
        padded_intrinsics[0, 2] += float(pad_left)
        padded_intrinsics[1, 2] += float(pad_top)
        metadata_values = _encode_relative_metadata(
            relative_c2w[frame_idx],
            padded_intrinsics,
            near_m=near_m,
            far_m=far_m,
        )
        packed_frame = _assemble_kinect_frame(
            color_rgb=color_rgb,
            depth_rgb=depth_rgb,
            metadata=metadata_values,
            metadata_width=metadata_width,
        )
        packed_height, packed_width = packed_frame.shape[:2]
        frame_path = os.path.join(frames_dir, f"{frame_idx:06d}.png")
        Image.fromarray(packed_frame, mode="RGB").save(frame_path)

    sequence_info_path = _write_sequence_info(
        output_dir=output_dir,
        frames_dir=frames_dir,
        fps=fps,
        encoding=depth_encoding,
        metadata_width=metadata_width,
        num_frames=images.shape[0],
        packed_width=packed_width,
        packed_height=packed_height,
    )

    return {
        "output_dir": output_dir,
        "frames_dir": frames_dir,
        "sequence_info_path": sequence_info_path,
        "num_frames": int(images.shape[0]),
        "packed_width": int(packed_width),
        "packed_height": int(packed_height),
        "fps": int(fps),
    }


def export_stage0_kinect_image_sequence(
    *,
    scene_root: str,
    output_dir: str | None = None,
    fps: int = 30,
    metadata_width: int = _METADATA_WIDTH_DEFAULT,
    overwrite: bool = False,
) -> str:
    if output_dir is None or str(output_dir).strip() == "":
        output_dir = os.path.join(os.path.abspath(scene_root), "exports", "kinect_rgbd_sequence")
    result = _export_packed_kinect_frames(
        scene_root=scene_root,
        output_dir=output_dir,
        fps=fps,
        metadata_width=metadata_width,
        depth_encoding=_PACKED_SEQUENCE_ENCODING_RGB16,
        overwrite=overwrite,
    )
    logger.info("Exported Kinect-layout Stage 0 image sequence to %s", result["output_dir"])
    return str(result["output_dir"])


def export_stage0_kinect_image_sequence_depth8(
    *,
    scene_root: str,
    output_dir: str | None = None,
    fps: int = 30,
    metadata_width: int = _METADATA_WIDTH_DEFAULT,
    overwrite: bool = False,
) -> str:
    if output_dir is None or str(output_dir).strip() == "":
        output_dir = os.path.join(os.path.abspath(scene_root), "exports", "kinect_rgbd_sequence_depth8")
    result = _export_packed_kinect_frames(
        scene_root=scene_root,
        output_dir=output_dir,
        fps=fps,
        metadata_width=metadata_width,
        depth_encoding=_PACKED_SEQUENCE_ENCODING_GRAY8,
        overwrite=overwrite,
    )
    logger.info("Exported Kinect-layout Stage 0 8-bit depth image sequence to %s", result["output_dir"])
    return str(result["output_dir"])


def export_stage0_kinect_video(
    *,
    scene_root: str,
    output_dir: str | None = None,
    fps: int = 30,
    metadata_width: int = _METADATA_WIDTH_DEFAULT,
    overwrite: bool = False,
) -> str:
    scene_root = os.path.abspath(scene_root)
    if output_dir is None or str(output_dir).strip() == "":
        output_dir = os.path.join(scene_root, "exports", "kinect_rgbd_video")
    output_dir = os.path.abspath(output_dir)
    video_path = os.path.join(output_dir, _OUTPUT_VIDEO_FILENAME)

    result = _export_packed_kinect_frames(
        scene_root=scene_root,
        output_dir=output_dir,
        fps=fps,
        metadata_width=metadata_width,
        depth_encoding=_PACKED_SEQUENCE_ENCODING_RGB16,
        overwrite=overwrite,
    )

    _build_video_from_frames(frames_dir=str(result["frames_dir"]), output_video_path=video_path, fps=fps)

    logger.info("Exported Kinect-layout Stage 0 video to %s", output_dir)
    return output_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Stage 0 DA3 outputs as a Kinect-layout HAP Q RGBD video.")
    parser.add_argument("--scene_root", required=True, help="Scene root containing exports/npz/results.npz")
    parser.add_argument("--output_dir", default=None, help="Optional output directory")
    parser.add_argument("--fps", type=int, default=30, help="Video frame rate")
    parser.add_argument("--metadata_width", type=int, default=_METADATA_WIDTH_DEFAULT, help="Barcode strip width")
    parser.add_argument("--overwrite", action="store_true", help="Replace any existing export directory")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    export_stage0_kinect_video(
        scene_root=args.scene_root,
        output_dir=args.output_dir,
        fps=args.fps,
        metadata_width=args.metadata_width,
        overwrite=args.overwrite,
    )
