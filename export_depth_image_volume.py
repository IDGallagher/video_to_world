"""
Export a reconstructed scene as a DepthImageVolume for Unreal Engine.

Takes the canonical point cloud from Stage 1 or 2 and rasterizes each
per-frame point segment back into depth/color/normal images using the
refined camera pose and intrinsics.

Important:
- Stage 1/2 local deformation can move points laterally in camera space, so
  the original pixel indices are no longer exact.
- This exporter therefore reprojects each refined point to its actual image
  location and writes the closest single-layer approximation that fits the
  current DepthImageVolume representation.

Usage:
    python export_depth_image_volume.py \
        --scene_root /path/to/scene \
        --run run_name \
        --output_dir /path/to/output

Or call export_depth_image_volume() programmatically from the Gradio UI.
"""

import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

_DEPTH_U16_INVALID = np.uint16(0)
_DEPTH_U16_VALID_MIN = np.uint16(1)
_DEPTH_U16_VALID_MAX = np.uint16(65535)
_DEPTH_U16_VALID_RANGE = float(int(_DEPTH_U16_VALID_MAX) - int(_DEPTH_U16_VALID_MIN))


def _rasterize_projected_points(
    *,
    pts_cam: torch.Tensor,
    colors: torch.Tensor,
    normals_cam: torch.Tensor,
    K: torch.Tensor,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Rasterize refined camera-space points onto a single-layer image grid.

    We round each projected point to its nearest pixel and keep the front-most
    sample per pixel. This is the closest representation available with the
    current single-depth-per-pixel export format.
    """
    if pts_cam.ndim != 2 or pts_cam.shape[-1] != 3:
        raise ValueError(f"pts_cam must be (N,3), got {tuple(pts_cam.shape)}")

    pts_cam_np = pts_cam.detach().cpu().numpy()
    colors_np = colors.detach().cpu().numpy()
    normals_np = normals_cam.detach().cpu().numpy()
    K_np = K.detach().cpu().numpy()

    Z = pts_cam_np[:, 2]
    finite = np.isfinite(pts_cam_np).all(axis=1) & (Z > 0)
    if not np.any(finite):
        empty_depth = np.zeros((height, width), dtype=np.float32)
        empty_color = np.zeros((height, width, 3), dtype=np.float32)
        empty_normal = np.zeros((height, width, 3), dtype=np.float32)
        return empty_depth, empty_color, empty_normal, {
            "num_input_points": int(pts_cam_np.shape[0]),
            "num_positive_depth": 0,
            "num_in_bounds": 0,
            "num_pixels_written": 0,
            "num_collisions": 0,
            "mean_pixel_shift": float("nan"),
            "median_pixel_shift": float("nan"),
            "p95_pixel_shift": float("nan"),
        }

    pts_cam_np = pts_cam_np[finite]
    colors_np = colors_np[finite]
    normals_np = normals_np[finite]
    Z = Z[finite]

    fx = float(K_np[0, 0])
    fy = float(K_np[1, 1])
    cx = float(K_np[0, 2])
    cy = float(K_np[1, 2])

    u = fx * (pts_cam_np[:, 0] / Z) + cx
    v = fy * (pts_cam_np[:, 1] / Z) + cy
    pixel_shift = np.sqrt((u - np.round(u)) ** 2 + (v - np.round(v)) ** 2)

    u_round = np.rint(u).astype(np.int64)
    v_round = np.rint(v).astype(np.int64)
    in_bounds = (
        np.isfinite(u)
        & np.isfinite(v)
        & (u_round >= 0)
        & (u_round < width)
        & (v_round >= 0)
        & (v_round < height)
    )

    if not np.any(in_bounds):
        empty_depth = np.zeros((height, width), dtype=np.float32)
        empty_color = np.zeros((height, width, 3), dtype=np.float32)
        empty_normal = np.zeros((height, width, 3), dtype=np.float32)
        return empty_depth, empty_color, empty_normal, {
            "num_input_points": int(pts_cam.shape[0]),
            "num_positive_depth": int(finite.sum()),
            "num_in_bounds": 0,
            "num_pixels_written": 0,
            "num_collisions": 0,
            "mean_pixel_shift": float(pixel_shift.mean()) if pixel_shift.size else float("nan"),
            "median_pixel_shift": float(np.median(pixel_shift)) if pixel_shift.size else float("nan"),
            "p95_pixel_shift": float(np.quantile(pixel_shift, 0.95)) if pixel_shift.size else float("nan"),
        }

    u_round = u_round[in_bounds]
    v_round = v_round[in_bounds]
    Z = Z[in_bounds]
    colors_np = colors_np[in_bounds]
    normals_np = normals_np[in_bounds]
    pixel_shift = pixel_shift[in_bounds]

    linear = v_round * width + u_round
    # Primary key: pixel index. Secondary: front-most depth. Tertiary: closest
    # to the quantized pixel center.
    order = np.lexsort((pixel_shift, Z, linear))
    linear_sorted = linear[order]

    keep_sorted = np.ones(linear_sorted.shape[0], dtype=bool)
    if linear_sorted.shape[0] > 1:
        keep_sorted[1:] = linear_sorted[1:] != linear_sorted[:-1]
    keep = order[keep_sorted]

    kept_linear = linear[keep]
    depth_map = np.zeros(height * width, dtype=np.float32)
    color_map = np.zeros((height * width, 3), dtype=np.float32)
    normal_map = np.zeros((height * width, 3), dtype=np.float32)
    depth_map[kept_linear] = Z[keep].astype(np.float32, copy=False)
    color_map[kept_linear] = colors_np[keep].astype(np.float32, copy=False)
    normal_map[kept_linear] = normals_np[keep].astype(np.float32, copy=False)

    return (
        depth_map.reshape(height, width),
        color_map.reshape(height, width, 3),
        normal_map.reshape(height, width, 3),
        {
            "num_input_points": int(pts_cam.shape[0]),
            "num_positive_depth": int(finite.sum()),
            "num_in_bounds": int(in_bounds.sum()),
            "num_pixels_written": int(keep.shape[0]),
            "num_collisions": int(in_bounds.sum() - keep.shape[0]),
            "mean_pixel_shift": float(pixel_shift.mean()),
            "median_pixel_shift": float(np.median(pixel_shift)),
            "p95_pixel_shift": float(np.quantile(pixel_shift, 0.95)),
        },
    )


def _load_refined_c2w_poses(checkpoint_dir: str, device: str) -> list[torch.Tensor]:
    """Load per-frame refined c2w poses from SE3 twist checkpoints."""
    from utils.geometry import se3_exp

    poses = []
    i = 0
    while True:
        path = os.path.join(checkpoint_dir, f"per_frame_global_deform_{i:05d}.pt")
        if not os.path.exists(path):
            path = os.path.join(checkpoint_dir, f"per_frame_global_rigid_{i:05d}.pt")
        if not os.path.exists(path):
            break
        xi = torch.load(path, map_location=device, weights_only=True).detach()
        with torch.no_grad():
            R, t = se3_exp(xi.unsqueeze(0))
            c2w = torch.eye(4, device=device, dtype=torch.float32)
            c2w[:3, :3] = R[0]
            c2w[:3, 3] = t[0]
        poses.append(c2w)
        i += 1

    if not poses:
        raise FileNotFoundError(
            f"No per_frame_global_deform_*.pt or per_frame_global_rigid_*.pt "
            f"files found in {checkpoint_dir}"
        )
    logger.info("Loaded %d refined c2w poses from %s", len(poses), checkpoint_dir)
    return poses


def _encode_depth_map_u16(
    depth_map: np.ndarray,
    *,
    near: float,
    far: float,
) -> np.ndarray:
    """Encode metric depth to uint16 with 0 reserved for invalid pixels.

    Valid depths are mapped linearly onto codes [1, 65535].
    The matching decode formula is:

        depth = near + ((code - 1) / 65534) * (far - near)

    where code == 0 denotes an invalid pixel.
    """
    if depth_map.ndim != 2:
        raise ValueError(f"depth_map must be 2-D, got shape {depth_map.shape}")
    if not np.isfinite(near) or not np.isfinite(far) or far <= near:
        raise ValueError(f"Invalid depth range: near={near}, far={far}")

    valid_mask = depth_map > 0
    encoded = np.zeros(depth_map.shape, dtype=np.uint16)
    if not np.any(valid_mask):
        return encoded

    depth_norm = np.empty(depth_map.shape, dtype=np.float32)
    depth_norm.fill(0.0)
    depth_norm[valid_mask] = (depth_map[valid_mask] - near) / (far - near)
    depth_norm[valid_mask] = np.clip(depth_norm[valid_mask], 0.0, 1.0)

    # Use the full valid code range [1, 65535] while preserving 0 as invalid.
    encoded_valid = np.rint(depth_norm[valid_mask] * _DEPTH_U16_VALID_RANGE + int(_DEPTH_U16_VALID_MIN))
    encoded[valid_mask] = encoded_valid.clip(
        int(_DEPTH_U16_VALID_MIN),
        int(_DEPTH_U16_VALID_MAX),
    ).astype(np.uint16)
    return encoded


def export_depth_image_volume(
    scene_root: str,
    run: str,
    output_dir: str,
    device: str = "cuda",
    dedup_enable: bool = True,
    dedup_radius: float = 0.001,
    normals_k: int = 16,
    normals_chunk_size: int = 50000,
    resolution_scale: int = 1,
) -> str:
    from data.checkpoint_loading import (
        load_aligned_point_cloud,
        load_alignment_data_params,
    )

    with torch.no_grad():
        # Find the best checkpoint (prefer stage 2, fall back to stage 1)
        run_dir = os.path.join(scene_root, run)
        stage2_dir = os.path.join(run_dir, "after_global_optimization")
        stage1_dir = os.path.join(run_dir, "after_non_rigid_icp")

        if os.path.exists(os.path.join(stage2_dir, "aligned_points.ply")):
            checkpoint_dir = stage2_dir
            checkpoint_stage = "stage2_after_global_optimization"
            logger.info("Using Stage 2 checkpoint: %s", checkpoint_dir)
        elif os.path.exists(os.path.join(stage1_dir, "aligned_points.ply")):
            checkpoint_dir = stage1_dir
            checkpoint_stage = "stage1_after_non_rigid_icp"
            logger.info("Using Stage 1 checkpoint: %s", checkpoint_dir)
        else:
            raise FileNotFoundError(
                f"No aligned_points.ply found in {stage2_dir} or {stage1_dir}."
            )

        # Load alignment data params
        data_params = load_alignment_data_params(scene_root, run)

        # Load canonical point cloud (the full merged model)
        canonical_points, canonical_colors = load_aligned_point_cloud(checkpoint_dir, device)
        N_points = canonical_points.shape[0]
        logger.info("Loaded canonical point cloud: %d points", N_points)

        # Load per-frame segment boundaries.
        # model_frame_segments: [(start_0, end_0), ...] — which points belong to each frame.
        # These are always saved by Stage 1; Stage 2 preserves the same segments.
        seg_path = os.path.join(checkpoint_dir, "model_frame_segments.pt")
        if not os.path.exists(seg_path):
            raise FileNotFoundError(f"model_frame_segments.pt not found in {checkpoint_dir}")
        frame_segments = torch.load(seg_path, map_location="cpu", weights_only=True)

        num_frames = len(frame_segments)
        logger.info("Frame segments: %d frames", num_frames)

        # --- PCA normal estimation on the full canonical cloud ---
        from scipy.spatial import cKDTree

        pts_np = canonical_points.detach().cpu().numpy()
        col_np = (
            canonical_colors.detach().cpu().numpy()
            if canonical_colors is not None
            else np.zeros((N_points, 3))
        )

        K_normals = max(3, int(normals_k))
        chunk = max(10000, int(normals_chunk_size))
        normals_np = np.zeros_like(pts_np, dtype=np.float32)
        centroid_np = pts_np.mean(axis=0)
        tree = cKDTree(pts_np)
        logger.info("Computing normals (k=%d) for %d points...", K_normals, N_points)

        for s in range(0, N_points, chunk):
            e = min(N_points, s + chunk)
            _, idx = tree.query(pts_np[s:e], k=K_normals + 1)
            idx = idx[:, 1:]
            nbrs = pts_np[idx]
            cent = nbrs.mean(axis=1, keepdims=True)
            X = nbrs - cent
            cov = np.einsum("mki,mkj->mij", X, X) / max(1, K_normals - 1)
            w, v = np.linalg.eigh(cov)
            n = v[:, :, 0]
            view_dir = pts_np[s:e] - centroid_np
            sign = np.sign(np.sum(n * view_dir, axis=1, keepdims=True) + 1e-12)
            normals_np[s:e] = (n * sign).astype(np.float32)

        logger.info("Normal estimation complete.")

        canonical_normals = torch.from_numpy(normals_np).to(device=device, dtype=torch.float32)

        # Load refined c2w poses
        refined_c2w_poses = _load_refined_c2w_poses(checkpoint_dir, device)

        # Load intrinsics and image dimensions from DA3
        predictions = np.load(os.path.join(scene_root, "exports", "npz", "results.npz"))
        N_total = predictions["conf"].shape[0]
        all_indices = np.arange(data_params.offset, N_total, data_params.stride)[: data_params.num_frames]
        intrinsics_np = predictions["intrinsics"][all_indices]
        intrinsics_t = torch.from_numpy(intrinsics_np).to(device=device, dtype=torch.float32)
        H_base, W_base = predictions["depth"].shape[1], predictions["depth"].shape[2]

        resolution_scale = int(resolution_scale)
        if resolution_scale < 1:
            raise ValueError(f"resolution_scale must be >= 1, got {resolution_scale}")
        H = int(H_base * resolution_scale)
        W = int(W_base * resolution_scale)

        N_cams = min(len(all_indices), num_frames, len(refined_c2w_poses))

        # Compute centroid for recentering in metadata
        centroid = canonical_points.mean(dim=0)

        METERS_TO_CM = 100.0

        os.makedirs(output_dir, exist_ok=True)
        frames_json = []

        for frame_idx in range(N_cams):
            logger.info("Processing frame %d/%d", frame_idx + 1, N_cams)

            # Get this frame's slice of the canonical cloud
            start, end = frame_segments[frame_idx]
            frame_points = canonical_points[start:end]  # (M, 3) world-space
            frame_colors = canonical_colors[start:end]   # (M, 3) in [0,1]
            frame_normals = canonical_normals[start:end]  # (M, 3)

            M_frame = frame_points.shape[0]
            if M_frame == 0:
                logger.warning("Frame %d: no points, skipping", frame_idx)
                continue

            # Project this frame's canonical points into camera space using refined pose
            c2w = refined_c2w_poses[frame_idx]
            w2c = torch.linalg.inv(c2w)
            K = intrinsics_t[frame_idx].clone()
            if resolution_scale != 1:
                K[:2, :] *= float(resolution_scale)

            pts_hom = torch.cat([frame_points, torch.ones(M_frame, 1, device=device)], dim=1)
            pts_cam = (w2c @ pts_hom.T).T[:, :3]  # (M, 3)

            # Transform normals to camera space before rasterization.
            R_w2c = w2c[:3, :3]
            normals_cam = (R_w2c @ frame_normals.T).T

            # Project the refined points to their actual image locations and keep
            # the front-most point per pixel.
            depth_map, color_map, normal_map, raster_stats = _rasterize_projected_points(
                pts_cam=pts_cam,
                colors=frame_colors,
                normals_cam=normals_cam,
                K=K,
                height=H,
                width=W,
            )

            valid_mask = depth_map > 0
            if int(valid_mask.sum()) == 0:
                logger.warning("Frame %d: empty depth map, skipping", frame_idx)
                continue

            near = float(depth_map[valid_mask].min())
            far = float(depth_map[valid_mask].max())
            if far - near < 0.001:
                far = near + 1.0

            # Save depth as 16-bit PNG using the full valid code range [1, 65535].
            depth_u16 = _encode_depth_map_u16(depth_map, near=near, far=far)
            depth_filename = f"depth_{frame_idx:04d}.png"
            Image.fromarray(depth_u16, mode='I;16').save(os.path.join(output_dir, depth_filename))

            # Save normal map as RGB PNG: camera-space normals [-1,1] -> [0,255]
            normal_rgb = ((normal_map * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
            no_depth = depth_map == 0
            normal_rgb[no_depth] = [128, 128, 255]
            normal_filename = f"normal_{frame_idx:04d}.png"
            Image.fromarray(normal_rgb).save(os.path.join(output_dir, normal_filename))

            # Save color image at the refined projected pixel locations.
            color_np = (color_map * 255).clip(0, 255).astype(np.uint8)
            color_filename = f"color_{frame_idx:04d}.png"
            Image.fromarray(color_np).save(os.path.join(output_dir, color_filename))

            # Camera-to-world for metadata
            c2w_np = c2w.detach().cpu().numpy().copy()
            c2w_np[:3, 3] -= centroid.detach().cpu().numpy()
            c2w_np[:3, 3] *= METERS_TO_CM
            K_np = K.detach().cpu().numpy()

            frames_json.append({
                "file_path": depth_filename,
                "color_path": color_filename,
                "normal_path": normal_filename,
                "transform_matrix": c2w_np.tolist(),
                "fl_x": float(K_np[0, 0]),
                "fl_y": float(K_np[1, 1]),
                "cx": float(K_np[0, 2]),
                "cy": float(K_np[1, 2]),
                "near": float(near * METERS_TO_CM),
                "far": float(far * METERS_TO_CM),
            })

            logger.info(
                (
                    "  Frame %d: %d input points, %d in bounds, %d pixels written, "
                    "%d collisions, depth [%.3f, %.3f]m, pixel shift mean=%.2f px p95=%.2f px"
                ),
                frame_idx,
                M_frame,
                raster_stats["num_in_bounds"],
                raster_stats["num_pixels_written"],
                raster_stats["num_collisions"],
                near,
                far,
                raster_stats["mean_pixel_shift"],
                raster_stats["p95_pixel_shift"],
            )

        K_first = intrinsics_t[0].clone()
        if resolution_scale != 1:
            K_first[:2, :] *= float(resolution_scale)
        K_first = K_first.detach().cpu().numpy()
        centroid_np = centroid.detach().cpu().numpy()
        cameras = {
            "w": int(W),
            "h": int(H),
            "fl_x": float(K_first[0, 0]),
            "fl_y": float(K_first[1, 1]),
            "cx": float(K_first[0, 2]),
            "cy": float(K_first[1, 2]),
            "source_checkpoint_dir": checkpoint_dir,
            "source_checkpoint_stage": checkpoint_stage,
            "resolution_scale": int(resolution_scale),
            "coordinate_units": "centimeters",
            "canonical_point_cloud_centroid_m": centroid_np.tolist(),
            "transform_translation_units": "centimeters",
            "point_cloud_space": (
                "Reconstructed points from depth images live in canonical point-cloud space, "
                "centered by canonical_point_cloud_centroid_m and scaled to centimeters."
            ),
            "depth_encoding": {
                "dtype": "uint16",
                "invalid_code": int(_DEPTH_U16_INVALID),
                "valid_code_min": int(_DEPTH_U16_VALID_MIN),
                "valid_code_max": int(_DEPTH_U16_VALID_MAX),
                "units": "centimeters",
                "decode_formula_export_units": "near + ((code - 1) / 65534) * (far - near)",
            },
            "frames": frames_json,
        }

        cameras_path = os.path.join(output_dir, "cameras.json")
        with open(cameras_path, "w") as f:
            json.dump(cameras, f, indent=2)

        logger.info("Export complete: %d frames written to %s", len(frames_json), output_dir)
        return output_dir


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Export DepthImageVolume from reconstructed scene")
    parser.add_argument("--scene_root", required=True, help="Scene root directory")
    parser.add_argument("--run", required=True, help="Run name")
    parser.add_argument("--output_dir", required=True, help="Output directory for the export")
    parser.add_argument("--device", default="cuda", help="Torch device")
    parser.add_argument("--resolution_scale", type=int, default=1, help="Integer export resolution scale (1=base)")
    args = parser.parse_args()

    export_depth_image_volume(
        args.scene_root,
        args.run,
        args.output_dir,
        args.device,
        resolution_scale=args.resolution_scale,
    )
