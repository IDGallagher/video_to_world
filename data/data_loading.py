import glob
import os
import shutil

import json
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from PIL import Image
from torchvision.transforms import ToTensor

from utils.logging import get_logger

logger = get_logger(__name__)


def _depth_edge_suffix(
    *,
    enabled: bool,
    edge_rtol: float | None,
    edge_atol: float | None,
    edge_kernel_size: int,
) -> str:
    if not enabled:
        return ""

    parts = [f"edge_k{int(edge_kernel_size)}"]
    if edge_rtol is not None:
        parts.append(f"r{float(edge_rtol):.4f}")
    if edge_atol is not None:
        parts.append(f"a{float(edge_atol):.4f}")
    return "_" + "_".join(parts)


def _max_depth_suffix(
    *,
    enabled: bool,
    max_depth_rtol: float | None,
    max_depth_atol: float | None,
) -> str:
    if not enabled:
        return ""

    parts = ["maxdepth"]
    if max_depth_rtol is not None:
        parts.append(f"r{float(max_depth_rtol):.6f}")
    if max_depth_atol is not None:
        parts.append(f"a{float(max_depth_atol):.6f}")
    return "_" + "_".join(parts)


def _sky_depth_band_suffix(*, enabled: bool, band_percent: float) -> str:
    if not enabled:
        return ""
    return f"_skyband_p{float(band_percent):.3f}"


def _sky_mask_suffix(*, enabled: bool) -> str:
    return "_sky" if enabled else ""


def _write_boolean_mask_png(path: str, mask: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    image.save(path)


def _write_debug_mask_pngs(
    *,
    output_dir: str,
    depth_shape: tuple[int, int, int],
    sky_mask: np.ndarray | None,
    valid_flat_indices_all: list[np.ndarray],
) -> None:
    num_frames, height, width = depth_shape
    if len(valid_flat_indices_all) != num_frames:
        raise ValueError(
            f"Expected valid indices for {num_frames} frames, got {len(valid_flat_indices_all)}."
        )

    sky_dir = os.path.join(output_dir, "sky")
    kept_dir = os.path.join(output_dir, "kept")
    os.makedirs(kept_dir, exist_ok=True)
    if sky_mask is not None:
        os.makedirs(sky_dir, exist_ok=True)

    for i in range(num_frames):
        kept_mask = np.zeros((height, width), dtype=bool)
        flat_indices = valid_flat_indices_all[i]
        if flat_indices.size > 0:
            kept_mask.reshape(-1)[flat_indices.astype(np.int64)] = True
        _write_boolean_mask_png(os.path.join(kept_dir, f"frame_{i:05d}.png"), kept_mask)

        if sky_mask is not None:
            _write_boolean_mask_png(os.path.join(sky_dir, f"frame_{i:05d}.png"), sky_mask[i].astype(bool, copy=False))


def _apply_sky_mask_suppression(
    depth: np.ndarray,
    conf: np.ndarray,
    *,
    sky_mask: np.ndarray | None,
    enabled: bool,
) -> tuple[np.ndarray, np.ndarray]:
    base_valid = np.isfinite(depth) & (depth > 0)
    if not enabled:
        return conf, base_valid
    if sky_mask is None:
        logger.warning("Sky-mask suppression requested, but results.npz does not contain a 'sky' array. Skipping.")
        return conf, base_valid
    if sky_mask.shape != depth.shape:
        raise ValueError(f"Sky mask shape {sky_mask.shape} does not match depth shape {depth.shape}.")

    conf_filtered = conf.copy()
    sky_mask = sky_mask.astype(bool, copy=False)
    conf_filtered[sky_mask] = 0.0
    valid_filtered = base_valid & (~sky_mask)
    return conf_filtered, valid_filtered


def _apply_sky_depth_band_suppression(
    depth: np.ndarray,
    conf: np.ndarray,
    *,
    sky_mask: np.ndarray | None,
    enabled: bool,
    band_percent: float,
    base_valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    base_valid = base_valid_mask if base_valid_mask is not None else (np.isfinite(depth) & (depth > 0))
    if not enabled:
        return conf, base_valid
    if sky_mask is None:
        logger.warning(
            "Sky-depth-band suppression requested, but results.npz does not contain a 'sky' array. Skipping."
        )
        return conf, base_valid
    if sky_mask.shape != depth.shape:
        raise ValueError(f"Sky mask shape {sky_mask.shape} does not match depth shape {depth.shape}.")
    if band_percent < 0.0 or band_percent > 100.0:
        raise ValueError(f"Sky depth band percent must be between 0 and 100, got {band_percent}.")

    conf_filtered = conf.copy()
    valid_filtered = base_valid.copy()
    band_ratio = float(band_percent) / 100.0

    for i in range(depth.shape[0]):
        vm = base_valid[i]
        depth_valid = np.isfinite(depth[i]) & (depth[i] > 0)
        sm = sky_mask[i].astype(bool, copy=False) & depth_valid
        if not np.any(sm):
            continue
        frame_depth = depth[i]
        sky_depth_ref = float(frame_depth[sm].min())
        threshold = sky_depth_ref * max(0.0, 1.0 - band_ratio)
        band_mask = vm & (frame_depth >= threshold)
        conf_filtered[i][band_mask] = 0.0
        valid_filtered[i][band_mask] = False

    return conf_filtered, valid_filtered


def _compute_depth_edge_mask(
    depth: np.ndarray,
    *,
    edge_rtol: float | None,
    edge_atol: float | None,
    edge_kernel_size: int,
    valid_mask: np.ndarray | None,
) -> np.ndarray:
    # Treat zero/negative tolerances as disabled (atol=0 would flag every pixel
    # with any depth variation; rtol=0 is similarly degenerate).
    if edge_atol is not None and edge_atol <= 0:
        edge_atol = None
    if edge_rtol is not None and edge_rtol <= 0:
        edge_rtol = None
    if edge_rtol is None and edge_atol is None:
        return np.zeros_like(depth, dtype=bool)

    depth_t = torch.from_numpy(depth.astype(np.float32, copy=False))
    depth_t = depth_t.reshape(-1, 1, *depth_t.shape[-2:])

    mask_t = None
    if valid_mask is not None:
        mask_t = torch.from_numpy(valid_mask.astype(bool, copy=False))
        mask_t = mask_t.reshape(-1, 1, *mask_t.shape[-2:])

    if mask_t is None:
        diff = F.max_pool2d(depth_t, edge_kernel_size, stride=1, padding=edge_kernel_size // 2) + F.max_pool2d(
            -depth_t, edge_kernel_size, stride=1, padding=edge_kernel_size // 2
        )
    else:
        neg_inf = torch.full_like(depth_t, float("-inf"))
        diff = F.max_pool2d(
            torch.where(mask_t, depth_t, neg_inf),
            edge_kernel_size,
            stride=1,
            padding=edge_kernel_size // 2,
        ) + F.max_pool2d(
            torch.where(mask_t, -depth_t, neg_inf),
            edge_kernel_size,
            stride=1,
            padding=edge_kernel_size // 2,
        )

    edge = torch.zeros_like(depth_t, dtype=torch.bool)
    if edge_atol is not None:
        edge |= diff > float(edge_atol)
    if edge_rtol is not None:
        edge |= (diff / depth_t.abs().clamp_min(1e-12)).nan_to_num_() > float(edge_rtol)
    if mask_t is not None:
        edge &= mask_t

    return edge.reshape(depth.shape).cpu().numpy()


def _apply_depth_edge_suppression(
    depth: np.ndarray,
    conf: np.ndarray,
    *,
    enabled: bool,
    edge_rtol: float | None,
    edge_atol: float | None,
    edge_kernel_size: int,
    base_valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    base_valid = base_valid_mask if base_valid_mask is not None else (np.isfinite(depth) & (depth > 0))
    if not enabled:
        return conf, base_valid
    if int(edge_kernel_size) <= 0 or int(edge_kernel_size) % 2 == 0:
        raise ValueError(f"conf_edge_kernel_size must be a positive odd integer, got {edge_kernel_size}")

    edge_mask = _compute_depth_edge_mask(
        depth,
        edge_rtol=edge_rtol,
        edge_atol=edge_atol,
        edge_kernel_size=edge_kernel_size,
        valid_mask=base_valid,
    )
    conf_filtered = conf.copy()
    conf_filtered[edge_mask] = 0.0
    valid_filtered = base_valid & (~edge_mask)
    return conf_filtered, valid_filtered


def _apply_max_depth_suppression(
    depth: np.ndarray,
    conf: np.ndarray,
    *,
    enabled: bool,
    max_depth_rtol: float | None,
    max_depth_atol: float | None,
    base_valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    base_valid = base_valid_mask if base_valid_mask is not None else (np.isfinite(depth) & (depth > 0))
    if not enabled:
        return conf, base_valid
    if max_depth_rtol is None and max_depth_atol is None:
        raise ValueError("Max-depth suppression needs a relative or absolute threshold.")

    conf_filtered = conf.copy()
    valid_filtered = base_valid.copy()

    rtol = float(max_depth_rtol) if max_depth_rtol is not None else 0.0
    atol = float(max_depth_atol) if max_depth_atol is not None else 0.0

    for i in range(depth.shape[0]):
        vm = base_valid[i]
        if not np.any(vm):
            continue
        frame_depth = depth[i]
        frame_max = float(frame_depth[vm].max())
        tol = max(atol, abs(frame_max) * rtol)
        plateau_mask = vm & (frame_depth >= (frame_max - tol))
        conf_filtered[i][plateau_mask] = 0.0
        valid_filtered[i][plateau_mask] = False

    return conf_filtered, valid_filtered


def _find_preprocess_frames_dir(root_path: str) -> str:
    """
    Locate the frame folder produced/used by `preprocess_video.py`.

    Preferred: read `<scene_root>/preprocess_frames.json` written by
    `preprocess_video.py` to find the actual frames directory.

    Fallback: `<scene_root>/frames/`.
    """
    meta_path = os.path.join(root_path, "preprocess_frames.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            frames_dir = meta.get("frames_dir", None)
            if isinstance(frames_dir, str) and os.path.isdir(frames_dir):
                return frames_dir
        except Exception as e:
            logger.warning("Failed reading %s (%s); falling back to <scene_root>/frames", meta_path, e)

    frames_dir = os.path.join(root_path, "frames")
    if not os.path.isdir(frames_dir):
        raise FileNotFoundError(
            "Requested original images/intrinsics, but could not find the preprocessing frames folder. "
            f"Tried '{meta_path}' and then '{frames_dir}'."
        )
    return frames_dir


def load_image(path: str) -> torch.Tensor:
    """Load an image file as a float tensor in [0,1], shape (C,H,W)."""
    image = Image.open(path)
    return ToTensor()(image)


def load_point_cloud(path: str, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load a point cloud from file (PLY, PCD, XYZ, etc.) using Open3D.

    Returns:
        points: (N,3) float32 tensor on the given device
        colors: (N,3) float32 tensor on the given device
    """
    pcd = o3d.io.read_point_cloud(path)
    if not pcd.has_points():
        raise ValueError(f"Could not load point cloud from {path}")
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    return (
        torch.tensor(points, dtype=torch.float32, device=device),
        torch.tensor(colors, dtype=torch.float32, device=device),
    )


def torch_to_o3d_pcd(points: torch.Tensor, colors: torch.Tensor | None = None) -> o3d.geometry.PointCloud:
    """Convert torch (N,3) points (+ optional colors) to an Open3D PointCloud."""
    pts = points.detach().cpu().numpy().reshape(-1, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    if colors is not None:
        cols = colors.detach().cpu().numpy().reshape(-1, 3)
        pcd.colors = o3d.utility.Vector3dVector(cols)
    return pcd


def _as_homogeneous44(ext: np.ndarray) -> np.ndarray:
    """
    Accept (4,4) or (3,4) extrinsic parameters, return (4,4) homogeneous matrix.
    """
    if ext.shape == (4, 4):
        return ext
    if ext.shape == (3, 4):
        H = np.eye(4, dtype=ext.dtype)
        H[:3, :4] = ext
        return H
    raise ValueError(f"extrinsic must be (4,4) or (3,4), got {ext.shape}")


def depths_to_world_points_with_colors(
    depth: np.ndarray,
    K: np.ndarray,
    ext_w2c: np.ndarray,
    images_u8: np.ndarray,
    conf: np.ndarray | None = None,
    conf_thr: float = 0.0,
    valid_mask: np.ndarray | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Back-project depth maps to world-space 3-D points with colours.

    For each frame, transform (u,v,1) through K^{-1} to get rays,
    multiply by depth to get camera-frame points, then use (w2c)^{-1}
    to transform to world frame.  Simultaneously extract colours.

    Args:
        depth: (N, H, W) depth maps.
        K: (N, 3, 3) intrinsic matrices.
        ext_w2c: (N, 3, 4) or (N, 4, 4) world-to-camera extrinsics.
        images_u8: (N, H, W, 3) uint8 images.
        conf: (N, H, W) confidence maps (optional).
        conf_thr: confidence threshold – pixels with conf < conf_thr are
            discarded.  Ignored when *conf* is None and *valid_mask* is None.
        valid_mask: optional boolean mask (N, H, W). When provided, only pixels
            with valid_mask == True are kept, in addition to finite/positive
            depth. In this case *conf*/*conf_thr* are ignored.

    Returns:
        pts_all: list of (M_i, 3) float32 arrays – world points per frame.
        col_all: list of (M_i, 3) uint8 arrays – colours per frame.
    """
    N, H, W = depth.shape
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    ones = np.ones_like(us)
    pix = np.stack([us, vs, ones], axis=-1).reshape(-1, 3)  # (H*W, 3)

    pts_all: list[np.ndarray] = []
    col_all: list[np.ndarray] = []

    for i in tqdm(range(N), desc="Back-projecting frames"):
        d = depth[i]  # (H, W)
        valid = np.isfinite(d) & (d > 0)
        if valid_mask is not None:
            valid &= valid_mask[i]
        elif conf is not None:
            valid &= conf[i] >= conf_thr
        if not np.any(valid):
            pts_all.append(np.zeros((0, 3), dtype=np.float32))
            col_all.append(np.zeros((0, 3), dtype=np.uint8))
            continue

        d_flat = d.reshape(-1)
        vidx = np.flatnonzero(valid.reshape(-1))

        K_inv = np.linalg.inv(K[i])  # (3, 3)
        c2w = np.linalg.inv(_as_homogeneous44(ext_w2c[i]))  # (4, 4)

        rays = K_inv @ pix[vidx].T  # (3, M)
        Xc = rays * d_flat[vidx][None, :]  # (3, M)
        Xc_h = np.vstack([Xc, np.ones((1, Xc.shape[1]))])
        Xw = (c2w @ Xc_h)[:3].T.astype(np.float32)  # (M, 3)

        cols = images_u8[i].reshape(-1, 3)[vidx].astype(np.uint8)  # (M, 3)

        pts_all.append(Xw)
        col_all.append(cols)

    if len(pts_all) == 0:
        return [np.zeros((0, 3), dtype=np.float32)], [np.zeros((0, 3), dtype=np.uint8)]

    return pts_all, col_all


def _voxelized_conf_filter_da3(
    all_depth: np.ndarray,
    all_conf: np.ndarray,
    all_intrinsics: np.ndarray,
    all_extrinsics: np.ndarray,
    all_images: np.ndarray,
    voxel_size: float,
    local_percentile: float,
    global_percentile: float | None = None,
    min_count_percentile: float | None = None,
    valid_mask: np.ndarray | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """
    Compute per-voxel confidence thresholds in world space and filter points.

    Returns per-frame:
        pts3d_per_frame: list of (M_i, 3) float32 arrays
        colors_per_frame: list of (M_i, 3) float32 arrays (0–255)
        valid_flat_indices_per_frame: list of 1-D int arrays of flat pixel ids
    """
    N_total, H, W = all_depth.shape
    # First, back-project all *depth-valid* pixels using the shared helper.
    depth_valid = valid_mask if valid_mask is not None else (np.isfinite(all_depth) & (all_depth > 0))
    pts3d_per_frame, colors_per_frame = depths_to_world_points_with_colors(
        all_depth,
        all_intrinsics,
        all_extrinsics,
        all_images,
        conf=None,
        conf_thr=0.0,
        valid_mask=depth_valid,
    )

    pts_all_list: list[np.ndarray] = []
    conf_all_list: list[np.ndarray] = []
    color_all_list: list[np.ndarray] = []
    frame_ids_list: list[np.ndarray] = []
    pix_ids_list: list[np.ndarray] = []

    valid_flat_indices_per_frame: list[np.ndarray] = []

    for i in range(N_total):
        vm = depth_valid[i]
        if not np.any(vm):
            valid_flat_indices_per_frame.append(np.zeros((0,), dtype=np.int64))
            continue

        c = all_conf[i]  # (H, W)
        vidx = np.flatnonzero(vm.reshape(-1)).astype(np.int64)

        conf_flat = c.reshape(-1)[vidx]
        pts_i = pts3d_per_frame[i]
        cols_i = colors_per_frame[i].astype(np.float32)

        valid_flat_indices_per_frame.append(vidx)

        pts_all_list.append(pts_i)
        conf_all_list.append(conf_flat)
        color_all_list.append(cols_i)
        frame_ids_list.append(np.full(vidx.shape[0], i, dtype=np.int64))
        pix_ids_list.append(vidx)

    if len(pts_all_list) == 0:
        # No valid points at all.
        return (
            [np.zeros((0, 3), dtype=np.float32)] * N_total,
            [np.zeros((0, 3), dtype=np.float32)] * N_total,
            [np.zeros((0,), dtype=np.int64)] * N_total,
        )

    pts_all = np.concatenate(pts_all_list, axis=0)  # (P, 3)
    conf_all = np.concatenate(conf_all_list, axis=0)  # (P,)
    colors_all = np.concatenate(color_all_list, axis=0)  # (P, 3)
    frame_ids_all = np.concatenate(frame_ids_list, axis=0)  # (P,)
    pix_ids_all = np.concatenate(pix_ids_list, axis=0)  # (P,)

    # Compute voxel indices in world space.
    voxel_indices = np.floor(pts_all / float(voxel_size)).astype(np.int64)  # (P, 3)
    unique_vox, inv = np.unique(voxel_indices, axis=0, return_inverse=True)
    num_vox = unique_vox.shape[0]

    # Per-voxel occupancy counts (for optional min_count_percentile filtering).
    voxel_counts = np.bincount(inv, minlength=num_vox)

    # Optional global threshold (across all frames/voxels).
    global_thr = None
    if global_percentile is not None:
        global_thr = float(np.percentile(conf_all, global_percentile))

    # Compute per-voxel confidence thresholds.
    order = np.argsort(inv)
    inv_sorted = inv[order]
    conf_sorted = conf_all[order]

    change = np.r_[True, inv_sorted[1:] != inv_sorted[:-1]]
    starts = np.flatnonzero(change)
    ends = np.r_[starts[1:], len(inv_sorted)]

    voxel_thr = np.empty(num_vox, dtype=np.float32)
    for j, (s, e) in enumerate(zip(starts, ends)):
        vals = conf_sorted[s:e]
        if vals.size == 0:
            voxel_thr[j] = -np.inf
            continue
        local_thr = float(np.percentile(vals, local_percentile))
        if global_thr is not None:
            voxel_thr[j] = max(global_thr, local_thr)
        else:
            voxel_thr[j] = local_thr

    # Optional voxel removal based on occupancy percentile.
    if min_count_percentile is not None:
        count_thr = float(np.percentile(voxel_counts, min_count_percentile))
        keep_voxels = voxel_counts >= count_thr  # (num_vox,)
    else:
        keep_voxels = np.ones(num_vox, dtype=bool)

    # Map thresholds back to individual points.
    point_thr = voxel_thr[inv]  # (P,)
    keep_mask = (conf_all >= point_thr) & keep_voxels[inv]  # (P,)

    # Rebuild per-frame outputs using the filtered points.
    pts3d_filtered_per_frame: list[np.ndarray] = []
    colors_filtered_per_frame: list[np.ndarray] = []
    valid_flat_filtered_per_frame: list[np.ndarray] = []
    for i in range(N_total):
        sel = (frame_ids_all == i) & keep_mask
        if not np.any(sel):
            pts3d_filtered_per_frame.append(np.zeros((0, 3), dtype=np.float32))
            colors_filtered_per_frame.append(np.zeros((0, 3), dtype=np.float32))
            valid_flat_filtered_per_frame.append(np.zeros((0,), dtype=np.int64))
        else:
            pts3d_filtered_per_frame.append(pts_all[sel])
            colors_filtered_per_frame.append(colors_all[sel])
            valid_flat_filtered_per_frame.append(pix_ids_all[sel])

    return (
        pts3d_filtered_per_frame,
        colors_filtered_per_frame,
        valid_flat_filtered_per_frame,
    )


def load_data(
    root_path: str,
    num_frames: int = 10,
    stride: int = 1,
    device: str = "cpu",
    conf_thresh_percentile: float = 40.0,
    *,
    conf_mode: str = "global",
    conf_local_percentile: float | None = None,
    conf_global_percentile: float | None = None,
    voxel_size: float = 0.1,
    voxel_min_count_percentile: float | None = None,
    conf_mask_sky: bool = False,
    conf_mask_sky_depth_band: bool = False,
    conf_sky_depth_band_percent: float = 2.0,
    conf_mask_depth_edges: bool = False,
    conf_edge_rtol: float | None = 0.03,
    conf_edge_atol: float | None = None,
    conf_edge_kernel_size: int = 3,
    conf_mask_max_depth: bool = False,
    conf_max_depth_rtol: float | None = 0.001,
    conf_max_depth_atol: float | None = None,
    offset: int = 0,
    load_original_images_and_intrinsics: bool = False,
):
    predictions = np.load(os.path.join(root_path, "exports", "npz", "results.npz"))
    N_total = predictions["conf"].shape[0]
    all_indices = np.arange(N_total)
    if (N_total - offset) < num_frames:
        stride = 1
    indices = all_indices[offset::stride][:num_frames]
    extrinsics = predictions["extrinsics"][indices]  # w2c matrices as np array of shape (N, 3, 4)
    intrinsics = predictions["intrinsics"][indices]  # intrinsics in pixel space as np array of shape (N, 3, 3)
    images = predictions["image"][indices]  # np array of shape (N, H, W, 3) in uint8 format

    # Load depth and conf for computing valid pixel indices
    all_depth = predictions["depth"]  # (N_total, H, W)
    all_conf = predictions["conf"]  # (N_total, H, W)
    sky_mask = predictions["sky"] if "sky" in predictions.files else None
    all_conf_filtered, base_valid_mask = _apply_sky_mask_suppression(
        all_depth,
        all_conf,
        sky_mask=sky_mask,
        enabled=conf_mask_sky,
    )
    if conf_mask_sky and sky_mask is not None:
        masked_pixels = int((np.isfinite(all_depth) & (all_depth > 0) & sky_mask.astype(bool)).sum())
        logger.info("Sky-mask suppression removed %d pixels before confidence filtering", masked_pixels)
    all_conf_filtered, base_valid_mask = _apply_sky_depth_band_suppression(
        all_depth,
        all_conf_filtered,
        sky_mask=sky_mask,
        enabled=conf_mask_sky_depth_band,
        band_percent=conf_sky_depth_band_percent,
        base_valid_mask=base_valid_mask,
    )
    if conf_mask_sky_depth_band and sky_mask is not None:
        masked_pixels = int((np.isfinite(all_depth) & (all_depth > 0) & (~base_valid_mask)).sum())
        logger.info(
            "Sky-depth-band suppression removed %d pixels before confidence filtering (band_percent=%s)",
            masked_pixels,
            str(conf_sky_depth_band_percent),
        )

    all_conf_filtered, base_valid_mask = _apply_max_depth_suppression(
        all_depth,
        all_conf_filtered,
        enabled=conf_mask_max_depth,
        max_depth_rtol=conf_max_depth_rtol,
        max_depth_atol=conf_max_depth_atol,
        base_valid_mask=base_valid_mask,
    )
    if conf_mask_max_depth:
        masked_pixels = int((np.isfinite(all_depth) & (all_depth > 0) & (~base_valid_mask)).sum())
        logger.info(
            "Max-depth suppression removed %d pixels before confidence filtering (rtol=%s, atol=%s)",
            masked_pixels,
            str(conf_max_depth_rtol),
            str(conf_max_depth_atol),
        )

    all_conf_filtered, base_valid_mask = _apply_depth_edge_suppression(
        all_depth,
        all_conf_filtered,
        enabled=conf_mask_depth_edges,
        edge_rtol=conf_edge_rtol,
        edge_atol=conf_edge_atol,
        edge_kernel_size=conf_edge_kernel_size,
        base_valid_mask=base_valid_mask,
    )
    if conf_mask_depth_edges:
        masked_pixels = int((np.isfinite(all_depth) & (all_depth > 0) & (~base_valid_mask)).sum())
        logger.info(
            "Depth-edge suppression removed %d pixels before confidence filtering (kernel=%d, rtol=%s, atol=%s)",
            masked_pixels,
            int(conf_edge_kernel_size),
            str(conf_edge_rtol),
            str(conf_edge_atol),
        )

    pcl_folder = os.path.join(root_path, "exports", "ply")
    if not os.path.exists(pcl_folder):
        os.makedirs(pcl_folder, exist_ok=True)

    # Resolve local/global percentiles (for guided modes) with sensible defaults.
    if conf_local_percentile is None:
        conf_local_percentile = conf_thresh_percentile
    if conf_global_percentile is None:
        conf_global_percentile = conf_thresh_percentile

    conf_mode = conf_mode.lower()
    if conf_mode not in {
        "global",
        "per_frame",
        "per_frame_guided",
        "voxel",
        "voxel_guided",
        "voxel_or",
    }:
        raise ValueError(
            f"Unknown conf_mode '{conf_mode}'. "
            "Expected one of: 'global', 'per_frame', 'per_frame_guided', 'voxel', 'voxel_guided'."
        )

    # Naming scheme for confidence-filtered PLY folders.
    if conf_mode == "global":
        pcl_conf_folder = os.path.join(pcl_folder, f"conf_percentile_{conf_thresh_percentile}")
    elif conf_mode == "per_frame":
        pcl_conf_folder = os.path.join(pcl_folder, f"conf_perframe_{conf_thresh_percentile}")
    elif conf_mode == "per_frame_guided":
        pcl_conf_folder = os.path.join(
            pcl_folder,
            f"conf_perframe_guided_g{conf_global_percentile}_l{conf_local_percentile}",
        )
    elif conf_mode == "voxel":
        if voxel_min_count_percentile is None:
            pcl_conf_folder = os.path.join(
                pcl_folder,
                f"conf_voxel_vs{voxel_size}_p{conf_thresh_percentile}",
            )
        else:
            pcl_conf_folder = os.path.join(
                pcl_folder,
                f"conf_voxel_vs{voxel_size}_p{conf_thresh_percentile}_min{voxel_min_count_percentile}",
            )
    elif conf_mode == "voxel_guided":
        pcl_conf_folder = os.path.join(
            pcl_folder,
            f"conf_voxel_guided_vs{voxel_size}_g{conf_global_percentile}_l{conf_local_percentile}",
        )
    else:  # "voxel_or"
        name = (
            f"conf_voxel_or_vs{voxel_size}_g{conf_global_percentile}_l{conf_local_percentile}_p{conf_thresh_percentile}"
        )
        if voxel_min_count_percentile is not None:
            name += f"_min{voxel_min_count_percentile}"
        pcl_conf_folder = os.path.join(pcl_folder, name)

    edge_suffix = _depth_edge_suffix(
        enabled=conf_mask_depth_edges,
        edge_rtol=conf_edge_rtol,
        edge_atol=conf_edge_atol,
        edge_kernel_size=conf_edge_kernel_size,
    )
    if edge_suffix:
        pcl_conf_folder = f"{pcl_conf_folder}{edge_suffix}"
    max_depth_suffix = _max_depth_suffix(
        enabled=conf_mask_max_depth,
        max_depth_rtol=conf_max_depth_rtol,
        max_depth_atol=conf_max_depth_atol,
    )
    if max_depth_suffix:
        pcl_conf_folder = f"{pcl_conf_folder}{max_depth_suffix}"
    sky_suffix = _sky_mask_suffix(enabled=conf_mask_sky)
    if sky_suffix:
        pcl_conf_folder = f"{pcl_conf_folder}{sky_suffix}"
    sky_depth_band_suffix = _sky_depth_band_suffix(
        enabled=conf_mask_sky_depth_band,
        band_percent=conf_sky_depth_band_percent,
    )
    if sky_depth_band_suffix:
        pcl_conf_folder = f"{pcl_conf_folder}{sky_depth_band_suffix}"

    if not os.path.exists(pcl_conf_folder):
        os.makedirs(pcl_conf_folder, exist_ok=True)

        logger.info("Preprocessing {image, depth, extrinsics, intrinsics} to point clouds...")

        all_extrinsics = predictions["extrinsics"]  # w2c matrices as np array of shape (N, 3, 4)
        all_intrinsics = predictions["intrinsics"]  # intrinsics in pixel space as np array of shape (N, 3, 3)
        all_images = predictions["image"]  # np array of shape (N, H, W, 3) in uint8 format

        # When we build PLYs from scratch we also collect the corresponding
        # flat pixel indices once and store them on disk for future reuse.
        valid_flat_indices_all: list[np.ndarray]

        if conf_mode == "global":
            conf_thresh = np.percentile(all_conf_filtered, conf_thresh_percentile)
            valid_mask = base_valid_mask & (all_conf_filtered >= conf_thresh)
            pts3d, colors = depths_to_world_points_with_colors(
                all_depth,
                all_intrinsics,
                all_extrinsics,
                all_images,
                conf=None,
                conf_thr=0.0,
                valid_mask=valid_mask,
            )
            valid_flat_indices_all = [
                np.flatnonzero(valid_mask[i].reshape(-1)).astype(np.int64) for i in range(N_total)
            ]
        elif conf_mode == "per_frame":
            valid_mask = np.zeros_like(all_depth, dtype=bool)
            for i in tqdm(range(N_total), desc="Computing per-frame conf masks"):
                conf_thr_i = np.percentile(all_conf_filtered[i], conf_thresh_percentile)
                c = all_conf_filtered[i]
                vm = base_valid_mask[i] & (c >= conf_thr_i)
                valid_mask[i] = vm
            pts3d, colors = depths_to_world_points_with_colors(
                all_depth,
                all_intrinsics,
                all_extrinsics,
                all_images,
                conf=None,
                conf_thr=0.0,
                valid_mask=valid_mask,
            )
            valid_flat_indices_all = [
                np.flatnonzero(valid_mask[i].reshape(-1)).astype(np.int64) for i in range(N_total)
            ]
        elif conf_mode == "per_frame_guided":
            global_thr = np.percentile(all_conf_filtered, conf_global_percentile)
            valid_mask = np.zeros_like(all_depth, dtype=bool)
            for i in tqdm(range(N_total), desc="Computing per-frame guided conf masks"):
                c = all_conf_filtered[i]
                local_thr = np.percentile(c, conf_local_percentile)
                conf_thr_i = max(global_thr, local_thr)
                vm = base_valid_mask[i] & (c >= conf_thr_i)
                valid_mask[i] = vm
            pts3d, colors = depths_to_world_points_with_colors(
                all_depth,
                all_intrinsics,
                all_extrinsics,
                all_images,
                conf=None,
                conf_thr=0.0,
                valid_mask=valid_mask,
            )
            valid_flat_indices_all = [
                np.flatnonzero(valid_mask[i].reshape(-1)).astype(np.int64) for i in range(N_total)
            ]
        elif conf_mode == "voxel":
            pts3d, colors, valid_flat_indices_all = _voxelized_conf_filter_da3(
                all_depth,
                all_conf_filtered,
                all_intrinsics,
                all_extrinsics,
                all_images,
                voxel_size,
                local_percentile=conf_thresh_percentile,
                global_percentile=None,
                min_count_percentile=voxel_min_count_percentile,
                valid_mask=base_valid_mask,
            )
        elif conf_mode == "voxel_guided":
            pts3d, colors, valid_flat_indices_all = _voxelized_conf_filter_da3(
                all_depth,
                all_conf_filtered,
                all_intrinsics,
                all_extrinsics,
                all_images,
                voxel_size,
                local_percentile=conf_local_percentile,
                global_percentile=conf_global_percentile,
                valid_mask=base_valid_mask,
            )
        else:  # "voxel_or" – OR-combine voxel_guided and voxel(min-count) selections
            # First: voxel_guided branch (global/local guided)
            pts3d_g, colors_g, flat_idx_g = _voxelized_conf_filter_da3(
                all_depth,
                all_conf_filtered,
                all_intrinsics,
                all_extrinsics,
                all_images,
                voxel_size,
                local_percentile=conf_local_percentile,
                global_percentile=conf_global_percentile,
                valid_mask=base_valid_mask,
            )
            # Second: plain voxel branch with strong local/min-count filtering
            pts3d_v, colors_v, flat_idx_v = _voxelized_conf_filter_da3(
                all_depth,
                all_conf_filtered,
                all_intrinsics,
                all_extrinsics,
                all_images,
                voxel_size,
                local_percentile=conf_thresh_percentile,
                global_percentile=None,
                min_count_percentile=voxel_min_count_percentile,
                valid_mask=base_valid_mask,
            )

            pts3d = []
            colors = []
            valid_flat_indices_all = []
            for i in range(N_total):
                idx_g = flat_idx_g[i]
                idx_v = flat_idx_v[i]
                if idx_g.size == 0 and idx_v.size == 0:
                    pts3d.append(np.zeros((0, 3), dtype=np.float32))
                    colors.append(np.zeros((0, 3), dtype=np.float32))
                    valid_flat_indices_all.append(np.zeros((0,), dtype=np.int64))
                    continue

                # Concatenate indices and corresponding points/colors, then
                # deduplicate by flat index to realise the OR-combination.
                idx_all = np.concatenate([idx_g, idx_v], axis=0)
                pts_all = np.concatenate([pts3d_g[i], pts3d_v[i]], axis=0)
                cols_all = np.concatenate([colors_g[i], colors_v[i]], axis=0)

                order = np.argsort(idx_all)
                idx_sorted = idx_all[order]
                pts_sorted = pts_all[order]
                cols_sorted = cols_all[order]

                first_of_run = np.r_[True, idx_sorted[1:] != idx_sorted[:-1]]
                idx_unique = idx_sorted[first_of_run]
                pts_unique = pts_sorted[first_of_run]
                cols_unique = cols_sorted[first_of_run]

                valid_flat_indices_all.append(idx_unique.astype(np.int64))
                pts3d.append(pts_unique.astype(np.float32))
                colors.append(cols_unique.astype(np.float32))

        def make_pcd(p: np.ndarray, c: np.ndarray) -> o3d.geometry.PointCloud:
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(p)
            pc.colors = o3d.utility.Vector3dVector(c / 255.0 if c.size > 0 and c.max() > 1 else c)
            return pc

        pcls = [make_pcd(pts3d[i], colors[i].astype(np.float32)) for i in range(len(pts3d))]

        for i in tqdm(range(len(pcls)), desc="Saving point clouds"):
            o3d.io.write_point_cloud(os.path.join(pcl_conf_folder, f"frame_{i:05d}.ply"), pcls[i])

        # Persist valid pixel indices for all frames alongside the PLYs so that
        # future data loading never has to recompute them.
        valid_indices_path = os.path.join(pcl_conf_folder, "valid_pixel_indices.npz")
        np.savez_compressed(
            valid_indices_path,
            **{f"frame_{i:05d}": arr for i, arr in enumerate(valid_flat_indices_all)},
        )
        if conf_mask_sky:
            _write_debug_mask_pngs(
                output_dir=os.path.join(pcl_conf_folder, "debug_masks"),
                depth_shape=all_depth.shape,
                sky_mask=sky_mask,
                valid_flat_indices_all=valid_flat_indices_all,
            )

    pcls = [o3d.io.read_point_cloud(os.path.join(pcl_conf_folder, f"frame_{i:05d}.ply")) for i in indices]
    extrinsics = [ext for ext in extrinsics]
    intrinsics = [intr for intr in intrinsics]
    images = torch.from_numpy(images).permute(0, 3, 1, 2).to(torch.float32).to(device) / 255.0

    # Load valid pixel indices for each frame from disk (same filtering as for PLYs).
    valid_indices_path = os.path.join(pcl_conf_folder, "valid_pixel_indices.npz")

    def _load_valid_indices() -> list[torch.Tensor]:
        if not os.path.exists(valid_indices_path):
            raise FileNotFoundError(f"Expected precomputed valid pixel indices at {valid_indices_path}.")
        valid_indices_npz = np.load(valid_indices_path)
        result: list[torch.Tensor] = []
        for frame_idx in indices:
            key = f"frame_{frame_idx:05d}"
            if key not in valid_indices_npz:
                raise KeyError(f"Missing key {key} in {valid_indices_path}.")
            flat_indices = valid_indices_npz[key]
            result.append(torch.from_numpy(flat_indices).long().to(device))
        return result

    try:
        valid_pixel_indices = _load_valid_indices()
    except (FileNotFoundError, KeyError) as e:
        logger.warning(
            "Valid pixel indices missing/corrupted at %s (error: %s). "
            "Deleting PLY folder and recomputing preprocessing.",
            valid_indices_path,
            e,
        )
        # Best-effort cleanup of stale/conflicting data, then rerun this loader
        # which will recreate PLYs + valid_pixel_indices from scratch.
        shutil.rmtree(pcl_conf_folder, ignore_errors=True)
        return load_data(
            root_path=root_path,
            num_frames=num_frames,
            stride=stride,
            device=device,
            conf_thresh_percentile=conf_thresh_percentile,
            conf_mode=conf_mode,
            conf_local_percentile=conf_local_percentile,
            conf_global_percentile=conf_global_percentile,
            voxel_size=voxel_size,
            voxel_min_count_percentile=voxel_min_count_percentile,
            conf_mask_sky=conf_mask_sky,
            conf_mask_sky_depth_band=conf_mask_sky_depth_band,
            conf_sky_depth_band_percent=conf_sky_depth_band_percent,
            conf_mask_depth_edges=conf_mask_depth_edges,
            conf_edge_rtol=conf_edge_rtol,
            conf_edge_atol=conf_edge_atol,
            conf_edge_kernel_size=conf_edge_kernel_size,
            conf_mask_max_depth=conf_mask_max_depth,
            conf_max_depth_rtol=conf_max_depth_rtol,
            conf_max_depth_atol=conf_max_depth_atol,
            offset=offset,
            load_original_images_and_intrinsics=load_original_images_and_intrinsics,
        )

    debug_mask_dir = os.path.join(pcl_conf_folder, "debug_masks")
    need_sky_dir = sky_mask is not None
    if conf_mask_sky and (
        not os.path.isdir(os.path.join(debug_mask_dir, "kept"))
        or (need_sky_dir and not os.path.isdir(os.path.join(debug_mask_dir, "sky")))
    ):
        valid_indices_npz = np.load(valid_indices_path)
        valid_flat_indices_all = []
        for frame_idx in range(N_total):
            key = f"frame_{frame_idx:05d}"
            valid_flat_indices_all.append(valid_indices_npz[key] if key in valid_indices_npz else np.zeros((0,), dtype=np.int64))
        _write_debug_mask_pngs(
            output_dir=debug_mask_dir,
            depth_shape=all_depth.shape,
            sky_mask=sky_mask,
            valid_flat_indices_all=valid_flat_indices_all,
        )

    logger.info("Computed valid pixel indices for %d frames", len(valid_pixel_indices))
    logger.info(
        "  Points per frame: min=%d, max=%d, avg=%.0f",
        min(len(v) for v in valid_pixel_indices),
        max(len(v) for v in valid_pixel_indices),
        float(np.mean([len(v) for v in valid_pixel_indices])),
    )

    # Optionally load original-resolution images from the preprocessing frames
    # folder (<root_path>/frames) and derive matching intrinsics by scaling the
    # DA3 prediction intrinsics based on resolution.
    orig_images_t = None
    orig_intrinsics_np = None
    if load_original_images_and_intrinsics:
        frames_dir = _find_preprocess_frames_dir(root_path)
        orig_images_t, orig_intrinsics_np = load_da3_original_images_from_folder(
            root_path=root_path,
            images_dir=frames_dir,
            num_frames=num_frames,
            stride=stride,
            device=device,
            offset=offset,
        )

    depth_conf = all_conf_filtered[indices]  # (N, H, W)
    depth_maps = all_depth[indices]  # (N, H, W)

    return (
        pcls,
        extrinsics,
        intrinsics,
        images,
        valid_pixel_indices,
        depth_conf,
        depth_maps,
        orig_images_t,
        orig_intrinsics_np,
    )


def load_da3_camera_images(
    root_path: str,
    num_frames: int = 10,
    stride: int = 1,
    device: str = "cpu",
    use_original_images_and_intrinsics: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Lightweight DA3 camera/image loading (no PLY generation).

    Returns:
        images: (N,3,H,W) float in [0,1]
        poses_c2w: (N,4,4)
        intrinsics: (N,3,3)
    """
    predictions = np.load(os.path.join(root_path, "exports", "npz", "results.npz"))
    N_total = predictions["conf"].shape[0]
    indices = np.arange(N_total)[::stride][:num_frames]

    extrinsics = predictions["extrinsics"][indices]  # w2c (N, 3, 4)
    intrinsics = predictions["intrinsics"][indices]  # (N, 3, 3)
    images = predictions["image"][indices]  # (N, H, W, 3) uint8

    if use_original_images_and_intrinsics:
        frames_dir = _find_preprocess_frames_dir(root_path)
        images_t, intrinsics_list = load_da3_original_images_from_folder(
            root_path=root_path,
            images_dir=frames_dir,
            num_frames=num_frames,
            stride=stride,
            device=device,
            offset=0,
        )
        intrinsics_t = torch.stack(
            [torch.from_numpy(K).to(device=device, dtype=torch.float32) for K in intrinsics_list],
            dim=0,
        )
    else:
        images_t = torch.from_numpy(images.astype(np.float32) / 255.0).to(device).permute(0, 3, 1, 2)
        intrinsics_t = torch.from_numpy(intrinsics).to(device).float()

    # Convert extrinsics to homogeneous and then to c2w
    N = extrinsics.shape[0]
    extrinsics_hom = np.concatenate([extrinsics, np.tile([[[0, 0, 0, 1]]], (N, 1, 1))], axis=1)
    extrinsics_t = torch.from_numpy(extrinsics_hom).to(device).float()
    poses_c2w = torch.linalg.inv(extrinsics_t)

    return images_t, poses_c2w, intrinsics_t


def load_depth_maps_da3(
    root_path: str,
    num_frames: int = 10,
    stride: int = 1,
    offset: int = 0,
    device: str = "cpu",
    conf_thresh_percentile: float = 40.0,
    conf_mask_depth_edges: bool = False,
    conf_edge_rtol: float | None = 0.03,
    conf_edge_atol: float | None = None,
    conf_edge_kernel_size: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load raw depth maps and per-pixel validity masks from a DA3 NPZ file.

    The depth values are **camera z-depth** (the z-coordinate in camera space),
    not Euclidean distance from the camera origin.  This matches the expected
    depth rendered by ``gsplat.rasterization_2dgs`` with ``render_mode="ED"``
    or ``"RGB+ED"``.

    Returns:
        depth_maps:  (N, H, W) float32 tensor of raw depth values.
        valid_masks: (N, H, W) bool tensor —
            ``True`` where ``isfinite(depth) & (depth > 0) & (conf >= thresh)``.
    """
    predictions = np.load(os.path.join(root_path, "exports", "npz", "results.npz"))
    N_total = predictions["conf"].shape[0]
    if (N_total - offset) < num_frames:
        stride = 1
    indices = np.arange(N_total)[::stride][:num_frames]

    all_depth = predictions["depth"]  # (N_total, H, W)
    all_conf = predictions["conf"]  # (N_total, H, W)
    all_conf_filtered, base_valid_mask = _apply_depth_edge_suppression(
        all_depth,
        all_conf,
        enabled=conf_mask_depth_edges,
        edge_rtol=conf_edge_rtol,
        edge_atol=conf_edge_atol,
        edge_kernel_size=conf_edge_kernel_size,
    )
    conf_thresh = np.percentile(all_conf_filtered, conf_thresh_percentile)

    depth_selected = all_depth[indices]  # (N, H, W)
    conf_selected = all_conf_filtered[indices]  # (N, H, W)
    valid_base_selected = base_valid_mask[indices]  # (N, H, W)

    valid = valid_base_selected & (conf_selected >= conf_thresh)

    depth_maps = torch.from_numpy(depth_selected.astype(np.float32)).to(device)
    valid_masks = torch.from_numpy(valid).to(device)

    logger.info(
        "Loaded depth maps: %d frames, %dx%d, %.1f%% valid pixels",
        depth_maps.shape[0],
        depth_maps.shape[1],
        depth_maps.shape[2],
        100.0 * valid.mean(),
    )

    return depth_maps, valid_masks


def load_da3_original_images_from_folder(
    root_path: str,
    images_dir: str,
    num_frames: int = 10,
    stride: int = 1,
    device: str = "cpu",
    offset: int = 0,
):
    """
    Load original-resolution images from a folder and derive matching intrinsics
    by scaling DA3 prediction intrinsics based on resolution.

    This mirrors the frame indexing logic of ``load_data_da3`` and assumes that
    the sorted image list in ``images_dir`` is aligned with the DA3 prediction
    ordering.

    Returns:
        images: (N, 3, H_orig, W_orig) float32 tensor in [0, 1] on ``device``.
        intrinsics: list of (3, 3) float32 numpy arrays (length N).
    """
    predictions = np.load(os.path.join(root_path, "exports", "npz", "results.npz"))
    N_total = predictions["conf"].shape[0]
    all_indices = np.arange(N_total)
    if (N_total - offset) < num_frames:
        stride = 1
    indices = all_indices[offset::stride][:num_frames]

    base_images = predictions["image"][indices]  # (N, H_pred, W_pred, 3) uint8
    base_intrinsics = predictions["intrinsics"][indices]  # (N, 3, 3)

    # Support common image extensions written by preprocess_video.py.
    exts = ["png", "jpg", "jpeg", "webp"]
    image_paths: list[str] = []
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(images_dir, f"*.{ext}")))
        image_paths.extend(glob.glob(os.path.join(images_dir, f"*.{ext.upper()}")))
    image_paths = sorted(set(image_paths))
    if len(image_paths) == 0:
        raise FileNotFoundError(
            f"No images found in folder: {images_dir}. "
            "Expected the folder to contain per-frame images aligned with DA3 predictions.",
        )

    max_required_index = int(indices.max())
    if len(image_paths) <= max_required_index:
        raise ValueError(
            f"Not enough images in {images_dir}: need at least {max_required_index + 1}, but found {len(image_paths)}.",
        )

    selected_paths = [image_paths[i] for i in indices]
    imgs = [load_image(p) for p in selected_paths]  # list of (3, H_orig, W_orig)
    if not imgs:
        raise RuntimeError("No images loaded from folder; this should be unreachable if checks above passed.")

    # Assume all folder images share the same resolution.
    _, H_orig, W_orig = imgs[0].shape

    images_t = torch.stack(imgs, dim=0).to(device=device, dtype=torch.float32)

    # Use DA3 prediction resolution to compute scaling factors.
    _, H_pred, W_pred, _ = base_images.shape
    sx = float(W_orig) / float(W_pred)
    sy = float(H_orig) / float(H_pred)
    if not np.isfinite(sx) or not np.isfinite(sy):
        raise ValueError(
            f"Non-finite scaling factors when comparing DA3 resolution ({H_pred}x{W_pred}) "
            f"to folder resolution ({H_orig}x{W_orig}).",
        )
    if abs(sx - sy) > 1e-3:
        logger.warning(
            "Anisotropic image scaling between DA3 predictions and folder images "
            "(sx=%.6f, sy=%.6f). Using sx for fx/cx and sy for fy/cy.",
            sx,
            sy,
        )

    intrinsics_scaled: list[np.ndarray] = []
    for i in range(base_intrinsics.shape[0]):
        K = base_intrinsics[i].astype(np.float32).copy()
        K[0, 0] *= sx
        K[0, 2] *= sx
        K[1, 1] *= sy
        K[1, 2] *= sy
        intrinsics_scaled.append(K)

    logger.info(
        "Loaded %d original images from folder %s (orig_res=%dx%d, da3_res=%dx%d, sx=%.6f, sy=%.6f)",
        images_t.shape[0],
        images_dir,
        H_orig,
        W_orig,
        H_pred,
        W_pred,
        sx,
        sy,
    )

    return images_t, intrinsics_scaled


def load_nerf_transforms_json(
    transforms_path: str,
    device: str = "cpu",
    *,
    override_width: int | None = None,
    override_height: int | None = None,
    blender_opengl_to_opencv: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, int | None, int | None]:
    """Load poses + intrinsics from a NeRF-style transforms JSON.

    This matches the common format used by NeRF / Gaussian Splatting datasets:
    - top-level ``fl_x``, ``fl_y``, ``cx``, ``cy`` (or ``camera_angle_x`` + ``w``)
    - top-level optional ``w``, ``h``
    - ``frames`` list with per-frame ``transform_matrix`` (camera-to-world, 4x4)

    Args:
        transforms_path: Path to JSON (e.g. ``.../gs_video/0000_extend_transforms.json``).
        device: Torch device for returned tensors.
        override_width/override_height: If set, returned intrinsics are rescaled
            from the JSON's (w,h) to this target resolution.
        blender_opengl_to_opencv: If True, convert the pose convention used by
            Blender/OpenGL-style NeRF datasets to an OpenCV-style camera frame.
            This flips the camera Y and Z axes in camera coordinates:
            ``c2w_opencv = c2w_blender @ diag([1, -1, -1, 1])``.

    Returns:
        poses_c2w: (N, 4, 4) float32
        intrinsics: (N, 3, 3) float32
        width: width used for intrinsics scaling (may be None)
        height: height used for intrinsics scaling (may be None)
    """
    with open(transforms_path, "r") as f:
        data = json.load(f)

    w_json = data.get("w", None)
    h_json = data.get("h", None)
    # Optional global scale applied to camera translations.
    # Many NeRF-style pipelines store poses in a normalised unit cube and
    # record the scene scale separately.
    scale_factor = float(data.get("scale_factor", 1.0))

    # Intrinsics: prefer explicit focal lengths, fallback to camera_angle_x.
    if "fl_x" in data and "fl_y" in data:
        fl_x = float(data["fl_x"])
        fl_y = float(data["fl_y"])
    elif "camera_angle_x" in data and w_json is not None:
        # fl_x = 0.5*w / tan(0.5*angle_x). Assume square-ish pixels when fl_y missing.
        angle_x = float(data["camera_angle_x"])
        fl_x = 0.5 * float(w_json) / float(np.tan(0.5 * angle_x))
        fl_y = fl_x
    else:
        raise ValueError(f"Unsupported transforms format at {transforms_path}: need fl_x/fl_y or camera_angle_x+w")

    if "cx" in data and "cy" in data:
        cx = float(data["cx"])
        cy = float(data["cy"])
    elif w_json is not None and h_json is not None:
        cx = float(w_json) * 0.5
        cy = float(h_json) * 0.5
    else:
        raise ValueError(f"Unsupported transforms format at {transforms_path}: need cx/cy or w/h")

    # Rescale intrinsics if overriding resolution.
    sx = 1.0 if cx > 1 else w_json
    sy = 1.0 if cy > 1 else h_json
    out_w = w_json
    out_h = h_json
    if override_width is not None or override_height is not None:
        if w_json is None or h_json is None:
            raise ValueError("override_width/override_height require the JSON to contain w and h")
        out_w = int(override_width if override_width is not None else w_json)
        out_h = int(override_height if override_height is not None else h_json)
        sx = float(out_w) / float(w_json)
        sy = float(out_h) / float(h_json)

    K_single = np.array(
        [
            [fl_x * sx, 0.0, cx * sx],
            [0.0, fl_y * sy, cy * sy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    frames = data.get("frames", None)
    if not isinstance(frames, list) or len(frames) == 0:
        raise ValueError(f"No frames found in transforms file: {transforms_path}")

    poses = []
    for fr in frames:
        tm = fr.get("transform_matrix", None)
        if tm is None:
            raise ValueError(f"Frame missing transform_matrix in {transforms_path}")
        c2w = np.array(tm, dtype=np.float32).reshape(4, 4)
        # Apply global scale to translation (keep rotation intact).
        c2w[:3, 3] *= scale_factor
        if blender_opengl_to_opencv:
            # Convert camera coordinate basis: x stays, y/z flip.
            # Right-multiply because c2w maps from camera coords to world coords.
            c2w = c2w @ np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
        poses.append(c2w)

    poses_c2w = torch.from_numpy(np.stack(poses, axis=0)).to(device=device, dtype=torch.float32)
    intrinsics = torch.from_numpy(np.stack([K_single] * poses_c2w.shape[0], axis=0)).to(
        device=device, dtype=torch.float32
    )

    return (
        poses_c2w,
        intrinsics,
        (int(out_w) if out_w is not None else None),
        (int(out_h) if out_h is not None else None),
    )


def load_worldexplorer_camera_path_json(
    camera_path_json: str,
    device: str = "cpu",
    *,
    height: int = 1080,
    override_width: int | None = None,
    override_height: int | None = None,
    blender_opengl_to_opencv: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Load poses + intrinsics from a WorldExplorer-style camera path JSON.

    Format (as in scripts/worldexplorer/downscale_camera_path_json.py):
      - keyframes[*].matrix         – 4×4 camera-to-world (fallback if no camera_path)
      - camera_path[*].camera_to_world – 4×4 camera-to-world (preferred for trajectory)
      - default_fov: float           – horizontal field of view in degrees (e.g. 75.0)
      - aspect: float                – width/height (e.g. 1.7777 for 16:9)

    Intrinsics K are built from default_fov and aspect:
      width = height * aspect (or overridden)
      fx = fy = 0.5 * width / tan(0.5 * fov_rad)
      cx = width/2, cy = height/2

    Returns:
        poses_c2w: (N, 4, 4) float32
        intrinsics: (N, 3, 3) float32 (same K for all frames)
        width, height: int
    """
    import math

    with open(camera_path_json, "r") as f:
        data = json.load(f)

    default_fov = float(data.get("default_fov", 75.0))
    aspect = float(data.get("aspect", 1.777777))  # 16/9

    # Resolution (ceil so e.g. aspect 1.777777 + height 1080 → 1920, not 1919)
    h = int(override_height) if override_height is not None else height
    w = int(override_width) if override_width is not None else int(math.ceil(h * aspect))
    w = max(1, w)
    h = max(1, h)

    # Intrinsics from FOV (assume horizontal FOV) and aspect
    fov_rad = math.radians(default_fov)
    fx = 0.5 * float(w) / math.tan(0.5 * fov_rad)
    fy = fx
    cx = 0.5 * float(w)
    cy = 0.5 * float(h)

    K_single = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    poses = []
    camera_path = data.get("camera_path", None)
    if isinstance(camera_path, list) and len(camera_path) > 0:
        for item in camera_path:
            mat = item.get("camera_to_world", None)
            if mat is None:
                continue
            c2w = np.array(mat, dtype=np.float32).reshape(4, 4)
            poses.append(c2w)
    keyframes = data.get("keyframes", None)
    if (not poses) and isinstance(keyframes, list) and len(keyframes) > 0:
        for item in keyframes:
            mat = item.get("matrix", None)
            if mat is None:
                continue
            c2w = np.array(mat, dtype=np.float32).reshape(4, 4)
            poses.append(c2w)

    if not poses:
        raise ValueError(
            f"No poses found in {camera_path_json}: need camera_path[*].camera_to_world or keyframes[*].matrix"
        )

    if blender_opengl_to_opencv:
        basis = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        poses = [c2w @ basis for c2w in poses]

    poses_c2w = torch.from_numpy(np.stack(poses, axis=0)).to(device=device, dtype=torch.float32)
    intrinsics = torch.from_numpy(np.stack([K_single] * poses_c2w.shape[0], axis=0)).to(
        device=device, dtype=torch.float32
    )

    return poses_c2w, intrinsics, int(w), int(h)
