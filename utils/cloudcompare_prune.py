from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import torch

from configs.common import AlignmentDataConfig
from data.data_loading import load_data
from utils.stage1_preparation import load_stage0_prep_alignment, prepare_stage1_inputs


CC_SOURCE_BEFORE = "before_non_rigid_icp"
CC_SOURCE_STAGE1 = "after_non_rigid_icp"
CC_SOURCE_STAGE2 = "after_global_optimization"
CC_SOURCES = {CC_SOURCE_BEFORE, CC_SOURCE_STAGE1, CC_SOURCE_STAGE2}


def _safe_suffix(value: str | None) -> str:
    import re

    text = str(value or "ccpruned").strip()
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    return safe or "ccpruned"


def _unique_child_dir(parent: Path, name: str) -> Path:
    candidate = parent / name
    if not candidate.exists():
        return candidate
    for idx in range(1, 10000):
        candidate = parent / f"{name}_{idx:03d}"
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"Could not find an available run directory under {parent}")


def _read_pcd(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    pcd = o3d.io.read_point_cloud(str(path))
    points = np.asarray(pcd.points, dtype=np.float64)
    colors = np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() else None
    return points, colors


def _write_pcd(path: Path, points: np.ndarray, colors: np.ndarray | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64).clip(0.0, 1.0))
    o3d.io.write_point_cloud(str(path), pcd)


def _normalize_segments(raw_segments: Any) -> list[tuple[int, int]]:
    if isinstance(raw_segments, torch.Tensor):
        raw_segments = raw_segments.detach().cpu().tolist()
    return [(int(seg[0]), int(seg[1])) for seg in raw_segments]


def _load_segments(path: Path) -> list[tuple[int, int]]:
    if not path.exists():
        raise FileNotFoundError(f"model_frame_segments.pt not found: {path}")
    return _normalize_segments(torch.load(path, map_location="cpu", weights_only=False))


def _load_model_valid_pixel_indices(path: Path) -> list[torch.Tensor] | None:
    if not path.exists():
        return None
    raw = torch.load(path, map_location="cpu", weights_only=False)
    return [torch.as_tensor(v, dtype=torch.long) for v in raw]


def _alignment_from_run(scene_root: Path, run: str) -> AlignmentDataConfig:
    run_dir = scene_root / run
    alignment = load_stage0_prep_alignment(str(run_dir))
    if alignment is not None:
        return alignment

    config_path = run_dir / CC_SOURCE_STAGE1 / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"No stage0_prep_config.json or after_non_rigid_icp/config.json found for run: {run_dir}"
        )
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    alignment_payload = payload.get("alignment")
    if not isinstance(alignment_payload, dict):
        raise ValueError(f"Stage 1 config has no alignment object: {config_path}")
    defaults = asdict(AlignmentDataConfig())
    defaults.update(alignment_payload)
    return AlignmentDataConfig(**defaults)


def _load_valid_pixel_indices(scene_root: Path, alignment: AlignmentDataConfig) -> tuple[list[int], list[torch.Tensor]]:
    predictions = np.load(scene_root / "exports" / "npz" / "results.npz")
    n_total = int(predictions["conf"].shape[0])
    stride = int(alignment.stride)
    if (n_total - int(alignment.offset)) < int(alignment.num_frames):
        stride = 1
    selected_indices = np.arange(n_total)[int(alignment.offset) :: stride][: int(alignment.num_frames)]

    (_, _, _, _, valid_pixel_indices, _, _, _, _) = load_data(
        str(scene_root),
        alignment.num_frames,
        alignment.stride,
        "cpu",
        alignment.conf_thresh_percentile,
        conf_mode=alignment.conf_mode,
        conf_local_percentile=alignment.conf_local_percentile,
        conf_global_percentile=alignment.conf_global_percentile,
        voxel_size=alignment.conf_voxel_size,
        voxel_min_count_percentile=alignment.conf_voxel_min_count_percentile,
        conf_mask_sky=alignment.conf_mask_sky,
        conf_mask_sky_depth_band=alignment.conf_mask_sky_depth_band,
        conf_sky_depth_band_percent=alignment.conf_sky_depth_band_percent,
        conf_mask_min_depth_range_percent=alignment.conf_mask_min_depth_range_percent,
        conf_min_depth_range_percent=alignment.conf_min_depth_range_percent,
        conf_mask_min_depth_range_meters=alignment.conf_mask_min_depth_range_meters,
        conf_min_depth_range_meters=alignment.conf_min_depth_range_meters,
        conf_mask_depth_edges=alignment.conf_mask_depth_edges,
        conf_edge_rtol=alignment.conf_edge_rtol,
        conf_edge_atol=alignment.conf_edge_atol,
        conf_edge_kernel_size=alignment.conf_edge_kernel_size,
        conf_mask_max_depth=alignment.conf_mask_max_depth,
        conf_max_depth_rtol=alignment.conf_max_depth_rtol,
        conf_max_depth_atol=alignment.conf_max_depth_atol,
        conf_mask_white_background=alignment.conf_mask_white_background,
        conf_white_bg_min_rgb=alignment.conf_white_bg_min_rgb,
        conf_white_bg_max_channel_delta=alignment.conf_white_bg_max_channel_delta,
        conf_white_bg_grow_px=alignment.conf_white_bg_grow_px,
        manual_valid_indices_path=alignment.manual_valid_indices_path,
        offset=alignment.offset,
        write_ply_cache=False,
        load_point_clouds=False,
        write_debug_masks=False,
    )
    return [int(i) for i in selected_indices], valid_pixel_indices


def _frame_ids_and_pixel_ids_from_segments(
    point_count: int,
    segments: list[tuple[int, int]],
    pixel_indices: list[torch.Tensor] | None,
) -> tuple[np.ndarray, np.ndarray]:
    frame_ids = np.full(point_count, -1, dtype=np.int32)
    pixel_ids = np.full(point_count, -1, dtype=np.int32)
    for frame_idx, (start, end) in enumerate(segments):
        if start < 0 or end < start or end > point_count:
            raise ValueError(f"Invalid frame segment {frame_idx}: {(start, end)} for {point_count} points")
        frame_ids[start:end] = int(frame_idx)
        if pixel_indices is not None and frame_idx < len(pixel_indices):
            pix = pixel_indices[frame_idx].detach().cpu().numpy().astype(np.int64, copy=False)
            if pix.shape[0] == end - start:
                pixel_ids[start:end] = pix.astype(np.int32, copy=False)
    return frame_ids, pixel_ids


def _write_cloudcompare_ply(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray | None,
    frame_ids: np.ndarray,
    pixel_ids: np.ndarray,
) -> None:
    point_count = int(points.shape[0])
    if point_count >= np.iinfo(np.int32).max:
        raise ValueError("CloudCompare edit export only supports fewer than 2^31 points.")

    rgb = np.zeros((point_count, 3), dtype=np.uint8)
    if colors is not None and colors.shape[0] == point_count:
        rgb = np.rint(np.asarray(colors).clip(0.0, 1.0) * 255.0).astype(np.uint8)

    arr = np.empty(
        point_count,
        dtype=[
            ("x", "<f8"),
            ("y", "<f8"),
            ("z", "<f8"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("v2w_point_id", "<i4"),
            ("v2w_frame_id", "<i4"),
            ("v2w_pixel_index", "<i4"),
        ],
    )
    arr["x"] = points[:, 0]
    arr["y"] = points[:, 1]
    arr["z"] = points[:, 2]
    arr["red"] = rgb[:, 0]
    arr["green"] = rgb[:, 1]
    arr["blue"] = rgb[:, 2]
    arr["v2w_point_id"] = np.arange(point_count, dtype=np.int32)
    arr["v2w_frame_id"] = frame_ids.astype(np.int32, copy=False)
    arr["v2w_pixel_index"] = pixel_ids.astype(np.int32, copy=False)

    path.parent.mkdir(parents=True, exist_ok=True)
    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            "comment video_to_world CloudCompare deletion edit file",
            "comment Delete points only; transforms/resampling are ignored on import.",
            f"element vertex {point_count}",
            "property double x",
            "property double y",
            "property double z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "property int v2w_point_id",
            "property int v2w_frame_id",
            "property int v2w_pixel_index",
            "end_header",
            "",
        ]
    ).encode("ascii")
    with open(path, "wb") as f:
        f.write(header)
        arr.tofile(f)


_PLY_TYPE_TO_DTYPE = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


def _read_ply_vertex_properties(path: Path) -> dict[str, np.ndarray]:
    with open(path, "rb") as f:
        first = f.readline().decode("ascii", errors="replace").strip()
        if first != "ply":
            raise ValueError(f"Not a PLY file: {path}")

        ply_format = None
        vertex_count = None
        vertex_props: list[tuple[str, str]] = []
        current_element = None

        while True:
            raw = f.readline()
            if not raw:
                raise ValueError(f"Unexpected EOF while reading PLY header: {path}")
            line = raw.decode("ascii", errors="replace").strip()
            if line == "end_header":
                break
            if not line or line.startswith("comment"):
                continue
            parts = line.split()
            if parts[0] == "format":
                ply_format = parts[1]
            elif parts[0] == "element":
                current_element = parts[1]
                if current_element == "vertex":
                    vertex_count = int(parts[2])
            elif parts[0] == "property" and current_element == "vertex":
                if parts[1] == "list":
                    raise ValueError("List properties on PLY vertices are not supported.")
                vertex_props.append((parts[2], parts[1]))

        if ply_format is None or vertex_count is None:
            raise ValueError(f"PLY header is missing format or vertex count: {path}")

        if ply_format == "ascii":
            columns = {name: [] for name, _type_name in vertex_props}
            for _ in range(vertex_count):
                values = f.readline().decode("ascii", errors="replace").strip().split()
                if len(values) < len(vertex_props):
                    raise ValueError(f"Malformed ASCII PLY vertex row in {path}")
                for value, (name, type_name) in zip(values, vertex_props):
                    if _PLY_TYPE_TO_DTYPE[type_name] in {"f4", "f8"}:
                        columns[name].append(float(value))
                    else:
                        columns[name].append(int(float(value)))
            return {name: np.asarray(values) for name, values in columns.items()}

        if ply_format not in {"binary_little_endian", "binary_big_endian"}:
            raise ValueError(f"Unsupported PLY format '{ply_format}' in {path}")

        endian = "<" if ply_format == "binary_little_endian" else ">"
        dtype_fields = []
        for name, type_name in vertex_props:
            base = _PLY_TYPE_TO_DTYPE.get(type_name)
            if base is None:
                raise ValueError(f"Unsupported PLY property type '{type_name}' in {path}")
            dtype_fields.append((name, base if base in {"i1", "u1"} else endian + base))
        vertex_dtype = np.dtype(dtype_fields)
        data = np.fromfile(f, dtype=vertex_dtype, count=vertex_count)
        return {name: data[name] for name, _type_name in vertex_props}


def _point_id_property(properties: dict[str, np.ndarray]) -> str | None:
    normalized = {name.lower(): name for name in properties}
    for candidate in ("v2w_point_id", "point_id", "scalar_v2w_point_id"):
        if candidate in normalized:
            return normalized[candidate]
    for name in properties:
        lowered = name.lower()
        if lowered.endswith("v2w_point_id") or lowered.endswith("point_id"):
            return name
    return None


def _kept_point_mask_from_edit(
    edited_ply: Path,
    point_count: int,
    *,
    source_points: np.ndarray | None = None,
    match_tolerance: float | None = None,
) -> np.ndarray:
    properties = _read_ply_vertex_properties(edited_ply)
    point_id_name = _point_id_property(properties)
    if point_id_name is not None:
        kept_ids = np.rint(np.asarray(properties[point_id_name], dtype=np.float64)).astype(np.int64)
        if kept_ids.size == 0:
            return np.zeros(point_count, dtype=bool)
        if int(kept_ids.min()) < 0 or int(kept_ids.max()) >= point_count:
            raise ValueError(
                f"Edited PLY point IDs are outside the source range 0..{point_count - 1}: {edited_ply}"
            )
        keep = np.zeros(point_count, dtype=bool)
        keep[kept_ids] = True
        return keep

    if source_points is None:
        raise ValueError(
            "Edited PLY has no `v2w_point_id` property and no source points were provided for geometry matching."
        )

    missing_xyz = [name for name in ("x", "y", "z") if name not in properties]
    if missing_xyz:
        raise ValueError(
            "Edited PLY has no `v2w_point_id` property and is missing coordinate properties "
            f"{missing_xyz}, so it cannot be matched back to the source cloud."
        )

    edited_points = np.column_stack(
        [
            np.asarray(properties["x"], dtype=np.float64),
            np.asarray(properties["y"], dtype=np.float64),
            np.asarray(properties["z"], dtype=np.float64),
        ]
    )
    if edited_points.shape[0] == 0:
        return np.zeros(point_count, dtype=bool)

    source_points = np.asarray(source_points, dtype=np.float64)
    if source_points.shape != (point_count, 3):
        raise ValueError(
            f"Source point array has shape {source_points.shape}, expected ({point_count}, 3)."
        )

    finite_source = np.isfinite(source_points).all(axis=1)
    finite_edited = np.isfinite(edited_points).all(axis=1)
    if not finite_edited.all():
        edited_points = edited_points[finite_edited]
    if edited_points.shape[0] == 0:
        return np.zeros(point_count, dtype=bool)

    from scipy.spatial import cKDTree

    finite_source_indices = np.flatnonzero(finite_source)
    tree = cKDTree(source_points[finite_source])
    distances, nearest = tree.query(edited_points, k=1, workers=-1)

    if match_tolerance is None:
        source_min = np.nanmin(source_points[finite_source], axis=0)
        source_max = np.nanmax(source_points[finite_source], axis=0)
        diagonal = float(np.linalg.norm(source_max - source_min))
        match_tolerance = max(1e-4, diagonal * 1e-6)

    matched = np.isfinite(distances) & (distances <= float(match_tolerance))
    if not np.any(matched):
        raise ValueError(
            "Edited PLY has no `v2w_point_id` property, and geometry matching found no source points "
            f"within tolerance {match_tolerance:g}. Make sure the CloudCompare edit only deletes points."
        )

    unmatched = int((~matched).sum())
    if unmatched:
        unmatched_ratio = unmatched / max(1, int(edited_points.shape[0]))
        if unmatched_ratio > 0.01:
            raise ValueError(
                "Edited PLY has no `v2w_point_id` property, and too many points failed geometry matching: "
                f"{unmatched:,}/{edited_points.shape[0]:,} unmatched at tolerance {match_tolerance:g}. "
                "Make sure the CloudCompare edit only deletes points and does not transform or resample the cloud."
            )

    keep = np.zeros(point_count, dtype=bool)
    keep[finite_source_indices[nearest[matched]]] = True
    return keep


def export_cloudcompare_edit_ply(
    scene_root: str | Path,
    run: str,
    source: str,
    output_filename: str | None = None,
) -> dict[str, Any]:
    scene_root_path = Path(scene_root).resolve()
    if source not in CC_SOURCES:
        raise ValueError(f"Unknown CloudCompare source '{source}'.")
    run_dir = scene_root_path / run
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    if source == CC_SOURCE_BEFORE:
        source_ply = run_dir / "before_non_rigid_icp.ply"
        if not source_ply.exists():
            raise FileNotFoundError(f"before_non_rigid_icp.ply not found: {source_ply}")
        points, colors = _read_pcd(source_ply)
        alignment = _alignment_from_run(scene_root_path, run)
        _selected_frame_indices, valid_pixel_indices = _load_valid_pixel_indices(scene_root_path, alignment)
        segments: list[tuple[int, int]] = []
        cursor = 0
        for indices in valid_pixel_indices:
            count = int(indices.numel())
            segments.append((cursor, cursor + count))
            cursor += count
        if cursor != points.shape[0]:
            raise ValueError(
                f"before_non_rigid_icp.ply has {points.shape[0]} points, but the run's valid-pixel lists "
                f"describe {cursor} points. Recreate Stage 0 prep before exporting an edit PLY."
            )
        frame_ids, pixel_ids = _frame_ids_and_pixel_ids_from_segments(points.shape[0], segments, valid_pixel_indices)
        default_name = "cloudcompare_before_non_rigid_edit.ply"
        output_dir = run_dir
    else:
        checkpoint_dir = run_dir / source
        source_ply = checkpoint_dir / "aligned_points.ply"
        if not source_ply.exists():
            raise FileNotFoundError(f"aligned_points.ply not found: {source_ply}")
        points, colors = _read_pcd(source_ply)
        segments = _load_segments(checkpoint_dir / "model_frame_segments.pt")
        valid_pixel_indices = _load_model_valid_pixel_indices(checkpoint_dir / "model_valid_pixel_indices_list.pt")
        frame_ids, pixel_ids = _frame_ids_and_pixel_ids_from_segments(points.shape[0], segments, valid_pixel_indices)
        default_name = f"cloudcompare_{source}_edit.ply"
        output_dir = checkpoint_dir

    filename = str(output_filename or default_name).strip() or default_name
    if not filename.lower().endswith(".ply"):
        filename += ".ply"
    output_path = output_dir / filename
    _write_cloudcompare_ply(output_path, points, colors, frame_ids, pixel_ids)
    return {
        "output_path": str(output_path),
        "source_path": str(source_ply),
        "source": source,
        "points": int(points.shape[0]),
    }


def _copy_file_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _copy_checkpoint_dir(src: Path, dst: Path) -> None:
    if dst.exists():
        raise FileExistsError(f"Destination checkpoint already exists: {dst}")
    shutil.copytree(src, dst)


def _apply_aligned_prune(
    scene_root_path: Path,
    run: str,
    source: str,
    edited_ply: Path,
    output_suffix: str | None,
) -> dict[str, Any]:
    run_dir = scene_root_path / run
    checkpoint_dir = run_dir / source
    source_ply = checkpoint_dir / "aligned_points.ply"
    points, colors = _read_pcd(source_ply)
    point_count = int(points.shape[0])
    keep = _kept_point_mask_from_edit(edited_ply, point_count, source_points=points)
    segments = _load_segments(checkpoint_dir / "model_frame_segments.pt")
    model_valid_pixel_indices = _load_model_valid_pixel_indices(checkpoint_dir / "model_valid_pixel_indices_list.pt")

    new_segments: list[tuple[int, int]] = []
    new_valid_pixel_indices: list[torch.Tensor] | None = [] if model_valid_pixel_indices is not None else None
    cursor = 0
    for frame_idx, (start, end) in enumerate(segments):
        frame_keep = keep[start:end]
        kept_count = int(frame_keep.sum())
        new_segments.append((cursor, cursor + kept_count))
        cursor += kept_count
        if new_valid_pixel_indices is not None:
            src_indices = model_valid_pixel_indices[frame_idx]
            if int(src_indices.numel()) == end - start:
                new_valid_pixel_indices.append(src_indices[torch.from_numpy(frame_keep)])
            else:
                new_valid_pixel_indices = None

    suffix = _safe_suffix(output_suffix)
    new_run_dir = _unique_child_dir(scene_root_path, f"{run}_{suffix}")
    new_run_dir.mkdir(parents=True)

    _copy_file_if_exists(run_dir / "stage0_prep_config.json", new_run_dir / "stage0_prep_config.json")
    _copy_file_if_exists(run_dir / "before_non_rigid_icp.ply", new_run_dir / "before_non_rigid_icp.ply")

    if source == CC_SOURCE_STAGE2:
        if (run_dir / CC_SOURCE_STAGE1).exists():
            _copy_checkpoint_dir(run_dir / CC_SOURCE_STAGE1, new_run_dir / CC_SOURCE_STAGE1)
        _copy_checkpoint_dir(checkpoint_dir, new_run_dir / source)
    else:
        _copy_checkpoint_dir(checkpoint_dir, new_run_dir / source)

    dst_checkpoint_dir = new_run_dir / source
    _write_pcd(dst_checkpoint_dir / "aligned_points.ply", points[keep], None if colors is None else colors[keep])
    torch.save(new_segments, dst_checkpoint_dir / "model_frame_segments.pt")
    if new_valid_pixel_indices is not None:
        torch.save([idx.cpu() for idx in new_valid_pixel_indices], dst_checkpoint_dir / "model_valid_pixel_indices_list.pt")

    manifest = {
        "type": "aligned_points_prune",
        "source_run": run,
        "source_checkpoint": source,
        "edited_ply": str(edited_ply),
        "source_point_count": point_count,
        "kept_point_count": int(keep.sum()),
        "removed_point_count": int(point_count - keep.sum()),
    }
    (new_run_dir / "cloudcompare_prune_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        **manifest,
        "new_run": new_run_dir.name,
        "new_run_dir": str(new_run_dir),
        "new_checkpoint_dir": str(dst_checkpoint_dir),
    }


def _apply_before_prune(
    scene_root_path: Path,
    run: str,
    edited_ply: Path,
    output_suffix: str | None,
) -> dict[str, Any]:
    run_dir = scene_root_path / run
    source_ply = run_dir / "before_non_rigid_icp.ply"
    points, _colors = _read_pcd(source_ply)
    point_count = int(points.shape[0])
    keep = _kept_point_mask_from_edit(edited_ply, point_count, source_points=points)
    alignment = _alignment_from_run(scene_root_path, run)
    selected_frame_indices, valid_pixel_indices = _load_valid_pixel_indices(scene_root_path, alignment)

    manual_payload: dict[str, np.ndarray] = {}
    cursor = 0
    for da3_frame_idx, frame_valid_indices in zip(selected_frame_indices, valid_pixel_indices):
        count = int(frame_valid_indices.numel())
        frame_keep = keep[cursor : cursor + count]
        manual_payload[f"frame_{int(da3_frame_idx):05d}"] = (
            frame_valid_indices.detach().cpu().numpy().astype(np.int64, copy=False)[frame_keep]
        )
        cursor += count
    if cursor != point_count:
        raise ValueError(
            f"before_non_rigid_icp.ply has {point_count} points, but the run's valid-pixel lists "
            f"describe {cursor} points."
        )

    suffix = _safe_suffix(output_suffix)
    new_run_dir = _unique_child_dir(scene_root_path, f"{run}_{suffix}")
    new_run_dir.mkdir(parents=True)
    manual_dir = new_run_dir / "cloudcompare_prune"
    manual_dir.mkdir(parents=True, exist_ok=True)
    manual_path = manual_dir / "manual_valid_pixel_indices.npz"
    np.savez_compressed(manual_path, **manual_payload)

    pruned_alignment = replace(alignment, manual_valid_indices_path=str(manual_path))
    prepare_stage1_inputs(
        root_path=str(scene_root_path),
        alignment=pruned_alignment,
        out_path=str(new_run_dir),
        overwrite_before_non_rigid=True,
        write_before_non_rigid=True,
        write_debug_masks=False,
    )

    manifest = {
        "type": "before_non_rigid_prune",
        "source_run": run,
        "source_checkpoint": CC_SOURCE_BEFORE,
        "edited_ply": str(edited_ply),
        "manual_valid_indices_path": str(manual_path),
        "source_point_count": point_count,
        "kept_point_count": int(keep.sum()),
        "removed_point_count": int(point_count - keep.sum()),
    }
    (new_run_dir / "cloudcompare_prune_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        **manifest,
        "new_run": new_run_dir.name,
        "new_run_dir": str(new_run_dir),
        "new_before_non_rigid": str(new_run_dir / "before_non_rigid_icp.ply"),
    }


def apply_cloudcompare_edit_ply(
    scene_root: str | Path,
    run: str,
    source: str,
    edited_ply: str | Path,
    output_suffix: str | None = None,
) -> dict[str, Any]:
    scene_root_path = Path(scene_root).resolve()
    if source not in CC_SOURCES:
        raise ValueError(f"Unknown CloudCompare source '{source}'.")
    edited_ply_path = Path(edited_ply).resolve()
    if not edited_ply_path.is_file():
        raise FileNotFoundError(f"Edited PLY not found: {edited_ply_path}")
    if source == CC_SOURCE_BEFORE:
        return _apply_before_prune(scene_root_path, run, edited_ply_path, output_suffix)
    return _apply_aligned_prune(scene_root_path, run, source, edited_ply_path, output_suffix)
