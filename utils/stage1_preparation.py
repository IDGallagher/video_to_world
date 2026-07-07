from __future__ import annotations

import json
import os
from dataclasses import asdict

import open3d as o3d

from configs.common import AlignmentDataConfig
from data.data_loading import load_data
from utils.logging import get_logger
from utils.pointcloud import merge_point_clouds


logger = get_logger(__name__)
STAGE0_PREP_CONFIG_FILENAME = "stage0_prep_config.json"


def resolve_stage1_out_path(
    root_path: str,
    alignment: AlignmentDataConfig,
    *,
    out_path: str | None = None,
    out_suffix: str = "",
) -> str:
    if out_path is not None and str(out_path).strip():
        return os.path.abspath(out_path)
    return os.path.join(
        os.path.abspath(root_path),
        f"frame_to_model_icp_{alignment.num_frames}_{alignment.stride}_offset{alignment.offset}{out_suffix}",
    )


def _stage0_prep_config_path(out_path: str) -> str:
    return os.path.join(os.path.abspath(out_path), STAGE0_PREP_CONFIG_FILENAME)


def write_stage0_prep_config(*, root_path: str, out_path: str, alignment: AlignmentDataConfig) -> str:
    config_path = _stage0_prep_config_path(out_path)
    payload = {
        "root_path": os.path.abspath(root_path),
        "alignment": asdict(alignment),
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Persisted Stage 0 prep config to %s", config_path)
    return config_path


def load_stage0_prep_alignment(out_path: str) -> AlignmentDataConfig | None:
    config_path = _stage0_prep_config_path(out_path)
    if not os.path.exists(config_path):
        return None

    with open(config_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    alignment_payload = payload.get("alignment", payload)
    if not isinstance(alignment_payload, dict):
        raise ValueError(f"Invalid Stage 0 prep config at '{config_path}': missing alignment object.")

    defaults = asdict(AlignmentDataConfig())
    defaults.update(
        {
            "conf_mask_min_depth_range_percent": False,
            "conf_min_depth_range_percent": 50.0,
            "conf_mask_min_depth_range_meters": False,
            "conf_min_depth_range_meters": 3.0,
        }
    )
    resolved_alignment = {
        key: alignment_payload.get(key, default_value)
        for key, default_value in defaults.items()
    }
    return AlignmentDataConfig(**resolved_alignment)


def prepare_stage1_inputs(
    *,
    root_path: str,
    alignment: AlignmentDataConfig,
    out_path: str | None = None,
    out_suffix: str = "",
    device: str = "cpu",
    overwrite_before_non_rigid: bool = False,
    write_before_non_rigid: bool = True,
    write_debug_masks: bool = True,
) -> tuple[str, str | None]:
    """Materialize Stage 1's pre-ICP inputs from Stage 0 outputs.

    This ensures the filtered `exports/ply/...` cache exists for the selected
    alignment/filter configuration, optionally writes the merged
    `before_non_rigid_icp.ply` file into the Stage 1 run directory, and
    persists the prep config alongside that run for Stage 1 reuse.
    """

    resolved_root = os.path.abspath(root_path)
    resolved_out_path = resolve_stage1_out_path(
        resolved_root,
        alignment,
        out_path=out_path,
        out_suffix=out_suffix,
    )
    os.makedirs(resolved_out_path, exist_ok=True)

    pcls, *_ = load_data(
        resolved_root,
        alignment.num_frames,
        alignment.stride,
        device,
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
        write_ply_cache=write_before_non_rigid,
        load_point_clouds=write_before_non_rigid,
        write_debug_masks=write_debug_masks,
    )

    before_non_rigid_path = None
    if write_before_non_rigid:
        before_non_rigid_path = os.path.join(resolved_out_path, "before_non_rigid_icp.ply")
        if overwrite_before_non_rigid or not os.path.exists(before_non_rigid_path):
            merged = merge_point_clouds(pcls)
            o3d.io.write_point_cloud(before_non_rigid_path, merged)
            logger.info("Prepared pre-ICP merged point cloud at %s", before_non_rigid_path)
        else:
            logger.info("Reusing existing pre-ICP merged point cloud at %s", before_non_rigid_path)
    else:
        logger.info("Skipped pre-ICP merged point cloud for %s", resolved_out_path)

    write_stage0_prep_config(
        root_path=resolved_root,
        out_path=resolved_out_path,
        alignment=alignment,
    )

    return resolved_out_path, before_non_rigid_path
