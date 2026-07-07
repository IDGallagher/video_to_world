from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class TensorboardConfig:
    tensorboard: bool = True
    tensorboard_log_dir: Optional[str] = None


@dataclass
class KnnBackendConfig:
    knn_backend: str = "cpu_kdtree"


@dataclass(frozen=True)
class AlignmentDataConfig:
    """Stage-1-produced data-loading/confidence-filtering params reused downstream."""

    num_frames: int = 50
    stride: int = 2
    offset: int = 0

    conf_thresh_percentile: float = 80.0
    conf_mode: str = "voxel_or"
    conf_global_percentile: Optional[float] = 10.0
    conf_local_percentile: Optional[float] = 10.0
    conf_voxel_size: float = 1.0
    conf_voxel_min_count_percentile: Optional[float] = 50.0
    conf_mask_sky: bool = False
    conf_mask_sky_depth_band: bool = False
    conf_sky_depth_band_percent: float = 2.0
    conf_mask_min_depth_range_percent: bool = True
    conf_min_depth_range_percent: float = 50.0
    conf_mask_min_depth_range_meters: bool = False
    conf_min_depth_range_meters: float = 3.0
    conf_mask_depth_edges: bool = True
    conf_edge_rtol: Optional[float] = 0.1
    conf_edge_atol: Optional[float] = None
    conf_edge_kernel_size: int = 3
    conf_mask_max_depth: bool = False
    conf_max_depth_rtol: Optional[float] = 0.001
    conf_max_depth_atol: Optional[float] = None
    conf_mask_white_background: bool = False
    conf_white_bg_min_rgb: float = 220.0
    conf_white_bg_max_channel_delta: float = 25.0
    conf_white_bg_grow_px: int = 0
    manual_valid_indices_path: Optional[str] = None
