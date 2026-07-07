from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Optional

try:
    import gradio as gr
except ImportError as exc:  # pragma: no cover - exercised at runtime.
    raise SystemExit(
        "Gradio is required for this UI. Install it with `pip install gradio` in the active environment."
    ) from exc

try:
    _GRADIO_MAJOR_VERSION = int(str(gr.__version__).split(".", 1)[0])
except (TypeError, ValueError):  # pragma: no cover - defensive runtime fallback.
    _GRADIO_MAJOR_VERSION = 0

if _GRADIO_MAJOR_VERSION >= 6:  # pragma: no cover - exercised at runtime.
    raise SystemExit(
        f"Detected gradio {gr.__version__}. This app currently requires `gradio<6` due to a frontend compatibility "
        'regression with Gradio 6. Install a 5.x release, for example `pip install "gradio<6"`.'
    )


PROJECT_ROOT = Path(__file__).resolve().parent
VIDEOS_ROOT = (PROJECT_ROOT / "videos").resolve()
RUNS_ROOT = PROJECT_ROOT / ".gradio_runs"
UPLOADS_ROOT = VIDEOS_ROOT / "_gradio_uploads"
DIVSTREAM_JOBS_ROOT = RUNS_ROOT / "divstream_jobs"
VDA_DIVSTREAM_JOBS_ROOT = RUNS_ROOT / "vda_divstream_jobs"
FLAT_DIVSTREAM_JOBS_ROOT = RUNS_ROOT / "flat_divstream_jobs"
DIVSTREAM_OUTPUTS_ROOT = RUNS_ROOT / "divstream_outputs"
PICKER_ROOT = VIDEOS_ROOT if VIDEOS_ROOT.exists() else PROJECT_ROOT.resolve()
APP_BUILD_TIME = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
APP_VERSION = APP_BUILD_TIME
WINDOWS_DRIVE_RE = re.compile(r"^(?P<drive>[a-zA-Z]):[\\/](?P<rest>.*)$")
WSL_MOUNT_RE = re.compile(r"^/mnt/(?P<drive>[a-zA-Z])/(?P<rest>.*)$")
MAX_LOG_LINES = 220
POLL_INTERVAL_SEC = 1.0
CATALOG_CACHE_TTL_SEC = 60.0
AUTO_SYNC_OUTPUT_COUNT = 28
STAGE_COMPLETION_CONTROL_OUTPUT_COUNT = 14
DEFAULT_STAGE0_MAX_FRAMES = 20
DEFAULT_STAGE0_MAX_STRIDE = 1
DEFAULT_STAGE0_STREAMING = True
DEFAULT_STAGE0_STREAMING_OVERLAP = 10
DEFAULT_STAGE0_STREAMING_GLOBAL_GUIDE = False
DEFAULT_STAGE0_REF_VIEW_STRATEGY = "first"
DEFAULT_STAGE0_RUNTIME_EXPORT_FORMAT = "none"
DIVSTREAM_PREP_NUM_FRAMES = 1
DIVSTREAM_PREP_STRIDE = 1
DIVSTREAM_PREP_OFFSET = 0
DIVSTREAM_DEBUG_PREP_NUM_FRAMES = 1_000_000_000
DEFAULT_PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
UPLOAD_SCENE_DIRNAME = "scene"
UPLOAD_FRAMES_DIRNAME = "frames"
RUNTIME_EXPORT_CHOICES = [
    ("DirectStorage Stream", "directstorage_stream"),
    ("Kinect RGBD Video (HAP Q)", "kinect_rgbd_video"),
    ("Packed Frame Sequence", "packed_frame_sequence"),
    ("Packed Frame Sequence (8-bit Depth)", "packed_frame_sequence_depth8"),
    ("None", "none"),
]
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv"}
IGNORED_VIDEO_DIR_NAMES = {"gs_video", "gs_video_eval"}
PRUNED_VIDEO_DISCOVERY_DIR_NAMES = {
    "__pycache__",
    "after_global_optimization",
    "after_non_rigid_icp",
    "debug_masks",
    "exports",
    "frames",
    "frames_subsampled",
    "gs_video",
    "gs_video_eval",
}

ACTIVE_RUNS: dict[str, subprocess.Popen] = {}
ACTIVE_RUNS_LOCK = threading.Lock()
CATALOG_CACHE: dict[str, tuple[float, list[Path]]] = {}
CATALOG_CACHE_LOCK = threading.Lock()


@dataclass
class SceneArtifacts:
    scene_root: Path
    run_dirs: list[Path]
    latest_run_dir: Optional[Path]
    da3_videos: list[Path]
    gs_videos: list[Path]
    inverse_dirs: list[Path]
    gs_dirs: list[Path]
    key_files: list[Path]
    notes: list[str]


def _ensure_workspace_dirs() -> None:
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    UPLOADS_ROOT.mkdir(parents=True, exist_ok=True)
    DIVSTREAM_JOBS_ROOT.mkdir(parents=True, exist_ok=True)
    VDA_DIVSTREAM_JOBS_ROOT.mkdir(parents=True, exist_ok=True)
    FLAT_DIVSTREAM_JOBS_ROOT.mkdir(parents=True, exist_ok=True)
    DIVSTREAM_OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)


def _clear_catalog_cache() -> None:
    with CATALOG_CACHE_LOCK:
        CATALOG_CACHE.clear()


def _cached_catalog(key: str, builder) -> list[Path]:
    now = time.time()
    with CATALOG_CACHE_LOCK:
        cached = CATALOG_CACHE.get(key)
        if cached is not None and now - cached[0] <= CATALOG_CACHE_TTL_SEC:
            return list(cached[1])

    values = builder()
    with CATALOG_CACHE_LOCK:
        CATALOG_CACHE[key] = (now, list(values))
    return list(values)


def _prepend_active_python_bin_to_path(env: Optional[dict[str, str]] = None) -> dict[str, str]:
    target_env = env if env is not None else os.environ
    python_bin = str(Path(sys.executable).resolve().parent)
    current_path = target_env.get("PATH", "")
    entries = current_path.split(os.pathsep) if current_path else []
    if python_bin not in entries:
        target_env["PATH"] = python_bin if not current_path else f"{python_bin}{os.pathsep}{current_path}"
    return target_env


def _strip_quotes(raw: object) -> str:
    if raw is None:
        return ""
    if isinstance(raw, list):
        raw = raw[0] if raw else ""
    return str(raw).strip().strip('"').strip("'")


def _normalize_path(raw: object, *, allow_missing: bool = False) -> Path:
    value = _strip_quotes(raw)
    if not value:
        raise ValueError("Path is empty.")

    candidate = Path(value).expanduser()
    search_paths = []
    if candidate.is_absolute():
        search_paths.append(candidate)
    else:
        search_paths.append((Path.cwd() / candidate).resolve())
        search_paths.append((PROJECT_ROOT / candidate).resolve())

    match = WINDOWS_DRIVE_RE.match(value)
    if match and os.name != "nt":
        converted = Path("/mnt") / match.group("drive").lower() / match.group("rest").replace("\\", "/")
        search_paths.append(converted)

    for path in search_paths:
        if path.exists():
            return path.resolve()

    if allow_missing:
        return search_paths[0]

    raise FileNotFoundError(f"Path does not exist: {value}")


def _resolve_existing_file(raw: object) -> Path:
    path = _normalize_path(raw, allow_missing=False)
    if not path.is_file():
        raise FileNotFoundError(f"Expected a file, found: {path}")
    return path


def _resolve_existing_dir(raw: object) -> Path:
    path = _normalize_path(raw, allow_missing=False)
    if not path.is_dir():
        raise FileNotFoundError(f"Expected a directory, found: {path}")
    return path


def _safe_stem(name: str) -> str:
    stem = Path(name).stem
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return safe or "scene"


def _safe_divstream_filename(raw_name: object, fallback_source: Path) -> str:
    raw = _strip_quotes(raw_name)
    stem_source = raw or fallback_source.stem
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(stem_source).stem).strip("._-")
    return f"{safe or 'depth_image_stream'}.divstream"


def _next_available_path(path: Path) -> Path:
    if not path.exists():
        return path

    parent = path.parent
    stem = path.stem
    suffix = path.suffix
    for idx in range(1, 10000):
        candidate = parent / f"{stem}_{idx:03d}{suffix}"
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"Could not find an available output filename for {path}")


def _ensure_pytorch_cuda_allocator_conf(env: dict[str, str]) -> None:
    current = str(env.get("PYTORCH_CUDA_ALLOC_CONF") or "").strip()
    if not current:
        env["PYTORCH_CUDA_ALLOC_CONF"] = DEFAULT_PYTORCH_CUDA_ALLOC_CONF
        return
    if "expandable_segments" not in current:
        env["PYTORCH_CUDA_ALLOC_CONF"] = f"{current},{DEFAULT_PYTORCH_CUDA_ALLOC_CONF}"


def _new_upload_uid() -> str:
    while True:
        candidate = uuid.uuid4().hex[:8]
        if not (UPLOADS_ROOT / candidate).exists():
            return candidate


def _is_managed_upload_path(path: Path) -> bool:
    try:
        path.resolve().relative_to(UPLOADS_ROOT.resolve())
        return True
    except ValueError:
        return False


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def _path_identity_key(raw: object) -> str:
    value = _strip_quotes(raw)
    if not value:
        return ""

    text = value.replace("\\", "/")
    wsl_match = WSL_MOUNT_RE.match(text)
    if wsl_match:
        return f"{wsl_match.group('drive').lower()}:/{wsl_match.group('rest').strip('/')}".lower()

    windows_match = WINDOWS_DRIVE_RE.match(value)
    if windows_match:
        rest = windows_match.group("rest").replace("\\", "/").strip("/")
        return f"{windows_match.group('drive').lower()}:/{rest}".lower()

    try:
        text = str(_normalize_path(value, allow_missing=True))
    except Exception:
        pass
    return text.replace("\\", "/").rstrip("/").lower()


def _default_scene_root_for_video(input_video: Path) -> Path:
    video_path = input_video.resolve()
    if _is_managed_upload_path(video_path) and video_path.parent != UPLOADS_ROOT.resolve():
        return (video_path.parent / UPLOAD_SCENE_DIRNAME).resolve()
    return video_path.with_suffix("").resolve()


def _default_scene_root_for_frames_dir(frames_dir: Path) -> Path:
    frames_path = frames_dir.resolve()
    if _is_managed_upload_path(frames_path) and frames_path.parent != UPLOADS_ROOT.resolve():
        return (frames_path.parent / UPLOAD_SCENE_DIRNAME).resolve()
    return frames_path.with_name(f"{frames_path.name}_preprocessed").resolve()


def _coerce_int(value: object, *, label: str, optional: bool = False) -> Optional[int]:
    if value in (None, ""):
        if optional:
            return None
        raise ValueError(f"{label} is required.")
    try:
        return int(float(str(value)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc


def _coerce_float(value: object, *, label: str, optional: bool = False) -> Optional[float]:
    if value in (None, ""):
        if optional:
            return None
        raise ValueError(f"{label} is required.")
    try:
        return float(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number.") from exc


def _parse_fractional_rate(raw: object) -> Optional[float]:
    text = str(raw or "").strip()
    if not text or text == "0/0":
        return None
    try:
        if "/" in text:
            numerator_text, denominator_text = text.split("/", 1)
            numerator = float(numerator_text)
            denominator = float(denominator_text)
            if denominator == 0.0:
                return None
            value = numerator / denominator
        else:
            value = float(text)
    except ValueError:
        return None
    if not math.isfinite(value) or value <= 0.0:
        return None
    return value


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


def _probe_video_fps(video_path: Path) -> float:
    ffprobe = _resolve_executable("ffprobe")

    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,r_frame_rate",
            "-of",
            "json",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout or "{}")
    streams = payload.get("streams") or []
    if not streams:
        raise ValueError(f"No video stream found in {video_path}")

    stream = streams[0]
    for key in ("avg_frame_rate", "r_frame_rate"):
        fps = _parse_fractional_rate(stream.get(key))
        if fps is not None:
            return fps
    raise ValueError(f"Could not read a valid FPS from {video_path}")


def _format_fps_for_cli(fps: float) -> str:
    return f"{float(fps):.12g}"


def _coerce_runtime_export_format(value: object, *, label: str = "Stage 0 Runtime Export") -> str:
    normalized = str(value or "none").strip()
    valid_values = {choice_value for _, choice_value in RUNTIME_EXPORT_CHOICES}
    if normalized not in valid_values:
        raise ValueError(f"{label} must be one of: {', '.join(sorted(valid_values))}.")
    return normalized


def _append_confidence_filter_args(
    command: list[str],
    *,
    prefix: str,
    profile: str,
    percentile: float,
    voxel_min_count_percentile: Optional[float] = 50.0,
) -> None:
    if percentile < 0.0 or percentile > 100.0:
        raise ValueError("DA3 confidence percentile must be between 0 and 100.")

    normalized = str(profile or "default_mixed").strip().lower()
    command.extend([f"--{prefix}.conf-thresh-percentile", str(percentile)])

    if normalized == "default_mixed":
        command.extend(
            [
                f"--{prefix}.conf-mode",
                "voxel_or",
                f"--{prefix}.conf-global-percentile",
                "10",
                f"--{prefix}.conf-local-percentile",
                "10",
                f"--{prefix}.conf-voxel-size",
                "1.0",
            ]
        )
        if voxel_min_count_percentile is None:
            command.extend([f"--{prefix}.conf-voxel-min-count-percentile", "none"])
        else:
            command.extend([f"--{prefix}.conf-voxel-min-count-percentile", str(voxel_min_count_percentile)])
        return

    if normalized == "da3_per_frame":
        command.extend([f"--{prefix}.conf-mode", "per_frame"])
        return

    if normalized == "da3_global":
        command.extend([f"--{prefix}.conf-mode", "global"])
        return

    if normalized == "da3_per_frame_guided":
        command.extend(
            [
                f"--{prefix}.conf-mode",
                "per_frame_guided",
                f"--{prefix}.conf-global-percentile",
                str(percentile),
                f"--{prefix}.conf-local-percentile",
                str(percentile),
            ]
        )
        return

    raise ValueError(f"Unknown confidence profile: {profile}")


def _append_depth_edge_args(
    command: list[str],
    *,
    prefix: str,
    enabled: bool,
    edge_rtol: Optional[float],
    edge_atol: Optional[float],
    edge_kernel_size: int,
) -> None:
    if not enabled:
        return
    if edge_rtol is None and edge_atol is None:
        raise ValueError("Depth-edge suppression needs a relative or absolute threshold.")
    if edge_kernel_size <= 0 or edge_kernel_size % 2 == 0:
        raise ValueError("Depth-edge kernel must be a positive odd integer.")

    command.append(f"--{prefix}.conf-mask-depth-edges")
    command.extend([f"--{prefix}.conf-edge-kernel-size", str(edge_kernel_size)])
    if edge_rtol is not None:
        command.extend([f"--{prefix}.conf-edge-rtol", str(edge_rtol)])
    if edge_atol is not None:
        command.extend([f"--{prefix}.conf-edge-atol", str(edge_atol)])


def _append_max_depth_args(
    command: list[str],
    *,
    prefix: str,
    enabled: bool,
    max_depth_rtol: Optional[float],
    max_depth_atol: Optional[float],
) -> None:
    if not enabled:
        return
    if max_depth_rtol is None and max_depth_atol is None:
        raise ValueError("Max-depth suppression needs a relative or absolute threshold.")

    command.append(f"--{prefix}.conf-mask-max-depth")
    if max_depth_rtol is not None:
        command.extend([f"--{prefix}.conf-max-depth-rtol", str(max_depth_rtol)])
    if max_depth_atol is not None:
        command.extend([f"--{prefix}.conf-max-depth-atol", str(max_depth_atol)])


def _append_sky_mask_args(
    command: list[str],
    *,
    prefix: str,
    enabled: bool,
) -> None:
    if enabled:
        command.append(f"--{prefix}.conf-mask-sky")


def _append_sky_depth_band_args(
    command: list[str],
    *,
    prefix: str,
    enabled: bool,
    band_percent: Optional[float],
) -> None:
    if not enabled:
        return
    command.append(f"--{prefix}.conf-mask-sky-depth-band")
    if band_percent is not None:
        command.extend([f"--{prefix}.conf-sky-depth-band-percent", str(band_percent)])


def _append_white_background_args(
    command: list[str],
    *,
    prefix: str,
    enabled: bool,
    min_rgb: Optional[float],
    max_channel_delta: Optional[float],
    grow_px: Optional[int],
) -> None:
    if not enabled:
        return
    command.append(f"--{prefix}.conf-mask-white-background")
    if min_rgb is not None:
        command.extend([f"--{prefix}.conf-white-bg-min-rgb", str(min_rgb)])
    if max_channel_delta is not None:
        command.extend([f"--{prefix}.conf-white-bg-max-channel-delta", str(max_channel_delta)])
    if grow_px is not None:
        command.extend([f"--{prefix}.conf-white-bg-grow-px", str(grow_px)])


def _append_min_depth_range_args(
    command: list[str],
    *,
    prefix: str,
    enabled_percent: bool,
    min_depth_range_percent: Optional[float],
    enabled_meters: bool,
    min_depth_range_meters: Optional[float],
) -> None:
    _append_tyro_bool_arg(
        command,
        prefix=prefix,
        name="conf_mask_min_depth_range_percent",
        value=enabled_percent,
        default=True,
    )
    if min_depth_range_percent is not None:
        _append_tyro_value_arg(
            command,
            prefix=prefix,
            name="conf_min_depth_range_percent",
            value=min_depth_range_percent,
            default=50.0,
        )
    _append_tyro_bool_arg(
        command,
        prefix=prefix,
        name="conf_mask_min_depth_range_meters",
        value=enabled_meters,
        default=False,
    )
    if min_depth_range_meters is not None:
        _append_tyro_value_arg(
            command,
            prefix=prefix,
            name="conf_min_depth_range_meters",
            value=min_depth_range_meters,
            default=3.0,
        )


def _append_prepare_bool_arg(command: list[str], *, name: str, value: bool, default: bool) -> None:
    if value == default:
        return
    command.append(f"--{name}" if value else f"--no-{name}")


def _append_prepare_alignment_args(
    command: list[str],
    *,
    num_frames: int,
    stride: int,
    offset: int,
    conf_profile: str,
    conf_percentile: float,
    conf_mask_sky: bool,
    conf_mask_sky_depth_band: bool,
    conf_sky_depth_band_percent: Optional[float],
    conf_mask_white_background: bool,
    conf_white_bg_min_rgb: Optional[float],
    conf_white_bg_max_channel_delta: Optional[float],
    conf_white_bg_grow_px: Optional[int],
    conf_mask_min_depth_range_percent: bool,
    conf_min_depth_range_percent: Optional[float],
    conf_mask_min_depth_range_meters: bool,
    conf_min_depth_range_meters: Optional[float],
    conf_mask_depth_edges: bool,
    conf_edge_rtol: Optional[float],
    conf_edge_atol: Optional[float],
    conf_edge_kernel_size: int,
    conf_mask_max_depth: bool,
    conf_max_depth_rtol: Optional[float],
    conf_max_depth_atol: Optional[float],
    conf_voxel_min_count_percentile: Optional[float] = 50.0,
) -> None:
    command.extend(
        [
            "--prepare_stage1_inputs",
            "--prepare_num_frames",
            str(num_frames),
            "--prepare_stride",
            str(stride),
            "--prepare_offset",
            str(offset),
        ]
    )

    normalized = str(conf_profile or "default_mixed").strip().lower()
    command.extend(["--prepare_conf_thresh_percentile", str(conf_percentile)])
    if normalized == "default_mixed":
        command.extend(
            [
                "--prepare_conf_mode",
                "voxel_or",
                "--prepare_conf_global_percentile",
                "10",
                "--prepare_conf_local_percentile",
                "10",
                "--prepare_conf_voxel_size",
                "1.0",
            ]
        )
        if conf_voxel_min_count_percentile is None:
            command.extend(["--prepare_conf_voxel_min_count_percentile", "none"])
        else:
            command.extend(["--prepare_conf_voxel_min_count_percentile", str(conf_voxel_min_count_percentile)])
    elif normalized == "da3_per_frame":
        command.extend(["--prepare_conf_mode", "per_frame"])
    elif normalized == "da3_global":
        command.extend(["--prepare_conf_mode", "global"])
    elif normalized == "da3_per_frame_guided":
        command.extend(
            [
                "--prepare_conf_mode",
                "per_frame_guided",
                "--prepare_conf_global_percentile",
                str(conf_percentile),
                "--prepare_conf_local_percentile",
                str(conf_percentile),
            ]
        )
    else:
        raise ValueError(f"Unknown confidence profile: {conf_profile}")

    if conf_mask_sky:
        command.append("--prepare_conf_mask_sky")
    if conf_mask_sky_depth_band:
        command.append("--prepare_conf_mask_sky_depth_band")
        if conf_sky_depth_band_percent is not None:
            command.extend(["--prepare_conf_sky_depth_band_percent", str(conf_sky_depth_band_percent)])
    if conf_mask_white_background:
        if conf_white_bg_min_rgb is None:
            raise ValueError("White-background suppression needs a minimum RGB value.")
        if conf_white_bg_max_channel_delta is None:
            raise ValueError("White-background suppression needs a max channel delta.")
        if conf_white_bg_grow_px is None:
            conf_white_bg_grow_px = 0
        if int(conf_white_bg_grow_px) < 0:
            raise ValueError("White-background grow pixels must be non-negative.")
        command.append("--prepare_conf_mask_white_background")
        command.extend(["--prepare_conf_white_bg_min_rgb", str(conf_white_bg_min_rgb)])
        command.extend(["--prepare_conf_white_bg_max_channel_delta", str(conf_white_bg_max_channel_delta)])
        command.extend(["--prepare_conf_white_bg_grow_px", str(int(conf_white_bg_grow_px))])
    _append_prepare_bool_arg(
        command,
        name="prepare_conf_mask_min_depth_range_percent",
        value=conf_mask_min_depth_range_percent,
        default=True,
    )
    if conf_min_depth_range_percent is not None:
        command.extend(["--prepare_conf_min_depth_range_percent", str(conf_min_depth_range_percent)])
    _append_prepare_bool_arg(
        command,
        name="prepare_conf_mask_min_depth_range_meters",
        value=conf_mask_min_depth_range_meters,
        default=False,
    )
    if conf_min_depth_range_meters is not None:
        command.extend(["--prepare_conf_min_depth_range_meters", str(conf_min_depth_range_meters)])
    if conf_mask_depth_edges:
        if conf_edge_rtol is None and conf_edge_atol is None:
            raise ValueError("Depth-edge suppression needs a relative or absolute threshold.")
        if conf_edge_kernel_size <= 0 or conf_edge_kernel_size % 2 == 0:
            raise ValueError("Depth-edge kernel must be a positive odd integer.")
        command.append("--prepare_conf_mask_depth_edges")
        command.extend(["--prepare_conf_edge_kernel_size", str(conf_edge_kernel_size)])
        command.extend(["--prepare_conf_edge_rtol", "none" if conf_edge_rtol is None else str(conf_edge_rtol)])
        command.extend(["--prepare_conf_edge_atol", "none" if conf_edge_atol is None else str(conf_edge_atol)])
    if conf_mask_max_depth:
        if conf_max_depth_rtol is None and conf_max_depth_atol is None:
            raise ValueError("Max-depth suppression needs a relative or absolute threshold.")
        command.append("--prepare_conf_mask_max_depth")
        command.extend(
            ["--prepare_conf_max_depth_rtol", "none" if conf_max_depth_rtol is None else str(conf_max_depth_rtol)]
        )
        command.extend(
            ["--prepare_conf_max_depth_atol", "none" if conf_max_depth_atol is None else str(conf_max_depth_atol)]
        )


def _append_tyro_bool_arg(
    command: list[str],
    *,
    prefix: str,
    name: str,
    value: bool,
    default: bool,
) -> None:
    kebab = name.replace("_", "-")
    if value == default:
        return
    if value:
        command.append(f"--{prefix}.{kebab}")
    else:
        command.append(f"--{prefix}.no-{kebab}")


def _append_tyro_value_arg(
    command: list[str],
    *,
    prefix: str,
    name: str,
    value: object,
    default: object,
) -> None:
    if value is None or value == default:
        return
    command.extend([f"--{prefix}.{name.replace('_', '-')}", str(value)])


def _append_stage1_extra_args(command: list[str], *, prefix: str, settings: dict[str, object]) -> None:
    _append_tyro_bool_arg(
        command,
        prefix=f"{prefix}.roma",
        name="use_roma_matching",
        value=bool(settings["use_roma_matching"]),
        default=True,
    )
    _append_tyro_value_arg(
        command, prefix=f"{prefix}.roma", name="roma_version", value=settings["roma_version"], default="v2"
    )
    _append_tyro_value_arg(
        command, prefix=f"{prefix}.roma", name="roma_model", value=settings["roma_model"], default="indoor"
    )
    _append_tyro_value_arg(
        command, prefix=f"{prefix}.roma", name="roma_num_samples", value=settings["roma_num_samples"], default=5000
    )
    _append_tyro_value_arg(
        command,
        prefix=f"{prefix}.roma",
        name="roma_certainty_threshold",
        value=settings["roma_certainty_threshold"],
        default=0.5,
    )
    _append_tyro_value_arg(
        command, prefix=f"{prefix}.roma", name="roma_max_references", value=settings["roma_max_references"], default=20
    )
    _append_tyro_value_arg(
        command,
        prefix=f"{prefix}.roma",
        name="roma_reference_sampling",
        value=settings["roma_reference_sampling"],
        default="recent_and_strided",
    )
    _append_tyro_value_arg(
        command, prefix=f"{prefix}.roma", name="roma_loss_weight", value=settings["roma_loss_weight"], default=1.0
    )
    _append_tyro_value_arg(
        command, prefix=f"{prefix}.roma", name="roma_max_corr_dist", value=settings["roma_max_corr_dist"], default=1.0
    )
    _append_tyro_bool_arg(command, prefix=prefix, name="tensorboard", value=bool(settings["tensorboard"]), default=True)
    _append_tyro_value_arg(
        command, prefix=prefix, name="knn_backend", value=settings["knn_backend"], default="cpu_kdtree"
    )
    _append_tyro_value_arg(command, prefix=prefix, name="max_corr_dist", value=settings["max_corr_dist"], default=0.03)
    _append_tyro_value_arg(
        command, prefix=prefix, name="merge_voxel_size", value=settings["merge_voxel_size"], default=0.001
    )
    _append_tyro_value_arg(command, prefix=prefix, name="icp_n_iter", value=settings["icp_n_iter"], default=100)
    _append_tyro_value_arg(
        command,
        prefix=prefix,
        name="icp_early_stopping_patience",
        value=settings["icp_early_stopping_patience"],
        default=5,
    )
    _append_tyro_value_arg(
        command,
        prefix=prefix,
        name="icp_early_stopping_min_iters",
        value=settings["icp_early_stopping_min_iters"],
        default=25,
    )
    _append_tyro_value_arg(
        command,
        prefix=prefix,
        name="icp_early_stopping_min_delta",
        value=settings["icp_early_stopping_min_delta"],
        default=None,
    )
    _append_tyro_value_arg(command, prefix=prefix, name="icp_lr", value=settings["icp_lr"], default=1e-3)
    _append_tyro_value_arg(
        command, prefix=prefix, name="icp_method", value=settings["icp_method"], default="point2plane"
    )
    _append_tyro_value_arg(
        command, prefix=prefix, name="icp_local_twist_reg", value=settings["icp_local_twist_reg"], default=0.0
    )
    _append_tyro_value_arg(command, prefix=prefix, name="icp_tv_reg", value=settings["icp_tv_reg"], default=50.0)
    _append_tyro_value_arg(
        command, prefix=prefix, name="icp_tv_voxel_size", value=settings["icp_tv_voxel_size"], default=0.01
    )
    _append_tyro_value_arg(command, prefix=prefix, name="icp_tv_every_k", value=settings["icp_tv_every_k"], default=1)
    _append_tyro_value_arg(
        command, prefix=prefix, name="icp_tv_sample_ratio", value=settings["icp_tv_sample_ratio"], default=0.1
    )
    _append_tyro_value_arg(
        command, prefix=prefix, name="icp_color_icp_weight", value=settings["icp_color_icp_weight"], default=0.02
    )
    _append_tyro_value_arg(
        command,
        prefix=prefix,
        name="icp_color_icp_max_color_dist",
        value=settings["icp_color_icp_max_color_dist"],
        default=0.1,
    )
    _append_tyro_value_arg(
        command, prefix=prefix, name="icp_color_icp_k", value=settings["icp_color_icp_k"], default=10
    )
    _append_tyro_value_arg(
        command, prefix=prefix, name="save_intermediate_every", value=settings["save_intermediate_every"], default=10
    )
    _append_tyro_value_arg(
        command, prefix=prefix, name="deform_log2_hashmap_size", value=settings["deform_log2_hashmap_size"], default=19
    )
    _append_tyro_value_arg(
        command, prefix=prefix, name="deform_num_levels", value=settings["deform_num_levels"], default=24
    )
    _append_tyro_value_arg(
        command, prefix=prefix, name="deform_n_neurons", value=settings["deform_n_neurons"], default=64
    )
    _append_tyro_value_arg(
        command, prefix=prefix, name="deform_n_hidden_layers", value=settings["deform_n_hidden_layers"], default=4
    )
    _append_tyro_value_arg(command, prefix=prefix, name="deform_min_res", value=settings["deform_min_res"], default=16)
    _append_tyro_value_arg(
        command, prefix=prefix, name="deform_max_res", value=settings["deform_max_res"], default=2048
    )
    _append_tyro_bool_arg(
        command, prefix=prefix, name="filter_points", value=bool(settings["filter_points"]), default=False
    )
    _append_tyro_value_arg(
        command, prefix=prefix, name="filter_geom_sigma", value=settings["filter_geom_sigma"], default=2.5
    )
    _append_tyro_value_arg(
        command, prefix=prefix, name="filter_color_sigma", value=settings["filter_color_sigma"], default=1.5
    )
    _append_tyro_value_arg(
        command, prefix=prefix, name="filter_worst_pct", value=settings["filter_worst_pct"], default=0.2
    )
    _append_tyro_value_arg(
        command, prefix=prefix, name="filter_min_frames", value=settings["filter_min_frames"], default=2
    )
    _append_tyro_value_arg(
        command, prefix=prefix, name="filter_base_percentile", value=settings["filter_base_percentile"], default="p75"
    )


def _append_stage2_extra_args(command: list[str], *, prefix: str, settings: dict[str, object]) -> None:
    _append_tyro_bool_arg(command, prefix=prefix, name="tensorboard", value=bool(settings["tensorboard"]), default=True)
    _append_tyro_value_arg(
        command, prefix=prefix, name="knn_backend", value=settings["knn_backend"], default="cpu_kdtree"
    )
    _append_tyro_value_arg(
        command, prefix=prefix, name="loo_loss_weight", value=settings["loo_loss_weight"], default=1.0
    )
    _append_tyro_value_arg(command, prefix=prefix, name="loo_k_neighbors", value=settings["loo_k_neighbors"], default=5)
    _append_tyro_value_arg(
        command, prefix=prefix, name="loo_max_corr_dist", value=settings["loo_max_corr_dist"], default=0.03125
    )
    _append_tyro_value_arg(command, prefix=prefix, name="loo_normal_k", value=settings["loo_normal_k"], default=20)
    _append_tyro_value_arg(
        command, prefix=prefix, name="loo_kdtree_rebuild_every", value=settings["loo_kdtree_rebuild_every"], default=50
    )
    _append_tyro_value_arg(
        command, prefix=prefix, name="loo_max_pairs_per_iter", value=settings["loo_max_pairs_per_iter"], default=200000
    )
    _append_tyro_value_arg(
        command, prefix=prefix, name="loo_pairs_per_src", value=settings["loo_pairs_per_src"], default=1
    )
    _append_tyro_value_arg(
        command, prefix=prefix, name="deform_chunk_size", value=settings["deform_chunk_size"], default=200000
    )
    _append_tyro_value_arg(
        command, prefix=prefix, name="anchor_loss_weight", value=settings["anchor_loss_weight"], default=1000.0
    )
    _append_tyro_value_arg(
        command, prefix=prefix, name="anchor_n_samples", value=settings["anchor_n_samples"], default=4096
    )
    _append_tyro_value_arg(command, prefix=prefix, name="tv_reg", value=settings["tv_reg"], default=50.0)
    _append_tyro_value_arg(command, prefix=prefix, name="tv_voxel_size", value=settings["tv_voxel_size"], default=0.01)
    _append_tyro_value_arg(command, prefix=prefix, name="tv_every_k", value=settings["tv_every_k"], default=1)
    _append_tyro_value_arg(
        command, prefix=prefix, name="tv_sample_ratio", value=settings["tv_sample_ratio"], default=0.1
    )
    _append_tyro_value_arg(
        command, prefix=prefix, name="loo_color_icp_weight", value=settings["loo_color_icp_weight"], default=0.02
    )
    _append_tyro_value_arg(
        command, prefix=prefix, name="loo_color_icp_k", value=settings["loo_color_icp_k"], default=10
    )
    _append_tyro_value_arg(
        command,
        prefix=prefix,
        name="loo_color_icp_max_color_dist",
        value=settings["loo_color_icp_max_color_dist"],
        default=0.1,
    )
    _append_tyro_value_arg(
        command, prefix=prefix, name="thin_shell_weight", value=settings["thin_shell_weight"], default=1000.0
    )
    _append_tyro_value_arg(command, prefix=prefix, name="lr", value=settings["lr"], default=1e-3)
    _append_tyro_value_arg(command, prefix=prefix, name="n_iters", value=settings["n_iters"], default=150)
    _append_tyro_value_arg(
        command,
        prefix=prefix,
        name="save_intermediate_every_n",
        value=settings["save_intermediate_every_n"],
        default=50,
    )


def _append_stage31_extra_args(command: list[str], *, prefix: str, settings: dict[str, object]) -> None:
    _append_tyro_bool_arg(command, prefix=prefix, name="tensorboard", value=bool(settings["tensorboard"]), default=True)
    _append_tyro_value_arg(
        command, prefix=prefix, name="knn_backend", value=settings["knn_backend"], default="cpu_kdtree"
    )
    _append_tyro_value_arg(command, prefix=prefix, name="batch_size", value=settings["batch_size"], default=8192)
    _append_tyro_value_arg(command, prefix=prefix, name="lr", value=settings["lr"], default=1e-3)
    _append_tyro_value_arg(command, prefix=prefix, name="cycle_weight", value=settings["cycle_weight"], default=0.1)
    _append_tyro_value_arg(
        command, prefix=prefix, name="magnitude_weight", value=settings["magnitude_weight"], default=1e-3
    )
    _append_tyro_value_arg(
        command, prefix=prefix, name="smoothness_weight", value=settings["smoothness_weight"], default=1e-3
    )
    _append_tyro_value_arg(
        command, prefix=prefix, name="num_forward_samples", value=settings["num_forward_samples"], default=10000
    )
    _append_tyro_value_arg(
        command, prefix=prefix, name="num_interp_samples", value=settings["num_interp_samples"], default=5000
    )
    _append_tyro_value_arg(
        command, prefix=prefix, name="regenerate_every", value=settings["regenerate_every"], default=10
    )
    _append_tyro_value_arg(command, prefix=prefix, name="view_embed_dim", value=settings["view_embed_dim"], default=32)
    _append_tyro_value_arg(command, prefix=prefix, name="min_res", value=settings["min_res"], default=16)
    _append_tyro_value_arg(command, prefix=prefix, name="max_res", value=settings["max_res"], default=2048)
    _append_tyro_value_arg(command, prefix=prefix, name="num_levels", value=settings["num_levels"], default=16)
    _append_tyro_value_arg(
        command, prefix=prefix, name="log2_hashmap_size", value=settings["log2_hashmap_size"], default=19
    )
    _append_tyro_value_arg(command, prefix=prefix, name="n_neurons", value=settings["n_neurons"], default=64)
    _append_tyro_value_arg(command, prefix=prefix, name="n_hidden_layers", value=settings["n_hidden_layers"], default=3)
    _append_tyro_bool_arg(
        command, prefix=prefix, name="save_validation_plys", value=bool(settings["save_validation_plys"]), default=True
    )


def _append_gs_extra_args(command: list[str], *, prefix: str, settings: dict[str, object]) -> None:
    _append_tyro_bool_arg(command, prefix=prefix, name="tensorboard", value=bool(settings["tensorboard"]), default=True)
    _append_tyro_value_arg(command, prefix=prefix, name="sh_degree", value=settings["sh_degree"], default=3)
    _append_tyro_value_arg(
        command, prefix=prefix, name="sh_increase_every", value=settings["sh_increase_every"], default=0
    )
    _append_tyro_value_arg(
        command, prefix=prefix, name="sh_full_from_iter", value=settings["sh_full_from_iter"], default=5000
    )
    _append_tyro_bool_arg(
        command,
        prefix=prefix,
        name="sh_freeze_means_when_full_sh",
        value=bool(settings["sh_freeze_means_when_full_sh"]),
        default=True,
    )
    _append_tyro_value_arg(command, prefix=prefix, name="sh_reg_weight", value=settings["sh_reg_weight"], default=10.0)
    _append_tyro_value_arg(
        command, prefix=prefix, name="target_num_points", value=settings["target_num_points"], default=4000000
    )
    _append_tyro_bool_arg(
        command, prefix=prefix, name="optimize_cams", value=bool(settings["optimize_cams"]), default=True
    )
    _append_tyro_value_arg(command, prefix=prefix, name="lr_cams", value=settings["lr_cams"], default=1e-4)
    _append_tyro_bool_arg(
        command, prefix=prefix, name="optimize_positions", value=bool(settings["optimize_positions"]), default=True
    )
    _append_tyro_value_arg(command, prefix=prefix, name="lr_positions", value=settings["lr_positions"], default=1e-5)
    _append_tyro_value_arg(command, prefix=prefix, name="lr_colors", value=settings["lr_colors"], default=2.5e-3)
    _append_tyro_value_arg(command, prefix=prefix, name="lr_opacities", value=settings["lr_opacities"], default=5e-2)
    _append_tyro_value_arg(command, prefix=prefix, name="lr_scales", value=settings["lr_scales"], default=5e-3)
    _append_tyro_value_arg(command, prefix=prefix, name="lr_quats", value=settings["lr_quats"], default=1e-3)
    _append_tyro_value_arg(command, prefix=prefix, name="lr_sh0", value=settings["lr_sh0"], default=2.5e-3)
    _append_tyro_value_arg(command, prefix=prefix, name="lr_shN", value=settings["lr_shn"], default=2.5e-3 / 20.0)
    _append_tyro_bool_arg(
        command,
        prefix=prefix,
        name="deform_inverse_rotations",
        value=bool(settings["deform_inverse_rotations"]),
        default=True,
    )
    _append_tyro_value_arg(
        command, prefix=prefix, name="initial_opacity", value=settings["initial_opacity"], default=0.5
    )
    _append_tyro_value_arg(command, prefix=prefix, name="initial_scale", value=settings["initial_scale"], default=0.005)
    _append_tyro_value_arg(
        command, prefix=prefix, name="initial_flat_ratio", value=settings["initial_flat_ratio"], default=0.1
    )
    _append_tyro_value_arg(command, prefix=prefix, name="scale_init", value=settings["scale_init"], default="knn")
    _append_tyro_value_arg(command, prefix=prefix, name="knn_neighbors", value=settings["knn_neighbors"], default=4)
    _append_tyro_value_arg(command, prefix=prefix, name="normal_k", value=settings["normal_k"], default=20)
    _append_tyro_value_arg(command, prefix=prefix, name="l1_weight", value=settings["l1_weight"], default=0.8)
    _append_tyro_value_arg(command, prefix=prefix, name="lpips_weight", value=settings["lpips_weight"], default=0.2)
    _append_tyro_value_arg(
        command, prefix=prefix, name="opacity_reg_weight", value=settings["opacity_reg_weight"], default=0.0
    )
    _append_tyro_value_arg(
        command, prefix=prefix, name="scale_reg_weight", value=settings["scale_reg_weight"], default=0.0
    )
    _append_tyro_value_arg(
        command,
        prefix=prefix,
        name="normal_consistency_weight",
        value=settings["normal_consistency_weight"],
        default=0.05,
    )
    _append_tyro_value_arg(
        command, prefix=prefix, name="distortion_weight", value=settings["distortion_weight"], default=0.01
    )
    _append_tyro_value_arg(
        command, prefix=prefix, name="alpha_reg_weight", value=settings["alpha_reg_weight"], default=0.0
    )
    _append_tyro_value_arg(command, prefix=prefix, name="frames_per_iter", value=settings["frames_per_iter"], default=1)
    _append_tyro_value_arg(command, prefix=prefix, name="log_every", value=settings["log_every"], default=50)
    _append_tyro_value_arg(command, prefix=prefix, name="save_every", value=settings["save_every"], default=5000)
    _append_tyro_value_arg(command, prefix=prefix, name="eval_every", value=settings["eval_every"], default=1000)
    _append_tyro_value_arg(command, prefix=prefix, name="lr_decay", value=settings["lr_decay"], default=0.1)
    _append_tyro_bool_arg(command, prefix=prefix, name="auto_eval", value=bool(settings["auto_eval"]), default=True)


def _copy_uploaded_video(uploaded_path: str) -> Path:
    source = _resolve_existing_file(uploaded_path)
    if _is_managed_upload_path(source):
        return source.resolve()

    upload_dir = UPLOADS_ROOT / _new_upload_uid()
    upload_dir.mkdir(parents=True, exist_ok=False)
    target = upload_dir / f"{upload_dir.name}{source.suffix.lower()}"
    shutil.copy2(source, target)
    _clear_catalog_cache()
    return target.resolve()


def _copy_uploaded_frames_dir(frames_dir_path: object) -> Path:
    source = _resolve_existing_dir(frames_dir_path)
    if _is_managed_upload_path(source):
        return source.resolve()

    upload_dir = UPLOADS_ROOT / _new_upload_uid()
    target = upload_dir / UPLOAD_FRAMES_DIRNAME
    shutil.copytree(source, target)
    _clear_catalog_cache()
    return target.resolve()


def _ensure_unique_scene_root(path: Path) -> Path:
    candidate = path.resolve()
    if not candidate.exists():
        return candidate

    suffix = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    return candidate.with_name(f"{candidate.name}_{suffix}")


def _compose_scene_root_override(parent_dir_selection: object, scene_name: str, *, default_stem: str) -> Optional[Path]:
    parent_dir_text = _strip_quotes(parent_dir_selection)
    scene_name_text = _strip_quotes(scene_name)
    if not parent_dir_text and not scene_name_text:
        return None

    if parent_dir_text:
        parent_dir = _resolve_existing_dir(parent_dir_text)
    else:
        parent_dir = PICKER_ROOT

    final_name = _safe_stem(scene_name_text) if scene_name_text else default_stem
    return (parent_dir / final_name).resolve()


def _cache_uploaded_video_value(uploaded_path: object) -> str:
    raw_path = _strip_quotes(uploaded_path)
    if not raw_path:
        return ""
    return str(_copy_uploaded_video(raw_path))


def _display_path(path: Path, *, root: Optional[Path] = None) -> str:
    base = (root or PICKER_ROOT).resolve()
    try:
        rel = path.resolve().relative_to(base)
    except ValueError:
        return str(path.resolve())
    rel_text = rel.as_posix()
    if not rel_text or rel_text == ".":
        return f"{base.name}/"
    return rel_text


def _choice_tuples(paths: list[Path], *, include_empty_label: Optional[str] = None) -> list[tuple[str, str]]:
    choices: list[tuple[str, str]] = []
    if include_empty_label is not None:
        choices.append((include_empty_label, ""))

    seen: set[str] = set()
    for path in paths:
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        choices.append((_display_path(path), resolved))
    return choices


def _update_dropdown_choices(choices: list[tuple[str, str]], current_value: object = None):
    valid_values = [value for _, value in choices]
    selected = _strip_quotes(current_value)
    if selected not in valid_values:
        if "" in valid_values:
            selected = ""
        else:
            selected = valid_values[0] if valid_values else None
    return gr.update(choices=choices, value=selected)


def _should_prune_video_discovery_dir(path: Path) -> bool:
    name = path.name.lower()
    if name in PRUNED_VIDEO_DISCOVERY_DIR_NAMES:
        return True
    if name.startswith("frame_to_model_icp_") or name.startswith("inverse_deformation"):
        return True
    return (path / "exports" / "npz" / "results.npz").is_file()


def _discover_existing_videos_uncached() -> list[Path]:
    if not PICKER_ROOT.is_dir():
        return []

    candidates: list[Path] = []
    for dirpath_text, dirnames, filenames in os.walk(PICKER_ROOT):
        dirpath = Path(dirpath_text)
        dirnames[:] = [name for name in dirnames if not _should_prune_video_discovery_dir(dirpath / name)]
        for filename in filenames:
            path = dirpath / filename
            if path.suffix.lower() not in VIDEO_SUFFIXES:
                continue
            candidates.append(path.resolve())

    return sorted(candidates, key=lambda item: (-item.stat().st_mtime, _display_path(item)))


def _discover_existing_videos() -> list[Path]:
    return _cached_catalog("videos", _discover_existing_videos_uncached)


def _discover_scene_roots_uncached() -> list[Path]:
    if not PICKER_ROOT.is_dir():
        return []

    scene_roots: dict[str, Path] = {}
    for npz_path in PICKER_ROOT.rglob("results.npz"):
        try:
            if npz_path.parent.name != "npz" or npz_path.parent.parent.name != "exports":
                continue
            scene_root = npz_path.parents[2].resolve()
        except IndexError:
            continue
        scene_roots[str(scene_root)] = scene_root

    return sorted(scene_roots.values(), key=lambda item: (-item.stat().st_mtime, _display_path(item)))


def _discover_scene_roots() -> list[Path]:
    return _cached_catalog("scene_roots", _discover_scene_roots_uncached)


def _scene_root_has_stage0(scene_root: Path) -> bool:
    return (scene_root / "exports" / "npz" / "results.npz").is_file()


def _scene_activity_mtime(scene_root: Path) -> float:
    candidates = [scene_root]
    results_npz = scene_root / "exports" / "npz" / "results.npz"
    if results_npz.exists():
        candidates.append(results_npz)
    candidates.extend(_find_run_dirs(scene_root))

    mtimes: list[float] = []
    for path in candidates:
        try:
            mtimes.append(path.stat().st_mtime)
        except OSError:
            continue
    return max(mtimes) if mtimes else 0.0


def _choose_best_scene_root(candidates: list[Path]) -> Optional[Path]:
    unique: dict[str, Path] = {}
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        unique[str(resolved)] = resolved

    valid = [path for path in unique.values() if _scene_root_has_stage0(path)]
    if not valid:
        return None

    return max(valid, key=lambda path: (len(_find_run_dirs(path)), _scene_activity_mtime(path)))


def _load_scene_preprocess_metadata(scene_root: Path) -> dict[str, object]:
    meta_path = scene_root / "preprocess_frames.json"
    if not meta_path.is_file():
        return {}
    try:
        with meta_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _find_scene_root_for_video(video_selection: object) -> Optional[Path]:
    video_path = _resolve_existing_file(video_selection)
    candidates = [
        _default_scene_root_for_video(video_path),
        video_path.with_suffix("").resolve(),
    ]
    if _is_managed_upload_path(video_path):
        candidates.extend(
            [
                (video_path.parent / UPLOAD_SCENE_DIRNAME).resolve(),
                (video_path.parent / video_path.stem).resolve(),
            ]
        )

    video_key = _path_identity_key(video_path)
    for scene_root in _discover_scene_roots():
        meta = _load_scene_preprocess_metadata(scene_root)
        if _path_identity_key(meta.get("source_input_path")) == video_key:
            candidates.append(scene_root)

    return _choose_best_scene_root(candidates)


def _discover_output_parent_dirs() -> list[Path]:
    if not PICKER_ROOT.is_dir():
        return []

    directories = [PICKER_ROOT.resolve()]
    directories.extend(sorted([path.resolve() for path in PICKER_ROOT.iterdir() if path.is_dir()], key=_display_path))
    return directories


def _video_dropdown_choices() -> list[tuple[str, str]]:
    return _choice_tuples(_discover_existing_videos(), include_empty_label="Select video")


def _scene_dropdown_choices(extra_scene_root: object = None) -> list[tuple[str, str]]:
    scene_roots = _discover_scene_roots()
    extra_scene_text = _strip_quotes(extra_scene_root)
    if extra_scene_text:
        try:
            extra_scene = _resolve_existing_dir(extra_scene_text)
        except Exception:
            extra_scene = None
        if extra_scene is not None and _scene_root_has_stage0(extra_scene):
            known = {str(path.resolve()) for path in scene_roots}
            if str(extra_scene.resolve()) not in known:
                scene_roots = [extra_scene.resolve(), *scene_roots]
    return _choice_tuples(scene_roots, include_empty_label="Select after Stage 0")


def _output_parent_dropdown_choices() -> list[tuple[str, str]]:
    return _choice_tuples(_discover_output_parent_dirs(), include_empty_label="Default (automatic)")


def _refresh_pipeline_catalogs(current_video: object, current_scene: object, current_parent: object):
    return (
        _update_dropdown_choices(_video_dropdown_choices(), current_video),
        _update_dropdown_choices(_scene_dropdown_choices(), current_scene),
        _update_dropdown_choices(_output_parent_dropdown_choices(), current_parent),
    )


def _refresh_stage_catalogs(current_video: object, current_parent: object, current_scene: object):
    return (
        _update_dropdown_choices(_video_dropdown_choices(), current_video),
        _update_dropdown_choices(_output_parent_dropdown_choices(), current_parent),
        _update_dropdown_choices(_scene_dropdown_choices(), current_scene),
    )


def _refresh_inspect_catalog(current_scene: object):
    return _update_dropdown_choices(_scene_dropdown_choices(), current_scene)


def _sync_catalogs_and_scene_views(
    preferred_scene_root: object,
    pipeline_current_video: object,
    pipeline_current_scene: object,
    pipeline_current_parent: object,
    stage_current_video: object,
    stage_current_parent: object,
    stage_current_scene: object,
    inspect_current_scene: object,
):
    preferred_scene_text = _strip_quotes(preferred_scene_root)
    video_choices = _video_dropdown_choices()
    stage_scene_text = _strip_quotes(stage_current_scene)
    scene_choices = _scene_dropdown_choices(preferred_scene_text or stage_scene_text)
    output_parent_choices = _output_parent_dropdown_choices()

    if preferred_scene_text:
        scene_values = [value for _, value in scene_choices]
        if preferred_scene_text not in scene_values:
            try:
                preferred_scene_path = _resolve_existing_dir(preferred_scene_text)
            except Exception:
                preferred_scene_path = None
            if preferred_scene_path is not None and _scene_root_has_stage0(preferred_scene_path):
                _clear_catalog_cache()
                video_choices = _video_dropdown_choices()
                scene_choices = _scene_dropdown_choices(preferred_scene_text or stage_scene_text)
                output_parent_choices = _output_parent_dropdown_choices()

    pipeline_video_update = _update_dropdown_choices(video_choices, pipeline_current_video)
    pipeline_parent_update = _update_dropdown_choices(output_parent_choices, pipeline_current_parent)
    stage_video_update = _update_dropdown_choices(video_choices, stage_current_video)
    stage_parent_update = _update_dropdown_choices(output_parent_choices, stage_current_parent)

    scene_values = [value for _, value in scene_choices]
    selected_scene = None
    if preferred_scene_text == "" and stage_scene_text == "" and "" in scene_values:
        selected_scene = ""
    for candidate in (
        preferred_scene_root,
        stage_current_scene,
        pipeline_current_scene,
        inspect_current_scene,
    ):
        if selected_scene == "":
            break
        value = _strip_quotes(candidate)
        if value and value in scene_values:
            selected_scene = value
            break
    if selected_scene is None and scene_values:
        selected_scene = scene_values[0]

    pipeline_scene_update = _update_dropdown_choices(scene_choices, selected_scene)
    stage_scene_update = _update_dropdown_choices(scene_choices, selected_scene)
    inspect_scene_update = _update_dropdown_choices(scene_choices, selected_scene)

    empty_dropdown = _empty_dropdown_update()
    disabled = _stage_button_update(active=False, enabled=False)
    if not selected_scene:
        return (
            pipeline_video_update,
            pipeline_scene_update,
            pipeline_parent_update,
            stage_video_update,
            stage_parent_update,
            stage_scene_update,
            inspect_scene_update,
            "",
            "",
            empty_dropdown,
            "",
            empty_dropdown,
            empty_dropdown,
            empty_dropdown,
            "",
            "**Recommended Next Step**: `Stage 0`\n\nStart from a video to create the scene root, DA3 outputs, filtered point-cloud cache, and pre-ICP merge.",
            _stage_button_update(active=True, enabled=True),
            disabled,
            disabled,
            disabled,
            disabled,
            "",
            "",
            "",
            None,
            None,
            [],
            "Select a scene to inspect existing outputs.",
        )

    stage_refresh = _refresh_stage_scene(selected_scene)
    inspect_refresh = _inspect_existing_scene(selected_scene)
    return (
        pipeline_video_update,
        pipeline_scene_update,
        pipeline_parent_update,
        stage_video_update,
        stage_parent_update,
        stage_scene_update,
        inspect_scene_update,
        *stage_refresh,
        *inspect_refresh,
    )


def _sync_catalogs_after_stage_run(
    run_state: dict[str, str],
    pipeline_current_video: object,
    pipeline_current_scene: object,
    pipeline_current_parent: object,
    stage_current_video: object,
    stage_current_parent: object,
    stage_current_scene: object,
    inspect_current_scene: object,
):
    _clear_catalog_cache()
    preferred_scene_root = (run_state or {}).get("scene_root", "") or stage_current_scene
    return _sync_catalogs_and_scene_views(
        preferred_scene_root,
        pipeline_current_video,
        pipeline_current_scene,
        pipeline_current_parent,
        stage_current_video,
        stage_current_parent,
        stage_current_scene,
        inspect_current_scene,
    )


def _sync_catalogs_after_live_stage_run(
    preferred_scene_root: object,
    pipeline_current_video: object,
    pipeline_current_scene: object,
    pipeline_current_parent: object,
    stage_current_video: object,
    stage_current_parent: object,
    stage_current_scene: object,
    inspect_current_scene: object,
):
    _clear_catalog_cache()
    if not _strip_quotes(preferred_scene_root) and not _strip_quotes(stage_current_scene):
        return tuple(gr.update() for _ in range(AUTO_SYNC_OUTPUT_COUNT))
    return _sync_catalogs_and_scene_views(
        preferred_scene_root,
        pipeline_current_video,
        pipeline_current_scene,
        pipeline_current_parent,
        stage_current_video,
        stage_current_parent,
        stage_current_scene,
        inspect_current_scene,
    )


def _update_pipeline_source_mode(source_mode: str):
    normalized = str(source_mode or "upload_video").strip().lower()
    return (
        gr.update(visible=normalized == "upload_video"),
        gr.update(visible=normalized == "existing_video"),
        gr.update(visible=normalized == "existing_scene"),
        gr.update(visible=normalized != "existing_scene"),
    )


def _update_stage0_source_mode(source_mode: str):
    normalized = str(source_mode or "upload_video").strip().lower()
    return (
        gr.update(visible=normalized == "upload_video"),
        gr.update(visible=normalized == "existing_video"),
        gr.update(visible=normalized == "existing_frames"),
    )


def _tail_text(path: Path, *, max_lines: int = MAX_LOG_LINES) -> str:
    if not path.exists():
        return ""
    lines: deque[str] = deque(maxlen=max_lines)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            lines.append(line.rstrip())
    return "\n".join(lines)


def _format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


def _find_run_dirs(scene_root: Path) -> list[Path]:
    if not scene_root.is_dir():
        return []
    return sorted(
        [path for path in scene_root.iterdir() if path.is_dir() and path.name.startswith("frame_to_model_icp_")],
        key=lambda path: path.stat().st_mtime,
    )


def _list_inverse_dirs(run_dir: Optional[Path]) -> list[Path]:
    if run_dir is None or not run_dir.is_dir():
        return []
    return sorted(
        [path for path in run_dir.iterdir() if path.is_dir() and path.name.startswith("inverse_deformation")],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def _list_gs_dirs(run_dir: Optional[Path]) -> list[Path]:
    if run_dir is None or not run_dir.is_dir():
        return []
    return sorted(
        [path for path in run_dir.iterdir() if path.is_dir() and path.name.startswith("gs_")],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def _checkpoint_subdir_choices(run_dir: Optional[Path]) -> list[str]:
    if run_dir is None or not run_dir.is_dir():
        return []
    preferred = ["after_global_optimization", "after_non_rigid_icp"]
    return [name for name in preferred if (run_dir / name).is_dir()]


def _default_original_images_dir(scene_root: Path) -> str:
    frames_dir = scene_root / "frames_subsampled"
    return str(frames_dir.resolve()) if frames_dir.is_dir() else ""


def _collect_scene_artifacts(scene_root: Path) -> SceneArtifacts:
    notes: list[str] = []
    run_dirs = _find_run_dirs(scene_root)
    latest_run_dir = run_dirs[-1] if run_dirs else None

    da3_videos = sorted((scene_root / "gs_video").glob("*.mp4")) if (scene_root / "gs_video").is_dir() else []
    gs_videos: list[Path] = []
    gs_dirs: list[Path] = []
    inverse_dirs: list[Path] = []
    key_files: list[Path] = []

    results_npz = scene_root / "exports" / "npz" / "results.npz"
    if results_npz.exists():
        key_files.append(results_npz)
    else:
        notes.append("Stage 0 results.npz not found yet.")

    if latest_run_dir is None:
        notes.append("No Stage 1 run directory found yet.")
    else:
        after_non_rigid = latest_run_dir / "after_non_rigid_icp"
        after_global = latest_run_dir / "after_global_optimization"
        if after_non_rigid.is_dir():
            key_files.append(after_non_rigid / "config.json")
        else:
            notes.append("Stage 1 checkpoint directory not found yet.")
        if after_global.is_dir():
            key_files.append(after_global / "config.json")
        else:
            notes.append("Stage 2 checkpoint directory not found yet.")

        debug_mask_dir = latest_run_dir / "exports" / "ply"
        debug_mask_paths = sorted(debug_mask_dir.glob("*/debug_masks")) if debug_mask_dir.is_dir() else []
        if debug_mask_paths:
            key_files.append(debug_mask_paths[-1])

        inverse_dirs = _list_inverse_dirs(latest_run_dir)
        for inverse_dir in inverse_dirs:
            model_path = inverse_dir / "inverse_local.pt"
            if model_path.exists():
                key_files.append(model_path)

        gs_dirs = _list_gs_dirs(latest_run_dir)
        for gs_dir in gs_dirs:
            gs_videos.extend(sorted((gs_dir / "gs_video_eval").glob("*.mp4")))
            model_final = gs_dir / "model_final.pt"
            renderer_name = gs_dir.name.split("_", 1)[1] if "_" in gs_dir.name else gs_dir.name
            splat_ply = gs_dir / f"splats_{renderer_name}.ply"
            if model_final.exists():
                key_files.append(model_final)
            checkpoint_paths = sorted(gs_dir.glob("checkpoint_*.pt"))
            if checkpoint_paths:
                key_files.append(checkpoint_paths[-1])
            if splat_ply.exists():
                key_files.append(splat_ply)

    key_files.extend(da3_videos[:1])
    key_files.extend(gs_videos[:2])

    deduped_key_files: list[Path] = []
    seen = set()
    for path in key_files:
        if not path.exists():
            continue
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        deduped_key_files.append(path.resolve())

    return SceneArtifacts(
        scene_root=scene_root.resolve(),
        run_dirs=run_dirs,
        latest_run_dir=latest_run_dir.resolve() if latest_run_dir else None,
        da3_videos=[path.resolve() for path in da3_videos],
        gs_videos=[path.resolve() for path in gs_videos],
        inverse_dirs=[path.resolve() for path in inverse_dirs],
        gs_dirs=[path.resolve() for path in gs_dirs],
        key_files=deduped_key_files[:10],
        notes=notes,
    )


def _placeholder_artifacts(scene_root: Path) -> SceneArtifacts:
    return SceneArtifacts(
        scene_root=scene_root.resolve(),
        run_dirs=[],
        latest_run_dir=None,
        da3_videos=[],
        gs_videos=[],
        inverse_dirs=[],
        gs_dirs=[],
        key_files=[],
        notes=["Scene root has not been created yet."],
    )


def _choose_primary_preview(artifacts: SceneArtifacts) -> Optional[str]:
    if artifacts.gs_videos:
        return str(artifacts.gs_videos[0])
    if artifacts.da3_videos:
        return str(artifacts.da3_videos[0])
    return None


def _choose_secondary_preview(artifacts: SceneArtifacts) -> Optional[str]:
    if artifacts.gs_videos and artifacts.da3_videos:
        return str(artifacts.da3_videos[0])
    if len(artifacts.gs_videos) > 1:
        return str(artifacts.gs_videos[1])
    return None


def _format_scene_report(artifacts: SceneArtifacts, *, selected_run_name: Optional[str] = None) -> str:
    lines = [
        f"Scene root: `{artifacts.scene_root}`",
        f"Stage 1 runs: `{len(artifacts.run_dirs)}`",
        f"Latest run: `{artifacts.latest_run_dir}`" if artifacts.latest_run_dir else "Latest run: `not available`",
        f"DA3 preview videos: `{len(artifacts.da3_videos)}`",
        f"GS preview videos: `{len(artifacts.gs_videos)}`",
        f"Inverse deformation dirs: `{len(artifacts.inverse_dirs)}`",
        f"GS dirs: `{len(artifacts.gs_dirs)}`",
    ]

    selected_run_dir: Optional[Path] = None
    if selected_run_name:
        candidate = artifacts.scene_root / selected_run_name
        if candidate.is_dir():
            selected_run_dir = candidate.resolve()
    elif artifacts.latest_run_dir is not None:
        selected_run_dir = artifacts.latest_run_dir

    if selected_run_dir is not None:
        checkpoint_dirs = _checkpoint_subdir_choices(selected_run_dir)
        inverse_names = [path.name for path in _list_inverse_dirs(selected_run_dir)]
        gs_names = [path.name for path in _list_gs_dirs(selected_run_dir)]
        lines.extend(
            [
                "",
                "Selected run:",
                f"- `{selected_run_dir}`",
                f"- Checkpoint inputs: `{', '.join(checkpoint_dirs) if checkpoint_dirs else 'none'}`",
                f"- Inverse dirs: `{', '.join(inverse_names) if inverse_names else 'none'}`",
                f"- GS dirs: `{', '.join(gs_names) if gs_names else 'none'}`",
            ]
        )

    if artifacts.notes:
        lines.append("")
        lines.append("Notes:")
        lines.extend(f"- {note}" for note in artifacts.notes)
    return "\n".join(lines)


def _latest_run_dir_text(artifacts: SceneArtifacts) -> str:
    return str(artifacts.latest_run_dir) if artifacts.latest_run_dir else ""


def _key_files_as_strings(artifacts: SceneArtifacts) -> list[str]:
    return [str(path) for path in artifacts.key_files]


def _detect_current_stage(log_text: str, *, default_stage: str) -> str:
    if not log_text.strip():
        return default_stage

    pipeline_lines = [line.strip() for line in log_text.splitlines() if "[PIPELINE]" in line]
    for line in reversed(pipeline_lines):
        if "Done! All outputs are in:" in line:
            return "Complete"
        if "Stage 3.2" in line:
            return "Stage 3.2"
        if "Stage 3.1" in line:
            return "Stage 3.1"
        if "Stage 2" in line:
            return "Stage 2"
        if "Stage 1" in line:
            return "Stage 1"
        if "Stage 0" in line:
            return "Stage 0"

    lowered = log_text.lower()
    if "da3 preprocessing complete" in lowered or "results.npz" in lowered:
        return "Stage 0"
    if "frame_to_model_icp" in lowered or "non-rigid icp" in lowered:
        return "Stage 1"
    if "global optimization" in lowered or "rebuilding knn/normals" in lowered or "estimate_normals" in lowered:
        return "Stage 2"
    if "inverse deformation" in lowered or "validation_roundtrip" in lowered:
        return "Stage 3.1"
    if "gs training" in lowered or "train_gs" in lowered or "lpips" in lowered:
        return "Stage 3.2"
    if "done! all outputs are in:" in lowered:
        return "Complete"
    return default_stage


def _build_status_markdown(
    *,
    state: str,
    stage: str,
    elapsed_seconds: float,
    scene_root: Path,
    log_path: Path,
    run_dir: str,
    command: list[str],
    extra: Optional[str] = None,
) -> str:
    lines = [
        f"**State**: {state}",
        f"**Stage**: {stage}",
        f"**Elapsed**: `{_format_duration(elapsed_seconds)}`",
        f"**Scene Root**: `{scene_root}`",
        f"**Latest Run Dir**: `{run_dir or 'not available yet'}`",
        f"**Log**: `{log_path}`",
        "",
        "**Command**",
        f"`{' '.join(command)}`",
    ]
    if extra:
        lines.extend(["", extra])
    return "\n".join(lines)


def _build_pipeline_command(
    *,
    input_video: Optional[Path],
    existing_scene_root: Optional[Path],
    scene_root_override: Optional[Path],
    mode: str,
    renderer_choice: str,
    preprocess_overwrite: bool,
    preprocess_max_frames: int,
    preprocess_max_stride: int,
    preprocess_streaming: bool,
    preprocess_streaming_overlap: int,
    preprocess_streaming_global_guide: bool,
    preprocess_image_ext: str,
    preprocess_model_name: str,
    preprocess_process_res: int,
    preprocess_process_res_method: str,
    preprocess_export_gs_video: bool,
    preprocess_runtime_export_format: str,
    preprocess_runtime_export_fps: int,
    preprocess_use_ray_pose: bool,
    preprocess_ref_view_strategy: str,
    alignment_num_frames: int,
    alignment_stride: int,
    alignment_offset: int,
    alignment_conf_profile: str,
    alignment_conf_percentile: float,
    conf_mask_sky: bool,
    conf_mask_sky_depth_band: bool,
    conf_sky_depth_band_percent: Optional[float],
    conf_mask_white_background: bool,
    conf_white_bg_min_rgb: Optional[float],
    conf_white_bg_max_channel_delta: Optional[float],
    conf_white_bg_grow_px: Optional[int],
    conf_mask_min_depth_range_percent: bool,
    conf_min_depth_range_percent: Optional[float],
    conf_mask_min_depth_range_meters: bool,
    conf_min_depth_range_meters: Optional[float],
    conf_mask_depth_edges: bool,
    conf_edge_rtol: Optional[float],
    conf_edge_atol: Optional[float],
    conf_edge_kernel_size: int,
    conf_mask_max_depth: bool,
    conf_max_depth_rtol: Optional[float],
    conf_max_depth_atol: Optional[float],
    stage1_extra: dict[str, object],
    stage2_extra: dict[str, object],
    stage31_extra: dict[str, object],
    gs_extra: dict[str, object],
    inverse_epochs: Optional[int],
    gs_num_iters: Optional[int],
    dry_run: bool,
) -> list[str]:
    command = [sys.executable, "-u", str(PROJECT_ROOT / "run_reconstruction.py")]

    if existing_scene_root is not None:
        command += ["--config.root-path", str(existing_scene_root)]
    elif input_video is not None:
        command += ["--config.input-video", str(input_video)]
        if scene_root_override is not None:
            command += ["--config.scene-root", str(scene_root_override)]
    else:
        raise ValueError("Either input_video or existing_scene_root must be provided.")

    command += ["--config.mode", mode]
    if renderer_choice != "auto":
        command += ["--config.renderer", renderer_choice]

    command += ["--config.preprocess-max-frames", str(preprocess_max_frames)]
    command += ["--config.preprocess-max-stride", str(preprocess_max_stride)]
    if preprocess_streaming:
        command += ["--config.preprocess-streaming"]
    if preprocess_streaming_global_guide:
        command += ["--config.preprocess-streaming-global-guide"]
    command += ["--config.preprocess-streaming-overlap", str(preprocess_streaming_overlap)]
    command += ["--config.preprocess-image-ext", preprocess_image_ext]
    command += ["--config.preprocess-model-name", preprocess_model_name]
    command += ["--config.preprocess-process-res", str(preprocess_process_res)]
    command += ["--config.preprocess-process-res-method", preprocess_process_res_method]
    command += ["--config.preprocess-ref-view-strategy", preprocess_ref_view_strategy]
    command += ["--config.preprocess-runtime-export-format", preprocess_runtime_export_format]
    command += ["--config.preprocess-runtime-export-fps", str(preprocess_runtime_export_fps)]
    if preprocess_export_gs_video:
        command += ["--config.preprocess-export-gs-video"]
    if preprocess_use_ray_pose:
        command += ["--config.preprocess-use-ray-pose"]
    command += ["--config.stage0-alignment.num-frames", str(alignment_num_frames)]
    command += ["--config.stage0-alignment.stride", str(alignment_stride)]
    command += ["--config.stage0-alignment.offset", str(alignment_offset)]
    _append_confidence_filter_args(
        command,
        prefix="config.stage0-alignment",
        profile=alignment_conf_profile,
        percentile=alignment_conf_percentile,
    )
    _append_sky_mask_args(
        command,
        prefix="config.stage0-alignment",
        enabled=conf_mask_sky,
    )
    _append_sky_depth_band_args(
        command,
        prefix="config.stage0-alignment",
        enabled=conf_mask_sky_depth_band,
        band_percent=conf_sky_depth_band_percent,
    )
    _append_white_background_args(
        command,
        prefix="config.stage0-alignment",
        enabled=conf_mask_white_background,
        min_rgb=conf_white_bg_min_rgb,
        max_channel_delta=conf_white_bg_max_channel_delta,
        grow_px=conf_white_bg_grow_px,
    )
    _append_min_depth_range_args(
        command,
        prefix="config.stage0-alignment",
        enabled_percent=conf_mask_min_depth_range_percent,
        min_depth_range_percent=conf_min_depth_range_percent,
        enabled_meters=conf_mask_min_depth_range_meters,
        min_depth_range_meters=conf_min_depth_range_meters,
    )
    _append_depth_edge_args(
        command,
        prefix="config.stage0-alignment",
        enabled=conf_mask_depth_edges,
        edge_rtol=conf_edge_rtol,
        edge_atol=conf_edge_atol,
        edge_kernel_size=conf_edge_kernel_size,
    )
    _append_max_depth_args(
        command,
        prefix="config.stage0-alignment",
        enabled=conf_mask_max_depth,
        max_depth_rtol=conf_max_depth_rtol,
        max_depth_atol=conf_max_depth_atol,
    )
    _append_stage1_extra_args(command, prefix="config.stage1", settings=stage1_extra)
    _append_stage2_extra_args(command, prefix="config.stage2", settings=stage2_extra)
    _append_stage31_extra_args(command, prefix="config.stage31", settings=stage31_extra)
    _append_gs_extra_args(command, prefix="config.gs", settings=gs_extra)

    if preprocess_overwrite:
        command += ["--config.preprocess-overwrite"]
    if inverse_epochs is not None:
        command += ["--config.stage31.n-epochs", str(inverse_epochs)]
    if gs_num_iters is not None:
        command += ["--config.gs.num-iters", str(gs_num_iters)]
    if dry_run:
        command += ["--config.dry-run"]

    return command


def _build_stage0_command(
    *,
    input_video: Optional[Path],
    frames_dir: Optional[Path],
    scene_root_override: Optional[Path],
    preprocess_overwrite: bool,
    preprocess_max_frames: int,
    preprocess_max_stride: int,
    preprocess_streaming: bool,
    preprocess_streaming_overlap: int,
    preprocess_streaming_global_guide: bool,
    preprocess_image_ext: str,
    preprocess_model_name: str,
    preprocess_process_res: int = 768,
    preprocess_process_res_method: str = "upper_bound_resize",
    preprocess_export_gs_video: bool = False,
    preprocess_runtime_export_format: str = DEFAULT_STAGE0_RUNTIME_EXPORT_FORMAT,
    preprocess_runtime_export_fps: float = 30.0,
    preprocess_use_ray_pose: bool = False,
    preprocess_fixed_camera: bool = False,
    preprocess_fixed_camera_fov_degrees: float = 60.0,
    preprocess_ref_view_strategy: str = DEFAULT_STAGE0_REF_VIEW_STRATEGY,
    alignment_num_frames: int = 50,
    alignment_stride: int = 2,
    alignment_offset: int = 0,
    conf_profile: str = "default_mixed",
    conf_percentile: float = 80.0,
    conf_mask_sky: bool = False,
    conf_mask_sky_depth_band: bool = False,
    conf_sky_depth_band_percent: Optional[float] = None,
    conf_mask_white_background: bool = False,
    conf_white_bg_min_rgb: Optional[float] = 220.0,
    conf_white_bg_max_channel_delta: Optional[float] = 25.0,
    conf_white_bg_grow_px: Optional[int] = 0,
    conf_mask_min_depth_range_percent: bool = True,
    conf_min_depth_range_percent: Optional[float] = 50.0,
    conf_mask_min_depth_range_meters: bool = False,
    conf_min_depth_range_meters: Optional[float] = 3.0,
    conf_mask_depth_edges: bool = True,
    conf_edge_rtol: Optional[float] = 0.1,
    conf_edge_atol: Optional[float] = None,
    conf_edge_kernel_size: int = 3,
    conf_mask_max_depth: bool = False,
    conf_max_depth_rtol: Optional[float] = 0.001,
    conf_max_depth_atol: Optional[float] = None,
    conf_voxel_min_count_percentile: Optional[float] = 50.0,
    prepare_skip_before_non_rigid: bool = False,
    prepare_skip_debug_masks: Optional[bool] = None,
    skip_frame_materialization: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "preprocess_video.py"),
        "--image_ext",
        preprocess_image_ext,
        "--model_name",
        preprocess_model_name,
        "--streaming_overlap",
        str(preprocess_streaming_overlap),
        "--runtime_export_format",
        preprocess_runtime_export_format,
        "--runtime_export_fps",
        _format_fps_for_cli(float(preprocess_runtime_export_fps)),
        "--process_res",
        str(preprocess_process_res),
        "--process_res_method",
        preprocess_process_res_method,
        "--ref_view_strategy",
        preprocess_ref_view_strategy,
    ]
    if preprocess_export_gs_video:
        command += ["--export_gs_video"]
    if preprocess_use_ray_pose:
        command += ["--use_ray_pose"]
    if preprocess_fixed_camera:
        command += [
            "--fixed_camera",
            "--fixed_camera_fov_degrees",
            _format_fps_for_cli(float(preprocess_fixed_camera_fov_degrees)),
        ]
    if preprocess_streaming:
        command += ["--streaming"]
    if preprocess_streaming_global_guide:
        command += ["--streaming_global_guide"]
    if input_video is not None:
        command += [
            "--input_video",
            str(input_video),
            "--max_frames",
            str(preprocess_max_frames),
            "--max_stride",
            str(preprocess_max_stride),
        ]
    elif frames_dir is not None:
        command += ["--frames_dir", str(frames_dir)]
    else:
        raise ValueError("Stage 0 needs either an input video or a frames directory.")
    if scene_root_override is not None:
        command += ["--scene_root", str(scene_root_override)]
    if preprocess_overwrite:
        command += ["--overwrite"]
    _append_prepare_alignment_args(
        command,
        num_frames=alignment_num_frames,
        stride=alignment_stride,
        offset=alignment_offset,
        conf_profile=conf_profile,
        conf_percentile=conf_percentile,
        conf_mask_sky=conf_mask_sky,
        conf_mask_sky_depth_band=conf_mask_sky_depth_band,
        conf_sky_depth_band_percent=conf_sky_depth_band_percent,
        conf_mask_white_background=conf_mask_white_background,
        conf_white_bg_min_rgb=conf_white_bg_min_rgb,
        conf_white_bg_max_channel_delta=conf_white_bg_max_channel_delta,
        conf_white_bg_grow_px=conf_white_bg_grow_px,
        conf_mask_min_depth_range_percent=conf_mask_min_depth_range_percent,
        conf_min_depth_range_percent=conf_min_depth_range_percent,
        conf_mask_min_depth_range_meters=conf_mask_min_depth_range_meters,
        conf_min_depth_range_meters=conf_min_depth_range_meters,
        conf_mask_depth_edges=conf_mask_depth_edges,
        conf_edge_rtol=conf_edge_rtol,
        conf_edge_atol=conf_edge_atol,
        conf_edge_kernel_size=conf_edge_kernel_size,
        conf_mask_max_depth=conf_mask_max_depth,
        conf_max_depth_rtol=conf_max_depth_rtol,
        conf_max_depth_atol=conf_max_depth_atol,
        conf_voxel_min_count_percentile=conf_voxel_min_count_percentile,
    )
    if prepare_skip_before_non_rigid:
        command.append("--prepare_skip_before_non_rigid")
    if skip_frame_materialization:
        command.append("--skip_frame_materialization")
    if prepare_skip_debug_masks is None:
        prepare_skip_debug_masks = prepare_skip_before_non_rigid
    if prepare_skip_debug_masks:
        command.append("--prepare_skip_debug_masks")
    return command


def _build_stage1_command(
    *,
    scene_root: Path,
    run_dir: Path,
    stage1_extra: dict[str, object],
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "-m",
        "frame_to_model_icp",
        "--config.root-path",
        str(scene_root),
        "--config.out-path",
        str(run_dir),
    ]
    _append_stage1_extra_args(command, prefix="config", settings=stage1_extra)
    return command


def _build_stage2_command(*, scene_root: Path, run_name: str, stage2_extra: dict[str, object]) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "-m",
        "global_optimization",
        "--config.root-path",
        str(scene_root),
        "--config.run",
        run_name,
    ]
    _append_stage2_extra_args(command, prefix="config", settings=stage2_extra)
    return command


def _build_stage31_out_path(run_dir: Path, checkpoint_subdir: str) -> Path:
    return (run_dir / f"inverse_deformation_{checkpoint_subdir}").resolve()


def _build_stage31_command(
    *,
    scene_root: Path,
    run_name: str,
    run_dir: Path,
    checkpoint_subdir: str,
    inverse_epochs: Optional[int],
    stage31_extra: dict[str, object],
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "-m",
        "train_inverse_deformation",
        "--config.root-path",
        str(scene_root),
        "--config.run",
        run_name,
        "--config.checkpoint-subdir",
        checkpoint_subdir,
        "--config.out-path",
        str(_build_stage31_out_path(run_dir, checkpoint_subdir)),
    ]
    _append_stage31_extra_args(command, prefix="config", settings=stage31_extra)
    if inverse_epochs is not None:
        command += ["--config.n-epochs", str(inverse_epochs)]
    return command


def _build_stage32_out_dir(run_dir: Path, renderer: str, checkpoint_subdir: str) -> Path:
    return (run_dir / f"gs_{renderer}_{checkpoint_subdir}").resolve()


def _build_stage32_command(
    *,
    scene_root: Path,
    run_name: str,
    run_dir: Path,
    checkpoint_subdir: str,
    inverse_dir: Path,
    renderer: str,
    gs_num_iters: Optional[int],
    gs_extra: dict[str, object],
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "-m",
        "train_gs",
        "--config.root-path",
        str(scene_root),
        "--config.run",
        run_name,
        "--config.global-opt-subdir",
        checkpoint_subdir,
        "--config.inverse-deform-dir",
        str(inverse_dir),
        "--config.renderer",
        renderer,
        "--config.out-dir",
        str(_build_stage32_out_dir(run_dir, renderer, checkpoint_subdir)),
    ]
    _append_gs_extra_args(command, prefix="config", settings=gs_extra)
    original_images_dir = _default_original_images_dir(scene_root)
    if original_images_dir:
        command += ["--config.original-images-dir", original_images_dir]
    if gs_num_iters is not None:
        command += ["--config.num-iters", str(gs_num_iters)]
    return command


def _prepare_pipeline_run(
    *,
    source_mode: str,
    uploaded_video: Optional[str],
    existing_video_selection: object,
    existing_scene_root_selection: object,
    output_parent_selection: object,
    custom_scene_name: str,
) -> tuple[Optional[Path], Optional[Path], Optional[Path], Path, str]:
    normalized = str(source_mode or "").strip().lower()

    if normalized == "upload_video":
        if not uploaded_video:
            raise ValueError("Upload a video or choose a different source mode.")
        input_video = _copy_uploaded_video(uploaded_video)
        scene_root_override = _compose_scene_root_override(
            output_parent_selection,
            custom_scene_name,
            default_stem=input_video.parent.name
            if _is_managed_upload_path(input_video)
            else _safe_stem(input_video.name),
        )
        scene_root_override = (
            _ensure_unique_scene_root(scene_root_override) if scene_root_override is not None else None
        )
        effective_scene_root = scene_root_override or _ensure_unique_scene_root(
            _default_scene_root_for_video(input_video)
        )
        return input_video, None, scene_root_override, effective_scene_root, f"Copied upload to `{input_video}`"

    if normalized == "existing_video":
        if not _strip_quotes(existing_video_selection):
            raise ValueError("Choose an existing video or switch source mode.")
        input_video = _resolve_existing_file(existing_video_selection)
        scene_root_override = _compose_scene_root_override(
            output_parent_selection,
            custom_scene_name,
            default_stem=input_video.parent.name
            if _is_managed_upload_path(input_video)
            else _safe_stem(input_video.name),
        )
        effective_scene_root = scene_root_override or _default_scene_root_for_video(input_video)
        return input_video, None, scene_root_override, effective_scene_root, f"Using existing video `{input_video}`"

    if normalized != "existing_scene":
        raise ValueError("Choose a source mode: upload video, existing video, or existing scene root.")

    if not _strip_quotes(existing_scene_root_selection):
        raise ValueError("Choose an existing scene root or switch source mode.")
    existing_scene_root = _resolve_existing_dir(existing_scene_root_selection)
    return None, existing_scene_root, None, existing_scene_root, f"Using existing scene root `{existing_scene_root}`"


def _prepare_stage0_run(
    *,
    source_mode: str,
    uploaded_video: Optional[str],
    existing_video_selection: object,
    existing_frames_dir: object,
    output_parent_selection: object,
    custom_scene_name: str,
) -> tuple[Optional[Path], Optional[Path], Optional[Path], Path, str]:
    normalized = str(source_mode or "").strip().lower()

    if normalized == "upload_video":
        if not uploaded_video:
            raise ValueError("Upload a video or choose a different Stage 0 source mode.")
        input_video = _copy_uploaded_video(uploaded_video)
        scene_root_override = _compose_scene_root_override(
            output_parent_selection,
            custom_scene_name,
            default_stem=input_video.parent.name
            if _is_managed_upload_path(input_video)
            else _safe_stem(input_video.name),
        )
        scene_root_override = (
            _ensure_unique_scene_root(scene_root_override) if scene_root_override is not None else None
        )
        effective_scene_root = scene_root_override or _ensure_unique_scene_root(
            _default_scene_root_for_video(input_video)
        )
        return input_video, None, scene_root_override, effective_scene_root, f"Copied upload to `{input_video}`"

    if normalized == "existing_video":
        if not _strip_quotes(existing_video_selection):
            raise ValueError("Choose an existing video or switch Stage 0 source mode.")
        input_video = _resolve_existing_file(existing_video_selection)
        scene_root_override = _compose_scene_root_override(
            output_parent_selection,
            custom_scene_name,
            default_stem=input_video.parent.name
            if _is_managed_upload_path(input_video)
            else _safe_stem(input_video.name),
        )
        effective_scene_root = scene_root_override or _default_scene_root_for_video(input_video)
        return input_video, None, scene_root_override, effective_scene_root, f"Using existing video `{input_video}`"

    if normalized != "existing_frames":
        raise ValueError("Choose a Stage 0 source mode: upload video, existing video, or existing image folder.")

    if not _strip_quotes(existing_frames_dir):
        raise ValueError("Enter an existing image-folder path or switch Stage 0 source mode.")
    source_frames_dir = _resolve_existing_dir(existing_frames_dir)
    frames_dir = _copy_uploaded_frames_dir(source_frames_dir)
    scene_root_override = _compose_scene_root_override(
        output_parent_selection,
        custom_scene_name,
        default_stem=frames_dir.parent.name if _is_managed_upload_path(frames_dir) else _safe_stem(frames_dir.name),
    )
    effective_scene_root = scene_root_override or _ensure_unique_scene_root(
        _default_scene_root_for_frames_dir(frames_dir)
    )
    return (
        None,
        frames_dir,
        scene_root_override,
        effective_scene_root,
        f"Copied image folder `{source_frames_dir}` to `{frames_dir}`",
    )


def _resolve_run(scene_root: Path, run_name: str) -> tuple[str, Path]:
    run_name_text = _strip_quotes(run_name)
    if not run_name_text:
        raise ValueError("Choose a Stage 1 run directory.")
    run_dir = (scene_root / run_name_text).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    return run_name_text, run_dir


def _resolve_checkpoint_subdir(run_dir: Path, checkpoint_subdir: str) -> str:
    checkpoint_text = _strip_quotes(checkpoint_subdir)
    if not checkpoint_text:
        choices = _checkpoint_subdir_choices(run_dir)
        if not choices:
            raise ValueError("No checkpoint input directories are available for this run.")
        return choices[0]
    checkpoint_dir = run_dir / checkpoint_text
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint input directory not found: {checkpoint_dir}")
    return checkpoint_text


def _resolve_inverse_dir(run_dir: Path, inverse_dir_name: str) -> Path:
    inverse_name_text = _strip_quotes(inverse_dir_name)
    if not inverse_name_text:
        inverse_dirs = _list_inverse_dirs(run_dir)
        if not inverse_dirs:
            raise ValueError("No inverse deformation directory is available for this run.")
        return inverse_dirs[0]
    inverse_dir = (run_dir / inverse_name_text).resolve()
    if not inverse_dir.is_dir():
        raise FileNotFoundError(f"Inverse deformation directory not found: {inverse_dir}")
    return inverse_dir


def _run_command_generator(
    *,
    command: list[str],
    effective_scene_root: Path,
    input_note: str,
    stage_hint: str,
):
    _ensure_workspace_dirs()
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    log_path = RUNS_ROOT / f"{run_id}.log"
    started_at = time.time()
    state = {
        "run_id": run_id,
        "log_path": str(log_path),
        "scene_root": str(effective_scene_root),
    }

    env = os.environ.copy()
    _prepend_active_python_bin_to_path(env)
    _ensure_pytorch_cuda_allocator_conf(env)
    env["PYTHONUNBUFFERED"] = "1"

    with log_path.open("w", encoding="utf-8", buffering=1) as log_handle:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
        )

        with ACTIVE_RUNS_LOCK:
            ACTIVE_RUNS[run_id] = process

        try:
            while True:
                artifacts = (
                    _collect_scene_artifacts(effective_scene_root)
                    if effective_scene_root.exists()
                    else _placeholder_artifacts(effective_scene_root)
                )
                log_text = _tail_text(log_path)
                stage = _detect_current_stage(log_text, default_stage=stage_hint)
                elapsed = time.time() - started_at
                current_run_dir = _latest_run_dir_text(artifacts)
                primary_video = _choose_primary_preview(artifacts)
                secondary_video = _choose_secondary_preview(artifacts)
                report = _format_scene_report(artifacts)

                return_code = process.poll()
                if return_code is None:
                    status = _build_status_markdown(
                        state="Running",
                        stage=stage,
                        elapsed_seconds=elapsed,
                        scene_root=effective_scene_root,
                        log_path=log_path,
                        run_dir=current_run_dir,
                        command=command,
                        extra=input_note,
                    )
                    yield (
                        state,
                        status,
                        stage,
                        str(effective_scene_root),
                        current_run_dir,
                        report,
                        primary_video,
                        secondary_video,
                        _key_files_as_strings(artifacts),
                        log_text,
                    )
                    time.sleep(POLL_INTERVAL_SEC)
                    continue

                succeeded = return_code == 0
                final_state = "Finished" if succeeded else f"Exited with code {return_code}"
                status = _build_status_markdown(
                    state=final_state,
                    stage="Complete" if succeeded else stage,
                    elapsed_seconds=elapsed,
                    scene_root=effective_scene_root,
                    log_path=log_path,
                    run_dir=current_run_dir,
                    command=command,
                    extra=input_note,
                )
                yield (
                    state,
                    status,
                    "Complete" if succeeded else stage,
                    str(effective_scene_root),
                    current_run_dir,
                    report,
                    primary_video,
                    secondary_video,
                    _key_files_as_strings(artifacts),
                    log_text,
                )
                break
        finally:
            with ACTIVE_RUNS_LOCK:
                ACTIVE_RUNS.pop(run_id, None)


def _simple_divstream_status(
    *,
    state: str,
    phase: str,
    elapsed_seconds: float,
    output_path: Path,
    log_path: Path,
    extra: str = "",
) -> str:
    lines = [
        f"**State**: {state}",
        f"**Phase**: {phase}",
        f"**Elapsed**: `{_format_duration(elapsed_seconds)}`",
        f"**Output**: `{output_path}`",
        f"**Log**: `{log_path}`",
    ]
    if extra:
        lines.extend(["", extra])
    return "\n".join(lines)


def _cleanup_simple_divstream_job(job_root: Path) -> None:
    try:
        resolved_job = job_root.resolve()
        allowed_parents = [
            DIVSTREAM_JOBS_ROOT.resolve(),
            VDA_DIVSTREAM_JOBS_ROOT.resolve(),
            FLAT_DIVSTREAM_JOBS_ROOT.resolve(),
        ]
        if not any(_is_relative_to(resolved_job, parent) for parent in allowed_parents):
            return
    except Exception:
        return
    shutil.rmtree(resolved_job, ignore_errors=True)


def _run_simple_divstream_generator(
    uploaded_video,
    output_filename,
    divstream_compression_level,
    divstream_workers,
    export_before_non_rigid_ply,
    export_streaming_guide_ply,
    stage0_max_frames,
    stage0_max_stride,
    stage0_streaming,
    stage0_streaming_overlap,
    stage0_streaming_global_guide,
    stage0_image_ext,
    stage0_model_name,
    stage0_process_res,
    stage0_process_res_method,
    stage0_ref_view_strategy,
    stage0_use_ray_pose,
    fixed_camera,
    fixed_camera_fov_degrees,
    filter_conf_profile,
    filter_conf_percentile,
    filter_mask_sky,
    filter_mask_sky_depth_band,
    filter_sky_depth_band_percent,
    filter_mask_white_background,
    filter_white_bg_min_rgb,
    filter_white_bg_max_channel_delta,
    filter_white_bg_grow_px,
    filter_mask_min_depth_range_percent,
    filter_min_depth_range_percent,
    filter_mask_min_depth_range_meters,
    filter_min_depth_range_meters,
    filter_mask_depth_edges,
    filter_edge_rtol,
    filter_edge_atol,
    filter_edge_kernel_size,
    filter_mask_max_depth,
    filter_max_depth_rtol,
    filter_max_depth_atol,
):
    _ensure_workspace_dirs()

    try:
        input_video = _resolve_existing_file(uploaded_video)
        compression_level = _coerce_int(divstream_compression_level, label="Divstream Compression Level")
        if compression_level is None or compression_level < 1 or compression_level > 12:
            raise ValueError("Divstream Compression Level must be between 1 and 12.")
        workers = _coerce_int(divstream_workers, label="Divstream Workers")
        if workers is None or workers < 0:
            raise ValueError("Divstream Workers must be 0 or greater.")

        parsed_stage0_max_frames = _coerce_int(stage0_max_frames, label="DA3 Input Max Frames / Chunk Size")
        parsed_stage0_max_stride = _coerce_int(stage0_max_stride, label="DA3 Input Stride")
        if parsed_stage0_max_stride is None or parsed_stage0_max_stride < 1:
            raise ValueError("DA3 Input Stride must be at least 1.")
        source_fps = _probe_video_fps(input_video)
        fps = source_fps / float(parsed_stage0_max_stride)
        if not math.isfinite(fps) or fps <= 0.0:
            raise ValueError("Calculated output FPS must be greater than 0.")
        parsed_stage0_overlap = _coerce_int(stage0_streaming_overlap, label="DA3 Streaming Overlap")
        parsed_stage0_process_res = _coerce_int(stage0_process_res, label="DA3 Processing Resolution")
        parsed_stage0_image_ext = str(stage0_image_ext or "").strip().lstrip(".")
        parsed_stage0_model_name = str(stage0_model_name or "").strip()
        parsed_stage0_process_res_method = str(stage0_process_res_method or "").strip()
        parsed_stage0_ref_view_strategy = str(stage0_ref_view_strategy or "").strip()
        if not parsed_stage0_image_ext:
            raise ValueError("Image Extension is required.")
        if not parsed_stage0_model_name:
            raise ValueError("DA3 Model Name is required.")
        if not parsed_stage0_process_res_method:
            raise ValueError("DA3 Resolution Method is required.")
        if not parsed_stage0_ref_view_strategy:
            raise ValueError("DA3 Reference View is required.")

        parsed_filter_conf_profile = str(filter_conf_profile or "").strip()
        if not parsed_filter_conf_profile:
            raise ValueError("Confidence Mode is required.")
        parsed_filter_conf_percentile = _coerce_float(filter_conf_percentile, label="DA3 Confidence Percentile")
        parsed_filter_sky_depth_band_percent = _coerce_float(
            filter_sky_depth_band_percent,
            label="Sky Depth Band Percent",
            optional=True,
        )
        parsed_filter_white_bg_min_rgb = _coerce_float(
            filter_white_bg_min_rgb,
            label="White BG Min RGB",
            optional=True,
        )
        parsed_filter_white_bg_max_channel_delta = _coerce_float(
            filter_white_bg_max_channel_delta,
            label="White BG Max Channel Delta",
            optional=True,
        )
        parsed_filter_white_bg_grow_px = _coerce_int(filter_white_bg_grow_px, label="White BG Grow Pixels")
        if parsed_filter_white_bg_grow_px is None:
            parsed_filter_white_bg_grow_px = 0
        parsed_filter_min_depth_range_percent = _coerce_float(
            filter_min_depth_range_percent,
            label="Min Depth Range Percent",
            optional=True,
        )
        parsed_filter_min_depth_range_meters = _coerce_float(
            filter_min_depth_range_meters,
            label="Min Depth Range Metres",
            optional=True,
        )
        parsed_filter_edge_rtol = _coerce_float(filter_edge_rtol, label="Depth Edge Rel Threshold", optional=True)
        parsed_filter_edge_atol = _coerce_float(filter_edge_atol, label="Depth Edge Abs Threshold", optional=True)
        parsed_filter_edge_kernel_size = _coerce_int(filter_edge_kernel_size, label="Depth Edge Kernel")
        parsed_filter_max_depth_rtol = _coerce_float(
            filter_max_depth_rtol, label="Max Depth Rel Threshold", optional=True
        )
        parsed_filter_max_depth_atol = _coerce_float(
            filter_max_depth_atol, label="Max Depth Abs Threshold", optional=True
        )
        parsed_fixed_camera_fov_degrees = _coerce_float(
            fixed_camera_fov_degrees,
            label="Fixed Camera Horizontal FOV",
        )
        if parsed_fixed_camera_fov_degrees is None or not (1.0 < parsed_fixed_camera_fov_degrees < 179.0):
            raise ValueError("Fixed Camera Horizontal FOV must be between 1 and 179 degrees.")
        if export_streaming_guide_ply and not (bool(stage0_streaming) and bool(stage0_streaming_global_guide)):
            raise ValueError("Export Guide Pass PLY requires Use DA3 Streaming and Use Global Guide Pass.")
    except Exception as exc:
        yield {}, f"**Export failed:** `{exc}`", None, "", "", ""
        return

    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    job_root = (DIVSTREAM_JOBS_ROOT / run_id).resolve()
    scene_root = job_root / "scene"
    requested_output_path = (DIVSTREAM_OUTPUTS_ROOT / _safe_divstream_filename(output_filename, input_video)).resolve()
    output_path = _next_available_path(requested_output_path).resolve()
    debug_ply_output_path = None
    if export_before_non_rigid_ply:
        debug_ply_output_path = _next_available_path(
            DIVSTREAM_OUTPUTS_ROOT / f"{output_path.stem}_before_non_rigid_icp.ply"
        ).resolve()
    guide_ply_output_path = None
    if export_streaming_guide_ply:
        guide_ply_output_path = _next_available_path(
            DIVSTREAM_OUTPUTS_ROOT / f"{output_path.stem}_streaming_guide.ply"
        ).resolve()
    prep_num_frames = DIVSTREAM_DEBUG_PREP_NUM_FRAMES if export_before_non_rigid_ply else DIVSTREAM_PREP_NUM_FRAMES
    prep_run = scene_root / (
        f"frame_to_model_icp_{prep_num_frames}_{DIVSTREAM_PREP_STRIDE}_offset{DIVSTREAM_PREP_OFFSET}"
    )
    log_path = RUNS_ROOT / f"{run_id}_divstream.log"
    state = {
        "run_id": run_id,
        "log_path": str(log_path),
        "scene_root": str(scene_root),
        "output_path": str(output_path),
        "debug_ply_output_path": (str(debug_ply_output_path) if debug_ply_output_path is not None else ""),
        "guide_ply_output_path": (str(guide_ply_output_path) if guide_ply_output_path is not None else ""),
    }

    preprocess_command = _build_stage0_command(
        input_video=input_video,
        frames_dir=None,
        scene_root_override=scene_root,
        preprocess_overwrite=True,
        preprocess_max_frames=int(parsed_stage0_max_frames),
        preprocess_max_stride=int(parsed_stage0_max_stride),
        preprocess_streaming=bool(stage0_streaming),
        preprocess_streaming_overlap=int(parsed_stage0_overlap),
        preprocess_streaming_global_guide=bool(stage0_streaming_global_guide),
        preprocess_image_ext=parsed_stage0_image_ext,
        preprocess_model_name=parsed_stage0_model_name,
        preprocess_process_res=int(parsed_stage0_process_res),
        preprocess_process_res_method=parsed_stage0_process_res_method,
        preprocess_export_gs_video=False,
        preprocess_runtime_export_format="none",
        preprocess_runtime_export_fps=fps,
        preprocess_use_ray_pose=bool(stage0_use_ray_pose),
        preprocess_fixed_camera=bool(fixed_camera),
        preprocess_fixed_camera_fov_degrees=float(parsed_fixed_camera_fov_degrees),
        preprocess_ref_view_strategy=parsed_stage0_ref_view_strategy,
        alignment_num_frames=prep_num_frames,
        alignment_stride=DIVSTREAM_PREP_STRIDE,
        alignment_offset=DIVSTREAM_PREP_OFFSET,
        conf_profile=parsed_filter_conf_profile,
        conf_percentile=float(parsed_filter_conf_percentile),
        conf_mask_sky=bool(filter_mask_sky),
        conf_mask_sky_depth_band=bool(filter_mask_sky_depth_band),
        conf_sky_depth_band_percent=parsed_filter_sky_depth_band_percent,
        conf_mask_white_background=bool(filter_mask_white_background),
        conf_white_bg_min_rgb=parsed_filter_white_bg_min_rgb,
        conf_white_bg_max_channel_delta=parsed_filter_white_bg_max_channel_delta,
        conf_white_bg_grow_px=int(parsed_filter_white_bg_grow_px),
        conf_mask_min_depth_range_percent=bool(filter_mask_min_depth_range_percent),
        conf_min_depth_range_percent=parsed_filter_min_depth_range_percent,
        conf_mask_min_depth_range_meters=bool(filter_mask_min_depth_range_meters),
        conf_min_depth_range_meters=parsed_filter_min_depth_range_meters,
        conf_mask_depth_edges=bool(filter_mask_depth_edges),
        conf_edge_rtol=parsed_filter_edge_rtol,
        conf_edge_atol=parsed_filter_edge_atol,
        conf_edge_kernel_size=int(parsed_filter_edge_kernel_size),
        conf_mask_max_depth=bool(filter_mask_max_depth),
        conf_max_depth_rtol=parsed_filter_max_depth_rtol,
        conf_max_depth_atol=parsed_filter_max_depth_atol,
        conf_voxel_min_count_percentile=(None if int(parsed_stage0_max_stride) > 1 else 50.0),
        prepare_skip_before_non_rigid=not bool(export_before_non_rigid_ply),
        prepare_skip_debug_masks=True,
        skip_frame_materialization=True,
    )
    export_command = [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "export_depth_image_stream_bc7.py"),
        str(scene_root),
        "--output",
        str(output_path),
        "--fps",
        _format_fps_for_cli(fps),
        "--compression-level",
        str(compression_level),
        "--workers",
        str(workers),
        "--prep-run",
        str(prep_run),
        "--require-stage1-filters",
    ]
    if fixed_camera:
        export_command.append("--fixed-camera")
    guide_ply_command = None
    if guide_ply_output_path is not None:
        guide_ply_command = [
            sys.executable,
            "-u",
            str(PROJECT_ROOT / "export_streaming_guide_ply.py"),
            str(scene_root),
            "--output",
            str(guide_ply_output_path),
        ]

    env = os.environ.copy()
    _prepend_active_python_bin_to_path(env)
    _ensure_pytorch_cuda_allocator_conf(env)
    env["PYTHONUNBUFFERED"] = "1"
    started_at = time.time()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def run_phase(command: list[str], phase: str):
        with log_path.open("a", encoding="utf-8", buffering=1) as log_handle:
            log_handle.write(f"\n\n=== {phase} ===\n")
            log_handle.write(" ".join(command) + "\n\n")
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=env,
            )
            with ACTIVE_RUNS_LOCK:
                ACTIVE_RUNS[run_id] = process
            try:
                while True:
                    return_code = process.poll()
                    log_text = _tail_text(log_path)
                    elapsed = time.time() - started_at
                    if return_code is None:
                        yield (
                            None,
                            _simple_divstream_status(
                                state="Running",
                                phase=phase,
                                elapsed_seconds=elapsed,
                                output_path=output_path,
                                log_path=log_path,
                            ),
                            None,
                            log_text,
                        )
                        time.sleep(POLL_INTERVAL_SEC)
                        continue
                    if return_code != 0:
                        yield (
                            return_code,
                            _simple_divstream_status(
                                state=f"Exited with code {return_code}",
                                phase=phase,
                                elapsed_seconds=elapsed,
                                output_path=output_path,
                                log_path=log_path,
                                extra="The log below has the failure details.",
                            ),
                            None,
                            log_text,
                        )
                        return
                    yield (
                        0,
                        _simple_divstream_status(
                            state="Running",
                            phase=f"{phase} complete",
                            elapsed_seconds=elapsed,
                            output_path=output_path,
                            log_path=log_path,
                        ),
                        None,
                        log_text,
                    )
                    return
            finally:
                with ACTIVE_RUNS_LOCK:
                    ACTIVE_RUNS.pop(run_id, None)

    try:
        debug_ply_path_value = ""
        guide_ply_path_value = ""
        for return_code, status, file_value, log_text in run_phase(preprocess_command, "Preparing video"):
            yield state, status, file_value, debug_ply_path_value, guide_ply_path_value, log_text
            if return_code not in (None, 0):
                return

        if guide_ply_command is not None:
            for return_code, status, file_value, log_text in run_phase(guide_ply_command, "Writing guide PLY"):
                guide_ply_path_value = str(guide_ply_output_path) if guide_ply_output_path.is_file() else ""
                yield state, status, file_value, debug_ply_path_value, guide_ply_path_value, log_text
                if return_code not in (None, 0):
                    return

        if debug_ply_output_path is not None:
            debug_ply_source_path = prep_run / "before_non_rigid_icp.ply"
            if not debug_ply_source_path.is_file():
                yield (
                    state,
                    f"**Export failed:** expected debug PLY was not written: `{debug_ply_source_path}`",
                    None,
                    debug_ply_path_value,
                    guide_ply_path_value,
                    _tail_text(log_path),
                )
                return
            debug_ply_output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(debug_ply_source_path, debug_ply_output_path)
            debug_ply_path_value = str(debug_ply_output_path)

        for return_code, status, file_value, log_text in run_phase(export_command, "Writing divstream"):
            yield state, status, file_value, debug_ply_path_value, guide_ply_path_value, log_text
            if return_code not in (None, 0):
                return

        if not output_path.is_file():
            yield (
                state,
                f"**Export failed:** expected output was not written: `{output_path}`",
                None,
                debug_ply_path_value,
                guide_ply_path_value,
                _tail_text(log_path),
            )
            return

        _cleanup_simple_divstream_job(job_root)
        elapsed = time.time() - started_at
        extra_lines = [
            f"Output FPS: `{_format_fps_for_cli(fps)}` "
            f"(source `{_format_fps_for_cli(source_fps)}` / stride `{parsed_stage0_max_stride}`)",
            "",
            f"Size: `{output_path.stat().st_size / (1024.0 * 1024.0):.2f} MiB`",
        ]
        if debug_ply_path_value:
            extra_lines.extend(["", f"Debug PLY: `{debug_ply_path_value}`"])
        if guide_ply_path_value:
            extra_lines.extend(["", f"Guide PLY: `{guide_ply_path_value}`"])
        yield (
            state,
            _simple_divstream_status(
                state="Finished",
                phase="Complete",
                elapsed_seconds=elapsed,
                output_path=output_path,
                log_path=log_path,
                extra="\n".join(extra_lines),
            ),
            str(output_path),
            debug_ply_path_value,
            guide_ply_path_value,
            _tail_text(log_path),
        )
    except Exception as exc:
        yield state, f"**Export failed:** `{exc}`", None, "", "", _tail_text(log_path)


def _append_bool_cli_arg(command: list[str], *, name: str, value: bool, default: bool) -> None:
    if bool(value) == bool(default):
        return
    command.append(f"--{name}" if value else f"--no-{name}")


def _append_optional_cli_value(command: list[str], *, name: str, value: object) -> None:
    if value is None:
        command.extend([f"--{name}", "none"])
        return
    command.extend([f"--{name}", str(value)])


def _preview_flat_background_removal(
    uploaded_video,
    preview_frame_index,
    max_res,
    background_min_rgb,
    background_max_rgb,
    background_grow_px,
):
    try:
        input_video = _resolve_existing_file(uploaded_video)
        parsed_frame_index = _coerce_int(preview_frame_index, label="Preview Frame Index")
        parsed_max_res = _coerce_int(max_res, label="Video Max Resolution")
        parsed_background_min_rgb = _coerce_float(background_min_rgb, label="Background RGB Min")
        parsed_background_max_rgb = _coerce_float(background_max_rgb, label="Background RGB Max")
        parsed_background_grow_px = _coerce_int(background_grow_px, label="Background Grow Pixels")
        if parsed_frame_index is None or parsed_frame_index < 0:
            raise ValueError("Preview Frame Index must be non-negative.")
        if parsed_max_res is None or parsed_max_res == 0 or parsed_max_res < -1:
            raise ValueError("Video Max Resolution must be -1 or a positive integer.")
        if parsed_background_grow_px is None or parsed_background_grow_px < 0:
            raise ValueError("Background Grow Pixels must be non-negative.")

        from export_flat_background_divstream import build_background_removal_preview

        source, preview, status = build_background_removal_preview(
            input_video,
            frame_index=int(parsed_frame_index),
            max_res=int(parsed_max_res),
            background_min_rgb=float(parsed_background_min_rgb),
            background_max_rgb=float(parsed_background_max_rgb),
            background_grow_px=int(parsed_background_grow_px),
        )
        return source, preview, status
    except Exception as exc:
        return None, None, f"**Preview failed:** `{exc}`"


def _run_flat_divstream_generator(
    uploaded_video,
    output_filename,
    divstream_compression_level,
    divstream_workers,
    stride,
    max_frames,
    max_res,
    flat_depth_meters,
    fixed_camera_fov_degrees,
    background_min_rgb,
    background_max_rgb,
    background_grow_px,
):
    _ensure_workspace_dirs()

    try:
        input_video = _resolve_existing_file(uploaded_video)
        compression_level = _coerce_int(divstream_compression_level, label="Divstream Compression Level")
        if compression_level is None or compression_level < 1 or compression_level > 12:
            raise ValueError("Divstream Compression Level must be between 1 and 12.")
        workers = _coerce_int(divstream_workers, label="Divstream Workers")
        if workers is None or workers < 0:
            raise ValueError("Divstream Workers must be 0 or greater.")
        parsed_stride = _coerce_int(stride, label="Input Stride")
        if parsed_stride is None or parsed_stride < 1:
            raise ValueError("Input Stride must be at least 1.")
        parsed_max_frames = _coerce_int(max_frames, label="Max Output Frames")
        if parsed_max_frames is None or parsed_max_frames == 0 or parsed_max_frames < -1:
            raise ValueError("Max Output Frames must be -1 or a positive integer.")
        parsed_max_res = _coerce_int(max_res, label="Video Max Resolution")
        if parsed_max_res is None or parsed_max_res == 0 or parsed_max_res < -1:
            raise ValueError("Video Max Resolution must be -1 or a positive integer.")
        parsed_flat_depth_meters = _coerce_float(flat_depth_meters, label="Flat Depth Metres")
        if (
            parsed_flat_depth_meters is None
            or not math.isfinite(parsed_flat_depth_meters)
            or parsed_flat_depth_meters <= 0.0
        ):
            raise ValueError("Flat Depth Metres must be greater than 0.")
        parsed_fixed_camera_fov_degrees = _coerce_float(
            fixed_camera_fov_degrees,
            label="Fixed Camera Horizontal FOV",
        )
        if parsed_fixed_camera_fov_degrees is None or not (1.0 < parsed_fixed_camera_fov_degrees < 179.0):
            raise ValueError("Fixed Camera Horizontal FOV must be between 1 and 179 degrees.")
        parsed_background_min_rgb = _coerce_float(background_min_rgb, label="Background RGB Min")
        parsed_background_max_rgb = _coerce_float(background_max_rgb, label="Background RGB Max")
        if (
            parsed_background_min_rgb is None
            or parsed_background_max_rgb is None
            or parsed_background_min_rgb < 0.0
            or parsed_background_min_rgb > 255.0
            or parsed_background_max_rgb < 0.0
            or parsed_background_max_rgb > 255.0
            or parsed_background_min_rgb > parsed_background_max_rgb
        ):
            raise ValueError("Background RGB Min/Max must be a valid 0..255 range.")
        parsed_background_grow_px = _coerce_int(background_grow_px, label="Background Grow Pixels")
        if parsed_background_grow_px is None or parsed_background_grow_px < 0:
            raise ValueError("Background Grow Pixels must be non-negative.")

        source_fps = _probe_video_fps(input_video)
        fps = source_fps / float(parsed_stride)
        if not math.isfinite(fps) or fps <= 0.0:
            raise ValueError("Calculated output FPS must be greater than 0.")
    except Exception as exc:
        yield {}, f"**Export failed:** `{exc}`", None, ""
        return

    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    job_root = (FLAT_DIVSTREAM_JOBS_ROOT / run_id).resolve()
    scene_root = job_root / "scene"
    requested_output_path = (DIVSTREAM_OUTPUTS_ROOT / _safe_divstream_filename(output_filename, input_video)).resolve()
    output_path = _next_available_path(requested_output_path).resolve()
    log_path = RUNS_ROOT / f"{run_id}_flat_divstream.log"
    state = {
        "run_id": run_id,
        "log_path": str(log_path),
        "scene_root": str(scene_root),
        "output_path": str(output_path),
    }

    command = [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "export_flat_background_divstream.py"),
        "--input-video",
        str(input_video),
        "--scene-root",
        str(scene_root),
        "--output",
        str(output_path),
        "--stride",
        str(int(parsed_stride)),
        "--max-frames",
        str(int(parsed_max_frames)),
        "--max-res",
        str(int(parsed_max_res)),
        "--flat-depth-meters",
        str(float(parsed_flat_depth_meters)),
        "--fixed-camera-fov-degrees",
        _format_fps_for_cli(float(parsed_fixed_camera_fov_degrees)),
        "--background-min-rgb",
        str(float(parsed_background_min_rgb)),
        "--background-max-rgb",
        str(float(parsed_background_max_rgb)),
        "--background-grow-px",
        str(int(parsed_background_grow_px)),
        "--compression-level",
        str(int(compression_level)),
        "--workers",
        str(int(workers)),
    ]

    env = os.environ.copy()
    _prepend_active_python_bin_to_path(env)
    env["PYTHONUNBUFFERED"] = "1"
    started_at = time.time()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", encoding="utf-8", buffering=1) as log_handle:
        log_handle.write("\n\n=== Flat background divstream export ===\n")
        log_handle.write(" ".join(command) + "\n\n")
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
        )
        with ACTIVE_RUNS_LOCK:
            ACTIVE_RUNS[run_id] = process
        try:
            while True:
                return_code = process.poll()
                log_text = _tail_text(log_path)
                elapsed = time.time() - started_at
                if return_code is None:
                    yield (
                        state,
                        _simple_divstream_status(
                            state="Running",
                            phase="Writing flat-depth divstream",
                            elapsed_seconds=elapsed,
                            output_path=output_path,
                            log_path=log_path,
                        ),
                        None,
                        log_text,
                    )
                    time.sleep(POLL_INTERVAL_SEC)
                    continue
                if return_code != 0:
                    yield (
                        state,
                        _simple_divstream_status(
                            state=f"Exited with code {return_code}",
                            phase="Failed",
                            elapsed_seconds=elapsed,
                            output_path=output_path,
                            log_path=log_path,
                            extra="The log below has the failure details.",
                        ),
                        None,
                        log_text,
                    )
                    return
                break
        finally:
            with ACTIVE_RUNS_LOCK:
                ACTIVE_RUNS.pop(run_id, None)

    if not output_path.is_file():
        yield state, f"**Export failed:** expected output was not written: `{output_path}`", None, _tail_text(log_path)
        return

    _cleanup_simple_divstream_job(job_root)
    elapsed = time.time() - started_at
    yield (
        state,
        _simple_divstream_status(
            state="Finished",
            phase="Complete",
            elapsed_seconds=elapsed,
            output_path=output_path,
            log_path=log_path,
            extra=(
                f"Output FPS: `{_format_fps_for_cli(fps)}` "
                f"(source `{_format_fps_for_cli(source_fps)}` / stride `{parsed_stride}`)\n\n"
                f"Flat depth: `{float(parsed_flat_depth_meters):.6g}` metres\n\n"
                f"Background RGB range: `{float(parsed_background_min_rgb):.6g}.."
                f"{float(parsed_background_max_rgb):.6g}`; grow `{int(parsed_background_grow_px)}` px\n\n"
                f"Size: `{output_path.stat().st_size / (1024.0 * 1024.0):.2f} MiB`"
            ),
        ),
        str(output_path),
        _tail_text(log_path),
    )


def _run_vda_divstream_generator(
    uploaded_video,
    output_filename,
    divstream_compression_level,
    divstream_workers,
    vda_encoder,
    vda_metric,
    vda_relative_depth_inverse,
    vda_input_size,
    vda_decoder_micro_batch_size,
    vda_max_res,
    vda_max_frames,
    vda_stride,
    vda_fp32,
    vda_download_checkpoint,
    vda_fixed_camera_fov_degrees,
    vda_depth_scale,
    vda_depth_offset,
    filter_mask_min_depth_range_percent,
    filter_min_depth_range_percent,
    filter_mask_max_depth_range_percent,
    filter_max_depth_range_percent,
    filter_mask_min_depth_range_meters,
    filter_min_depth_range_meters,
    filter_mask_depth_edges,
    filter_edge_rtol,
    filter_edge_atol,
    filter_edge_kernel_size,
    filter_mask_max_depth,
    filter_max_depth_rtol,
    filter_max_depth_atol,
):
    _ensure_workspace_dirs()

    try:
        input_video = _resolve_existing_file(uploaded_video)
        compression_level = _coerce_int(divstream_compression_level, label="Divstream Compression Level")
        if compression_level is None or compression_level < 1 or compression_level > 12:
            raise ValueError("Divstream Compression Level must be between 1 and 12.")
        workers = _coerce_int(divstream_workers, label="Divstream Workers")
        if workers is None or workers < 0:
            raise ValueError("Divstream Workers must be 0 or greater.")

        parsed_encoder = str(vda_encoder or "").strip()
        if parsed_encoder not in {"vits", "vitb", "vitl"}:
            raise ValueError("VDA Encoder must be one of vits, vitb, or vitl.")
        parsed_input_size = _coerce_int(vda_input_size, label="VDA Input Size")
        parsed_decoder_micro_batch_size = _coerce_int(
            vda_decoder_micro_batch_size,
            label="VDA Decoder Micro-Batch Size",
        )
        parsed_max_res = _coerce_int(vda_max_res, label="VDA Max Resolution")
        parsed_max_frames = _coerce_int(vda_max_frames, label="VDA Max Frames")
        parsed_stride = _coerce_int(vda_stride, label="VDA Input Stride")
        if parsed_input_size is None or parsed_input_size < 14:
            raise ValueError("VDA Input Size must be at least 14.")
        if parsed_decoder_micro_batch_size is None or parsed_decoder_micro_batch_size < 1:
            raise ValueError("VDA Decoder Micro-Batch Size must be at least 1.")
        if parsed_max_res is None:
            parsed_max_res = -1
        if parsed_max_frames is None:
            parsed_max_frames = -1
        if parsed_stride is None or parsed_stride < 1:
            raise ValueError("VDA Input Stride must be at least 1.")

        source_fps = _probe_video_fps(input_video)
        fps = source_fps / float(parsed_stride)
        if not math.isfinite(fps) or fps <= 0.0:
            raise ValueError("Calculated output FPS must be greater than 0.")

        parsed_fixed_camera_fov_degrees = _coerce_float(
            vda_fixed_camera_fov_degrees,
            label="Fixed Camera Horizontal FOV",
        )
        if parsed_fixed_camera_fov_degrees is None or not (1.0 < parsed_fixed_camera_fov_degrees < 179.0):
            raise ValueError("Fixed Camera Horizontal FOV must be between 1 and 179 degrees.")
        parsed_depth_scale = _coerce_float(vda_depth_scale, label="VDA Depth Scale")
        parsed_depth_offset = _coerce_float(vda_depth_offset, label="VDA Depth Offset")
        if parsed_depth_scale is None or not math.isfinite(parsed_depth_scale):
            raise ValueError("VDA Depth Scale must be a finite number.")
        if parsed_depth_offset is None or not math.isfinite(parsed_depth_offset):
            raise ValueError("VDA Depth Offset must be a finite number.")

        parsed_filter_min_depth_range_percent = _coerce_float(
            filter_min_depth_range_percent,
            label="Min Depth Range Percent",
            optional=True,
        )
        parsed_filter_max_depth_range_percent = _coerce_float(
            filter_max_depth_range_percent,
            label="Max Depth Range Percent",
            optional=True,
        )
        parsed_filter_min_depth_range_meters = _coerce_float(
            filter_min_depth_range_meters,
            label="Min Depth Range Metres",
            optional=True,
        )
        parsed_filter_edge_rtol = _coerce_float(filter_edge_rtol, label="Depth Edge Rel Threshold", optional=True)
        parsed_filter_edge_atol = _coerce_float(filter_edge_atol, label="Depth Edge Abs Threshold", optional=True)
        parsed_filter_edge_kernel_size = _coerce_int(filter_edge_kernel_size, label="Depth Edge Kernel")
        parsed_filter_max_depth_rtol = _coerce_float(
            filter_max_depth_rtol, label="Max Depth Rel Threshold", optional=True
        )
        parsed_filter_max_depth_atol = _coerce_float(
            filter_max_depth_atol, label="Max Depth Abs Threshold", optional=True
        )
    except Exception as exc:
        yield {}, f"**Export failed:** `{exc}`", None, ""
        return

    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    job_root = (VDA_DIVSTREAM_JOBS_ROOT / run_id).resolve()
    scene_root = job_root / "scene"
    requested_output_path = (DIVSTREAM_OUTPUTS_ROOT / _safe_divstream_filename(output_filename, input_video)).resolve()
    output_path = _next_available_path(requested_output_path).resolve()
    log_path = RUNS_ROOT / f"{run_id}_vda_divstream.log"
    state = {
        "run_id": run_id,
        "log_path": str(log_path),
        "scene_root": str(scene_root),
        "output_path": str(output_path),
    }

    command = [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "export_vda_divstream.py"),
        "--input-video",
        str(input_video),
        "--scene-root",
        str(scene_root),
        "--output",
        str(output_path),
        "--encoder",
        parsed_encoder,
        "--input-size",
        str(int(parsed_input_size)),
        "--decoder-micro-batch-size",
        str(int(parsed_decoder_micro_batch_size)),
        "--max-res",
        str(int(parsed_max_res)),
        "--max-frames",
        str(int(parsed_max_frames)),
        "--stride",
        str(int(parsed_stride)),
        "--fixed-camera-fov-degrees",
        _format_fps_for_cli(float(parsed_fixed_camera_fov_degrees)),
        "--depth-scale",
        str(float(parsed_depth_scale)),
        "--depth-offset",
        str(float(parsed_depth_offset)),
        "--compression-level",
        str(int(compression_level)),
        "--workers",
        str(int(workers)),
    ]
    _append_bool_cli_arg(command, name="metric", value=bool(vda_metric), default=True)
    _append_bool_cli_arg(
        command,
        name="relative-depth-inverse",
        value=bool(vda_relative_depth_inverse),
        default=True,
    )
    _append_bool_cli_arg(command, name="download-checkpoint", value=bool(vda_download_checkpoint), default=True)
    if vda_fp32:
        command.append("--fp32")
    _append_bool_cli_arg(
        command,
        name="mask-min-depth-range-percent",
        value=bool(filter_mask_min_depth_range_percent),
        default=False,
    )
    command.extend(
        [
            "--min-depth-range-percent",
            str(50.0 if parsed_filter_min_depth_range_percent is None else parsed_filter_min_depth_range_percent),
        ]
    )
    _append_bool_cli_arg(
        command,
        name="mask-max-depth-range-percent",
        value=bool(filter_mask_max_depth_range_percent),
        default=False,
    )
    command.extend(
        [
            "--max-depth-range-percent",
            str(50.0 if parsed_filter_max_depth_range_percent is None else parsed_filter_max_depth_range_percent),
        ]
    )
    _append_bool_cli_arg(
        command,
        name="mask-min-depth-range-meters",
        value=bool(filter_mask_min_depth_range_meters),
        default=False,
    )
    command.extend(
        [
            "--min-depth-range-meters",
            str(3.0 if parsed_filter_min_depth_range_meters is None else parsed_filter_min_depth_range_meters),
        ]
    )
    _append_bool_cli_arg(command, name="mask-depth-edges", value=bool(filter_mask_depth_edges), default=True)
    _append_optional_cli_value(command, name="edge-rtol", value=parsed_filter_edge_rtol)
    _append_optional_cli_value(command, name="edge-atol", value=parsed_filter_edge_atol)
    command.extend(
        [
            "--edge-kernel-size",
            str(int(3 if parsed_filter_edge_kernel_size is None else parsed_filter_edge_kernel_size)),
        ]
    )
    _append_bool_cli_arg(command, name="mask-max-depth", value=bool(filter_mask_max_depth), default=False)
    _append_optional_cli_value(command, name="max-depth-rtol", value=parsed_filter_max_depth_rtol)
    _append_optional_cli_value(command, name="max-depth-atol", value=parsed_filter_max_depth_atol)

    env = os.environ.copy()
    _prepend_active_python_bin_to_path(env)
    _ensure_pytorch_cuda_allocator_conf(env)
    env["PYTHONUNBUFFERED"] = "1"
    started_at = time.time()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", encoding="utf-8", buffering=1) as log_handle:
        log_handle.write("\n\n=== VDA divstream export ===\n")
        log_handle.write(" ".join(command) + "\n\n")
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
        )
        with ACTIVE_RUNS_LOCK:
            ACTIVE_RUNS[run_id] = process
        try:
            while True:
                return_code = process.poll()
                log_text = _tail_text(log_path)
                elapsed = time.time() - started_at
                if return_code is None:
                    yield (
                        state,
                        _simple_divstream_status(
                            state="Running",
                            phase="Running Video Depth Anything",
                            elapsed_seconds=elapsed,
                            output_path=output_path,
                            log_path=log_path,
                        ),
                        None,
                        log_text,
                    )
                    time.sleep(POLL_INTERVAL_SEC)
                    continue
                if return_code != 0:
                    yield (
                        state,
                        _simple_divstream_status(
                            state=f"Exited with code {return_code}",
                            phase="Failed",
                            elapsed_seconds=elapsed,
                            output_path=output_path,
                            log_path=log_path,
                            extra="The log below has the failure details.",
                        ),
                        None,
                        log_text,
                    )
                    return
                break
        finally:
            with ACTIVE_RUNS_LOCK:
                ACTIVE_RUNS.pop(run_id, None)

    if not output_path.is_file():
        yield state, f"**Export failed:** expected output was not written: `{output_path}`", None, _tail_text(log_path)
        return

    _cleanup_simple_divstream_job(job_root)
    elapsed = time.time() - started_at
    yield (
        state,
        _simple_divstream_status(
            state="Finished",
            phase="Complete",
            elapsed_seconds=elapsed,
            output_path=output_path,
            log_path=log_path,
            extra=(
                f"Output FPS: `{_format_fps_for_cli(fps)}` "
                f"(source `{_format_fps_for_cli(source_fps)}` / stride `{parsed_stride}`)\n\n"
                f"Size: `{output_path.stat().st_size / (1024.0 * 1024.0):.2f} MiB`"
            ),
        ),
        str(output_path),
        _tail_text(log_path),
    )


def _emit_launch_failure(message: str):
    empty_state: dict[str, str] = {}
    yield empty_state, message, "Idle", "", "", "", None, None, [], ""


def _emit_stage_launch_failure(message: str):
    for outputs in _emit_launch_failure(message):
        yield (*outputs, *_stage_completion_control_no_updates())


def _run_stage_command_generator(
    *,
    command: list[str],
    effective_scene_root: Path,
    input_note: str,
    stage_hint: str,
):
    for outputs in _run_command_generator(
        command=command,
        effective_scene_root=effective_scene_root,
        input_note=input_note,
        stage_hint=stage_hint,
    ):
        control_outputs = (
            _stage_completion_control_outputs(outputs[3])
            if outputs[2] == "Complete"
            else _stage_completion_control_no_updates()
        )
        yield (*outputs, *control_outputs)


def _run_pipeline_generator(*args, **kwargs):
    (
        pipeline_source_mode,
        uploaded_video,
        existing_video_selection,
        existing_scene_root_selection,
        output_parent_selection,
        custom_scene_name,
        mode,
        renderer_choice,
        preprocess_overwrite,
        preprocess_max_frames,
        preprocess_max_stride,
        preprocess_streaming,
        preprocess_streaming_overlap,
        preprocess_streaming_global_guide,
        preprocess_image_ext,
        preprocess_model_name,
        preprocess_process_res,
        preprocess_process_res_method,
        preprocess_export_gs_video,
        preprocess_runtime_export_format,
        preprocess_runtime_export_fps,
        preprocess_use_ray_pose,
        preprocess_ref_view_strategy,
        alignment_num_frames,
        alignment_stride,
        alignment_offset,
        alignment_conf_profile,
        alignment_conf_percentile,
        conf_mask_sky,
        conf_mask_sky_depth_band,
        conf_sky_depth_band_percent,
        conf_mask_white_background,
        conf_white_bg_min_rgb,
        conf_white_bg_max_channel_delta,
        conf_white_bg_grow_px,
        conf_mask_min_depth_range_percent,
        conf_min_depth_range_percent,
        conf_mask_min_depth_range_meters,
        conf_min_depth_range_meters,
        conf_mask_depth_edges,
        conf_edge_rtol,
        conf_edge_atol,
        conf_edge_kernel_size,
        conf_mask_max_depth,
        conf_max_depth_rtol,
        conf_max_depth_atol,
        stage1_use_roma_matching,
        stage1_roma_version,
        stage1_roma_model,
        stage1_roma_num_samples,
        stage1_roma_certainty_threshold,
        stage1_roma_max_references,
        stage1_roma_reference_sampling,
        stage1_roma_loss_weight,
        stage1_roma_max_corr_dist,
        stage1_knn_backend,
        stage1_tensorboard,
        stage1_max_corr_dist,
        stage1_merge_voxel_size,
        stage1_icp_n_iter,
        stage1_icp_early_stopping_patience,
        stage1_icp_early_stopping_min_iters,
        stage1_icp_early_stopping_min_delta,
        stage1_icp_lr,
        stage1_icp_method,
        stage1_icp_local_twist_reg,
        stage1_icp_tv_reg,
        stage1_icp_tv_voxel_size,
        stage1_icp_tv_every_k,
        stage1_icp_tv_sample_ratio,
        stage1_icp_color_icp_weight,
        stage1_icp_color_icp_max_color_dist,
        stage1_icp_color_icp_k,
        stage1_save_intermediate_every,
        stage1_deform_log2_hashmap_size,
        stage1_deform_num_levels,
        stage1_deform_n_neurons,
        stage1_deform_n_hidden_layers,
        stage1_deform_min_res,
        stage1_deform_max_res,
        stage1_filter_points,
        stage1_filter_geom_sigma,
        stage1_filter_color_sigma,
        stage1_filter_worst_pct,
        stage1_filter_min_frames,
        stage1_filter_base_percentile,
        stage2_tensorboard,
        stage2_knn_backend,
        stage2_loo_loss_weight,
        stage2_loo_k_neighbors,
        stage2_loo_max_corr_dist,
        stage2_loo_normal_k,
        stage2_loo_kdtree_rebuild_every,
        stage2_loo_max_pairs_per_iter,
        stage2_loo_pairs_per_src,
        stage2_deform_chunk_size,
        stage2_anchor_loss_weight,
        stage2_anchor_n_samples,
        stage2_tv_reg,
        stage2_tv_voxel_size,
        stage2_tv_every_k,
        stage2_tv_sample_ratio,
        stage2_loo_color_icp_weight,
        stage2_loo_color_icp_k,
        stage2_loo_color_icp_max_color_dist,
        stage2_thin_shell_weight,
        stage2_lr,
        stage2_n_iters,
        stage2_save_intermediate_every_n,
        stage31_tensorboard,
        stage31_knn_backend,
        inverse_epochs,
        stage31_batch_size,
        stage31_lr,
        stage31_cycle_weight,
        stage31_magnitude_weight,
        stage31_smoothness_weight,
        stage31_num_forward_samples,
        stage31_num_interp_samples,
        stage31_regenerate_every,
        stage31_view_embed_dim,
        stage31_min_res,
        stage31_max_res,
        stage31_num_levels,
        stage31_log2_hashmap_size,
        stage31_n_neurons,
        stage31_n_hidden_layers,
        stage31_save_validation_plys,
        gs_tensorboard,
        gs_num_iters,
        gs_sh_degree,
        gs_sh_increase_every,
        gs_sh_full_from_iter,
        gs_sh_freeze_means_when_full_sh,
        gs_sh_reg_weight,
        gs_target_num_points,
        gs_optimize_cams,
        gs_lr_cams,
        gs_optimize_positions,
        gs_lr_positions,
        gs_lr_colors,
        gs_lr_opacities,
        gs_lr_scales,
        gs_lr_quats,
        gs_lr_sh0,
        gs_lr_shn,
        gs_deform_inverse_rotations,
        gs_initial_opacity,
        gs_initial_scale,
        gs_initial_flat_ratio,
        gs_scale_init,
        gs_knn_neighbors,
        gs_normal_k,
        gs_l1_weight,
        gs_lpips_weight,
        gs_opacity_reg_weight,
        gs_scale_reg_weight,
        gs_normal_consistency_weight,
        gs_distortion_weight,
        gs_alpha_reg_weight,
        gs_frames_per_iter,
        gs_log_every,
        gs_save_every,
        gs_eval_every,
        gs_lr_decay,
        gs_auto_eval,
        dry_run,
    ) = args

    try:
        input_video, existing_scene_root, scene_root_override, effective_scene_root, input_note = _prepare_pipeline_run(
            source_mode=pipeline_source_mode,
            uploaded_video=uploaded_video,
            existing_video_selection=existing_video_selection,
            existing_scene_root_selection=existing_scene_root_selection,
            output_parent_selection=output_parent_selection,
            custom_scene_name=custom_scene_name,
        )
        command = _build_pipeline_command(
            input_video=input_video,
            existing_scene_root=existing_scene_root,
            scene_root_override=scene_root_override,
            mode=mode,
            renderer_choice=renderer_choice,
            preprocess_overwrite=preprocess_overwrite,
            preprocess_max_frames=_coerce_int(preprocess_max_frames, label="Stage 0 Max Frames")
            or DEFAULT_STAGE0_MAX_FRAMES,
            preprocess_max_stride=_coerce_int(preprocess_max_stride, label="Stage 0 Input Stride")
            or DEFAULT_STAGE0_MAX_STRIDE,
            preprocess_streaming=bool(preprocess_streaming),
            preprocess_streaming_overlap=(
                _coerce_int(preprocess_streaming_overlap, label="DA3 Streaming Overlap")
                or DEFAULT_STAGE0_STREAMING_OVERLAP
            ),
            preprocess_streaming_global_guide=bool(preprocess_streaming_global_guide),
            preprocess_image_ext=str(preprocess_image_ext or "png"),
            preprocess_model_name=str(preprocess_model_name or "depth-anything/DA3NESTED-GIANT-LARGE"),
            preprocess_process_res=_coerce_int(preprocess_process_res, label="DA3 Processing Resolution") or 768,
            preprocess_process_res_method=str(preprocess_process_res_method or "upper_bound_resize"),
            preprocess_export_gs_video=bool(preprocess_export_gs_video),
            preprocess_runtime_export_format=_coerce_runtime_export_format(preprocess_runtime_export_format),
            preprocess_runtime_export_fps=(
                _coerce_int(preprocess_runtime_export_fps, label="Stage 0 Runtime Export FPS") or 30
            ),
            preprocess_use_ray_pose=bool(preprocess_use_ray_pose),
            preprocess_ref_view_strategy=str(preprocess_ref_view_strategy or "auto"),
            alignment_num_frames=_coerce_int(alignment_num_frames, label="Stage 0 Prep Num Frames") or 50,
            alignment_stride=_coerce_int(alignment_stride, label="Stage 0 Prep Stride") or 1,
            alignment_offset=_coerce_int(alignment_offset, label="Stage 0 Prep Offset") or 0,
            alignment_conf_profile=str(alignment_conf_profile or "default_mixed"),
            alignment_conf_percentile=_coerce_float(
                alignment_conf_percentile,
                label="DA3 Confidence Percentile",
            )
            or 80.0,
            conf_mask_sky=bool(conf_mask_sky),
            conf_mask_sky_depth_band=bool(conf_mask_sky_depth_band),
            conf_sky_depth_band_percent=_coerce_float(
                conf_sky_depth_band_percent,
                label="Sky Depth Band Percent",
                optional=True,
            ),
            conf_mask_white_background=bool(conf_mask_white_background),
            conf_white_bg_min_rgb=_coerce_float(
                conf_white_bg_min_rgb,
                label="White BG Min RGB",
                optional=True,
            ),
            conf_white_bg_max_channel_delta=_coerce_float(
                conf_white_bg_max_channel_delta,
                label="White BG Max Channel Delta",
                optional=True,
            ),
            conf_white_bg_grow_px=_coerce_int(conf_white_bg_grow_px, label="White BG Grow Pixels"),
            conf_mask_min_depth_range_percent=bool(conf_mask_min_depth_range_percent),
            conf_min_depth_range_percent=_coerce_float(
                conf_min_depth_range_percent,
                label="Min Depth Range Percent",
                optional=True,
            ),
            conf_mask_min_depth_range_meters=bool(conf_mask_min_depth_range_meters),
            conf_min_depth_range_meters=_coerce_float(
                conf_min_depth_range_meters,
                label="Min Depth Range Meters",
                optional=True,
            ),
            conf_mask_depth_edges=bool(conf_mask_depth_edges),
            conf_edge_rtol=_coerce_float(conf_edge_rtol, label="Depth Edge Rel Threshold", optional=True),
            conf_edge_atol=_coerce_float(conf_edge_atol, label="Depth Edge Abs Threshold", optional=True),
            conf_edge_kernel_size=_coerce_int(conf_edge_kernel_size, label="Depth Edge Kernel") or 3,
            conf_mask_max_depth=bool(conf_mask_max_depth),
            conf_max_depth_rtol=_coerce_float(
                conf_max_depth_rtol,
                label="Max Depth Rel Threshold",
                optional=True,
            ),
            conf_max_depth_atol=_coerce_float(
                conf_max_depth_atol,
                label="Max Depth Abs Threshold",
                optional=True,
            ),
            stage1_extra={
                "use_roma_matching": bool(stage1_use_roma_matching),
                "roma_version": str(stage1_roma_version or "v2"),
                "roma_model": str(stage1_roma_model or "outdoor"),
                "roma_num_samples": _coerce_int(stage1_roma_num_samples, label="RoMa Num Samples") or 5000,
                "roma_certainty_threshold": _coerce_float(
                    stage1_roma_certainty_threshold, label="RoMa Certainty Threshold"
                )
                or 0.5,
                "roma_max_references": _coerce_int(stage1_roma_max_references, label="RoMa Max References") or 20,
                "roma_reference_sampling": str(stage1_roma_reference_sampling or "recent_and_strided"),
                "roma_loss_weight": _coerce_float(stage1_roma_loss_weight, label="RoMa Loss Weight") or 1.0,
                "roma_max_corr_dist": _coerce_float(
                    stage1_roma_max_corr_dist, label="RoMa Max Corr Dist", optional=True
                ),
                "knn_backend": str(stage1_knn_backend or "cpu_kdtree"),
                "tensorboard": bool(stage1_tensorboard),
                "max_corr_dist": _coerce_float(stage1_max_corr_dist, label="Stage 1 Max Corr Dist") or 0.03,
                "merge_voxel_size": _coerce_float(stage1_merge_voxel_size, label="Stage 1 Merge Voxel Size") or 0.001,
                "icp_n_iter": _coerce_int(stage1_icp_n_iter, label="Stage 1 ICP Iterations") or 100,
                "icp_early_stopping_patience": _coerce_int(
                    stage1_icp_early_stopping_patience, label="Stage 1 Early Stop Patience", optional=True
                ),
                "icp_early_stopping_min_iters": _coerce_int(
                    stage1_icp_early_stopping_min_iters, label="Stage 1 Early Stop Min Iters"
                )
                or 25,
                "icp_early_stopping_min_delta": _coerce_float(
                    stage1_icp_early_stopping_min_delta, label="Stage 1 Early Stop Min Delta", optional=True
                ),
                "icp_lr": _coerce_float(stage1_icp_lr, label="Stage 1 ICP LR") or 1e-3,
                "icp_method": str(stage1_icp_method or "point2plane"),
                "icp_local_twist_reg": _coerce_float(stage1_icp_local_twist_reg, label="Stage 1 Local Twist Reg")
                or 0.0,
                "icp_tv_reg": _coerce_float(stage1_icp_tv_reg, label="Stage 1 TV Reg") or 50.0,
                "icp_tv_voxel_size": _coerce_float(stage1_icp_tv_voxel_size, label="Stage 1 TV Voxel Size") or 0.01,
                "icp_tv_every_k": _coerce_int(stage1_icp_tv_every_k, label="Stage 1 TV Every K") or 1,
                "icp_tv_sample_ratio": _coerce_float(
                    stage1_icp_tv_sample_ratio, label="Stage 1 TV Sample Ratio", optional=True
                ),
                "icp_color_icp_weight": _coerce_float(stage1_icp_color_icp_weight, label="Stage 1 Color ICP Weight")
                or 0.02,
                "icp_color_icp_max_color_dist": _coerce_float(
                    stage1_icp_color_icp_max_color_dist, label="Stage 1 Color ICP Max Color Dist", optional=True
                ),
                "icp_color_icp_k": _coerce_int(stage1_icp_color_icp_k, label="Stage 1 Color ICP K") or 10,
                "save_intermediate_every": _coerce_int(
                    stage1_save_intermediate_every, label="Stage 1 Save Intermediate Every"
                )
                or 10,
                "deform_log2_hashmap_size": _coerce_int(
                    stage1_deform_log2_hashmap_size, label="Stage 1 Deform Log2 Hashmap"
                )
                or 19,
                "deform_num_levels": _coerce_int(stage1_deform_num_levels, label="Stage 1 Deform Num Levels") or 24,
                "deform_n_neurons": _coerce_int(stage1_deform_n_neurons, label="Stage 1 Deform Neurons") or 64,
                "deform_n_hidden_layers": _coerce_int(
                    stage1_deform_n_hidden_layers, label="Stage 1 Deform Hidden Layers"
                )
                or 4,
                "deform_min_res": _coerce_int(stage1_deform_min_res, label="Stage 1 Deform Min Res") or 16,
                "deform_max_res": _coerce_int(stage1_deform_max_res, label="Stage 1 Deform Max Res") or 2048,
                "filter_points": bool(stage1_filter_points),
                "filter_geom_sigma": _coerce_float(stage1_filter_geom_sigma, label="Stage 1 Filter Geom Sigma") or 2.5,
                "filter_color_sigma": _coerce_float(stage1_filter_color_sigma, label="Stage 1 Filter Color Sigma")
                or 1.5,
                "filter_worst_pct": _coerce_float(stage1_filter_worst_pct, label="Stage 1 Filter Worst Pct") or 0.2,
                "filter_min_frames": _coerce_int(stage1_filter_min_frames, label="Stage 1 Filter Min Frames") or 2,
                "filter_base_percentile": str(stage1_filter_base_percentile or "p75"),
            },
            stage2_extra={
                "tensorboard": bool(stage2_tensorboard),
                "knn_backend": str(stage2_knn_backend or "cpu_kdtree"),
                "loo_loss_weight": _coerce_float(stage2_loo_loss_weight, label="Stage 2 LOO Loss Weight") or 1.0,
                "loo_k_neighbors": _coerce_int(stage2_loo_k_neighbors, label="Stage 2 LOO K Neighbors") or 5,
                "loo_max_corr_dist": _coerce_float(stage2_loo_max_corr_dist, label="Stage 2 LOO Max Corr Dist")
                or 0.03125,
                "loo_normal_k": _coerce_int(stage2_loo_normal_k, label="Stage 2 LOO Normal K") or 20,
                "loo_kdtree_rebuild_every": _coerce_int(
                    stage2_loo_kdtree_rebuild_every, label="Stage 2 KDT Rebuild Every"
                )
                or 50,
                "loo_max_pairs_per_iter": _coerce_int(
                    stage2_loo_max_pairs_per_iter, label="Stage 2 Max Pairs Per Iter", optional=True
                ),
                "loo_pairs_per_src": _coerce_int(stage2_loo_pairs_per_src, label="Stage 2 Pairs Per Src") or 1,
                "deform_chunk_size": _coerce_int(stage2_deform_chunk_size, label="Stage 2 Deform Chunk Size") or 50000,
                "anchor_loss_weight": _coerce_float(stage2_anchor_loss_weight, label="Stage 2 Anchor Loss Weight")
                or 1000.0,
                "anchor_n_samples": _coerce_int(stage2_anchor_n_samples, label="Stage 2 Anchor Samples") or 4096,
                "tv_reg": _coerce_float(stage2_tv_reg, label="Stage 2 TV Reg") or 50.0,
                "tv_voxel_size": _coerce_float(stage2_tv_voxel_size, label="Stage 2 TV Voxel Size") or 0.01,
                "tv_every_k": _coerce_int(stage2_tv_every_k, label="Stage 2 TV Every K") or 1,
                "tv_sample_ratio": _coerce_float(
                    stage2_tv_sample_ratio, label="Stage 2 TV Sample Ratio", optional=True
                ),
                "loo_color_icp_weight": _coerce_float(stage2_loo_color_icp_weight, label="Stage 2 Color ICP Weight")
                or 0.02,
                "loo_color_icp_k": _coerce_int(stage2_loo_color_icp_k, label="Stage 2 Color ICP K") or 10,
                "loo_color_icp_max_color_dist": _coerce_float(
                    stage2_loo_color_icp_max_color_dist, label="Stage 2 Color ICP Max Color Dist", optional=True
                ),
                "thin_shell_weight": _coerce_float(stage2_thin_shell_weight, label="Stage 2 Thin Shell Weight")
                or 1000.0,
                "lr": _coerce_float(stage2_lr, label="Stage 2 LR") or 1e-3,
                "n_iters": _coerce_int(stage2_n_iters, label="Stage 2 Iterations") or 150,
                "save_intermediate_every_n": _coerce_int(
                    stage2_save_intermediate_every_n, label="Stage 2 Save Intermediate Every"
                )
                or 50,
            },
            stage31_extra={
                "tensorboard": bool(stage31_tensorboard),
                "knn_backend": str(stage31_knn_backend or "cpu_kdtree"),
                "batch_size": _coerce_int(stage31_batch_size, label="Stage 3.1 Batch Size") or 8192,
                "lr": _coerce_float(stage31_lr, label="Stage 3.1 LR") or 1e-3,
                "cycle_weight": _coerce_float(stage31_cycle_weight, label="Stage 3.1 Cycle Weight") or 0.1,
                "magnitude_weight": _coerce_float(stage31_magnitude_weight, label="Stage 3.1 Magnitude Weight") or 1e-3,
                "smoothness_weight": _coerce_float(stage31_smoothness_weight, label="Stage 3.1 Smoothness Weight")
                or 1e-3,
                "num_forward_samples": _coerce_int(stage31_num_forward_samples, label="Stage 3.1 Forward Samples")
                or 10000,
                "num_interp_samples": _coerce_int(stage31_num_interp_samples, label="Stage 3.1 Interp Samples") or 5000,
                "regenerate_every": _coerce_int(stage31_regenerate_every, label="Stage 3.1 Regenerate Every") or 10,
                "view_embed_dim": _coerce_int(stage31_view_embed_dim, label="Stage 3.1 View Embed Dim") or 32,
                "min_res": _coerce_int(stage31_min_res, label="Stage 3.1 Min Res") or 16,
                "max_res": _coerce_int(stage31_max_res, label="Stage 3.1 Max Res") or 2048,
                "num_levels": _coerce_int(stage31_num_levels, label="Stage 3.1 Num Levels") or 16,
                "log2_hashmap_size": _coerce_int(stage31_log2_hashmap_size, label="Stage 3.1 Log2 Hashmap") or 19,
                "n_neurons": _coerce_int(stage31_n_neurons, label="Stage 3.1 Neurons") or 64,
                "n_hidden_layers": _coerce_int(stage31_n_hidden_layers, label="Stage 3.1 Hidden Layers") or 3,
                "save_validation_plys": bool(stage31_save_validation_plys),
            },
            gs_extra={
                "tensorboard": bool(gs_tensorboard),
                "sh_degree": _coerce_int(gs_sh_degree, label="GS SH Degree") or 3,
                "sh_increase_every": _coerce_int(gs_sh_increase_every, label="GS SH Increase Every") or 0,
                "sh_full_from_iter": _coerce_int(gs_sh_full_from_iter, label="GS SH Full From Iter") or 5000,
                "sh_freeze_means_when_full_sh": bool(gs_sh_freeze_means_when_full_sh),
                "sh_reg_weight": _coerce_float(gs_sh_reg_weight, label="GS SH Reg Weight") or 10.0,
                "target_num_points": _coerce_int(gs_target_num_points, label="GS Target Num Points") or 4000000,
                "optimize_cams": bool(gs_optimize_cams),
                "lr_cams": _coerce_float(gs_lr_cams, label="GS LR Cams") or 1e-4,
                "optimize_positions": bool(gs_optimize_positions),
                "lr_positions": _coerce_float(gs_lr_positions, label="GS LR Positions") or 1e-5,
                "lr_colors": _coerce_float(gs_lr_colors, label="GS LR Colors") or 2.5e-3,
                "lr_opacities": _coerce_float(gs_lr_opacities, label="GS LR Opacities") or 5e-2,
                "lr_scales": _coerce_float(gs_lr_scales, label="GS LR Scales") or 5e-3,
                "lr_quats": _coerce_float(gs_lr_quats, label="GS LR Quats") or 1e-3,
                "lr_sh0": _coerce_float(gs_lr_sh0, label="GS LR SH0") or 2.5e-3,
                "lr_shn": _coerce_float(gs_lr_shn, label="GS LR SHN") or (2.5e-3 / 20.0),
                "deform_inverse_rotations": bool(gs_deform_inverse_rotations),
                "initial_opacity": _coerce_float(gs_initial_opacity, label="GS Initial Opacity") or 0.5,
                "initial_scale": _coerce_float(gs_initial_scale, label="GS Initial Scale") or 0.005,
                "initial_flat_ratio": _coerce_float(gs_initial_flat_ratio, label="GS Initial Flat Ratio") or 0.1,
                "scale_init": str(gs_scale_init or "knn"),
                "knn_neighbors": _coerce_int(gs_knn_neighbors, label="GS KNN Neighbors") or 4,
                "normal_k": _coerce_int(gs_normal_k, label="GS Normal K") or 20,
                "l1_weight": _coerce_float(gs_l1_weight, label="GS L1 Weight") or 0.8,
                "lpips_weight": _coerce_float(gs_lpips_weight, label="GS LPIPS Weight") or 0.2,
                "opacity_reg_weight": _coerce_float(gs_opacity_reg_weight, label="GS Opacity Reg Weight") or 0.0,
                "scale_reg_weight": _coerce_float(gs_scale_reg_weight, label="GS Scale Reg Weight") or 0.0,
                "normal_consistency_weight": _coerce_float(
                    gs_normal_consistency_weight, label="GS Normal Consistency Weight"
                )
                or 0.05,
                "distortion_weight": _coerce_float(gs_distortion_weight, label="GS Distortion Weight") or 0.01,
                "alpha_reg_weight": _coerce_float(gs_alpha_reg_weight, label="GS Alpha Reg Weight") or 0.0,
                "frames_per_iter": _coerce_int(gs_frames_per_iter, label="GS Frames Per Iter") or 1,
                "log_every": _coerce_int(gs_log_every, label="GS Log Every") or 50,
                "save_every": _coerce_int(gs_save_every, label="GS Save Every") or 5000,
                "eval_every": _coerce_int(gs_eval_every, label="GS Eval Every") or 1000,
                "lr_decay": _coerce_float(gs_lr_decay, label="GS LR Decay") or 0.1,
                "auto_eval": bool(gs_auto_eval),
            },
            inverse_epochs=_coerce_int(inverse_epochs, label="Stage 3.1 Epoch Override", optional=True),
            gs_num_iters=_coerce_int(gs_num_iters, label="Stage 3.2 Iter Override", optional=True),
            dry_run=dry_run,
        )
    except Exception as exc:
        yield from _emit_launch_failure(f"**State**: Failed before launch\n\n`{exc}`")
        return

    yield from _run_command_generator(
        command=command,
        effective_scene_root=effective_scene_root,
        input_note=input_note,
        stage_hint="Starting",
    )


def _export_depth_volume(
    scene_root_selection,
    run_name,
    ply_dedup_enable,
    ply_dedup_radius,
    ply_normals_k,
    ply_chunk_size,
    depth_volume_resolution_scale,
):
    """Export the reconstructed scene as a DepthImageVolume for Unreal Engine."""
    import traceback as _tb

    if not scene_root_selection:
        return "**Error:** No scene selected. Select a scene root first."

    scene_root = scene_root_selection
    if not run_name:
        return "**Error:** No run name specified."

    resolution_scale = _coerce_int(depth_volume_resolution_scale, label="Depth Volume Resolution Scale") or 1
    if resolution_scale < 1:
        return "**Error:** Depth Volume Resolution Scale must be at least 1."

    output_name = "depth_image_volume_export"
    if resolution_scale != 1:
        output_name = f"{output_name}_x{resolution_scale}"
    output_dir = os.path.join(scene_root, run_name, output_name)

    try:
        from export_depth_image_volume import export_depth_image_volume

        result_path = export_depth_image_volume(
            scene_root=scene_root,
            run=run_name,
            output_dir=output_dir,
            device="cuda",
            dedup_enable=bool(ply_dedup_enable),
            dedup_radius=float(ply_dedup_radius or 0.001),
            normals_k=int(ply_normals_k or 16),
            normals_chunk_size=int(ply_chunk_size or 50000),
            resolution_scale=resolution_scale,
        )
        return f"**Export complete!** Output at:\n\n`{result_path}`\n\nContains `cameras.json` + depth/color/normal images ready for Unreal Engine import."
    except Exception as exc:
        tb = _tb.format_exc()
        return f"**Export failed:**\n\n```\n{exc}\n```\n\n<details><summary>Full traceback</summary>\n\n```\n{tb}\n```\n\n</details>"


def _export_runtime_format(
    scene_root_selection,
    runtime_export_format,
    runtime_export_fps,
    run_name=None,
):
    import traceback as _tb

    if not scene_root_selection:
        return "**Error:** No scene selected. Select a scene root first."

    export_format = _coerce_runtime_export_format(runtime_export_format)
    fps = _coerce_int(runtime_export_fps, label="Stage 0 Runtime Export FPS") or 30

    try:
        if export_format == "directstorage_stream":
            from export_depth_image_stream_bc7 import export_depth_image_stream_bc7

            output_path = export_depth_image_stream_bc7(
                scene_root=scene_root_selection,
                fps=int(fps),
                overwrite=True,
                prep_run=(run_name or None),
            )
            return (
                f"**Stage 0 runtime export complete!**\n\n"
                f"- Format: `DirectStorage stream`\n"
                f"- Output: `{output_path}`\n\n"
                "Point `ADepthImageVolumeDirectStorageActor` at this `.divstream` file."
            )

        if export_format == "kinect_rgbd_video":
            from export_stage0_kinect_video import export_stage0_kinect_video

            output_dir = export_stage0_kinect_video(
                scene_root=scene_root_selection,
                fps=int(fps),
                overwrite=True,
            )
            return (
                f"**Stage 0 runtime export complete!**\n\n"
                f"- Format: `Kinect RGBD Video (HAP Q)`\n"
                f"- Root: `{output_dir}`\n"
                f"- Video: `{os.path.join(output_dir, 'kinect_rgbd_hapq.mov')}`\n"
                f"- Sequence info: `{os.path.join(output_dir, 'sequence_info.json')}`"
            )

        if export_format == "packed_frame_sequence":
            from export_stage0_kinect_video import export_stage0_kinect_image_sequence

            output_dir = export_stage0_kinect_image_sequence(
                scene_root=scene_root_selection,
                fps=int(fps),
                overwrite=True,
            )
            frames_dir = os.path.join(output_dir, "frames")
            sequence_info_path = os.path.join(output_dir, "sequence_info.json")
            return (
                f"**Stage 0 runtime export complete!**\n\n"
                f"- Format: `Packed frame sequence`\n"
                f"- Root: `{output_dir}`\n"
                f"- Frames: `{frames_dir}`\n"
                f"- Sequence info: `{sequence_info_path}`\n\n"
                "Point `ADepthImageVolumeImageSequenceActor` at the sequence root or `frames/` directory."
            )

        if export_format == "packed_frame_sequence_depth8":
            from export_stage0_kinect_video import export_stage0_kinect_image_sequence_depth8

            output_dir = export_stage0_kinect_image_sequence_depth8(
                scene_root=scene_root_selection,
                fps=int(fps),
                overwrite=True,
            )
            frames_dir = os.path.join(output_dir, "frames")
            sequence_info_path = os.path.join(output_dir, "sequence_info.json")
            return (
                f"**Stage 0 runtime export complete!**\n\n"
                f"- Format: `Packed frame sequence (8-bit depth)`\n"
                f"- Root: `{output_dir}`\n"
                f"- Frames: `{frames_dir}`\n"
                f"- Sequence info: `{sequence_info_path}`\n\n"
                "Point `ADepthImageVolumeImageSequenceActor` at the sequence root or `frames/` directory."
            )

        return "**Error:** Runtime export format is set to `none`."
    except Exception as exc:
        tb = _tb.format_exc()
        return f"**Export failed:**\n\n```\n{exc}\n```\n\n<details><summary>Full traceback</summary>\n\n```\n{tb}\n```\n\n</details>"


def _export_packed_frame_sequence(
    scene_root_selection,
    packed_sequence_fps,
):
    import traceback as _tb

    if not scene_root_selection:
        return "**Error:** No scene selected. Select a scene root first."

    try:
        from export_stage0_kinect_video import export_stage0_kinect_image_sequence

        fps = _coerce_int(packed_sequence_fps, label="Packed Sequence FPS") or 30
        output_dir = export_stage0_kinect_image_sequence(
            scene_root=scene_root_selection,
            fps=int(fps),
        )
        frames_dir = os.path.join(output_dir, "frames")
        sequence_info_path = os.path.join(output_dir, "sequence_info.json")
        return (
            f"**Packed frame sequence exported!**\n\n"
            f"- Root: `{output_dir}`\n"
            f"- Frames: `{frames_dir}`\n"
            f"- Sequence info: `{sequence_info_path}`\n\n"
            "Point `ADepthImageVolumeImageSequenceActor` at the sequence root or `frames/` directory."
        )
    except Exception as exc:
        tb = _tb.format_exc()
        return f"**Export failed:**\n\n```\n{exc}\n```\n\n<details><summary>Full traceback</summary>\n\n```\n{tb}\n```\n\n</details>"


def _export_packed_frame_sequence_depth8(
    scene_root_selection,
    packed_sequence_fps,
):
    import traceback as _tb

    if not scene_root_selection:
        return "**Error:** No scene selected. Select a scene root first."

    try:
        from export_stage0_kinect_video import export_stage0_kinect_image_sequence_depth8

        fps = _coerce_int(packed_sequence_fps, label="Packed Sequence FPS") or 30
        output_dir = export_stage0_kinect_image_sequence_depth8(
            scene_root=scene_root_selection,
            fps=int(fps),
        )
        frames_dir = os.path.join(output_dir, "frames")
        sequence_info_path = os.path.join(output_dir, "sequence_info.json")
        return (
            f"**Packed 8-bit depth frame sequence exported!**\n\n"
            f"- Root: `{output_dir}`\n"
            f"- Frames: `{frames_dir}`\n"
            f"- Sequence info: `{sequence_info_path}`\n\n"
            "Point `ADepthImageVolumeImageSequenceActor` at the sequence root or `frames/` directory."
        )
    except Exception as exc:
        tb = _tb.format_exc()
        return f"**Export failed:**\n\n```\n{exc}\n```\n\n<details><summary>Full traceback</summary>\n\n```\n{tb}\n```\n\n</details>"


def _export_directstorage_stream(
    scene_root_selection,
    stream_fps,
):
    import traceback as _tb

    if not scene_root_selection:
        return "**Error:** No scene selected. Select a scene root first."

    try:
        from export_depth_image_stream_bc7 import export_depth_image_stream_bc7

        fps = _coerce_int(stream_fps, label="DirectStorage Stream FPS")
        output_path = export_depth_image_stream_bc7(
            scene_root=scene_root_selection,
            fps=(int(fps) if fps else None),
            overwrite=True,
        )
        return (
            f"**DirectStorage stream exported!**\n\n"
            f"- Stream: `{output_path}`\n\n"
            "Point `ADepthImageVolumeDirectStorageActor` at this `.divstream` file."
        )
    except Exception as exc:
        tb = _tb.format_exc()
        return f"**Export failed:**\n\n```\n{exc}\n```\n\n<details><summary>Full traceback</summary>\n\n```\n{tb}\n```\n\n</details>"


def _export_ply_with_normals(
    scene_root_selection,
    run_name,
    ply_checkpoint_source,
    ply_filename,
    ply_dedup_enable,
    ply_dedup_radius,
    ply_normals_k,
    ply_chunk_size,
):
    """Export the aligned point cloud as a PLY with voxel dedup and PCA normals."""
    import traceback as _tb

    if not scene_root_selection:
        return "**Error:** No scene selected."
    if not run_name:
        return "**Error:** No run name specified."

    run_dir = os.path.join(scene_root_selection, run_name)
    if not os.path.isdir(run_dir):
        return f"**Error:** Run directory not found: `{run_dir}`"

    # Resolve source PLY
    source_subdir = str(ply_checkpoint_source or "auto")
    if source_subdir == "auto":
        for candidate in ("after_global_optimization", "after_non_rigid_icp"):
            ply_path = os.path.join(run_dir, candidate, "aligned_points.ply")
            if os.path.isfile(ply_path):
                source_subdir = candidate
                break
        else:
            return "**Error:** No `aligned_points.ply` found in `after_global_optimization/` or `after_non_rigid_icp/`."
    ply_path = os.path.join(run_dir, source_subdir, "aligned_points.ply")
    if not os.path.isfile(ply_path):
        return f"**Error:** `{ply_path}` not found."

    try:
        import numpy as np
        import open3d as o3d
        from scipy.spatial import cKDTree

        pcd = o3d.io.read_point_cloud(ply_path)
        P = np.asarray(pcd.points, dtype=np.float32)
        has_colors = pcd.has_colors()
        RGB = (np.asarray(pcd.colors) * 255).astype(np.uint8) if has_colors else np.zeros((len(P), 3), dtype=np.uint8)

        if P.shape[0] == 0:
            return "**Error:** Source PLY has no points."

        # Filter non-finite
        finite_mask = np.isfinite(P).all(axis=1)
        P = P[finite_mask]
        RGB = RGB[finite_mask]

        # Voxel dedup
        dedup_radius = float(ply_dedup_radius or 0.001)
        if ply_dedup_enable and dedup_radius > 0:
            keys = np.floor(P / dedup_radius).astype(np.int64)
            keys_view = keys.view([("x", np.int64), ("y", np.int64), ("z", np.int64)])
            _, uniq_idx = np.unique(keys_view, return_index=True)
            P = P[uniq_idx]
            RGB = RGB[uniq_idx]

        M = P.shape[0]
        if M == 0:
            return "**Error:** No points remain after deduplication."

        # Normal estimation via PCA
        K = max(3, int(ply_normals_k or 16))
        chunk = max(10000, int(ply_chunk_size or 50000))
        tree = cKDTree(P)
        normals = np.zeros_like(P, dtype=np.float32)

        # Use centroid as fallback view direction (no per-point camera info)
        centroid = P.mean(axis=0)

        for s in range(0, M, chunk):
            e = min(M, s + chunk)
            _, idx = tree.query(P[s:e], k=K + 1)
            idx = idx[:, 1:]  # drop self
            nbrs = P[idx]
            cent = nbrs.mean(axis=1, keepdims=True)
            X = nbrs - cent
            cov = np.einsum("mki,mkj->mij", X, X) / max(1, K - 1)
            w, v = np.linalg.eigh(cov)
            n = v[:, :, 0]  # smallest eigenvector
            # Orient normals toward centroid (outward-facing)
            view_dir = P[s:e] - centroid
            sign = np.sign(np.sum(n * view_dir, axis=1, keepdims=True) + 1e-12)
            n = n * sign
            normals[s:e] = n.astype(np.float32)

        # Write PLY
        filename = str(ply_filename or "export_cloud.ply").strip()
        if not filename.endswith(".ply"):
            filename += ".ply"
        out_path = os.path.join(run_dir, source_subdir, filename)

        with open(out_path, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {M}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property float nx\n")
            f.write("property float ny\n")
            f.write("property float nz\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            for i in range(M):
                x, y, z = P[i]
                nx, ny, nz = normals[i]
                r, g, b = RGB[i]
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {nx:.6f} {ny:.6f} {nz:.6f} {int(r)} {int(g)} {int(b)}\n")

        return f"**PLY exported!** `{out_path}`\n\n- Source: `{source_subdir}/aligned_points.ply`\n- Points: **{M:,}** (dedup={ply_dedup_enable}, radius={dedup_radius})\n- Normals: k={K}"

    except Exception as exc:
        tb = _tb.format_exc()
        return f"**Export failed:**\n\n```\n{exc}\n```\n\n<details><summary>Full traceback</summary>\n\n```\n{tb}\n```\n\n</details>"


def _export_cloudcompare_edit_ply(
    scene_root_selection,
    run_name,
    cloudcompare_source,
    cloudcompare_filename,
):
    import traceback as _tb

    if not scene_root_selection:
        yield "**Error:** No scene selected."
        return
    if not run_name:
        yield "**Error:** No run name specified."
        return

    try:
        from utils.cloudcompare_prune import export_cloudcompare_edit_ply

        scene_root = _resolve_existing_dir(scene_root_selection)
        source = str(cloudcompare_source or "before_non_rigid_icp")
        yield (
            "**CloudCompare edit PLY export started.**\n\n"
            f"- Scene: `{scene_root}`\n"
            f"- Run: `{run_name}`\n"
            f"- Source: `{source}`\n\n"
            "Large point clouds can take a while to read and write."
        )
        result = export_cloudcompare_edit_ply(
            scene_root=scene_root,
            run=str(run_name),
            source=source,
            output_filename=str(cloudcompare_filename or "").strip() or None,
        )
        yield (
            "**CloudCompare edit PLY exported.**\n\n"
            f"- File: `{result['output_path']}`\n"
            f"- Source: `{result['source']}`\n"
            f"- Points: **{int(result['points']):,}**\n\n"
            "Delete points in CloudCompare, save the PLY, then use `Apply CloudCompare Edit PLY`. "
            "Transforms, resampling, and decimation are intentionally ignored."
        )
    except Exception as exc:
        tb = _tb.format_exc()
        yield f"**Export failed:**\n\n```\n{exc}\n```\n\n<details><summary>Full traceback</summary>\n\n```\n{tb}\n```\n\n</details>"


def _apply_cloudcompare_edit_ply(
    scene_root_selection,
    run_name,
    cloudcompare_source,
    cloudcompare_edited_ply,
    cloudcompare_output_suffix,
):
    import traceback as _tb

    no_run_update = gr.update()
    if not scene_root_selection:
        yield "**Error:** No scene selected.", no_run_update, gr.update()
        return
    if not run_name:
        yield "**Error:** No run name specified.", no_run_update, gr.update()
        return
    if not cloudcompare_edited_ply:
        yield "**Error:** Upload or select the edited CloudCompare PLY first.", no_run_update, gr.update()
        return

    try:
        from utils.cloudcompare_prune import apply_cloudcompare_edit_ply

        scene_root = _resolve_existing_dir(scene_root_selection)
        source = str(cloudcompare_source or "before_non_rigid_icp")
        yield (
            "**Applying CloudCompare edit.**\n\n"
            f"- Scene: `{scene_root}`\n"
            f"- Run: `{run_name}`\n"
            f"- Source: `{source}`\n"
            f"- Edited PLY: `{cloudcompare_edited_ply}`\n\n"
            "This is building the retained point mask and writing a new pruned run.",
            no_run_update,
            gr.update(),
        )
        result = apply_cloudcompare_edit_ply(
            scene_root=scene_root,
            run=str(run_name),
            source=source,
            edited_ply=cloudcompare_edited_ply,
            output_suffix=str(cloudcompare_output_suffix or "").strip() or None,
        )
        _clear_catalog_cache()
        artifacts = _collect_scene_artifacts(scene_root)
        run_names = [path.name for path in artifacts.run_dirs]
        new_run = str(result["new_run"])
        new_run_dir = str(result["new_run_dir"])
        if new_run not in run_names:
            run_names.append(new_run)

        if result["type"] == "before_non_rigid_prune":
            next_step = "Run Stage 1 using the new run."
        else:
            next_step = "Export Depth Volume using the new run."

        message = (
            "**CloudCompare edit applied.**\n\n"
            f"- New run: `{new_run}`\n"
            f"- Removed: **{int(result['removed_point_count']):,}** / "
            f"**{int(result['source_point_count']):,}** points\n"
            f"- Kept: **{int(result['kept_point_count']):,}** points\n"
            f"- New run dir: `{new_run_dir}`\n\n"
            f"{next_step}"
        )
        yield message, _choices_dropdown_update(run_names, new_run), new_run_dir
    except Exception as exc:
        tb = _tb.format_exc()
        yield (
            f"**Apply failed:**\n\n```\n{exc}\n```\n\n<details><summary>Full traceback</summary>\n\n```\n{tb}\n```\n\n</details>",
            no_run_update,
            gr.update(),
        )


def _run_stage_generator(*args, **kwargs):
    (
        stage_key,
        stage0_source_mode,
        stage0_uploaded_video,
        stage0_existing_video_selection,
        stage0_existing_frames_dir,
        stage0_output_parent_selection,
        stage0_custom_scene_name,
        stage0_overwrite,
        stage0_max_frames,
        stage0_max_stride,
        stage0_streaming,
        stage0_streaming_overlap,
        stage0_streaming_global_guide,
        stage0_image_ext,
        stage0_model_name,
        stage0_process_res,
        stage0_process_res_method,
        stage0_export_gs_video,
        stage0_runtime_export_format,
        stage0_runtime_export_fps,
        stage0_use_ray_pose,
        stage0_ref_view_strategy,
        stage_scene_root_selection,
        stage_run_name,
        stage1_num_frames,
        stage1_stride,
        stage1_offset,
        stage1_conf_profile,
        stage1_conf_percentile,
        stage1_conf_mask_sky,
        stage1_conf_mask_sky_depth_band,
        stage1_conf_sky_depth_band_percent,
        stage1_conf_mask_white_background,
        stage1_conf_white_bg_min_rgb,
        stage1_conf_white_bg_max_channel_delta,
        stage1_conf_white_bg_grow_px,
        stage1_conf_mask_min_depth_range_percent,
        stage1_conf_min_depth_range_percent,
        stage1_conf_mask_min_depth_range_meters,
        stage1_conf_min_depth_range_meters,
        stage1_conf_mask_depth_edges,
        stage1_conf_edge_rtol,
        stage1_conf_edge_atol,
        stage1_conf_edge_kernel_size,
        stage1_conf_mask_max_depth,
        stage1_conf_max_depth_rtol,
        stage1_conf_max_depth_atol,
        stage1_use_roma_matching,
        stage1_roma_version,
        stage1_roma_model,
        stage1_roma_num_samples,
        stage1_roma_certainty_threshold,
        stage1_roma_max_references,
        stage1_roma_reference_sampling,
        stage1_roma_loss_weight,
        stage1_roma_max_corr_dist,
        stage1_knn_backend,
        stage1_tensorboard,
        stage1_max_corr_dist,
        stage1_merge_voxel_size,
        stage1_icp_n_iter,
        stage1_icp_early_stopping_patience,
        stage1_icp_early_stopping_min_iters,
        stage1_icp_early_stopping_min_delta,
        stage1_icp_lr,
        stage1_icp_method,
        stage1_icp_local_twist_reg,
        stage1_icp_tv_reg,
        stage1_icp_tv_voxel_size,
        stage1_icp_tv_every_k,
        stage1_icp_tv_sample_ratio,
        stage1_icp_color_icp_weight,
        stage1_icp_color_icp_max_color_dist,
        stage1_icp_color_icp_k,
        stage1_save_intermediate_every,
        stage1_deform_log2_hashmap_size,
        stage1_deform_num_levels,
        stage1_deform_n_neurons,
        stage1_deform_n_hidden_layers,
        stage1_deform_min_res,
        stage1_deform_max_res,
        stage1_filter_points,
        stage1_filter_geom_sigma,
        stage1_filter_color_sigma,
        stage1_filter_worst_pct,
        stage1_filter_min_frames,
        stage1_filter_base_percentile,
        stage2_tensorboard,
        stage2_knn_backend,
        stage2_loo_loss_weight,
        stage2_loo_k_neighbors,
        stage2_loo_max_corr_dist,
        stage2_loo_normal_k,
        stage2_loo_kdtree_rebuild_every,
        stage2_loo_max_pairs_per_iter,
        stage2_loo_pairs_per_src,
        stage2_deform_chunk_size,
        stage2_anchor_loss_weight,
        stage2_anchor_n_samples,
        stage2_tv_reg,
        stage2_tv_voxel_size,
        stage2_tv_every_k,
        stage2_tv_sample_ratio,
        stage2_loo_color_icp_weight,
        stage2_loo_color_icp_k,
        stage2_loo_color_icp_max_color_dist,
        stage2_thin_shell_weight,
        stage2_lr,
        stage2_n_iters,
        stage2_save_intermediate_every_n,
        stage31_checkpoint_subdir,
        stage31_epochs,
        stage31_tensorboard,
        stage31_knn_backend,
        stage31_batch_size,
        stage31_lr,
        stage31_cycle_weight,
        stage31_magnitude_weight,
        stage31_smoothness_weight,
        stage31_num_forward_samples,
        stage31_num_interp_samples,
        stage31_regenerate_every,
        stage31_view_embed_dim,
        stage31_min_res,
        stage31_max_res,
        stage31_num_levels,
        stage31_log2_hashmap_size,
        stage31_n_neurons,
        stage31_n_hidden_layers,
        stage31_save_validation_plys,
        stage32_checkpoint_subdir,
        stage32_inverse_dir_name,
        stage32_renderer,
        stage32_num_iters,
        gs_tensorboard,
        gs_sh_degree,
        gs_sh_increase_every,
        gs_sh_full_from_iter,
        gs_sh_freeze_means_when_full_sh,
        gs_sh_reg_weight,
        gs_target_num_points,
        gs_optimize_cams,
        gs_lr_cams,
        gs_optimize_positions,
        gs_lr_positions,
        gs_lr_colors,
        gs_lr_opacities,
        gs_lr_scales,
        gs_lr_quats,
        gs_lr_sh0,
        gs_lr_shn,
        gs_deform_inverse_rotations,
        gs_initial_opacity,
        gs_initial_scale,
        gs_initial_flat_ratio,
        gs_scale_init,
        gs_knn_neighbors,
        gs_normal_k,
        gs_l1_weight,
        gs_lpips_weight,
        gs_opacity_reg_weight,
        gs_scale_reg_weight,
        gs_normal_consistency_weight,
        gs_distortion_weight,
        gs_alpha_reg_weight,
        gs_frames_per_iter,
        gs_log_every,
        gs_save_every,
        gs_eval_every,
        gs_lr_decay,
        gs_auto_eval,
    ) = args

    try:
        if stage_key == "stage0":
            input_video, frames_dir, scene_root_override, effective_scene_root, input_note = _prepare_stage0_run(
                source_mode=stage0_source_mode,
                uploaded_video=stage0_uploaded_video,
                existing_video_selection=stage0_existing_video_selection,
                existing_frames_dir=stage0_existing_frames_dir,
                output_parent_selection=stage0_output_parent_selection,
                custom_scene_name=stage0_custom_scene_name,
            )
            command = _build_stage0_command(
                input_video=input_video,
                frames_dir=frames_dir,
                scene_root_override=scene_root_override,
                preprocess_overwrite=stage0_overwrite,
                preprocess_max_frames=_coerce_int(stage0_max_frames, label="Stage 0 Max Frames")
                or DEFAULT_STAGE0_MAX_FRAMES,
                preprocess_max_stride=_coerce_int(stage0_max_stride, label="Stage 0 Input Stride")
                or DEFAULT_STAGE0_MAX_STRIDE,
                preprocess_streaming=bool(stage0_streaming),
                preprocess_streaming_overlap=(
                    _coerce_int(stage0_streaming_overlap, label="DA3 Streaming Overlap")
                    or DEFAULT_STAGE0_STREAMING_OVERLAP
                ),
                preprocess_streaming_global_guide=bool(stage0_streaming_global_guide),
                preprocess_image_ext=str(stage0_image_ext or "png"),
                preprocess_model_name=str(stage0_model_name or "depth-anything/DA3NESTED-GIANT-LARGE"),
                preprocess_process_res=_coerce_int(stage0_process_res, label="DA3 Processing Resolution") or 768,
                preprocess_process_res_method=str(stage0_process_res_method or "upper_bound_resize"),
                preprocess_export_gs_video=bool(stage0_export_gs_video),
                preprocess_runtime_export_format=_coerce_runtime_export_format(stage0_runtime_export_format),
                preprocess_runtime_export_fps=(
                    _coerce_int(stage0_runtime_export_fps, label="Stage 0 Runtime Export FPS") or 30
                ),
                preprocess_use_ray_pose=bool(stage0_use_ray_pose),
                preprocess_ref_view_strategy=str(stage0_ref_view_strategy or "auto"),
                alignment_num_frames=_coerce_int(stage1_num_frames, label="Stage 0 Prep Num Frames") or 50,
                alignment_stride=_coerce_int(stage1_stride, label="Stage 0 Prep Stride") or 1,
                alignment_offset=_coerce_int(stage1_offset, label="Stage 0 Prep Offset") or 0,
                conf_profile=str(stage1_conf_profile or "default_mixed"),
                conf_percentile=_coerce_float(stage1_conf_percentile, label="DA3 Confidence Percentile") or 80.0,
                conf_mask_sky=bool(stage1_conf_mask_sky),
                conf_mask_sky_depth_band=bool(stage1_conf_mask_sky_depth_band),
                conf_sky_depth_band_percent=_coerce_float(
                    stage1_conf_sky_depth_band_percent,
                    label="Sky Depth Band Percent",
                    optional=True,
                ),
                conf_mask_white_background=bool(stage1_conf_mask_white_background),
                conf_white_bg_min_rgb=_coerce_float(
                    stage1_conf_white_bg_min_rgb,
                    label="White BG Min RGB",
                    optional=True,
                ),
                conf_white_bg_max_channel_delta=_coerce_float(
                    stage1_conf_white_bg_max_channel_delta,
                    label="White BG Max Channel Delta",
                    optional=True,
                ),
                conf_white_bg_grow_px=_coerce_int(stage1_conf_white_bg_grow_px, label="White BG Grow Pixels") or 0,
                conf_mask_min_depth_range_percent=bool(stage1_conf_mask_min_depth_range_percent),
                conf_min_depth_range_percent=_coerce_float(
                    stage1_conf_min_depth_range_percent,
                    label="Min Depth Range Percent",
                    optional=True,
                ),
                conf_mask_min_depth_range_meters=bool(stage1_conf_mask_min_depth_range_meters),
                conf_min_depth_range_meters=_coerce_float(
                    stage1_conf_min_depth_range_meters,
                    label="Min Depth Range Meters",
                    optional=True,
                ),
                conf_mask_depth_edges=bool(stage1_conf_mask_depth_edges),
                conf_edge_rtol=_coerce_float(
                    stage1_conf_edge_rtol,
                    label="Depth Edge Relative Threshold",
                    optional=True,
                ),
                conf_edge_atol=_coerce_float(
                    stage1_conf_edge_atol,
                    label="Depth Edge Absolute Threshold",
                    optional=True,
                ),
                conf_edge_kernel_size=_coerce_int(stage1_conf_edge_kernel_size, label="Depth Edge Kernel") or 3,
                conf_mask_max_depth=bool(stage1_conf_mask_max_depth),
                conf_max_depth_rtol=_coerce_float(
                    stage1_conf_max_depth_rtol,
                    label="Max Depth Relative Threshold",
                    optional=True,
                ),
                conf_max_depth_atol=_coerce_float(
                    stage1_conf_max_depth_atol,
                    label="Max Depth Absolute Threshold",
                    optional=True,
                ),
            )
            stage_hint = "Stage 0"
        else:
            scene_root = _resolve_existing_dir(stage_scene_root_selection)
            effective_scene_root = scene_root
            input_note = f"Using existing scene root `{scene_root}`"

            if stage_key == "stage1":
                run_name, run_dir = _resolve_run(scene_root, stage_run_name)
                input_note = f"Using existing scene root `{scene_root}` and prepared run `{run_name}`"
                command = _build_stage1_command(
                    scene_root=scene_root,
                    run_dir=run_dir,
                    stage1_extra={
                        "use_roma_matching": bool(stage1_use_roma_matching),
                        "roma_version": str(stage1_roma_version or "v2"),
                        "roma_model": str(stage1_roma_model or "outdoor"),
                        "roma_num_samples": _coerce_int(stage1_roma_num_samples, label="RoMa Num Samples") or 5000,
                        "roma_certainty_threshold": _coerce_float(
                            stage1_roma_certainty_threshold, label="RoMa Certainty Threshold"
                        )
                        or 0.5,
                        "roma_max_references": _coerce_int(stage1_roma_max_references, label="RoMa Max References")
                        or 20,
                        "roma_reference_sampling": str(stage1_roma_reference_sampling or "recent_and_strided"),
                        "roma_loss_weight": _coerce_float(stage1_roma_loss_weight, label="RoMa Loss Weight") or 1.0,
                        "roma_max_corr_dist": _coerce_float(
                            stage1_roma_max_corr_dist, label="RoMa Max Corr Dist", optional=True
                        ),
                        "knn_backend": str(stage1_knn_backend or "cpu_kdtree"),
                        "tensorboard": bool(stage1_tensorboard),
                        "max_corr_dist": _coerce_float(stage1_max_corr_dist, label="Stage 1 Max Corr Dist") or 0.03,
                        "merge_voxel_size": _coerce_float(stage1_merge_voxel_size, label="Stage 1 Merge Voxel Size")
                        or 0.001,
                        "icp_n_iter": _coerce_int(stage1_icp_n_iter, label="Stage 1 ICP Iterations") or 100,
                        "icp_early_stopping_patience": _coerce_int(
                            stage1_icp_early_stopping_patience, label="Stage 1 Early Stop Patience", optional=True
                        ),
                        "icp_early_stopping_min_iters": _coerce_int(
                            stage1_icp_early_stopping_min_iters, label="Stage 1 Early Stop Min Iters"
                        )
                        or 25,
                        "icp_early_stopping_min_delta": _coerce_float(
                            stage1_icp_early_stopping_min_delta, label="Stage 1 Early Stop Min Delta", optional=True
                        ),
                        "icp_lr": _coerce_float(stage1_icp_lr, label="Stage 1 ICP LR") or 1e-3,
                        "icp_method": str(stage1_icp_method or "point2plane"),
                        "icp_local_twist_reg": _coerce_float(
                            stage1_icp_local_twist_reg, label="Stage 1 Local Twist Reg"
                        )
                        or 0.0,
                        "icp_tv_reg": _coerce_float(stage1_icp_tv_reg, label="Stage 1 TV Reg") or 50.0,
                        "icp_tv_voxel_size": _coerce_float(stage1_icp_tv_voxel_size, label="Stage 1 TV Voxel Size")
                        or 0.01,
                        "icp_tv_every_k": _coerce_int(stage1_icp_tv_every_k, label="Stage 1 TV Every K") or 1,
                        "icp_tv_sample_ratio": _coerce_float(
                            stage1_icp_tv_sample_ratio, label="Stage 1 TV Sample Ratio", optional=True
                        ),
                        "icp_color_icp_weight": _coerce_float(
                            stage1_icp_color_icp_weight, label="Stage 1 Color ICP Weight"
                        )
                        or 0.02,
                        "icp_color_icp_max_color_dist": _coerce_float(
                            stage1_icp_color_icp_max_color_dist, label="Stage 1 Color ICP Max Color Dist", optional=True
                        ),
                        "icp_color_icp_k": _coerce_int(stage1_icp_color_icp_k, label="Stage 1 Color ICP K") or 10,
                        "save_intermediate_every": _coerce_int(
                            stage1_save_intermediate_every, label="Stage 1 Save Intermediate Every"
                        )
                        or 10,
                        "deform_log2_hashmap_size": _coerce_int(
                            stage1_deform_log2_hashmap_size, label="Stage 1 Deform Log2 Hashmap"
                        )
                        or 19,
                        "deform_num_levels": _coerce_int(stage1_deform_num_levels, label="Stage 1 Deform Num Levels")
                        or 24,
                        "deform_n_neurons": _coerce_int(stage1_deform_n_neurons, label="Stage 1 Deform Neurons") or 64,
                        "deform_n_hidden_layers": _coerce_int(
                            stage1_deform_n_hidden_layers, label="Stage 1 Deform Hidden Layers"
                        )
                        or 4,
                        "deform_min_res": _coerce_int(stage1_deform_min_res, label="Stage 1 Deform Min Res") or 16,
                        "deform_max_res": _coerce_int(stage1_deform_max_res, label="Stage 1 Deform Max Res") or 2048,
                        "filter_points": bool(stage1_filter_points),
                        "filter_geom_sigma": _coerce_float(stage1_filter_geom_sigma, label="Stage 1 Filter Geom Sigma")
                        or 2.5,
                        "filter_color_sigma": _coerce_float(
                            stage1_filter_color_sigma, label="Stage 1 Filter Color Sigma"
                        )
                        or 1.5,
                        "filter_worst_pct": _coerce_float(stage1_filter_worst_pct, label="Stage 1 Filter Worst Pct")
                        or 0.2,
                        "filter_min_frames": _coerce_int(stage1_filter_min_frames, label="Stage 1 Filter Min Frames")
                        or 2,
                        "filter_base_percentile": str(stage1_filter_base_percentile or "p75"),
                    },
                )
                stage_hint = "Stage 1"
            else:
                run_name, run_dir = _resolve_run(scene_root, stage_run_name)
                input_note = f"Using existing scene root `{scene_root}` and run `{run_name}`"

                if stage_key == "stage2":
                    command = _build_stage2_command(
                        scene_root=scene_root,
                        run_name=run_name,
                        stage2_extra={
                            "tensorboard": bool(stage2_tensorboard),
                            "knn_backend": str(stage2_knn_backend or "cpu_kdtree"),
                            "loo_loss_weight": _coerce_float(stage2_loo_loss_weight, label="Stage 2 LOO Loss Weight")
                            or 1.0,
                            "loo_k_neighbors": _coerce_int(stage2_loo_k_neighbors, label="Stage 2 LOO K Neighbors")
                            or 5,
                            "loo_max_corr_dist": _coerce_float(
                                stage2_loo_max_corr_dist, label="Stage 2 LOO Max Corr Dist"
                            )
                            or 0.03125,
                            "loo_normal_k": _coerce_int(stage2_loo_normal_k, label="Stage 2 LOO Normal K") or 20,
                            "loo_kdtree_rebuild_every": _coerce_int(
                                stage2_loo_kdtree_rebuild_every, label="Stage 2 KDT Rebuild Every"
                            )
                            or 50,
                            "loo_max_pairs_per_iter": _coerce_int(
                                stage2_loo_max_pairs_per_iter, label="Stage 2 Max Pairs Per Iter", optional=True
                            ),
                            "loo_pairs_per_src": _coerce_int(stage2_loo_pairs_per_src, label="Stage 2 Pairs Per Src")
                            or 1,
                            "deform_chunk_size": _coerce_int(
                                stage2_deform_chunk_size, label="Stage 2 Deform Chunk Size"
                            )
                            or 50000,
                            "anchor_loss_weight": _coerce_float(
                                stage2_anchor_loss_weight, label="Stage 2 Anchor Loss Weight"
                            )
                            or 1000.0,
                            "anchor_n_samples": _coerce_int(stage2_anchor_n_samples, label="Stage 2 Anchor Samples")
                            or 4096,
                            "tv_reg": _coerce_float(stage2_tv_reg, label="Stage 2 TV Reg") or 50.0,
                            "tv_voxel_size": _coerce_float(stage2_tv_voxel_size, label="Stage 2 TV Voxel Size") or 0.01,
                            "tv_every_k": _coerce_int(stage2_tv_every_k, label="Stage 2 TV Every K") or 1,
                            "tv_sample_ratio": _coerce_float(
                                stage2_tv_sample_ratio, label="Stage 2 TV Sample Ratio", optional=True
                            ),
                            "loo_color_icp_weight": _coerce_float(
                                stage2_loo_color_icp_weight, label="Stage 2 Color ICP Weight"
                            )
                            or 0.02,
                            "loo_color_icp_k": _coerce_int(stage2_loo_color_icp_k, label="Stage 2 Color ICP K") or 10,
                            "loo_color_icp_max_color_dist": _coerce_float(
                                stage2_loo_color_icp_max_color_dist,
                                label="Stage 2 Color ICP Max Color Dist",
                                optional=True,
                            ),
                            "thin_shell_weight": _coerce_float(
                                stage2_thin_shell_weight, label="Stage 2 Thin Shell Weight"
                            )
                            or 1000.0,
                            "lr": _coerce_float(stage2_lr, label="Stage 2 LR") or 1e-3,
                            "n_iters": _coerce_int(stage2_n_iters, label="Stage 2 Iterations") or 150,
                            "save_intermediate_every_n": _coerce_int(
                                stage2_save_intermediate_every_n, label="Stage 2 Save Intermediate Every"
                            )
                            or 50,
                        },
                    )
                    stage_hint = "Stage 2"
                elif stage_key == "stage31":
                    checkpoint_subdir = _resolve_checkpoint_subdir(run_dir, stage31_checkpoint_subdir)
                    command = _build_stage31_command(
                        scene_root=scene_root,
                        run_name=run_name,
                        run_dir=run_dir,
                        checkpoint_subdir=checkpoint_subdir,
                        inverse_epochs=_coerce_int(stage31_epochs, label="Stage 3.1 Epoch Override", optional=True),
                        stage31_extra={
                            "tensorboard": bool(stage31_tensorboard),
                            "knn_backend": str(stage31_knn_backend or "cpu_kdtree"),
                            "batch_size": _coerce_int(stage31_batch_size, label="Stage 3.1 Batch Size") or 8192,
                            "lr": _coerce_float(stage31_lr, label="Stage 3.1 LR") or 1e-3,
                            "cycle_weight": _coerce_float(stage31_cycle_weight, label="Stage 3.1 Cycle Weight") or 0.1,
                            "magnitude_weight": _coerce_float(
                                stage31_magnitude_weight, label="Stage 3.1 Magnitude Weight"
                            )
                            or 1e-3,
                            "smoothness_weight": _coerce_float(
                                stage31_smoothness_weight, label="Stage 3.1 Smoothness Weight"
                            )
                            or 1e-3,
                            "num_forward_samples": _coerce_int(
                                stage31_num_forward_samples, label="Stage 3.1 Forward Samples"
                            )
                            or 10000,
                            "num_interp_samples": _coerce_int(
                                stage31_num_interp_samples, label="Stage 3.1 Interp Samples"
                            )
                            or 5000,
                            "regenerate_every": _coerce_int(
                                stage31_regenerate_every, label="Stage 3.1 Regenerate Every"
                            )
                            or 10,
                            "view_embed_dim": _coerce_int(stage31_view_embed_dim, label="Stage 3.1 View Embed Dim")
                            or 32,
                            "min_res": _coerce_int(stage31_min_res, label="Stage 3.1 Min Res") or 16,
                            "max_res": _coerce_int(stage31_max_res, label="Stage 3.1 Max Res") or 2048,
                            "num_levels": _coerce_int(stage31_num_levels, label="Stage 3.1 Num Levels") or 16,
                            "log2_hashmap_size": _coerce_int(stage31_log2_hashmap_size, label="Stage 3.1 Log2 Hashmap")
                            or 19,
                            "n_neurons": _coerce_int(stage31_n_neurons, label="Stage 3.1 Neurons") or 64,
                            "n_hidden_layers": _coerce_int(stage31_n_hidden_layers, label="Stage 3.1 Hidden Layers")
                            or 3,
                            "save_validation_plys": bool(stage31_save_validation_plys),
                        },
                    )
                    stage_hint = "Stage 3.1"
                elif stage_key == "stage32":
                    checkpoint_subdir = _resolve_checkpoint_subdir(run_dir, stage32_checkpoint_subdir)
                    inverse_dir = _resolve_inverse_dir(run_dir, stage32_inverse_dir_name)
                    command = _build_stage32_command(
                        scene_root=scene_root,
                        run_name=run_name,
                        run_dir=run_dir,
                        checkpoint_subdir=checkpoint_subdir,
                        inverse_dir=inverse_dir,
                        renderer=stage32_renderer,
                        gs_num_iters=_coerce_int(stage32_num_iters, label="Stage 3.2 Iter Override", optional=True),
                        gs_extra={
                            "tensorboard": bool(gs_tensorboard),
                            "sh_degree": _coerce_int(gs_sh_degree, label="GS SH Degree") or 3,
                            "sh_increase_every": _coerce_int(gs_sh_increase_every, label="GS SH Increase Every") or 0,
                            "sh_full_from_iter": _coerce_int(gs_sh_full_from_iter, label="GS SH Full From Iter")
                            or 5000,
                            "sh_freeze_means_when_full_sh": bool(gs_sh_freeze_means_when_full_sh),
                            "sh_reg_weight": _coerce_float(gs_sh_reg_weight, label="GS SH Reg Weight") or 10.0,
                            "target_num_points": _coerce_int(gs_target_num_points, label="GS Target Num Points")
                            or 4000000,
                            "optimize_cams": bool(gs_optimize_cams),
                            "lr_cams": _coerce_float(gs_lr_cams, label="GS LR Cams") or 1e-4,
                            "optimize_positions": bool(gs_optimize_positions),
                            "lr_positions": _coerce_float(gs_lr_positions, label="GS LR Positions") or 1e-5,
                            "lr_colors": _coerce_float(gs_lr_colors, label="GS LR Colors") or 2.5e-3,
                            "lr_opacities": _coerce_float(gs_lr_opacities, label="GS LR Opacities") or 5e-2,
                            "lr_scales": _coerce_float(gs_lr_scales, label="GS LR Scales") or 5e-3,
                            "lr_quats": _coerce_float(gs_lr_quats, label="GS LR Quats") or 1e-3,
                            "lr_sh0": _coerce_float(gs_lr_sh0, label="GS LR SH0") or 2.5e-3,
                            "lr_shn": _coerce_float(gs_lr_shn, label="GS LR SHN") or (2.5e-3 / 20.0),
                            "deform_inverse_rotations": bool(gs_deform_inverse_rotations),
                            "initial_opacity": _coerce_float(gs_initial_opacity, label="GS Initial Opacity") or 0.5,
                            "initial_scale": _coerce_float(gs_initial_scale, label="GS Initial Scale") or 0.005,
                            "initial_flat_ratio": _coerce_float(gs_initial_flat_ratio, label="GS Initial Flat Ratio")
                            or 0.1,
                            "scale_init": str(gs_scale_init or "knn"),
                            "knn_neighbors": _coerce_int(gs_knn_neighbors, label="GS KNN Neighbors") or 4,
                            "normal_k": _coerce_int(gs_normal_k, label="GS Normal K") or 20,
                            "l1_weight": _coerce_float(gs_l1_weight, label="GS L1 Weight") or 0.8,
                            "lpips_weight": _coerce_float(gs_lpips_weight, label="GS LPIPS Weight") or 0.2,
                            "opacity_reg_weight": _coerce_float(gs_opacity_reg_weight, label="GS Opacity Reg Weight")
                            or 0.0,
                            "scale_reg_weight": _coerce_float(gs_scale_reg_weight, label="GS Scale Reg Weight") or 0.0,
                            "normal_consistency_weight": _coerce_float(
                                gs_normal_consistency_weight, label="GS Normal Consistency Weight"
                            )
                            or 0.05,
                            "distortion_weight": _coerce_float(gs_distortion_weight, label="GS Distortion Weight")
                            or 0.01,
                            "alpha_reg_weight": _coerce_float(gs_alpha_reg_weight, label="GS Alpha Reg Weight") or 0.0,
                            "frames_per_iter": _coerce_int(gs_frames_per_iter, label="GS Frames Per Iter") or 1,
                            "log_every": _coerce_int(gs_log_every, label="GS Log Every") or 50,
                            "save_every": _coerce_int(gs_save_every, label="GS Save Every") or 5000,
                            "eval_every": _coerce_int(gs_eval_every, label="GS Eval Every") or 1000,
                            "lr_decay": _coerce_float(gs_lr_decay, label="GS LR Decay") or 0.1,
                            "auto_eval": bool(gs_auto_eval),
                        },
                    )
                    stage_hint = "Stage 3.2"
                else:
                    raise ValueError(f"Unknown stage: {stage_key}")
    except Exception as exc:
        yield from _emit_stage_launch_failure(f"**State**: Failed before launch\n\n`{exc}`")
        return

    yield from _run_stage_command_generator(
        command=command,
        effective_scene_root=effective_scene_root,
        input_note=input_note,
        stage_hint=stage_hint,
    )


def _stop_active_run(run_state: dict[str, str]):
    run_id = (run_state or {}).get("run_id", "")
    if not run_id:
        return "No active run is registered in the UI state.", run_state

    with ACTIVE_RUNS_LOCK:
        process = ACTIVE_RUNS.get(run_id)

    if process is None:
        return f"Run `{run_id}` is not active. It may have already finished.", run_state

    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)

    return f"Stop signal sent to `{run_id}`.", run_state


def _inspect_existing_scene(scene_root_selection: object):
    try:
        scene_root = _resolve_existing_dir(scene_root_selection)
        artifacts = _collect_scene_artifacts(scene_root)
    except Exception as exc:
        return (
            f"**Inspect Failed**\n\n`{exc}`",
            "",
            "",
            None,
            None,
            [],
            "",
        )

    return (
        _format_scene_report(artifacts),
        str(artifacts.scene_root),
        _latest_run_dir_text(artifacts),
        _choose_primary_preview(artifacts),
        _choose_secondary_preview(artifacts),
        _key_files_as_strings(artifacts),
        "",
    )


def _refresh_stage_scene(scene_root_selection: object):
    if not _strip_quotes(scene_root_selection):
        return _blank_stage_scene_outputs()

    empty_dropdown = _empty_dropdown_update()
    disabled = _stage_button_update(active=False, enabled=False)
    try:
        scene_root = _resolve_existing_dir(scene_root_selection)
        artifacts = _collect_scene_artifacts(scene_root)
    except Exception as exc:
        return (
            f"**Scene Selection Failed**\n\n`{exc}`",
            "",
            empty_dropdown,
            "",
            empty_dropdown,
            empty_dropdown,
            empty_dropdown,
            "",
            f"**Recommended Next Step**: `Stage 0`\n\n`{exc}`",
            _stage_button_update(active=True, enabled=True),
            disabled,
            disabled,
            disabled,
            disabled,
        )

    run_names = [path.name for path in artifacts.run_dirs]
    default_run_name = artifacts.latest_run_dir.name if artifacts.latest_run_dir else None
    selected_run_dir = (scene_root / default_run_name).resolve() if default_run_name else None
    checkpoint_choices = _checkpoint_subdir_choices(selected_run_dir)
    inverse_names = [path.name for path in _list_inverse_dirs(selected_run_dir)]

    recommendation = _recommend_stage_action(str(scene_root.resolve()), default_run_name)

    return (
        _format_scene_report(artifacts, selected_run_name=default_run_name),
        str(scene_root.resolve()),
        _choices_dropdown_update(run_names, default_run_name),
        str(selected_run_dir) if selected_run_dir and selected_run_dir.is_dir() else "",
        _choices_dropdown_update(checkpoint_choices),
        _choices_dropdown_update(checkpoint_choices),
        _choices_dropdown_update(inverse_names),
        _default_original_images_dir(scene_root),
        *recommendation,
    )


def _stage_completion_control_no_updates():
    return tuple(gr.update() for _ in range(STAGE_COMPLETION_CONTROL_OUTPUT_COUNT))


def _stage_completion_control_outputs(scene_root_selection: object):
    scene_text = _strip_quotes(scene_root_selection)
    if not scene_text:
        return _stage_completion_control_no_updates()

    try:
        scene_root = _resolve_existing_dir(scene_text)
        if _scene_root_has_stage0(scene_root):
            _clear_catalog_cache()
        refresh = _refresh_stage_scene(str(scene_root.resolve()))
    except Exception:
        return _stage_completion_control_no_updates()

    scene_root_text = refresh[1] or str(scene_root.resolve())
    return (
        _update_dropdown_choices(_scene_dropdown_choices(scene_root_text), scene_root_text),
        scene_root_text,
        refresh[2],
        refresh[3],
        refresh[4],
        refresh[5],
        refresh[6],
        refresh[7],
        refresh[8],
        refresh[9],
        refresh[10],
        refresh[11],
        refresh[12],
        refresh[13],
    )


def _refresh_stage_run(scene_root_selection: object, run_name: str):
    if not _strip_quotes(scene_root_selection):
        return (
            "",
            _empty_dropdown_update(),
            _empty_dropdown_update(),
            _empty_dropdown_update(),
            "",
            _blank_stage_scene_outputs()[0],
            *_blank_stage_scene_outputs()[8:],
        )

    empty_dropdown = _empty_dropdown_update()
    disabled = _stage_button_update(active=False, enabled=False)
    try:
        scene_root = _resolve_existing_dir(scene_root_selection)
        artifacts = _collect_scene_artifacts(scene_root)
        if not run_name:
            return (
                "",
                empty_dropdown,
                empty_dropdown,
                empty_dropdown,
                _default_original_images_dir(scene_root),
                _format_scene_report(artifacts),
                *_recommend_stage_action(str(scene_root.resolve()), None),
            )
        _, run_dir = _resolve_run(scene_root, run_name)
    except Exception as exc:
        return (
            "",
            empty_dropdown,
            empty_dropdown,
            empty_dropdown,
            "",
            f"**Run Selection Failed**\n\n`{exc}`",
            f"**Recommended Next Step**: `Stage 1`\n\n`{exc}`",
            disabled,
            disabled,
            disabled,
            disabled,
            disabled,
        )

    checkpoint_choices = _checkpoint_subdir_choices(run_dir)
    inverse_names = [path.name for path in _list_inverse_dirs(run_dir)]
    return (
        str(run_dir),
        _choices_dropdown_update(checkpoint_choices),
        _choices_dropdown_update(checkpoint_choices),
        _choices_dropdown_update(inverse_names),
        _default_original_images_dir(scene_root),
        _format_scene_report(artifacts, selected_run_name=run_name),
        *_recommend_stage_action(str(scene_root.resolve()), run_name),
    )


STAGE_PARAMETER_OUTPUT_NAMES = [
    "stage0_max_frames",
    "stage0_max_stride",
    "stage0_streaming",
    "stage0_streaming_overlap",
    "stage0_streaming_global_guide",
    "stage0_image_ext",
    "stage0_model_name",
    "stage0_process_res",
    "stage0_process_res_method",
    "stage0_export_gs_video",
    "stage0_runtime_export_format",
    "stage0_runtime_export_fps",
    "stage0_use_ray_pose",
    "stage0_ref_view_strategy",
    "stage1_num_frames",
    "stage1_stride",
    "stage1_offset",
    "stage1_conf_profile",
    "stage1_conf_percentile",
    "stage1_conf_mask_sky",
    "stage1_conf_mask_sky_depth_band",
    "stage1_conf_sky_depth_band_percent",
    "stage1_conf_mask_white_background",
    "stage1_conf_white_bg_min_rgb",
    "stage1_conf_white_bg_max_channel_delta",
    "stage1_conf_white_bg_grow_px",
    "stage1_conf_mask_min_depth_range_percent",
    "stage1_conf_min_depth_range_percent",
    "stage1_conf_mask_min_depth_range_meters",
    "stage1_conf_min_depth_range_meters",
    "stage1_conf_mask_depth_edges",
    "stage1_conf_edge_rtol",
    "stage1_conf_edge_atol",
    "stage1_conf_edge_kernel_size",
    "stage1_conf_mask_max_depth",
    "stage1_conf_max_depth_rtol",
    "stage1_conf_max_depth_atol",
    "stage1_use_roma_matching",
    "stage1_roma_version",
    "stage1_roma_model",
    "stage1_roma_num_samples",
    "stage1_roma_certainty_threshold",
    "stage1_roma_max_references",
    "stage1_roma_reference_sampling",
    "stage1_roma_loss_weight",
    "stage1_roma_max_corr_dist",
    "stage1_knn_backend",
    "stage1_tensorboard",
    "stage1_max_corr_dist",
    "stage1_merge_voxel_size",
    "stage1_icp_n_iter",
    "stage1_icp_early_stopping_patience",
    "stage1_icp_early_stopping_min_iters",
    "stage1_icp_early_stopping_min_delta",
    "stage1_icp_lr",
    "stage1_icp_method",
    "stage1_icp_local_twist_reg",
    "stage1_icp_tv_reg",
    "stage1_icp_tv_voxel_size",
    "stage1_icp_tv_every_k",
    "stage1_icp_tv_sample_ratio",
    "stage1_icp_color_icp_weight",
    "stage1_icp_color_icp_max_color_dist",
    "stage1_icp_color_icp_k",
    "stage1_save_intermediate_every",
    "stage1_deform_log2_hashmap_size",
    "stage1_deform_num_levels",
    "stage1_deform_n_neurons",
    "stage1_deform_n_hidden_layers",
    "stage1_deform_min_res",
    "stage1_deform_max_res",
    "stage1_filter_points",
    "stage1_filter_geom_sigma",
    "stage1_filter_color_sigma",
    "stage1_filter_worst_pct",
    "stage1_filter_min_frames",
    "stage1_filter_base_percentile",
]


def _blank_stage_parameter_updates():
    return tuple(gr.update() for _ in STAGE_PARAMETER_OUTPUT_NAMES)


def _read_json_object(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _selected_or_latest_run_dir(scene_root: Path, run_name: object = None) -> Optional[Path]:
    run_name_text = _strip_quotes(run_name)
    if run_name_text:
        candidate = scene_root / run_name_text
        if candidate.is_dir():
            return candidate.resolve()

    run_dirs = _find_run_dirs(scene_root)
    return run_dirs[-1].resolve() if run_dirs else None


def _conf_profile_from_alignment(alignment: dict[str, object]) -> str:
    mode = str(alignment.get("conf_mode") or "").strip().lower()
    if mode == "per_frame":
        return "da3_per_frame"
    if mode == "global":
        return "da3_global"
    if mode == "per_frame_guided":
        return "da3_per_frame_guided"
    return "default_mixed"


def _apply_alignment_param_values(values: dict[str, object], alignment: dict[str, object]) -> None:
    if not alignment:
        return

    mapping = {
        "stage1_num_frames": "num_frames",
        "stage1_stride": "stride",
        "stage1_offset": "offset",
        "stage1_conf_percentile": "conf_thresh_percentile",
        "stage1_conf_mask_sky": "conf_mask_sky",
        "stage1_conf_mask_sky_depth_band": "conf_mask_sky_depth_band",
        "stage1_conf_sky_depth_band_percent": "conf_sky_depth_band_percent",
        "stage1_conf_mask_white_background": "conf_mask_white_background",
        "stage1_conf_white_bg_min_rgb": "conf_white_bg_min_rgb",
        "stage1_conf_white_bg_max_channel_delta": "conf_white_bg_max_channel_delta",
        "stage1_conf_white_bg_grow_px": "conf_white_bg_grow_px",
        "stage1_conf_mask_min_depth_range_percent": "conf_mask_min_depth_range_percent",
        "stage1_conf_min_depth_range_percent": "conf_min_depth_range_percent",
        "stage1_conf_mask_min_depth_range_meters": "conf_mask_min_depth_range_meters",
        "stage1_conf_min_depth_range_meters": "conf_min_depth_range_meters",
        "stage1_conf_mask_depth_edges": "conf_mask_depth_edges",
        "stage1_conf_edge_rtol": "conf_edge_rtol",
        "stage1_conf_edge_atol": "conf_edge_atol",
        "stage1_conf_edge_kernel_size": "conf_edge_kernel_size",
        "stage1_conf_mask_max_depth": "conf_mask_max_depth",
        "stage1_conf_max_depth_rtol": "conf_max_depth_rtol",
        "stage1_conf_max_depth_atol": "conf_max_depth_atol",
    }
    values["stage1_conf_profile"] = _conf_profile_from_alignment(alignment)
    for output_name, alignment_key in mapping.items():
        if alignment_key in alignment:
            values[output_name] = alignment.get(alignment_key)


def _load_stage0_values_from_log(scene_root: Path) -> dict[str, object]:
    scene_text = str(scene_root).replace("\\", "/")
    values: dict[str, object] = {}
    try:
        log_paths = sorted(RUNS_ROOT.glob("*.log"), key=lambda path: path.stat().st_mtime, reverse=True)
    except OSError:
        return values

    for log_path in log_paths[:80]:
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if scene_text not in text.replace("\\", "/"):
            continue

        match = re.search(
            r"DA3 inference settings:\s*"
            r"process_res=(?P<process_res>\d+),\s*"
            r"process_res_method=(?P<process_res_method>[^,]+),\s*"
            r"use_ray_pose=(?P<use_ray_pose>True|False),\s*"
            r"(?:fixed_camera=(?P<fixed_camera>True|False),\s*"
            r"fixed_camera_fov_degrees=(?P<fixed_camera_fov_degrees>[^,]+),\s*)?"
            r"ref_view_strategy=(?P<ref_view_strategy>[^,]+),\s*"
            r"export_gs_video=(?P<export_gs_video>True|False),\s*"
            r"streaming=(?P<streaming>True|False)"
            r"(?:,\s*streaming_global_guide=(?P<streaming_global_guide>True|False))?",
            text,
            flags=re.MULTILINE,
        )
        if not match:
            continue

        values["stage0_process_res"] = int(match.group("process_res"))
        values["stage0_process_res_method"] = match.group("process_res_method").strip()
        values["stage0_use_ray_pose"] = match.group("use_ray_pose") == "True"
        values["stage0_ref_view_strategy"] = match.group("ref_view_strategy").strip()
        values["stage0_export_gs_video"] = match.group("export_gs_video") == "True"
        values["stage0_streaming"] = match.group("streaming") == "True"
        if match.group("streaming_global_guide") is not None:
            values["stage0_streaming_global_guide"] = match.group("streaming_global_guide") == "True"
        return values

    return values


def _load_stage_parameter_values(scene_root: Path, run_name: object = None) -> dict[str, object]:
    values: dict[str, object] = {}

    meta = _read_json_object(scene_root / "preprocess_frames.json")
    stage0_mapping = {
        "stage0_max_frames": "max_frames",
        "stage0_max_stride": "max_stride",
        "stage0_streaming": "streaming_enabled",
        "stage0_streaming_overlap": "streaming_overlap",
        "stage0_streaming_global_guide": "streaming_global_guide",
        "stage0_image_ext": "image_ext",
        "stage0_model_name": "model_name",
        "stage0_process_res": "process_res",
        "stage0_process_res_method": "process_res_method",
        "stage0_export_gs_video": "export_gs_video",
        "stage0_runtime_export_format": "runtime_export_format",
        "stage0_runtime_export_fps": "runtime_export_fps",
        "stage0_use_ray_pose": "use_ray_pose",
        "stage0_ref_view_strategy": "ref_view_strategy",
    }
    for output_name, meta_key in stage0_mapping.items():
        if meta_key in meta and meta.get(meta_key) is not None:
            values[output_name] = meta.get(meta_key)
    for output_name, value in _load_stage0_values_from_log(scene_root).items():
        values.setdefault(output_name, value)
    if "stage0_streaming_overlap" not in values and meta.get("streaming_enabled") is False:
        values["stage0_streaming_overlap"] = DEFAULT_STAGE0_STREAMING_OVERLAP

    run_dir = _selected_or_latest_run_dir(scene_root, run_name)
    if run_dir is None:
        return values

    prep_payload = _read_json_object(run_dir / "stage0_prep_config.json")
    alignment = prep_payload.get("alignment")
    if isinstance(alignment, dict):
        _apply_alignment_param_values(values, alignment)

    stage1_config = _read_json_object(run_dir / "after_non_rigid_icp" / "config.json")
    stage1_alignment = stage1_config.get("alignment")
    if isinstance(stage1_alignment, dict):
        _apply_alignment_param_values(values, stage1_alignment)

    roma = stage1_config.get("roma")
    if isinstance(roma, dict):
        roma_mapping = {
            "stage1_use_roma_matching": "use_roma_matching",
            "stage1_roma_version": "roma_version",
            "stage1_roma_model": "roma_model",
            "stage1_roma_num_samples": "roma_num_samples",
            "stage1_roma_certainty_threshold": "roma_certainty_threshold",
            "stage1_roma_max_references": "roma_max_references",
            "stage1_roma_reference_sampling": "roma_reference_sampling",
            "stage1_roma_loss_weight": "roma_loss_weight",
            "stage1_roma_max_corr_dist": "roma_max_corr_dist",
        }
        for output_name, config_key in roma_mapping.items():
            if config_key in roma:
                values[output_name] = roma.get(config_key)

    stage1_mapping = {
        "stage1_knn_backend": "knn_backend",
        "stage1_tensorboard": "tensorboard",
        "stage1_max_corr_dist": "max_corr_dist",
        "stage1_merge_voxel_size": "merge_voxel_size",
        "stage1_icp_n_iter": "icp_n_iter",
        "stage1_icp_early_stopping_patience": "icp_early_stopping_patience",
        "stage1_icp_early_stopping_min_iters": "icp_early_stopping_min_iters",
        "stage1_icp_early_stopping_min_delta": "icp_early_stopping_min_delta",
        "stage1_icp_lr": "icp_lr",
        "stage1_icp_method": "icp_method",
        "stage1_icp_local_twist_reg": "icp_local_twist_reg",
        "stage1_icp_tv_reg": "icp_tv_reg",
        "stage1_icp_tv_voxel_size": "icp_tv_voxel_size",
        "stage1_icp_tv_every_k": "icp_tv_every_k",
        "stage1_icp_tv_sample_ratio": "icp_tv_sample_ratio",
        "stage1_icp_color_icp_weight": "icp_color_icp_weight",
        "stage1_icp_color_icp_max_color_dist": "icp_color_icp_max_color_dist",
        "stage1_icp_color_icp_k": "icp_color_icp_k",
        "stage1_save_intermediate_every": "save_intermediate_every",
        "stage1_deform_log2_hashmap_size": "deform_log2_hashmap_size",
        "stage1_deform_num_levels": "deform_num_levels",
        "stage1_deform_n_neurons": "deform_n_neurons",
        "stage1_deform_n_hidden_layers": "deform_n_hidden_layers",
        "stage1_deform_min_res": "deform_min_res",
        "stage1_deform_max_res": "deform_max_res",
        "stage1_filter_points": "filter_points",
        "stage1_filter_geom_sigma": "filter_geom_sigma",
        "stage1_filter_color_sigma": "filter_color_sigma",
        "stage1_filter_worst_pct": "filter_worst_pct",
        "stage1_filter_min_frames": "filter_min_frames",
        "stage1_filter_base_percentile": "filter_base_percentile",
    }
    for output_name, config_key in stage1_mapping.items():
        if config_key in stage1_config:
            values[output_name] = stage1_config.get(config_key)

    return values


def _stage_parameter_updates_for_scene(scene_root_selection: object, run_name: object = None):
    if not _strip_quotes(scene_root_selection):
        return _blank_stage_parameter_updates()
    try:
        scene_root = _resolve_existing_dir(scene_root_selection)
        values = _load_stage_parameter_values(scene_root, run_name)
    except Exception:
        return _blank_stage_parameter_updates()

    return tuple(
        gr.update(value=values[name]) if name in values else gr.update() for name in STAGE_PARAMETER_OUTPUT_NAMES
    )


def _stage_button_update(*, active: bool, enabled: bool) -> dict:
    return gr.update(variant="primary" if active else "secondary", interactive=True)


def _empty_dropdown_update() -> dict:
    return gr.update(choices=[""], value="", interactive=False)


def _choices_dropdown_update(choices: list[str], value: Optional[str] = None) -> dict:
    selected = value if value in choices else (choices[0] if choices else "")
    return gr.update(choices=choices if choices else [""], value=selected, interactive=bool(choices))


def _blank_stage_scene_outputs():
    disabled = _stage_button_update(active=False, enabled=False)
    return (
        (
            "**Stage 0 Source Selected**\n\n"
            "No scene is selected yet. Run `Stage 0` to create a new scene root and unlock the later stages."
        ),
        "",
        _empty_dropdown_update(),
        "",
        _empty_dropdown_update(),
        _empty_dropdown_update(),
        _empty_dropdown_update(),
        "",
        (
            "**Recommended Next Step**: `Stage 0`\n\n"
            "A new source video is selected, so the explicit-stage flow is reset to preprocessing."
        ),
        _stage_button_update(active=True, enabled=True),
        disabled,
        disabled,
        disabled,
        disabled,
    )


def _recommend_stage_action(scene_root_selection: object, run_name: str | None = None):
    default_note = (
        "**Recommended Next Step**: `Stage 0`\n\n"
        "Start from a video to create the scene root, extracted frames, DA3 depth/confidence, the filtered point-cloud cache, and `before_non_rigid_icp.ply`."
    )
    disabled = _stage_button_update(active=False, enabled=False)

    try:
        scene_root = _resolve_existing_dir(scene_root_selection)
    except Exception:
        return (
            default_note,
            _stage_button_update(active=True, enabled=True),
            disabled,
            disabled,
            disabled,
            disabled,
        )

    results_npz = scene_root / "exports" / "npz" / "results.npz"
    stage0_done = results_npz.exists()
    run_dirs = _find_run_dirs(scene_root)

    selected_run_dir: Optional[Path] = None
    if run_name:
        candidate = scene_root / run_name
        if candidate.is_dir():
            selected_run_dir = candidate.resolve()
    if selected_run_dir is None and run_dirs:
        selected_run_dir = run_dirs[-1].resolve()

    stage1_done = selected_run_dir is not None and (selected_run_dir / "after_non_rigid_icp").is_dir()
    stage2_done = selected_run_dir is not None and (selected_run_dir / "after_global_optimization").is_dir()
    checkpoint_choices = _checkpoint_subdir_choices(selected_run_dir)
    inverse_dirs = _list_inverse_dirs(selected_run_dir)
    gs_dirs = _list_gs_dirs(selected_run_dir)
    stage31_done = bool(inverse_dirs)
    stage32_done = bool(gs_dirs)

    if not stage0_done:
        note = (
            "**Recommended Next Step**: `Stage 0`\n\n"
            "Create the scene root plus Stage 0 exports: frames, DA3 depth/confidence, intrinsics, extrinsics, the filtered point-cloud cache, and `before_non_rigid_icp.ply`."
        )
        next_stage = "stage0"
    elif not run_dirs:
        note = (
            "**Recommended Next Step**: `Stage 0`\n\n"
            "This scene has DA3 outputs but no prepared Stage 1 run directory yet. Rerun Stage 0 with the desired pre-ICP settings to create the filtered point-cloud cache and `before_non_rigid_icp.ply`."
        )
        next_stage = "stage0"
    elif not stage1_done:
        note = (
            "**Recommended Next Step**: `Stage 1`\n\n"
            "Run non-rigid ICP on the prepared Stage 0 inputs and save `after_non_rigid_icp`."
        )
        next_stage = "stage1"
    elif not stage2_done:
        note = (
            "**Recommended Next Step**: `Stage 2`\n\n"
            "Refine the selected Stage 1 run jointly and write the improved checkpoint to `after_global_optimization`."
        )
        next_stage = "stage2"
    elif not stage31_done:
        note = (
            "**Recommended Next Step**: `Stage 3.1`\n\n"
            "Train the inverse deformation model from the selected checkpoint so Stage 3.2 can render a splat."
        )
        next_stage = "stage31"
    elif not stage32_done:
        note = (
            "**Recommended Next Step**: `Stage 3.2`\n\n"
            "Train the Gaussian splat from the selected checkpoint and inverse deformation directory."
        )
        next_stage = "stage32"
    else:
        note = (
            "**Recommended Next Step**: `Complete`\n\n"
            "This run already has Stage 0, Stage 1, Stage 2, Stage 3.1, and Stage 3.2 outputs."
        )
        next_stage = ""

    stage0_enabled = True
    stage1_enabled = selected_run_dir is not None
    stage2_enabled = stage1_done
    stage31_enabled = stage1_done and bool(checkpoint_choices)
    stage32_enabled = stage31_done and bool(checkpoint_choices)

    return (
        note,
        _stage_button_update(active=next_stage == "stage0", enabled=stage0_enabled),
        _stage_button_update(active=next_stage == "stage1", enabled=stage1_enabled),
        _stage_button_update(active=next_stage == "stage2", enabled=stage2_enabled),
        _stage_button_update(active=next_stage == "stage31", enabled=stage31_enabled),
        _stage_button_update(active=next_stage == "stage32", enabled=stage32_enabled),
    )


def _reset_stage_panel_for_new_source(
    source_mode: str,
    uploaded_video: object,
    existing_video_selection: object,
    existing_frames_dir: object,
):
    empty_dropdown = _empty_dropdown_update()
    source_name = ""

    normalized_mode = str(source_mode or "").strip().lower()
    if normalized_mode == "upload_video":
        raw_path = _strip_quotes(uploaded_video)
        if raw_path:
            source_name = Path(raw_path).name
    elif normalized_mode == "existing_video":
        raw_path = _strip_quotes(existing_video_selection)
        if raw_path:
            source_name = Path(raw_path).name
            try:
                scene_root = _find_scene_root_for_video(raw_path)
            except Exception:
                scene_root = None
            if scene_root is not None:
                run_dir = _selected_or_latest_run_dir(scene_root)
                run_name = run_dir.name if run_dir is not None else None
                return (
                    _update_dropdown_choices(_scene_dropdown_choices(), str(scene_root.resolve())),
                    *_refresh_stage_scene(str(scene_root.resolve())),
                    *_stage_parameter_updates_for_scene(str(scene_root.resolve()), run_name),
                )
    elif normalized_mode == "existing_frames":
        raw_path = _strip_quotes(existing_frames_dir)
        if raw_path:
            source_name = Path(raw_path).name

    source_note = f" for `{source_name}`" if source_name else ""
    return (
        _update_dropdown_choices(_scene_dropdown_choices(), ""),
        (
            "**Stage 0 Source Selected**\n\n"
            f"A new Stage 0 input is selected{source_note}. Run `Stage 0` next. "
            "Downstream scene, run, checkpoint, and inverse-deformation selections will repopulate after preprocessing finishes."
        ),
        "",
        empty_dropdown,
        "",
        empty_dropdown,
        empty_dropdown,
        empty_dropdown,
        "",
        (
            "**Recommended Next Step**: `Stage 0`\n\n"
            "A new source video is selected, so the explicit-stage flow is reset to preprocessing."
        ),
        _stage_button_update(active=True, enabled=True),
        _stage_button_update(active=False, enabled=False),
        _stage_button_update(active=False, enabled=False),
        _stage_button_update(active=False, enabled=False),
        _stage_button_update(active=False, enabled=False),
        *_blank_stage_parameter_updates(),
    )


def build_app() -> gr.Blocks:
    video_choices = _video_dropdown_choices()
    scene_choices = _scene_dropdown_choices()
    output_parent_choices = _output_parent_dropdown_choices()
    default_video = video_choices[0][1] if video_choices else None
    default_scene = scene_choices[0][1] if scene_choices else None

    with gr.Blocks(title="Video-to-World Studio") as demo:
        pipeline_run_state = gr.State({})
        stage_run_state = gr.State({})
        divstream_run_state = gr.State({})
        flat_divstream_run_state = gr.State({})
        vda_divstream_run_state = gr.State({})

        gr.Markdown(
            f"""
            # Video-to-World Studio
            Version: `{APP_VERSION}`

            Launch `video_to_world` from upload and dropdown-based inputs instead of manual path entry.

            Existing videos and scene roots are discovered under `{PICKER_ROOT}`.
            The explicit stage runner launches the native Stage 0 through Stage 3.2 commands directly.
            """
        )

        with gr.Tab("Divstream Export"):
            with gr.Row():
                simple_divstream_video = gr.File(
                    label="Video",
                    file_types=[".mp4", ".mov", ".avi", ".mkv"],
                    type="filepath",
                )
                with gr.Column():
                    simple_divstream_filename = gr.Textbox(
                        label="Output Filename",
                        placeholder="Optional. Defaults to the uploaded video name.",
                    )
                    simple_divstream_compression = gr.Number(label="Compression Level", value=9, precision=0)
                    simple_divstream_workers = gr.Number(
                        label="Compression Workers",
                        value=0,
                        precision=0,
                    )
                    simple_export_before_non_rigid_ply = gr.Checkbox(
                        label="Export Debug before_non_rigid_icp.ply",
                        value=False,
                        info="Also write and keep the filtered pre-ICP merged point cloud for all selected DA3 frames.",
                    )
                    simple_export_streaming_guide_ply = gr.Checkbox(
                        label="Export Guide Pass PLY",
                        value=False,
                        info="Write the sparse global-guide pass point cloud. Requires Use DA3 Streaming and Use Global Guide Pass.",
                    )
            with gr.Accordion("Stage 0 DA3", open=True):
                with gr.Row():
                    simple_stage0_max_frames = gr.Number(
                        label="DA3 Input Max Frames / Chunk Size",
                        value=DEFAULT_STAGE0_MAX_FRAMES,
                        precision=0,
                        info="Target DA3 frame count in standard mode. In streaming mode, this becomes the per-chunk DA3 batch size.",
                    )
                    simple_stage0_max_stride = gr.Number(
                        label="DA3 Input Stride",
                        value=DEFAULT_STAGE0_MAX_STRIDE,
                        precision=0,
                        info="Maximum raw-frame gap in standard mode. In streaming mode this is applied exactly before chunking.",
                    )
                    simple_stage0_streaming = gr.Checkbox(label="Use DA3 Streaming", value=DEFAULT_STAGE0_STREAMING)
                    simple_stage0_streaming_overlap = gr.Number(
                        label="DA3 Streaming Overlap",
                        value=DEFAULT_STAGE0_STREAMING_OVERLAP,
                        precision=0,
                    )
                    simple_stage0_streaming_global_guide = gr.Checkbox(
                        label="Use Global Guide Pass",
                        value=DEFAULT_STAGE0_STREAMING_GLOBAL_GUIDE,
                        info="Run one sparse whole-video DA3 pass and anchor dense chunks to exact matching guide frames.",
                    )
                with gr.Row():
                    simple_stage0_image_ext = gr.Textbox(label="Image Extension", value="png")
                    simple_stage0_model_name = gr.Textbox(
                        label="DA3 Model Name",
                        value="depth-anything/DA3NESTED-GIANT-LARGE",
                    )
                    simple_stage0_process_res = gr.Number(label="DA3 Processing Resolution", value=768, precision=0)
                with gr.Row():
                    simple_stage0_process_res_method = gr.Dropdown(
                        choices=["upper_bound_resize", "upper_bound_crop"],
                        value="upper_bound_resize",
                        label="DA3 Resolution Method",
                    )
                    simple_stage0_ref_view_strategy = gr.Dropdown(
                        choices=["first", "middle", "saddle_balanced", "saddle_sim_range"],
                        value=DEFAULT_STAGE0_REF_VIEW_STRATEGY,
                        label="DA3 Reference View",
                    )
                    simple_stage0_use_ray_pose = gr.Checkbox(label="Use DA3 Ray Pose", value=False)
                    simple_fixed_camera = gr.Checkbox(
                        label="Fixed Camera",
                        value=False,
                        info="Tell DA3 the camera is static by supplying identity extrinsics and estimated intrinsics.",
                    )
                    simple_fixed_camera_fov = gr.Number(
                        label="Fixed Camera HFOV",
                        value=60.0,
                        info="Horizontal field of view used to estimate DA3 intrinsics when Fixed Camera is enabled.",
                    )

            with gr.Accordion("Stage 0 Filters Included In Divstream", open=True):
                with gr.Row():
                    simple_filter_conf_profile = gr.Dropdown(
                        choices=[
                            ("Default Mixed (voxel + DA3)", "default_mixed"),
                            ("DA3 Only (Per Frame)", "da3_per_frame"),
                            ("DA3 Only (Global)", "da3_global"),
                            ("DA3 Only (Per Frame Guided)", "da3_per_frame_guided"),
                        ],
                        value="default_mixed",
                        label="Confidence Mode",
                    )
                    simple_filter_conf_percentile = gr.Number(label="DA3 Confidence Percentile", value=1.0)
                with gr.Row():
                    simple_filter_mask_sky = gr.Checkbox(label="Use DA3 Sky Mask", value=False)
                    simple_filter_mask_sky_depth_band = gr.Checkbox(label="Expand Sky By Depth Band", value=False)
                    simple_filter_sky_depth_band_percent = gr.Number(label="Sky Depth Band Percent", value=50.0)
                with gr.Row():
                    simple_filter_mask_white_background = gr.Checkbox(
                        label="Suppress White Background",
                        value=False,
                        info="Removes bright low-saturation pixels from the Stage 0 point cache.",
                    )
                    simple_filter_white_bg_min_rgb = gr.Number(label="White BG Min RGB", value=220.0)
                    simple_filter_white_bg_max_channel_delta = gr.Number(
                        label="White BG Max Channel Delta",
                        value=25.0,
                    )
                    simple_filter_white_bg_grow_px = gr.Number(
                        label="White BG Grow Pixels",
                        value=1,
                        precision=0,
                    )
                with gr.Row():
                    simple_filter_mask_min_depth_range_percent = gr.Checkbox(
                        label="Limit By Min Depth Range %", value=False
                    )
                    simple_filter_min_depth_range_percent = gr.Number(label="Min Depth Range Percent", value=50.0)
                    simple_filter_mask_min_depth_range_meters = gr.Checkbox(
                        label="Limit By Min Depth Metres", value=False
                    )
                    simple_filter_min_depth_range_meters = gr.Number(label="Min Depth Range Metres", value=3.0)
                with gr.Row():
                    simple_filter_mask_depth_edges = gr.Checkbox(label="Suppress Depth Edges", value=True)
                    simple_filter_edge_rtol = gr.Number(label="Depth Edge Rel Threshold", value=0.1)
                    simple_filter_edge_atol = gr.Number(label="Depth Edge Abs Threshold", value=0.0)
                    simple_filter_edge_kernel_size = gr.Number(label="Depth Edge Kernel", value=3, precision=0)
                with gr.Row():
                    simple_filter_mask_max_depth = gr.Checkbox(label="Suppress Max DA3 Depth Plateau", value=False)
                    simple_filter_max_depth_rtol = gr.Number(label="Max Depth Rel Threshold", value=0.001)
                    simple_filter_max_depth_atol = gr.Number(label="Max Depth Abs Threshold", value=None)
            with gr.Row():
                simple_divstream_button = gr.Button("Export Divstream", variant="primary")
                simple_divstream_stop_button = gr.Button("Stop", variant="stop")
            simple_divstream_status = gr.Markdown()
            simple_divstream_file = gr.File(label="Divstream", interactive=False)
            simple_before_non_rigid_ply_path = gr.Textbox(
                label="Debug before_non_rigid_icp.ply Path",
                interactive=False,
            )
            simple_streaming_guide_ply_path = gr.Textbox(
                label="Guide Pass PLY Path",
                interactive=False,
            )
            simple_divstream_stop_feedback = gr.Markdown()
            with gr.Accordion("Log", open=False):
                simple_divstream_log = gr.Textbox(lines=18, interactive=False)

            simple_divstream_button.click(
                fn=_run_simple_divstream_generator,
                inputs=[
                    simple_divstream_video,
                    simple_divstream_filename,
                    simple_divstream_compression,
                    simple_divstream_workers,
                    simple_export_before_non_rigid_ply,
                    simple_export_streaming_guide_ply,
                    simple_stage0_max_frames,
                    simple_stage0_max_stride,
                    simple_stage0_streaming,
                    simple_stage0_streaming_overlap,
                    simple_stage0_streaming_global_guide,
                    simple_stage0_image_ext,
                    simple_stage0_model_name,
                    simple_stage0_process_res,
                    simple_stage0_process_res_method,
                    simple_stage0_ref_view_strategy,
                    simple_stage0_use_ray_pose,
                    simple_fixed_camera,
                    simple_fixed_camera_fov,
                    simple_filter_conf_profile,
                    simple_filter_conf_percentile,
                    simple_filter_mask_sky,
                    simple_filter_mask_sky_depth_band,
                    simple_filter_sky_depth_band_percent,
                    simple_filter_mask_white_background,
                    simple_filter_white_bg_min_rgb,
                    simple_filter_white_bg_max_channel_delta,
                    simple_filter_white_bg_grow_px,
                    simple_filter_mask_min_depth_range_percent,
                    simple_filter_min_depth_range_percent,
                    simple_filter_mask_min_depth_range_meters,
                    simple_filter_min_depth_range_meters,
                    simple_filter_mask_depth_edges,
                    simple_filter_edge_rtol,
                    simple_filter_edge_atol,
                    simple_filter_edge_kernel_size,
                    simple_filter_mask_max_depth,
                    simple_filter_max_depth_rtol,
                    simple_filter_max_depth_atol,
                ],
                outputs=[
                    divstream_run_state,
                    simple_divstream_status,
                    simple_divstream_file,
                    simple_before_non_rigid_ply_path,
                    simple_streaming_guide_ply_path,
                    simple_divstream_log,
                ],
            )
            simple_divstream_stop_button.click(
                fn=_stop_active_run,
                inputs=[divstream_run_state],
                outputs=[simple_divstream_stop_feedback, divstream_run_state],
            )

        with gr.Tab("Flat Depth BG Divstream"):
            with gr.Row():
                flat_divstream_video = gr.File(
                    label="Video",
                    file_types=[".mp4", ".mov", ".avi", ".mkv"],
                    type="filepath",
                )
                with gr.Column():
                    flat_divstream_filename = gr.Textbox(
                        label="Output Filename",
                        placeholder="Optional. Defaults to the uploaded video name.",
                    )
                    flat_divstream_compression = gr.Number(label="Compression Level", value=9, precision=0)
                    flat_divstream_workers = gr.Number(label="Compression Workers", value=0, precision=0)
            with gr.Accordion("Video", open=True):
                with gr.Row():
                    flat_stride = gr.Number(
                        label="Input Stride",
                        value=1,
                        precision=0,
                        info="Take every Nth input frame. Output FPS is input FPS divided by this stride.",
                    )
                    flat_max_frames = gr.Number(
                        label="Max Output Frames",
                        value=-1,
                        precision=0,
                        info="-1 exports the whole video after stride.",
                    )
                    flat_max_res = gr.Number(
                        label="Video Max Resolution",
                        value=-1,
                        precision=0,
                        info="Resize input video so the longest side is at most this value. Use -1 for original resolution.",
                    )
                    flat_depth_meters = gr.Number(
                        label="Flat Depth Metres",
                        value=1.0,
                    )
                    flat_fixed_camera_fov = gr.Number(
                        label="Fixed Camera HFOV",
                        value=60.0,
                    )
            with gr.Accordion("Background Range", open=True):
                with gr.Row():
                    flat_background_min_rgb = gr.Number(
                        label="Background RGB Min",
                        value=0.0,
                    )
                    flat_background_max_rgb = gr.Number(
                        label="Background RGB Max",
                        value=32.0,
                    )
                    flat_background_grow_px = gr.Number(
                        label="Background Grow Pixels",
                        value=1,
                        precision=0,
                    )
            with gr.Accordion("Preview", open=True):
                with gr.Row():
                    flat_preview_frame_index = gr.Number(
                        label="Preview Frame Index",
                        value=0,
                        precision=0,
                    )
                    flat_preview_button = gr.Button("Preview Removal", variant="secondary")
                with gr.Row():
                    flat_source_frame = gr.Image(label="Source Frame", type="numpy", interactive=False)
                    flat_removal_preview = gr.Image(label="Removal Preview", type="numpy", interactive=False)
                flat_preview_status = gr.Markdown()
            with gr.Row():
                flat_divstream_button = gr.Button("Export Flat Divstream", variant="primary")
                flat_divstream_stop_button = gr.Button("Stop", variant="stop")
            flat_divstream_status = gr.Markdown()
            flat_divstream_file = gr.File(label="Divstream", interactive=False)
            flat_divstream_stop_feedback = gr.Markdown()
            with gr.Accordion("Log", open=False):
                flat_divstream_log = gr.Textbox(lines=18, interactive=False)

            flat_preview_button.click(
                fn=_preview_flat_background_removal,
                inputs=[
                    flat_divstream_video,
                    flat_preview_frame_index,
                    flat_max_res,
                    flat_background_min_rgb,
                    flat_background_max_rgb,
                    flat_background_grow_px,
                ],
                outputs=[
                    flat_source_frame,
                    flat_removal_preview,
                    flat_preview_status,
                ],
            )
            flat_divstream_button.click(
                fn=_run_flat_divstream_generator,
                inputs=[
                    flat_divstream_video,
                    flat_divstream_filename,
                    flat_divstream_compression,
                    flat_divstream_workers,
                    flat_stride,
                    flat_max_frames,
                    flat_max_res,
                    flat_depth_meters,
                    flat_fixed_camera_fov,
                    flat_background_min_rgb,
                    flat_background_max_rgb,
                    flat_background_grow_px,
                ],
                outputs=[
                    flat_divstream_run_state,
                    flat_divstream_status,
                    flat_divstream_file,
                    flat_divstream_log,
                ],
            )
            flat_divstream_stop_button.click(
                fn=_stop_active_run,
                inputs=[flat_divstream_run_state],
                outputs=[flat_divstream_stop_feedback, flat_divstream_run_state],
            )

        with gr.Tab("VDA Divstream Export"):
            with gr.Row():
                vda_divstream_video = gr.File(
                    label="Video",
                    file_types=[".mp4", ".mov", ".avi", ".mkv"],
                    type="filepath",
                )
                with gr.Column():
                    vda_divstream_filename = gr.Textbox(
                        label="Output Filename",
                        placeholder="Optional. Defaults to the uploaded video name.",
                    )
                    vda_divstream_compression = gr.Number(label="Compression Level", value=9, precision=0)
                    vda_divstream_workers = gr.Number(label="Compression Workers", value=0, precision=0)
            with gr.Accordion("Video Depth Anything", open=True):
                with gr.Row():
                    vda_encoder = gr.Dropdown(
                        choices=[
                            ("Small / vits", "vits"),
                            ("Base / vitb", "vitb"),
                            ("Large / vitl", "vitl"),
                        ],
                        value="vitl",
                        label="VDA Encoder",
                        info="Installed checkpoints are reused. Missing checkpoints only download when enabled below.",
                    )
                    vda_metric = gr.Checkbox(
                        label="Metric Depth",
                        value=False,
                        info="Uses metric_video_depth_anything_*.pth. Disable only if you want the separate relative-depth checkpoint.",
                    )
                    vda_relative_depth_inverse = gr.Checkbox(
                        label="Invert Non-Metric Depth",
                        value=True,
                        info="For relative VDA checkpoints, globally normalize model values and reverse them with 1 - depth before export.",
                    )
                    vda_fp32 = gr.Checkbox(
                        label="Use FP32",
                        value=False,
                        info="Higher precision but much slower and uses more VRAM. Leave off for Large/vitl unless you need to debug precision.",
                    )
                    vda_download_checkpoint = gr.Checkbox(
                        label="Download Missing Checkpoint",
                        value=False,
                        info="When off, the run fails instead of downloading a missing metric/relative encoder checkpoint.",
                    )
                with gr.Row():
                    vda_input_size = gr.Number(
                        label="VDA Input Size",
                        value=768,
                        precision=0,
                        info="Model inference size. Large/vitl at 768 can exceed 24 GB VRAM; use 384 or 518 if CUDA OOM.",
                    )
                    vda_decoder_micro_batch_size = gr.Number(
                        label="Decoder Micro-Batch Size",
                        value=4,
                        precision=0,
                        info="Lower uses less VRAM in VDA's decoder. Use 1 if CUDA OOM occurs.",
                    )
                    vda_max_res = gr.Number(
                        label="Video Max Resolution",
                        value=1280,
                        precision=0,
                        info="Resize input video so the longest side is at most this value. Use -1 for original resolution.",
                    )
                    vda_max_frames = gr.Number(
                        label="Max Output Frames",
                        value=-1,
                        precision=0,
                        info="-1 exports the whole video after stride.",
                    )
                    vda_stride = gr.Number(
                        label="Input Stride",
                        value=1,
                        precision=0,
                        info="Take every Nth input frame. Output FPS is input FPS divided by this stride.",
                    )
                with gr.Row():
                    vda_fixed_camera_fov = gr.Number(
                        label="Fixed Camera HFOV",
                        value=60.0,
                        info="VDA does not estimate camera poses, so the divstream uses identity poses and this pinhole HFOV.",
                    )
                    vda_depth_scale = gr.Number(
                        label="Depth Scale",
                        value=100.0,
                        info="Applied before export. For non-metric depth, far pseudo-depth is 1 before this scale.",
                    )
                    vda_depth_offset = gr.Number(
                        label="Depth Offset",
                        value=0.0,
                        info="Applied after scaling before invalid depths are removed.",
                    )
            with gr.Accordion("Depth Filters Included In Divstream", open=True):
                with gr.Row():
                    vda_filter_mask_min_depth_range_percent = gr.Checkbox(
                        label="Limit By Min Depth Range %",
                        value=False,
                        info="Uses the global valid depth range across all exported frames and removes depths beyond the near-side limit.",
                    )
                    vda_filter_min_depth_range_percent = gr.Number(label="Min Depth Range Percent", value=50.0)
                    vda_filter_mask_max_depth_range_percent = gr.Checkbox(
                        label="Limit By Max Depth Range %",
                        value=False,
                        info="Uses the global valid depth range across all exported frames and removes depths before the far-side limit.",
                    )
                    vda_filter_max_depth_range_percent = gr.Number(label="Max Depth Range Percent", value=50.0)
                with gr.Row():
                    vda_filter_mask_min_depth_range_meters = gr.Checkbox(
                        label="Limit By Min Depth Metres",
                        value=False,
                    )
                    vda_filter_min_depth_range_meters = gr.Number(label="Min Depth Range Metres", value=3.0)
                with gr.Row():
                    vda_filter_mask_depth_edges = gr.Checkbox(label="Suppress Depth Edges", value=True)
                    vda_filter_edge_rtol = gr.Number(label="Depth Edge Rel Threshold", value=0.1)
                    vda_filter_edge_atol = gr.Number(label="Depth Edge Abs Threshold", value=0.0)
                    vda_filter_edge_kernel_size = gr.Number(label="Depth Edge Kernel", value=3, precision=0)
                with gr.Row():
                    vda_filter_mask_max_depth = gr.Checkbox(label="Suppress Max Depth Plateau", value=False)
                    vda_filter_max_depth_rtol = gr.Number(label="Max Depth Rel Threshold", value=0.001)
                    vda_filter_max_depth_atol = gr.Number(label="Max Depth Abs Threshold", value=None)
            with gr.Row():
                vda_divstream_button = gr.Button("Export VDA Divstream", variant="primary")
                vda_divstream_stop_button = gr.Button("Stop", variant="stop")
            vda_divstream_status = gr.Markdown()
            vda_divstream_file = gr.File(label="Divstream", interactive=False)
            vda_divstream_stop_feedback = gr.Markdown()
            with gr.Accordion("Log", open=False):
                vda_divstream_log = gr.Textbox(lines=18, interactive=False)

            vda_divstream_button.click(
                fn=_run_vda_divstream_generator,
                inputs=[
                    vda_divstream_video,
                    vda_divstream_filename,
                    vda_divstream_compression,
                    vda_divstream_workers,
                    vda_encoder,
                    vda_metric,
                    vda_relative_depth_inverse,
                    vda_input_size,
                    vda_decoder_micro_batch_size,
                    vda_max_res,
                    vda_max_frames,
                    vda_stride,
                    vda_fp32,
                    vda_download_checkpoint,
                    vda_fixed_camera_fov,
                    vda_depth_scale,
                    vda_depth_offset,
                    vda_filter_mask_min_depth_range_percent,
                    vda_filter_min_depth_range_percent,
                    vda_filter_mask_max_depth_range_percent,
                    vda_filter_max_depth_range_percent,
                    vda_filter_mask_min_depth_range_meters,
                    vda_filter_min_depth_range_meters,
                    vda_filter_mask_depth_edges,
                    vda_filter_edge_rtol,
                    vda_filter_edge_atol,
                    vda_filter_edge_kernel_size,
                    vda_filter_mask_max_depth,
                    vda_filter_max_depth_rtol,
                    vda_filter_max_depth_atol,
                ],
                outputs=[
                    vda_divstream_run_state,
                    vda_divstream_status,
                    vda_divstream_file,
                    vda_divstream_log,
                ],
            )
            vda_divstream_stop_button.click(
                fn=_stop_active_run,
                inputs=[vda_divstream_run_state],
                outputs=[vda_divstream_stop_feedback, vda_divstream_run_state],
            )

        with gr.Tab("Run Full Pipeline"):
            gr.Markdown(
                "Pick one source. Existing videos, scene roots, and downstream stage inputs update automatically on load and after runs."
            )

            with gr.Row():
                pipeline_source_mode = gr.Radio(
                    choices=[
                        ("Upload Video", "upload_video"),
                        ("Existing Video", "existing_video"),
                        ("Existing Scene", "existing_scene"),
                    ],
                    value="upload_video",
                    label="Source Mode",
                )

            with gr.Row():
                with gr.Column(scale=6):
                    with gr.Group(visible=True) as pipeline_upload_group:
                        uploaded_video = gr.File(
                            label="Upload Video",
                            file_types=[".mp4", ".mov", ".avi", ".mkv"],
                            type="filepath",
                        )
                        uploaded_video_cached = gr.State("")
                    with gr.Group(visible=False) as pipeline_existing_video_group:
                        existing_video_selection = gr.Dropdown(
                            label="Existing Video",
                            choices=video_choices,
                            value=default_video,
                            info="Discovered under the videos workspace.",
                            allow_custom_value=True,
                        )
                    with gr.Group(visible=False) as pipeline_existing_scene_group:
                        existing_scene_root_selection = gr.Dropdown(
                            label="Existing Scene Root",
                            choices=scene_choices,
                            value=default_scene,
                            info="Only scene roots with Stage 0 outputs are listed.",
                        )

                with gr.Column(scale=4):
                    with gr.Group(visible=True) as pipeline_output_group:
                        output_parent_selection = gr.Dropdown(
                            label="Output Parent Directory",
                            choices=output_parent_choices,
                            value="",
                            info="Leave on automatic unless you want a custom destination.",
                        )
                        custom_scene_name = gr.Textbox(
                            label="Custom Scene Name",
                            placeholder="Optional. If blank, the video filename is used.",
                        )
                    mode = gr.Radio(["fast", "extensive"], value="fast", label="Mode")
                    renderer_choice = gr.Dropdown(
                        choices=["auto", "2dgs", "3dgs", "both"],
                        value="auto",
                        label="Renderer",
                    )
                    preprocess_overwrite = gr.Checkbox(label="Overwrite Stage 0 Outputs", value=False)
                    dry_run = gr.Checkbox(label="Dry Run", value=False)

            with gr.Accordion("Advanced", open=False):
                with gr.Accordion("Stage 0", open=False):
                    with gr.Row():
                        preprocess_max_frames = gr.Number(
                            label="DA3 Input Max Frames / Chunk Size",
                            value=DEFAULT_STAGE0_MAX_FRAMES,
                            precision=0,
                            info="Target DA3 frame count in standard mode. In streaming mode, this becomes the per-chunk DA3 batch size.",
                        )
                        preprocess_max_stride = gr.Number(
                            label="DA3 Input Stride",
                            value=DEFAULT_STAGE0_MAX_STRIDE,
                            precision=0,
                            info="Maximum raw-frame gap in standard mode. In streaming mode this is applied exactly before chunking.",
                        )
                        preprocess_streaming_overlap = gr.Number(
                            label="DA3 Streaming Overlap",
                            value=DEFAULT_STAGE0_STREAMING_OVERLAP,
                            precision=0,
                            info="Overlap between adjacent DA3 chunks when streaming mode is enabled.",
                        )
                        preprocess_image_ext = gr.Textbox(label="Image Extension", value="png")
                    preprocess_model_name = gr.Textbox(
                        label="DA3 Model Name", value="depth-anything/DA3NESTED-GIANT-LARGE"
                    )
                    preprocess_process_res = gr.Number(
                        label="DA3 Processing Resolution",
                        value=768,
                        precision=0,
                        info="Longest side is resized to this before DA3 runs. Higher = denser points, more VRAM. VRAM scales as (res/504)^2. Common: 504 (low), 768 (medium), 1024 (high).",
                    )
                    with gr.Row():
                        preprocess_process_res_method = gr.Dropdown(
                            choices=["upper_bound_resize", "upper_bound_crop"],
                            value="upper_bound_resize",
                            label="DA3 Resolution Method",
                            info="Resize policy before DA3 inference.",
                        )
                        preprocess_ref_view_strategy = gr.Dropdown(
                            choices=["first", "middle", "saddle_balanced", "saddle_sim_range"],
                            value=DEFAULT_STAGE0_REF_VIEW_STRATEGY,
                            label="DA3 Reference View",
                            info="Default is `first` for stable chunked divstream exports.",
                        )
                        preprocess_export_gs_video = gr.Checkbox(
                            label="Export DA3 GS Preview Video",
                            value=False,
                            info="Opt-in preview export. Leave off for a faster Stage 0.",
                        )
                        preprocess_runtime_export_format = gr.Dropdown(
                            choices=RUNTIME_EXPORT_CHOICES,
                            value=DEFAULT_STAGE0_RUNTIME_EXPORT_FORMAT,
                            label="Stage 0 Runtime Export",
                            info="Optional runtime-ready export written after Stage 0. Default is DirectStorage stream.",
                        )
                    with gr.Row():
                        preprocess_runtime_export_fps = gr.Number(
                            label="Stage 0 Runtime Export FPS",
                            value=30,
                            precision=0,
                            info="Frame rate metadata used by the selected Stage 0 runtime export.",
                        )
                        preprocess_use_ray_pose = gr.Checkbox(
                            label="Use DA3 Ray Pose",
                            value=False,
                            info="Use ray-based pose estimation instead of the camera decoder.",
                        )
                        preprocess_streaming = gr.Checkbox(
                            label="Use DA3 Streaming",
                            value=DEFAULT_STAGE0_STREAMING,
                            info="Process the full selected sequence in overlapping chunks instead of one global DA3 batch.",
                        )
                        preprocess_streaming_global_guide = gr.Checkbox(
                            label="Use Global Guide Pass",
                            value=DEFAULT_STAGE0_STREAMING_GLOBAL_GUIDE,
                            info="Run a sparse whole-video DA3 pass and anchor dense chunks to exact matching guide frames.",
                        )
                    gr.Markdown(
                        "Stage 0 samples from the original video before DA3 runs, then prepares the filtered point-cloud cache plus "
                        "`before_non_rigid_icp.ply` for the selected pre-ICP settings. The DA3 GS preview video is optional, and `Stage 0 Runtime Export` chooses the runtime-ready output to write after preprocessing. "
                        "When `Use DA3 Streaming` is enabled, `DA3 Input Max Frames / Chunk Size` becomes the chunk size and Stage 0 covers the selected strided clip with overlapping chunks."
                    )
                    gr.Markdown("Reference-view note: this app now defaults DA3 to `first`.")
                    gr.Markdown(
                        "Example: in standard mode, raw `500` frames with `DA3 Input Max Frames / Chunk Size=100`, `DA3 Input Stride=6` samples 100 frames across the clip with no gap above 6. "
                        "In streaming mode, `DA3 Input Stride=6` uses every sixth raw frame before chunking."
                    )
                    gr.Markdown(
                        "When available, Stage 0 also stores DA3's sky mask in `results.npz`. The pre-ICP filtering controls below use it to drop sky pixels before the non-rigid ICP stage."
                    )
                with gr.Accordion("Stage 0 Pre-ICP Filtering", open=False):
                    with gr.Row():
                        alignment_num_frames = gr.Number(
                            label="Prep Num Frames",
                            value=50,
                            precision=0,
                            info="Maximum number of Stage 0 frames to prepare for the non-rigid ICP stage.",
                        )
                        alignment_stride = gr.Number(
                            label="Prep Stride",
                            value=1,
                            precision=0,
                            info="Stride over the Stage 0 frame set, not over the raw video.",
                        )
                        alignment_offset = gr.Number(label="Prep Offset", value=0, precision=0)
                    gr.Markdown(
                        "Stage 0 applies `Prep Stride` first over the DA3 frame set, then truncates to `Prep Num Frames` when preparing the pre-ICP caches. "
                        "Example: if Stage 0 produced `100` DA3 frames, then `Prep Stride=3`, `Prep Num Frames=20` takes frames `0, 3, 6, ...` and stops after 20 selected frames."
                    )
                    with gr.Row():
                        alignment_conf_profile = gr.Dropdown(
                            choices=[
                                ("Default Mixed (voxel + DA3)", "default_mixed"),
                                ("DA3 Only (Per Frame)", "da3_per_frame"),
                                ("DA3 Only (Global)", "da3_global"),
                                ("DA3 Only (Per Frame Guided)", "da3_per_frame_guided"),
                            ],
                            value="default_mixed",
                            label="Confidence Mode",
                        )
                        alignment_conf_percentile = gr.Number(label="DA3 Confidence Percentile", value=1.0)
                        stage1_knn_backend = gr.Dropdown(
                            choices=["cpu_kdtree", "gpu_kdtree"], value="cpu_kdtree", label="KNN Backend"
                        )
                    with gr.Row():
                        conf_mask_sky = gr.Checkbox(
                            label="Use DA3 Sky Mask",
                            value=True,
                            info="Requires Stage 0 sky output. Excludes DA3 sky pixels before point-cloud generation.",
                        )
                        conf_mask_sky_depth_band = gr.Checkbox(
                            label="Expand Sky By Depth Band",
                            value=False,
                            info="After sky masking, also drop pixels in the top x% depth band of the sky depth plateau.",
                        )
                        conf_sky_depth_band_percent = gr.Number(label="Sky Depth Band Percent", value=50.0)
                        conf_mask_depth_edges = gr.Checkbox(label="Suppress Depth Edges", value=True)
                        conf_edge_rtol = gr.Number(label="Depth Edge Rel Threshold", value=0.1)
                        conf_edge_atol = gr.Number(label="Depth Edge Abs Threshold", value=None)
                        conf_edge_kernel_size = gr.Number(label="Depth Edge Kernel", value=3, precision=0)
                    with gr.Row():
                        conf_mask_white_background = gr.Checkbox(
                            label="Suppress White Background",
                            value=False,
                            info="Drops bright, low-saturation image pixels from the Stage 0 point cache.",
                        )
                        conf_white_bg_min_rgb = gr.Number(label="White BG Min RGB", value=220.0)
                        conf_white_bg_max_channel_delta = gr.Number(
                            label="White BG Max Channel Delta",
                            value=25.0,
                        )
                        conf_white_bg_grow_px = gr.Number(
                            label="White BG Grow Pixels",
                            value=1,
                            precision=0,
                        )
                    with gr.Row():
                        conf_mask_min_depth_range_percent = gr.Checkbox(
                            label="Limit By Min Depth Range %",
                            value=True,
                            info="Per frame, keep only pixels up to min_depth + x% of that frame's valid depth range.",
                        )
                        conf_min_depth_range_percent = gr.Number(label="Min Depth Range Percent", value=50.0)
                        conf_mask_min_depth_range_meters = gr.Checkbox(
                            label="Limit By Min Depth Metres",
                            value=False,
                            info="Per frame, keep only pixels within a fixed metric distance of the frame minimum depth.",
                        )
                        conf_min_depth_range_meters = gr.Number(label="Min Depth Range Metres", value=3.0)
                    with gr.Row():
                        conf_mask_max_depth = gr.Checkbox(label="Suppress Max DA3 Depth Plateau", value=False)
                        conf_max_depth_rtol = gr.Number(label="Max Depth Rel Threshold", value=0.001)
                        conf_max_depth_atol = gr.Number(label="Max Depth Abs Threshold", value=None)
                    gr.Markdown(
                        "If `Use DA3 Sky Mask` is enabled, Stage 0 prep also writes debug PNGs under "
                        "`exports/ply/<active-filter>/debug_masks/{sky,kept}/` so you can inspect the raw DA3 sky mask and the final kept-pixel mask."
                    )
                    gr.Markdown(
                        "`Expand Sky By Depth Band` uses the masked sky depth plateau as a reference and removes any pixel within the top `x%` of that depth range."
                    )
                    gr.Markdown(
                        "`Limit By Min Depth Range %` measures each frame's valid depth span after sky-based masking and keeps only points up to "
                        "`min_depth + x% * (max_depth - min_depth)`. `Limit By Min Depth Metres` keeps only points up to "
                        "`min_depth + metres`. If both are enabled, the stricter limit wins."
                    )
                with gr.Accordion("Stage 1 RoMa Matching", open=False):
                    gr.Markdown(
                        "RoMa is the cross-frame matcher used to add correspondence constraints during Stage 1."
                    )
                    with gr.Row():
                        stage1_use_roma_matching = gr.Checkbox(label="Use RoMa Matching", value=True)
                        stage1_roma_version = gr.Dropdown(choices=["v2", "v1"], value="v2", label="RoMa Version")
                        stage1_roma_model = gr.Dropdown(
                            choices=["indoor", "outdoor", "tiny"], value="outdoor", label="RoMa Model"
                        )
                    with gr.Row():
                        stage1_roma_num_samples = gr.Number(label="RoMa Samples Per Pair", value=5000, precision=0)
                        stage1_roma_certainty_threshold = gr.Number(label="RoMa Certainty Threshold", value=0.5)
                        stage1_roma_max_references = gr.Number(label="RoMa Max References", value=20, precision=0)
                    with gr.Row():
                        stage1_roma_reference_sampling = gr.Dropdown(
                            choices=["recent_and_strided", "recent", "strided", "all_previous"],
                            value="recent_and_strided",
                            label="RoMa Reference Sampling",
                        )
                        stage1_roma_loss_weight = gr.Number(label="RoMa Loss Weight", value=1.0)
                        stage1_roma_max_corr_dist = gr.Number(label="RoMa Max Corr Dist", value=1.0)
                with gr.Accordion("Stage 1 ICP / Deformation", open=False):
                    with gr.Row():
                        stage1_tensorboard = gr.Checkbox(label="TensorBoard", value=True)
                        stage1_max_corr_dist = gr.Number(label="Max Corr Dist", value=0.03)
                        stage1_merge_voxel_size = gr.Number(
                            label="Merge Voxel Size",
                            value=0.001,
                            info="Voxel grid size for spatial dedup when merging frame points into the model. Smaller = denser clouds. Pi-Long uses 0.001, original default was 0.05.",
                        )
                        stage1_icp_n_iter = gr.Number(label="ICP Iterations", value=100, precision=0)
                        stage1_icp_method = gr.Dropdown(
                            choices=["point2plane", "point2point"], value="point2plane", label="ICP Method"
                        )
                    with gr.Row():
                        stage1_icp_early_stopping_patience = gr.Number(
                            label="Early Stop Patience", value=5, precision=0
                        )
                        stage1_icp_early_stopping_min_iters = gr.Number(
                            label="Early Stop Min Iters", value=25, precision=0
                        )
                        stage1_icp_early_stopping_min_delta = gr.Number(label="Early Stop Min Delta", value=None)
                        stage1_icp_lr = gr.Number(label="ICP LR", value=1e-3)
                    with gr.Row():
                        stage1_icp_local_twist_reg = gr.Number(label="Local Twist Reg", value=0.0)
                        stage1_icp_tv_reg = gr.Number(label="TV Reg", value=50.0)
                        stage1_icp_tv_voxel_size = gr.Number(label="TV Voxel Size", value=0.01)
                        stage1_icp_tv_every_k = gr.Number(label="TV Every K", value=1, precision=0)
                        stage1_icp_tv_sample_ratio = gr.Number(label="TV Sample Ratio", value=0.1)
                    with gr.Row():
                        stage1_icp_color_icp_weight = gr.Number(label="Color ICP Weight", value=0.02)
                        stage1_icp_color_icp_max_color_dist = gr.Number(label="Color ICP Max Color Dist", value=0.1)
                        stage1_icp_color_icp_k = gr.Number(label="Color ICP K", value=10, precision=0)
                        stage1_save_intermediate_every = gr.Number(
                            label="Save Intermediate Every", value=10, precision=0
                        )
                    with gr.Row():
                        stage1_deform_log2_hashmap_size = gr.Number(label="Deform Log2 Hashmap", value=19, precision=0)
                        stage1_deform_num_levels = gr.Number(label="Deform Num Levels", value=24, precision=0)
                        stage1_deform_n_neurons = gr.Number(label="Deform Neurons", value=64, precision=0)
                        stage1_deform_n_hidden_layers = gr.Number(label="Deform Hidden Layers", value=4, precision=0)
                        stage1_deform_min_res = gr.Number(label="Deform Min Res", value=16, precision=0)
                        stage1_deform_max_res = gr.Number(label="Deform Max Res", value=2048, precision=0)
                with gr.Accordion("Stage 1 Point Filtering", open=False):
                    with gr.Row():
                        stage1_filter_points = gr.Checkbox(label="Filter Points", value=False)
                        stage1_filter_geom_sigma = gr.Number(label="Geom Sigma", value=2.5)
                        stage1_filter_color_sigma = gr.Number(label="Color Sigma", value=1.5)
                        stage1_filter_worst_pct = gr.Number(label="Worst Percent", value=0.2)
                        stage1_filter_min_frames = gr.Number(label="Min Frames", value=2, precision=0)
                        stage1_filter_base_percentile = gr.Dropdown(
                            choices=["p75", "p90", "p95", "p99"], value="p75", label="Base Percentile"
                        )
                with gr.Accordion("Stage 2", open=False):
                    with gr.Row():
                        stage2_tensorboard = gr.Checkbox(label="TensorBoard", value=True)
                        stage2_knn_backend = gr.Dropdown(
                            choices=["cpu_kdtree", "gpu_kdtree"], value="cpu_kdtree", label="KNN Backend"
                        )
                        stage2_n_iters = gr.Number(label="Iterations", value=150, precision=0)
                        stage2_lr = gr.Number(label="LR", value=1e-3)
                    with gr.Row():
                        stage2_loo_loss_weight = gr.Number(label="LOO Loss Weight", value=1.0)
                        stage2_loo_k_neighbors = gr.Number(label="LOO K Neighbors", value=5, precision=0)
                        stage2_loo_max_corr_dist = gr.Number(label="LOO Max Corr Dist", value=0.03125)
                        stage2_loo_normal_k = gr.Number(label="LOO Normal K", value=20, precision=0)
                        stage2_loo_kdtree_rebuild_every = gr.Number(label="KDT Rebuild Every", value=50, precision=0)
                    with gr.Row():
                        stage2_loo_max_pairs_per_iter = gr.Number(label="Max Pairs Per Iter", value=200000, precision=0)
                        stage2_loo_pairs_per_src = gr.Number(label="Pairs Per Src", value=1, precision=0)
                        stage2_deform_chunk_size = gr.Number(label="Deform Chunk Size", value=50000, precision=0)
                        stage2_anchor_loss_weight = gr.Number(label="Anchor Loss Weight", value=1000.0)
                        stage2_anchor_n_samples = gr.Number(label="Anchor Samples", value=4096, precision=0)
                    with gr.Row():
                        stage2_tv_reg = gr.Number(label="TV Reg", value=50.0)
                        stage2_tv_voxel_size = gr.Number(label="TV Voxel Size", value=0.01)
                        stage2_tv_every_k = gr.Number(label="TV Every K", value=1, precision=0)
                        stage2_tv_sample_ratio = gr.Number(label="TV Sample Ratio", value=0.1)
                        stage2_loo_color_icp_weight = gr.Number(label="Color ICP Weight", value=0.02)
                        stage2_loo_color_icp_k = gr.Number(label="Color ICP K", value=10, precision=0)
                    with gr.Row():
                        stage2_loo_color_icp_max_color_dist = gr.Number(label="Color ICP Max Color Dist", value=0.1)
                        stage2_thin_shell_weight = gr.Number(label="Thin Shell Weight", value=1000.0)
                        stage2_save_intermediate_every_n = gr.Number(
                            label="Save Intermediate Every", value=50, precision=0
                        )
                with gr.Accordion("Stage 3.1", open=False):
                    with gr.Row():
                        inverse_epochs = gr.Number(label="Epoch Override", value=None, precision=0)
                        stage31_tensorboard = gr.Checkbox(label="TensorBoard", value=True)
                        stage31_knn_backend = gr.Dropdown(
                            choices=["cpu_kdtree", "gpu_kdtree"], value="cpu_kdtree", label="KNN Backend"
                        )
                        stage31_batch_size = gr.Number(label="Batch Size", value=8192, precision=0)
                        stage31_lr = gr.Number(label="LR", value=1e-3)
                    with gr.Row():
                        stage31_cycle_weight = gr.Number(label="Cycle Weight", value=0.1)
                        stage31_magnitude_weight = gr.Number(label="Magnitude Weight", value=1e-3)
                        stage31_smoothness_weight = gr.Number(label="Smoothness Weight", value=1e-3)
                        stage31_num_forward_samples = gr.Number(label="Forward Samples", value=10000, precision=0)
                        stage31_num_interp_samples = gr.Number(label="Interp Samples", value=5000, precision=0)
                        stage31_regenerate_every = gr.Number(label="Regenerate Every", value=10, precision=0)
                    with gr.Row():
                        stage31_view_embed_dim = gr.Number(label="View Embed Dim", value=32, precision=0)
                        stage31_min_res = gr.Number(label="Min Res", value=16, precision=0)
                        stage31_max_res = gr.Number(label="Max Res", value=2048, precision=0)
                        stage31_num_levels = gr.Number(label="Num Levels", value=16, precision=0)
                        stage31_log2_hashmap_size = gr.Number(label="Log2 Hashmap", value=19, precision=0)
                        stage31_n_neurons = gr.Number(label="Neurons", value=64, precision=0)
                        stage31_n_hidden_layers = gr.Number(label="Hidden Layers", value=3, precision=0)
                        stage31_save_validation_plys = gr.Checkbox(label="Save Validation PLYs", value=True)
                with gr.Accordion("Stage 3.2", open=False):
                    with gr.Row():
                        gs_num_iters = gr.Number(label="Iter Override", value=None, precision=0)
                        gs_tensorboard = gr.Checkbox(label="TensorBoard", value=True)
                        gs_target_num_points = gr.Number(label="Target Num Points", value=4000000, precision=0)
                        gs_frames_per_iter = gr.Number(label="Frames Per Iter", value=1, precision=0)
                    with gr.Row():
                        gs_sh_degree = gr.Number(label="SH Degree", value=3, precision=0)
                        gs_sh_increase_every = gr.Number(label="SH Increase Every", value=0, precision=0)
                        gs_sh_full_from_iter = gr.Number(label="SH Full From Iter", value=5000, precision=0)
                        gs_sh_freeze_means_when_full_sh = gr.Checkbox(label="Freeze Means When Full SH", value=True)
                        gs_sh_reg_weight = gr.Number(label="SH Reg Weight", value=10.0)
                    with gr.Row():
                        gs_optimize_cams = gr.Checkbox(label="Optimize Cams", value=True)
                        gs_lr_cams = gr.Number(label="LR Cams", value=1e-4)
                        gs_optimize_positions = gr.Checkbox(label="Optimize Positions", value=True)
                        gs_lr_positions = gr.Number(label="LR Positions", value=1e-5)
                        gs_lr_colors = gr.Number(label="LR Colors", value=2.5e-3)
                        gs_lr_opacities = gr.Number(label="LR Opacities", value=5e-2)
                    with gr.Row():
                        gs_lr_scales = gr.Number(label="LR Scales", value=5e-3)
                        gs_lr_quats = gr.Number(label="LR Quats", value=1e-3)
                        gs_lr_sh0 = gr.Number(label="LR SH0", value=2.5e-3)
                        gs_lr_shn = gr.Number(label="LR SHN", value=2.5e-3 / 20.0)
                        gs_deform_inverse_rotations = gr.Checkbox(label="Deform Inverse Rotations", value=True)
                        gs_initial_opacity = gr.Number(label="Initial Opacity", value=0.5)
                        gs_initial_scale = gr.Number(label="Initial Scale", value=0.005)
                    with gr.Row():
                        gs_initial_flat_ratio = gr.Number(label="Initial Flat Ratio", value=0.1)
                        gs_scale_init = gr.Dropdown(choices=["knn", "fixed"], value="knn", label="Scale Init")
                        gs_knn_neighbors = gr.Number(label="KNN Neighbors", value=4, precision=0)
                        gs_normal_k = gr.Number(label="Normal K", value=20, precision=0)
                        gs_l1_weight = gr.Number(label="L1 Weight", value=0.8)
                        gs_lpips_weight = gr.Number(label="LPIPS Weight", value=0.2)
                    with gr.Row():
                        gs_opacity_reg_weight = gr.Number(label="Opacity Reg Weight", value=0.0)
                        gs_scale_reg_weight = gr.Number(label="Scale Reg Weight", value=0.0)
                        gs_normal_consistency_weight = gr.Number(label="Normal Consistency Weight", value=0.05)
                        gs_distortion_weight = gr.Number(label="Distortion Weight", value=0.01)
                        gs_alpha_reg_weight = gr.Number(label="Alpha Reg Weight", value=0.0)
                    with gr.Row():
                        gs_log_every = gr.Number(label="Log Every", value=50, precision=0)
                        gs_save_every = gr.Number(label="Save Every", value=5000, precision=0)
                        gs_eval_every = gr.Number(label="Eval Every", value=1000, precision=0)
                        gs_lr_decay = gr.Number(label="LR Decay", value=0.1)
                        gs_auto_eval = gr.Checkbox(label="Auto Eval", value=True)

            with gr.Row():
                run_pipeline_button = gr.Button("Start Full Pipeline", variant="primary")
                stop_pipeline_button = gr.Button("Stop Active Run", variant="stop")

            pipeline_stop_feedback = gr.Markdown()
            pipeline_status_md = gr.Markdown()

            with gr.Row():
                pipeline_stage_text = gr.Textbox(label="Current Stage", interactive=False)
                pipeline_scene_root_text = gr.Textbox(label="Resolved Scene Root", interactive=False)
                pipeline_latest_run_dir_text = gr.Textbox(label="Latest Run Dir", interactive=False)

            pipeline_scene_report_md = gr.Markdown()

            with gr.Row():
                pipeline_primary_preview = gr.Video(label="Primary Preview", interactive=False)
                pipeline_secondary_preview = gr.Video(label="Secondary Preview", interactive=False)

            pipeline_key_files = gr.Files(label="Key Files")
            pipeline_live_log = gr.Textbox(label="Live Log", lines=24, interactive=False)

            pipeline_source_mode.change(
                fn=_update_pipeline_source_mode,
                inputs=[pipeline_source_mode],
                outputs=[
                    pipeline_upload_group,
                    pipeline_existing_video_group,
                    pipeline_existing_scene_group,
                    pipeline_output_group,
                ],
            )
            pipeline_uploaded_event = uploaded_video.upload(
                fn=_cache_uploaded_video_value,
                inputs=[uploaded_video],
                outputs=[uploaded_video_cached],
            )
            pipeline_run_event = run_pipeline_button.click(
                fn=_run_pipeline_generator,
                inputs=[
                    pipeline_source_mode,
                    uploaded_video_cached,
                    existing_video_selection,
                    existing_scene_root_selection,
                    output_parent_selection,
                    custom_scene_name,
                    mode,
                    renderer_choice,
                    preprocess_overwrite,
                    preprocess_max_frames,
                    preprocess_max_stride,
                    preprocess_streaming,
                    preprocess_streaming_overlap,
                    preprocess_streaming_global_guide,
                    preprocess_image_ext,
                    preprocess_model_name,
                    preprocess_process_res,
                    preprocess_process_res_method,
                    preprocess_export_gs_video,
                    preprocess_runtime_export_format,
                    preprocess_runtime_export_fps,
                    preprocess_use_ray_pose,
                    preprocess_ref_view_strategy,
                    alignment_num_frames,
                    alignment_stride,
                    alignment_offset,
                    alignment_conf_profile,
                    alignment_conf_percentile,
                    conf_mask_sky,
                    conf_mask_sky_depth_band,
                    conf_sky_depth_band_percent,
                    conf_mask_white_background,
                    conf_white_bg_min_rgb,
                    conf_white_bg_max_channel_delta,
                    conf_white_bg_grow_px,
                    conf_mask_min_depth_range_percent,
                    conf_min_depth_range_percent,
                    conf_mask_min_depth_range_meters,
                    conf_min_depth_range_meters,
                    conf_mask_depth_edges,
                    conf_edge_rtol,
                    conf_edge_atol,
                    conf_edge_kernel_size,
                    conf_mask_max_depth,
                    conf_max_depth_rtol,
                    conf_max_depth_atol,
                    stage1_use_roma_matching,
                    stage1_roma_version,
                    stage1_roma_model,
                    stage1_roma_num_samples,
                    stage1_roma_certainty_threshold,
                    stage1_roma_max_references,
                    stage1_roma_reference_sampling,
                    stage1_roma_loss_weight,
                    stage1_roma_max_corr_dist,
                    stage1_knn_backend,
                    stage1_tensorboard,
                    stage1_max_corr_dist,
                    stage1_merge_voxel_size,
                    stage1_icp_n_iter,
                    stage1_icp_early_stopping_patience,
                    stage1_icp_early_stopping_min_iters,
                    stage1_icp_early_stopping_min_delta,
                    stage1_icp_lr,
                    stage1_icp_method,
                    stage1_icp_local_twist_reg,
                    stage1_icp_tv_reg,
                    stage1_icp_tv_voxel_size,
                    stage1_icp_tv_every_k,
                    stage1_icp_tv_sample_ratio,
                    stage1_icp_color_icp_weight,
                    stage1_icp_color_icp_max_color_dist,
                    stage1_icp_color_icp_k,
                    stage1_save_intermediate_every,
                    stage1_deform_log2_hashmap_size,
                    stage1_deform_num_levels,
                    stage1_deform_n_neurons,
                    stage1_deform_n_hidden_layers,
                    stage1_deform_min_res,
                    stage1_deform_max_res,
                    stage1_filter_points,
                    stage1_filter_geom_sigma,
                    stage1_filter_color_sigma,
                    stage1_filter_worst_pct,
                    stage1_filter_min_frames,
                    stage1_filter_base_percentile,
                    stage2_tensorboard,
                    stage2_knn_backend,
                    stage2_loo_loss_weight,
                    stage2_loo_k_neighbors,
                    stage2_loo_max_corr_dist,
                    stage2_loo_normal_k,
                    stage2_loo_kdtree_rebuild_every,
                    stage2_loo_max_pairs_per_iter,
                    stage2_loo_pairs_per_src,
                    stage2_deform_chunk_size,
                    stage2_anchor_loss_weight,
                    stage2_anchor_n_samples,
                    stage2_tv_reg,
                    stage2_tv_voxel_size,
                    stage2_tv_every_k,
                    stage2_tv_sample_ratio,
                    stage2_loo_color_icp_weight,
                    stage2_loo_color_icp_k,
                    stage2_loo_color_icp_max_color_dist,
                    stage2_thin_shell_weight,
                    stage2_lr,
                    stage2_n_iters,
                    stage2_save_intermediate_every_n,
                    stage31_tensorboard,
                    stage31_knn_backend,
                    inverse_epochs,
                    stage31_batch_size,
                    stage31_lr,
                    stage31_cycle_weight,
                    stage31_magnitude_weight,
                    stage31_smoothness_weight,
                    stage31_num_forward_samples,
                    stage31_num_interp_samples,
                    stage31_regenerate_every,
                    stage31_view_embed_dim,
                    stage31_min_res,
                    stage31_max_res,
                    stage31_num_levels,
                    stage31_log2_hashmap_size,
                    stage31_n_neurons,
                    stage31_n_hidden_layers,
                    stage31_save_validation_plys,
                    gs_tensorboard,
                    gs_num_iters,
                    gs_sh_degree,
                    gs_sh_increase_every,
                    gs_sh_full_from_iter,
                    gs_sh_freeze_means_when_full_sh,
                    gs_sh_reg_weight,
                    gs_target_num_points,
                    gs_optimize_cams,
                    gs_lr_cams,
                    gs_optimize_positions,
                    gs_lr_positions,
                    gs_lr_colors,
                    gs_lr_opacities,
                    gs_lr_scales,
                    gs_lr_quats,
                    gs_lr_sh0,
                    gs_lr_shn,
                    gs_deform_inverse_rotations,
                    gs_initial_opacity,
                    gs_initial_scale,
                    gs_initial_flat_ratio,
                    gs_scale_init,
                    gs_knn_neighbors,
                    gs_normal_k,
                    gs_l1_weight,
                    gs_lpips_weight,
                    gs_opacity_reg_weight,
                    gs_scale_reg_weight,
                    gs_normal_consistency_weight,
                    gs_distortion_weight,
                    gs_alpha_reg_weight,
                    gs_frames_per_iter,
                    gs_log_every,
                    gs_save_every,
                    gs_eval_every,
                    gs_lr_decay,
                    gs_auto_eval,
                    dry_run,
                ],
                outputs=[
                    pipeline_run_state,
                    pipeline_status_md,
                    pipeline_stage_text,
                    pipeline_scene_root_text,
                    pipeline_latest_run_dir_text,
                    pipeline_scene_report_md,
                    pipeline_primary_preview,
                    pipeline_secondary_preview,
                    pipeline_key_files,
                    pipeline_live_log,
                ],
            )

            stop_pipeline_button.click(
                fn=_stop_active_run,
                inputs=[pipeline_run_state],
                outputs=[pipeline_stop_feedback, pipeline_run_state],
            )

        with gr.Tab("Run Explicit Stage"):
            gr.Markdown(
                "Run the native stage commands directly. Stage 0 starts from a video. Stages 1 through 3.2 follow the currently selected scene automatically."
            )

            with gr.Accordion("Stage 0 Source", open=True):
                with gr.Row():
                    stage0_source_mode = gr.Radio(
                        choices=[
                            ("Upload Video", "upload_video"),
                            ("Existing Video", "existing_video"),
                            ("Existing Image Folder", "existing_frames"),
                        ],
                        value="upload_video",
                        label="Stage 0 Source Mode",
                    )
                with gr.Row():
                    with gr.Group(visible=True) as stage0_upload_group:
                        stage0_uploaded_video = gr.File(
                            label="Upload Video",
                            file_types=[".mp4", ".mov", ".avi", ".mkv"],
                            type="filepath",
                        )
                        stage0_uploaded_video_cached = gr.State("")
                    with gr.Group(visible=False) as stage0_existing_video_group:
                        stage0_existing_video_selection = gr.Dropdown(
                            label="Existing Video",
                            choices=video_choices,
                            value=default_video,
                            info="Discovered under the videos workspace.",
                            allow_custom_value=True,
                        )
                    with gr.Group(visible=False) as stage0_existing_frames_group:
                        stage0_existing_frames_dir = gr.Textbox(
                            label="Existing Image Folder",
                            placeholder=r"Absolute or project-relative path to a folder of images",
                            info="Stage 0 will use these images directly with `--frames_dir`.",
                        )
                with gr.Row():
                    stage0_output_parent_selection = gr.Dropdown(
                        label="Output Parent Directory",
                        choices=output_parent_choices,
                        value="",
                        info="Leave on automatic unless you want a custom destination.",
                    )
                    stage0_custom_scene_name = gr.Textbox(
                        label="Custom Scene Name",
                        placeholder="Optional. If blank, the video filename is used.",
                    )
                with gr.Row():
                    stage0_overwrite = gr.Checkbox(label="Overwrite Stage 0 Outputs", value=False)
                    stage0_max_frames = gr.Number(
                        label="DA3 Input Max Frames / Chunk Size",
                        value=5,
                        precision=0,
                        info="Target DA3 frame count in standard mode. In streaming mode, this becomes the per-chunk DA3 batch size.",
                    )
                    stage0_max_stride = gr.Number(
                        label="DA3 Input Stride",
                        value=1000,
                        precision=0,
                        info="Maximum raw-frame gap in standard mode. In streaming mode this is applied exactly before chunking.",
                    )
                    stage0_streaming_overlap = gr.Number(
                        label="DA3 Streaming Overlap",
                        value=DEFAULT_STAGE0_STREAMING_OVERLAP,
                        precision=0,
                        info="Overlap between adjacent DA3 chunks when streaming mode is enabled.",
                    )
                    stage0_image_ext = gr.Textbox(label="Image Extension", value="png")
                    stage0_model_name = gr.Textbox(label="DA3 Model Name", value="depth-anything/DA3NESTED-GIANT-LARGE")
                    stage0_process_res = gr.Number(
                        label="DA3 Processing Resolution",
                        value=768,
                        precision=0,
                        info="Longest side is resized to this before DA3 runs. Higher = denser points, more VRAM. VRAM scales as (res/504)^2. Common: 504 (low), 768 (medium), 1024 (high).",
                    )
                with gr.Row():
                    stage0_process_res_method = gr.Dropdown(
                        choices=["upper_bound_resize", "upper_bound_crop"],
                        value="upper_bound_resize",
                        label="DA3 Resolution Method",
                        info="Resize policy before DA3 inference.",
                    )
                    stage0_ref_view_strategy = gr.Dropdown(
                        choices=["first", "middle", "saddle_balanced", "saddle_sim_range"],
                        value="saddle_balanced",
                        label="DA3 Reference View",
                        info="Run Explicit Stage defaults to `saddle_balanced`.",
                    )
                    stage0_export_gs_video = gr.Checkbox(
                        label="Export DA3 GS Preview Video",
                        value=False,
                        info="Opt-in preview export. Leave off for a faster Stage 0.",
                    )
                    stage0_runtime_export_format = gr.Dropdown(
                        choices=RUNTIME_EXPORT_CHOICES,
                        value=DEFAULT_STAGE0_RUNTIME_EXPORT_FORMAT,
                        label="Stage 0 Runtime Export",
                        info="Optional runtime-ready export written after Stage 0. Default is DirectStorage stream.",
                    )
                with gr.Row():
                    stage0_runtime_export_fps = gr.Number(
                        label="Stage 0 Runtime Export FPS",
                        value=30,
                        precision=0,
                        info="Frame rate metadata used by the selected Stage 0 runtime export.",
                    )
                    stage0_use_ray_pose = gr.Checkbox(
                        label="Use DA3 Ray Pose",
                        value=False,
                        info="Use ray-based pose estimation instead of the camera decoder.",
                    )
                    stage0_streaming = gr.Checkbox(
                        label="Use DA3 Streaming",
                        value=False,
                        info="Process the full selected sequence in overlapping chunks instead of one global DA3 batch.",
                    )
                    stage0_streaming_global_guide = gr.Checkbox(
                        label="Use Global Guide Pass",
                        value=DEFAULT_STAGE0_STREAMING_GLOBAL_GUIDE,
                        info="Run a sparse whole-video DA3 pass and anchor dense chunks to exact matching guide frames.",
                    )
                    gr.Markdown(
                        "Stage 0 samples from the original video before DA3 runs, then prepares the filtered point-cloud cache plus `before_non_rigid_icp.ply`. "
                        "Example: in standard mode, raw `500` frames with `DA3 Input Max Frames / Chunk Size=100`, `DA3 Input Stride=6` samples 100 frames across the clip with no gap above 6. "
                        "In streaming mode, `DA3 Input Stride=6` uses every sixth raw frame before chunking. "
                        "If `Stage 0 Source Mode` is `Existing Image Folder`, the app first copies that folder into `videos/_gradio_uploads/<uid>/frames/`, then runs Stage 0 from the copied images. Uploaded videos likewise land under `videos/_gradio_uploads/<uid>/<uid>.*` with a short default scene root at `videos/_gradio_uploads/<uid>/scene/`. In that mode `DA3 Input Stride` is ignored; `DA3 Input Max Frames / Chunk Size` is only used when streaming mode is enabled. "
                        "The optional DA3 GS preview video is skipped by default. `Stage 0 Runtime Export` defaults to DirectStorage stream."
                    )
                gr.Markdown("Reference-view note: this app now defaults DA3 to `first`.")

            with gr.Accordion("Existing Scene / Run", open=True):
                with gr.Row():
                    stage_scene_root_selection = gr.Dropdown(
                        label="Existing Scene Root",
                        choices=scene_choices,
                        value=default_scene,
                        info="Only scene roots with Stage 0 outputs are listed.",
                        allow_custom_value=True,
                    )
                stage_scene_report_md = gr.Markdown()

                with gr.Row():
                    stage_scene_root_text = gr.Textbox(label="Resolved Scene Root", interactive=False)
                    stage_selected_run_dir_text = gr.Textbox(label="Selected Run Directory", interactive=False)

                stage_run_name = gr.Dropdown(
                    choices=[""],
                    value="",
                    label="Prepared Stage 1 Run",
                    allow_custom_value=True,
                )

            with gr.Accordion("Stage Parameters", open=False):
                with gr.Accordion("Stage 0 Pre-ICP Filtering", open=False):
                    with gr.Row():
                        stage1_num_frames = gr.Number(
                            label="Prep Num Frames",
                            value=50,
                            precision=0,
                            info="Maximum number of Stage 0 frames to prepare for the non-rigid ICP stage.",
                        )
                        stage1_stride = gr.Number(
                            label="Prep Stride",
                            value=1,
                            precision=0,
                            info="Stride over the Stage 0 frame set, not over the raw video.",
                        )
                        stage1_offset = gr.Number(label="Prep Offset", value=0, precision=0)
                    gr.Markdown(
                        "Stage 0 applies `Prep Stride` first over the DA3 frame set, then truncates to `Prep Num Frames` when preparing the pre-ICP caches. "
                        "Example: if Stage 0 produced `100` DA3 frames, then `Prep Stride=3`, `Prep Num Frames=20` takes frames `0, 3, 6, ...` and stops after 20 selected frames."
                    )
                    with gr.Row():
                        stage1_conf_profile = gr.Dropdown(
                            choices=[
                                ("Default Mixed (voxel + DA3)", "default_mixed"),
                                ("DA3 Only (Per Frame)", "da3_per_frame"),
                                ("DA3 Only (Global)", "da3_global"),
                                ("DA3 Only (Per Frame Guided)", "da3_per_frame_guided"),
                            ],
                            value="default_mixed",
                            label="Confidence Mode",
                        )
                        stage1_conf_percentile = gr.Number(label="DA3 Confidence Percentile", value=1.0)
                        stage1_knn_backend = gr.Dropdown(
                            choices=["cpu_kdtree", "gpu_kdtree"], value="cpu_kdtree", label="KNN Backend"
                        )
                    with gr.Row():
                        stage1_conf_mask_sky = gr.Checkbox(
                            label="Use DA3 Sky Mask",
                            value=True,
                            info="Requires Stage 0 sky output. Excludes DA3 sky pixels before point-cloud generation.",
                        )
                        stage1_conf_mask_sky_depth_band = gr.Checkbox(
                            label="Expand Sky By Depth Band",
                            value=False,
                            info="After sky masking, also drop pixels in the top x% depth band of the sky depth plateau.",
                        )
                        stage1_conf_sky_depth_band_percent = gr.Number(label="Sky Depth Band Percent", value=50.0)
                        stage1_conf_mask_depth_edges = gr.Checkbox(label="Suppress Depth Edges", value=True)
                        stage1_conf_edge_rtol = gr.Number(label="Depth Edge Rel Threshold", value=0.1)
                        stage1_conf_edge_atol = gr.Number(label="Depth Edge Abs Threshold", value=None)
                        stage1_conf_edge_kernel_size = gr.Number(label="Depth Edge Kernel", value=3, precision=0)
                    with gr.Row():
                        stage1_conf_mask_white_background = gr.Checkbox(
                            label="Suppress White Background",
                            value=False,
                            info="Drops bright, low-saturation image pixels from the Stage 0 point cache.",
                        )
                        stage1_conf_white_bg_min_rgb = gr.Number(label="White BG Min RGB", value=220.0)
                        stage1_conf_white_bg_max_channel_delta = gr.Number(
                            label="White BG Max Channel Delta",
                            value=25.0,
                        )
                        stage1_conf_white_bg_grow_px = gr.Number(
                            label="White BG Grow Pixels",
                            value=1,
                            precision=0,
                        )
                    with gr.Row():
                        stage1_conf_mask_min_depth_range_percent = gr.Checkbox(
                            label="Limit By Min Depth Range %",
                            value=True,
                            info="Per frame, keep only pixels up to min_depth + x% of that frame's valid depth range.",
                        )
                        stage1_conf_min_depth_range_percent = gr.Number(label="Min Depth Range Percent", value=50.0)
                        stage1_conf_mask_min_depth_range_meters = gr.Checkbox(
                            label="Limit By Min Depth Metres",
                            value=False,
                            info="Per frame, keep only pixels within a fixed metric distance of the frame minimum depth.",
                        )
                        stage1_conf_min_depth_range_meters = gr.Number(label="Min Depth Range Metres", value=3.0)
                    with gr.Row():
                        stage1_conf_mask_max_depth = gr.Checkbox(label="Suppress Max DA3 Depth Plateau", value=False)
                        stage1_conf_max_depth_rtol = gr.Number(label="Max Depth Rel Threshold", value=0.001)
                        stage1_conf_max_depth_atol = gr.Number(label="Max Depth Abs Threshold", value=None)
                    gr.Markdown(
                        "If `Use DA3 Sky Mask` is enabled, Stage 0 prep also writes debug PNGs under "
                        "`exports/ply/<active-filter>/debug_masks/{sky,kept}/` so you can inspect the raw DA3 sky mask and the final kept-pixel mask."
                    )
                    gr.Markdown(
                        "`Expand Sky By Depth Band` uses the masked sky depth plateau as a reference and removes any pixel within the top `x%` of that depth range."
                    )
                    gr.Markdown(
                        "`Limit By Min Depth Range %` measures each frame's valid depth span after sky-based masking and keeps only points up to "
                        "`min_depth + x% * (max_depth - min_depth)`. `Limit By Min Depth Metres` keeps only points up to "
                        "`min_depth + metres`. If both are enabled, the stricter limit wins."
                    )
                with gr.Accordion("Stage 1 RoMa Matching", open=False):
                    gr.Markdown(
                        "RoMa is the cross-frame matcher used to add correspondence constraints during Stage 1."
                    )
                    with gr.Row():
                        stage1_use_roma_matching = gr.Checkbox(label="Use RoMa Matching", value=True)
                        stage1_roma_version = gr.Dropdown(choices=["v2", "v1"], value="v2", label="RoMa Version")
                        stage1_roma_model = gr.Dropdown(
                            choices=["indoor", "outdoor", "tiny"], value="outdoor", label="RoMa Model"
                        )
                    with gr.Row():
                        stage1_roma_num_samples = gr.Number(label="RoMa Samples Per Pair", value=5000, precision=0)
                        stage1_roma_certainty_threshold = gr.Number(label="RoMa Certainty Threshold", value=0.5)
                        stage1_roma_max_references = gr.Number(label="RoMa Max References", value=20, precision=0)
                    with gr.Row():
                        stage1_roma_reference_sampling = gr.Dropdown(
                            choices=["recent_and_strided", "recent", "strided", "all_previous"],
                            value="recent_and_strided",
                            label="RoMa Reference Sampling",
                        )
                        stage1_roma_loss_weight = gr.Number(label="RoMa Loss Weight", value=1.0)
                        stage1_roma_max_corr_dist = gr.Number(label="RoMa Max Corr Dist", value=1.0)
                with gr.Accordion("Stage 1 ICP / Deformation", open=False):
                    with gr.Row():
                        stage1_tensorboard = gr.Checkbox(label="TensorBoard", value=True)
                        stage1_max_corr_dist = gr.Number(label="Max Corr Dist", value=0.03)
                        stage1_merge_voxel_size = gr.Number(
                            label="Merge Voxel Size",
                            value=0.001,
                            info="Voxel grid size for spatial dedup when merging frame points into the model. Smaller = denser clouds. Pi-Long uses 0.001, original default was 0.05.",
                        )
                        stage1_icp_n_iter = gr.Number(label="ICP Iterations", value=100, precision=0)
                        stage1_icp_method = gr.Dropdown(
                            choices=["point2plane", "point2point"], value="point2plane", label="ICP Method"
                        )
                    with gr.Row():
                        stage1_icp_early_stopping_patience = gr.Number(
                            label="Early Stop Patience", value=5, precision=0
                        )
                        stage1_icp_early_stopping_min_iters = gr.Number(
                            label="Early Stop Min Iters", value=25, precision=0
                        )
                        stage1_icp_early_stopping_min_delta = gr.Number(label="Early Stop Min Delta", value=None)
                        stage1_icp_lr = gr.Number(label="ICP LR", value=1e-3)
                    with gr.Row():
                        stage1_icp_local_twist_reg = gr.Number(label="Local Twist Reg", value=0.0)
                        stage1_icp_tv_reg = gr.Number(label="TV Reg", value=50.0)
                        stage1_icp_tv_voxel_size = gr.Number(label="TV Voxel Size", value=0.01)
                        stage1_icp_tv_every_k = gr.Number(label="TV Every K", value=1, precision=0)
                        stage1_icp_tv_sample_ratio = gr.Number(label="TV Sample Ratio", value=0.1)
                    with gr.Row():
                        stage1_icp_color_icp_weight = gr.Number(label="Color ICP Weight", value=0.02)
                        stage1_icp_color_icp_max_color_dist = gr.Number(label="Color ICP Max Color Dist", value=0.1)
                        stage1_icp_color_icp_k = gr.Number(label="Color ICP K", value=10, precision=0)
                        stage1_save_intermediate_every = gr.Number(
                            label="Save Intermediate Every", value=10, precision=0
                        )
                    with gr.Row():
                        stage1_deform_log2_hashmap_size = gr.Number(label="Deform Log2 Hashmap", value=19, precision=0)
                        stage1_deform_num_levels = gr.Number(label="Deform Num Levels", value=24, precision=0)
                        stage1_deform_n_neurons = gr.Number(label="Deform Neurons", value=64, precision=0)
                        stage1_deform_n_hidden_layers = gr.Number(label="Deform Hidden Layers", value=4, precision=0)
                        stage1_deform_min_res = gr.Number(label="Deform Min Res", value=16, precision=0)
                        stage1_deform_max_res = gr.Number(label="Deform Max Res", value=2048, precision=0)
                with gr.Accordion("Stage 1 Point Filtering", open=False):
                    with gr.Row():
                        stage1_filter_points = gr.Checkbox(label="Filter Points", value=False)
                        stage1_filter_geom_sigma = gr.Number(label="Geom Sigma", value=2.5)
                        stage1_filter_color_sigma = gr.Number(label="Color Sigma", value=1.5)
                        stage1_filter_worst_pct = gr.Number(label="Worst Percent", value=0.2)
                        stage1_filter_min_frames = gr.Number(label="Min Frames", value=2, precision=0)
                        stage1_filter_base_percentile = gr.Dropdown(
                            choices=["p75", "p90", "p95", "p99"], value="p75", label="Base Percentile"
                        )
                with gr.Accordion("Stage 2", open=False):
                    with gr.Row():
                        stage2_tensorboard = gr.Checkbox(label="TensorBoard", value=True)
                        stage2_knn_backend = gr.Dropdown(
                            choices=["cpu_kdtree", "gpu_kdtree"], value="cpu_kdtree", label="KNN Backend"
                        )
                        stage2_n_iters = gr.Number(label="Iterations", value=150, precision=0)
                        stage2_lr = gr.Number(label="LR", value=1e-3)
                    with gr.Row():
                        stage2_loo_loss_weight = gr.Number(label="LOO Loss Weight", value=1.0)
                        stage2_loo_k_neighbors = gr.Number(label="LOO K Neighbors", value=5, precision=0)
                        stage2_loo_max_corr_dist = gr.Number(label="LOO Max Corr Dist", value=0.03125)
                        stage2_loo_normal_k = gr.Number(label="LOO Normal K", value=20, precision=0)
                        stage2_loo_kdtree_rebuild_every = gr.Number(label="KDT Rebuild Every", value=50, precision=0)
                    with gr.Row():
                        stage2_loo_max_pairs_per_iter = gr.Number(label="Max Pairs Per Iter", value=200000, precision=0)
                        stage2_loo_pairs_per_src = gr.Number(label="Pairs Per Src", value=1, precision=0)
                        stage2_deform_chunk_size = gr.Number(label="Deform Chunk Size", value=50000, precision=0)
                        stage2_anchor_loss_weight = gr.Number(label="Anchor Loss Weight", value=1000.0)
                        stage2_anchor_n_samples = gr.Number(label="Anchor Samples", value=4096, precision=0)
                    with gr.Row():
                        stage2_tv_reg = gr.Number(label="TV Reg", value=50.0)
                        stage2_tv_voxel_size = gr.Number(label="TV Voxel Size", value=0.01)
                        stage2_tv_every_k = gr.Number(label="TV Every K", value=1, precision=0)
                        stage2_tv_sample_ratio = gr.Number(label="TV Sample Ratio", value=0.1)
                        stage2_loo_color_icp_weight = gr.Number(label="Color ICP Weight", value=0.02)
                        stage2_loo_color_icp_k = gr.Number(label="Color ICP K", value=10, precision=0)
                    with gr.Row():
                        stage2_loo_color_icp_max_color_dist = gr.Number(label="Color ICP Max Color Dist", value=0.1)
                        stage2_thin_shell_weight = gr.Number(label="Thin Shell Weight", value=1000.0)
                        stage2_save_intermediate_every_n = gr.Number(
                            label="Save Intermediate Every", value=50, precision=0
                        )
                with gr.Accordion("Stage 3.1", open=False):
                    with gr.Row():
                        stage31_checkpoint_subdir = gr.Dropdown(choices=[], label="Checkpoint Input")
                        stage31_epochs = gr.Number(label="Epoch Override", value=None, precision=0)
                        stage31_tensorboard = gr.Checkbox(label="TensorBoard", value=True)
                        stage31_knn_backend = gr.Dropdown(
                            choices=["cpu_kdtree", "gpu_kdtree"], value="cpu_kdtree", label="KNN Backend"
                        )
                    with gr.Row():
                        stage31_batch_size = gr.Number(label="Batch Size", value=8192, precision=0)
                        stage31_lr = gr.Number(label="LR", value=1e-3)
                        stage31_cycle_weight = gr.Number(label="Cycle Weight", value=0.1)
                        stage31_magnitude_weight = gr.Number(label="Magnitude Weight", value=1e-3)
                        stage31_smoothness_weight = gr.Number(label="Smoothness Weight", value=1e-3)
                    with gr.Row():
                        stage31_num_forward_samples = gr.Number(label="Forward Samples", value=10000, precision=0)
                        stage31_num_interp_samples = gr.Number(label="Interp Samples", value=5000, precision=0)
                        stage31_regenerate_every = gr.Number(label="Regenerate Every", value=10, precision=0)
                        stage31_view_embed_dim = gr.Number(label="View Embed Dim", value=32, precision=0)
                        stage31_min_res = gr.Number(label="Min Res", value=16, precision=0)
                        stage31_max_res = gr.Number(label="Max Res", value=2048, precision=0)
                    with gr.Row():
                        stage31_num_levels = gr.Number(label="Num Levels", value=16, precision=0)
                        stage31_log2_hashmap_size = gr.Number(label="Log2 Hashmap", value=19, precision=0)
                        stage31_n_neurons = gr.Number(label="Neurons", value=64, precision=0)
                        stage31_n_hidden_layers = gr.Number(label="Hidden Layers", value=3, precision=0)
                        stage31_save_validation_plys = gr.Checkbox(label="Save Validation PLYs", value=True)
                with gr.Accordion("Stage 3.2", open=False):
                    with gr.Row():
                        stage32_checkpoint_subdir = gr.Dropdown(choices=[], label="Checkpoint Input")
                        stage32_inverse_dir_name = gr.Dropdown(choices=[], label="Inverse Deformation Dir")
                        stage32_renderer = gr.Dropdown(choices=["2dgs", "3dgs"], value="3dgs", label="Renderer")
                        stage32_num_iters = gr.Number(label="Iter Override", value=None, precision=0)
                    with gr.Row():
                        gs_tensorboard = gr.Checkbox(label="TensorBoard", value=True)
                        gs_target_num_points = gr.Number(label="Target Num Points", value=4000000, precision=0)
                        gs_frames_per_iter = gr.Number(label="Frames Per Iter", value=1, precision=0)
                        gs_sh_degree = gr.Number(label="SH Degree", value=3, precision=0)
                        gs_sh_increase_every = gr.Number(label="SH Increase Every", value=0, precision=0)
                        gs_sh_full_from_iter = gr.Number(label="SH Full From Iter", value=5000, precision=0)
                    with gr.Row():
                        gs_sh_freeze_means_when_full_sh = gr.Checkbox(label="Freeze Means When Full SH", value=True)
                        gs_sh_reg_weight = gr.Number(label="SH Reg Weight", value=10.0)
                        gs_optimize_cams = gr.Checkbox(label="Optimize Cams", value=True)
                        gs_lr_cams = gr.Number(label="LR Cams", value=1e-4)
                        gs_optimize_positions = gr.Checkbox(label="Optimize Positions", value=True)
                        gs_lr_positions = gr.Number(label="LR Positions", value=1e-5)
                    with gr.Row():
                        gs_lr_colors = gr.Number(label="LR Colors", value=2.5e-3)
                        gs_lr_opacities = gr.Number(label="LR Opacities", value=5e-2)
                        gs_lr_scales = gr.Number(label="LR Scales", value=5e-3)
                        gs_lr_quats = gr.Number(label="LR Quats", value=1e-3)
                        gs_lr_sh0 = gr.Number(label="LR SH0", value=2.5e-3)
                        gs_lr_shn = gr.Number(label="LR SHN", value=2.5e-3 / 20.0)
                    with gr.Row():
                        gs_deform_inverse_rotations = gr.Checkbox(label="Deform Inverse Rotations", value=True)
                        gs_initial_opacity = gr.Number(label="Initial Opacity", value=0.5)
                        gs_initial_scale = gr.Number(label="Initial Scale", value=0.005)
                        gs_initial_flat_ratio = gr.Number(label="Initial Flat Ratio", value=0.1)
                        gs_scale_init = gr.Dropdown(choices=["knn", "fixed"], value="knn", label="Scale Init")
                        gs_knn_neighbors = gr.Number(label="KNN Neighbors", value=4, precision=0)
                        gs_normal_k = gr.Number(label="Normal K", value=20, precision=0)
                    with gr.Row():
                        gs_l1_weight = gr.Number(label="L1 Weight", value=0.8)
                        gs_lpips_weight = gr.Number(label="LPIPS Weight", value=0.2)
                        gs_opacity_reg_weight = gr.Number(label="Opacity Reg Weight", value=0.0)
                        gs_scale_reg_weight = gr.Number(label="Scale Reg Weight", value=0.0)
                        gs_normal_consistency_weight = gr.Number(label="Normal Consistency Weight", value=0.05)
                    with gr.Row():
                        gs_distortion_weight = gr.Number(label="Distortion Weight", value=0.01)
                        gs_alpha_reg_weight = gr.Number(label="Alpha Reg Weight", value=0.0)
                        gs_log_every = gr.Number(label="Log Every", value=50, precision=0)
                        gs_save_every = gr.Number(label="Save Every", value=5000, precision=0)
                        gs_eval_every = gr.Number(label="Eval Every", value=1000, precision=0)
                        gs_lr_decay = gr.Number(label="LR Decay", value=0.1)
                        gs_auto_eval = gr.Checkbox(label="Auto Eval", value=True)
                stage32_original_images_dir = gr.Textbox(
                    label="Stage 3.2 Original Images Dir",
                    interactive=False,
                    placeholder="Auto-resolved from <scene_root>/frames_subsampled when present.",
                )

            stage_button_guide_md = gr.Markdown(
                "\n".join(
                    [
                        "**Stage Guide**",
                        "- `Stage 0`: extract frames, run DA3 preprocessing, build the filtered point-cloud cache, and write `before_non_rigid_icp.ply`.",
                        "- `Stage 1`: run non-rigid ICP on the prepared Stage 0 inputs and write `after_non_rigid_icp`.",
                        "- `Stage 2`: jointly refine that run into `after_global_optimization`.",
                        "- `Stage 3.1`: train the inverse deformation model from the selected checkpoint.",
                        "- `Stage 3.2`: train the Gaussian splat from the selected checkpoint plus inverse deformation output.",
                    ]
                )
            )
            stage_next_step_md = gr.Markdown(
                "**Recommended Next Step**: `Stage 0`\n\nStart from a video to create the scene root, DA3 outputs, filtered point-cloud cache, and pre-ICP merge."
            )

            with gr.Row():
                run_stage0_button = gr.Button("Run Stage 0", variant="primary")
                run_stage1_button = gr.Button("Run Stage 1")
                run_stage2_button = gr.Button("Run Stage 2")
                run_stage31_button = gr.Button("Run Stage 3.1")
                run_stage32_button = gr.Button("Run Stage 3.2")
                stop_stage_button = gr.Button("Stop Active Run", variant="stop")
            with gr.Row():
                export_div_button = gr.Button("Export Depth Volume", variant="secondary")
                export_runtime_button = gr.Button("Export Stage 0 Runtime Format", variant="secondary")
                export_ply_button = gr.Button("Export PLY", variant="secondary")
                export_cloudcompare_button = gr.Button("Export CloudCompare Edit PLY", variant="secondary")
                apply_cloudcompare_button = gr.Button("Apply CloudCompare Edit PLY", variant="secondary")

            with gr.Accordion("PLY Export Settings", open=False):
                with gr.Row():
                    ply_checkpoint_source = gr.Dropdown(
                        choices=["auto", "after_global_optimization", "after_non_rigid_icp"],
                        value="auto",
                        label="Source Checkpoint",
                        info="Which aligned_points.ply to use. 'auto' prefers Stage 2 if available, else Stage 1.",
                    )
                    ply_filename = gr.Textbox(label="Output Filename", value="export_cloud.ply")
                with gr.Row():
                    ply_dedup_enable = gr.Checkbox(label="Voxel Dedup", value=True)
                    ply_dedup_radius = gr.Number(
                        label="Dedup Radius",
                        value=0.001,
                        info="Voxel grid size. Smaller = denser. Pi-Long default is 0.001.",
                    )
                    ply_normals_k = gr.Number(
                        label="Normals K", value=16, precision=0, info="Number of neighbors for PCA normal estimation."
                    )
                    ply_chunk_size = gr.Number(
                        label="Chunk Size",
                        value=50000,
                        precision=0,
                        info="Points processed per batch during normal estimation.",
                    )
                    depth_volume_resolution_scale = gr.Dropdown(
                        choices=["1", "2", "4"],
                        value="1",
                        label="Depth Volume Scale",
                        info="Scales export resolution and intrinsics. Higher reduces pixel rounding and collisions.",
                    )

            with gr.Accordion("CloudCompare Point Pruning", open=False):
                with gr.Row():
                    cloudcompare_source = gr.Dropdown(
                        choices=[
                            ("Before non-rigid ICP", "before_non_rigid_icp"),
                            ("Stage 1 aligned_points", "after_non_rigid_icp"),
                            ("Stage 2 aligned_points", "after_global_optimization"),
                        ],
                        value="before_non_rigid_icp",
                        label="Edit Source",
                        info="Before-ICP edits create a new prepared run for Stage 1. Aligned edits create a new run for export.",
                    )
                    cloudcompare_filename = gr.Textbox(
                        label="Export Filename",
                        value="cloudcompare_edit.ply",
                    )
                    cloudcompare_output_suffix = gr.Textbox(
                        label="New Run Suffix",
                        value="ccpruned",
                    )
                cloudcompare_edited_ply = gr.File(
                    label="Edited CloudCompare PLY",
                    file_types=[".ply"],
                    type="filepath",
                )

            stage_stop_feedback = gr.Markdown()
            stage_status_md = gr.Markdown()

            with gr.Row():
                stage_live_stage = gr.Textbox(label="Current Stage", interactive=False)
                stage_live_scene_root = gr.Textbox(label="Resolved Scene Root", interactive=False)
                stage_live_latest_run_dir = gr.Textbox(label="Latest Run Dir", interactive=False)

            with gr.Row():
                stage_primary_preview = gr.Video(label="Primary Preview", interactive=False)
                stage_secondary_preview = gr.Video(label="Secondary Preview", interactive=False)

            stage_key_files = gr.Files(label="Key Files")
            stage_live_log = gr.Textbox(label="Live Log", lines=24, interactive=False)

            stage0_source_mode.change(
                fn=_update_stage0_source_mode,
                inputs=[stage0_source_mode],
                outputs=[stage0_upload_group, stage0_existing_video_group, stage0_existing_frames_group],
            )
            stage0_uploaded_event = stage0_uploaded_video.upload(
                fn=_cache_uploaded_video_value,
                inputs=[stage0_uploaded_video],
                outputs=[stage0_uploaded_video_cached],
            )
            stage_scene_inputs = [
                stage0_source_mode,
                stage0_uploaded_video_cached,
                stage0_existing_video_selection,
                stage0_existing_frames_dir,
                stage0_output_parent_selection,
                stage0_custom_scene_name,
                stage0_overwrite,
                stage0_max_frames,
                stage0_max_stride,
                stage0_streaming,
                stage0_streaming_overlap,
                stage0_streaming_global_guide,
                stage0_image_ext,
                stage0_model_name,
                stage0_process_res,
                stage0_process_res_method,
                stage0_export_gs_video,
                stage0_runtime_export_format,
                stage0_runtime_export_fps,
                stage0_use_ray_pose,
                stage0_ref_view_strategy,
                stage_scene_root_selection,
                stage_run_name,
                stage1_num_frames,
                stage1_stride,
                stage1_offset,
                stage1_conf_profile,
                stage1_conf_percentile,
                stage1_conf_mask_sky,
                stage1_conf_mask_sky_depth_band,
                stage1_conf_sky_depth_band_percent,
                stage1_conf_mask_white_background,
                stage1_conf_white_bg_min_rgb,
                stage1_conf_white_bg_max_channel_delta,
                stage1_conf_white_bg_grow_px,
                stage1_conf_mask_min_depth_range_percent,
                stage1_conf_min_depth_range_percent,
                stage1_conf_mask_min_depth_range_meters,
                stage1_conf_min_depth_range_meters,
                stage1_conf_mask_depth_edges,
                stage1_conf_edge_rtol,
                stage1_conf_edge_atol,
                stage1_conf_edge_kernel_size,
                stage1_conf_mask_max_depth,
                stage1_conf_max_depth_rtol,
                stage1_conf_max_depth_atol,
                stage1_use_roma_matching,
                stage1_roma_version,
                stage1_roma_model,
                stage1_roma_num_samples,
                stage1_roma_certainty_threshold,
                stage1_roma_max_references,
                stage1_roma_reference_sampling,
                stage1_roma_loss_weight,
                stage1_roma_max_corr_dist,
                stage1_knn_backend,
                stage1_tensorboard,
                stage1_max_corr_dist,
                stage1_merge_voxel_size,
                stage1_icp_n_iter,
                stage1_icp_early_stopping_patience,
                stage1_icp_early_stopping_min_iters,
                stage1_icp_early_stopping_min_delta,
                stage1_icp_lr,
                stage1_icp_method,
                stage1_icp_local_twist_reg,
                stage1_icp_tv_reg,
                stage1_icp_tv_voxel_size,
                stage1_icp_tv_every_k,
                stage1_icp_tv_sample_ratio,
                stage1_icp_color_icp_weight,
                stage1_icp_color_icp_max_color_dist,
                stage1_icp_color_icp_k,
                stage1_save_intermediate_every,
                stage1_deform_log2_hashmap_size,
                stage1_deform_num_levels,
                stage1_deform_n_neurons,
                stage1_deform_n_hidden_layers,
                stage1_deform_min_res,
                stage1_deform_max_res,
                stage1_filter_points,
                stage1_filter_geom_sigma,
                stage1_filter_color_sigma,
                stage1_filter_worst_pct,
                stage1_filter_min_frames,
                stage1_filter_base_percentile,
                stage2_tensorboard,
                stage2_knn_backend,
                stage2_loo_loss_weight,
                stage2_loo_k_neighbors,
                stage2_loo_max_corr_dist,
                stage2_loo_normal_k,
                stage2_loo_kdtree_rebuild_every,
                stage2_loo_max_pairs_per_iter,
                stage2_loo_pairs_per_src,
                stage2_deform_chunk_size,
                stage2_anchor_loss_weight,
                stage2_anchor_n_samples,
                stage2_tv_reg,
                stage2_tv_voxel_size,
                stage2_tv_every_k,
                stage2_tv_sample_ratio,
                stage2_loo_color_icp_weight,
                stage2_loo_color_icp_k,
                stage2_loo_color_icp_max_color_dist,
                stage2_thin_shell_weight,
                stage2_lr,
                stage2_n_iters,
                stage2_save_intermediate_every_n,
                stage31_checkpoint_subdir,
                stage31_epochs,
                stage31_tensorboard,
                stage31_knn_backend,
                stage31_batch_size,
                stage31_lr,
                stage31_cycle_weight,
                stage31_magnitude_weight,
                stage31_smoothness_weight,
                stage31_num_forward_samples,
                stage31_num_interp_samples,
                stage31_regenerate_every,
                stage31_view_embed_dim,
                stage31_min_res,
                stage31_max_res,
                stage31_num_levels,
                stage31_log2_hashmap_size,
                stage31_n_neurons,
                stage31_n_hidden_layers,
                stage31_save_validation_plys,
                stage32_checkpoint_subdir,
                stage32_inverse_dir_name,
                stage32_renderer,
                stage32_num_iters,
                gs_tensorboard,
                gs_sh_degree,
                gs_sh_increase_every,
                gs_sh_full_from_iter,
                gs_sh_freeze_means_when_full_sh,
                gs_sh_reg_weight,
                gs_target_num_points,
                gs_optimize_cams,
                gs_lr_cams,
                gs_optimize_positions,
                gs_lr_positions,
                gs_lr_colors,
                gs_lr_opacities,
                gs_lr_scales,
                gs_lr_quats,
                gs_lr_sh0,
                gs_lr_shn,
                gs_deform_inverse_rotations,
                gs_initial_opacity,
                gs_initial_scale,
                gs_initial_flat_ratio,
                gs_scale_init,
                gs_knn_neighbors,
                gs_normal_k,
                gs_l1_weight,
                gs_lpips_weight,
                gs_opacity_reg_weight,
                gs_scale_reg_weight,
                gs_normal_consistency_weight,
                gs_distortion_weight,
                gs_alpha_reg_weight,
                gs_frames_per_iter,
                gs_log_every,
                gs_save_every,
                gs_eval_every,
                gs_lr_decay,
                gs_auto_eval,
            ]
            stage_scene_outputs = [
                stage_run_state,
                stage_status_md,
                stage_live_stage,
                stage_live_scene_root,
                stage_live_latest_run_dir,
                stage_scene_report_md,
                stage_primary_preview,
                stage_secondary_preview,
                stage_key_files,
                stage_live_log,
                stage_scene_root_selection,
                stage_scene_root_text,
                stage_run_name,
                stage_selected_run_dir_text,
                stage31_checkpoint_subdir,
                stage32_checkpoint_subdir,
                stage32_inverse_dir_name,
                stage32_original_images_dir,
                stage_next_step_md,
                run_stage0_button,
                run_stage1_button,
                run_stage2_button,
                run_stage31_button,
                run_stage32_button,
            ]

            stage0_run_event = run_stage0_button.click(
                fn=partial(_run_stage_generator, "stage0"),
                inputs=stage_scene_inputs,
                outputs=stage_scene_outputs,
            )
            stage1_run_event = run_stage1_button.click(
                fn=partial(_run_stage_generator, "stage1"),
                inputs=stage_scene_inputs,
                outputs=stage_scene_outputs,
            )
            stage2_run_event = run_stage2_button.click(
                fn=partial(_run_stage_generator, "stage2"),
                inputs=stage_scene_inputs,
                outputs=stage_scene_outputs,
            )
            stage31_run_event = run_stage31_button.click(
                fn=partial(_run_stage_generator, "stage31"),
                inputs=stage_scene_inputs,
                outputs=stage_scene_outputs,
            )
            stage32_run_event = run_stage32_button.click(
                fn=partial(_run_stage_generator, "stage32"),
                inputs=stage_scene_inputs,
                outputs=stage_scene_outputs,
            )

            stop_stage_button.click(
                fn=_stop_active_run,
                inputs=[stage_run_state],
                outputs=[stage_stop_feedback, stage_run_state],
            )

            export_div_button.click(
                fn=_export_depth_volume,
                inputs=[
                    stage_scene_root_selection,
                    stage_run_name,
                    ply_dedup_enable,
                    ply_dedup_radius,
                    ply_normals_k,
                    ply_chunk_size,
                    depth_volume_resolution_scale,
                ],
                outputs=[stage_status_md],
            )

            export_runtime_button.click(
                fn=_export_runtime_format,
                inputs=[
                    stage_scene_root_selection,
                    stage0_runtime_export_format,
                    stage0_runtime_export_fps,
                    stage_run_name,
                ],
                outputs=[stage_status_md],
            )

            export_ply_button.click(
                fn=_export_ply_with_normals,
                inputs=[
                    stage_scene_root_selection,
                    stage_run_name,
                    ply_checkpoint_source,
                    ply_filename,
                    ply_dedup_enable,
                    ply_dedup_radius,
                    ply_normals_k,
                    ply_chunk_size,
                ],
                outputs=[stage_status_md],
            )

            export_cloudcompare_button.click(
                fn=_export_cloudcompare_edit_ply,
                inputs=[
                    stage_scene_root_selection,
                    stage_run_name,
                    cloudcompare_source,
                    cloudcompare_filename,
                ],
                outputs=[stage_status_md],
            )

            apply_cloudcompare_button.click(
                fn=_apply_cloudcompare_edit_ply,
                inputs=[
                    stage_scene_root_selection,
                    stage_run_name,
                    cloudcompare_source,
                    cloudcompare_edited_ply,
                    cloudcompare_output_suffix,
                ],
                outputs=[
                    stage_status_md,
                    stage_run_name,
                    stage_selected_run_dir_text,
                ],
            )

            stage_parameter_outputs = [
                stage0_max_frames,
                stage0_max_stride,
                stage0_streaming,
                stage0_streaming_overlap,
                stage0_streaming_global_guide,
                stage0_image_ext,
                stage0_model_name,
                stage0_process_res,
                stage0_process_res_method,
                stage0_export_gs_video,
                stage0_runtime_export_format,
                stage0_runtime_export_fps,
                stage0_use_ray_pose,
                stage0_ref_view_strategy,
                stage1_num_frames,
                stage1_stride,
                stage1_offset,
                stage1_conf_profile,
                stage1_conf_percentile,
                stage1_conf_mask_sky,
                stage1_conf_mask_sky_depth_band,
                stage1_conf_sky_depth_band_percent,
                stage1_conf_mask_white_background,
                stage1_conf_white_bg_min_rgb,
                stage1_conf_white_bg_max_channel_delta,
                stage1_conf_white_bg_grow_px,
                stage1_conf_mask_min_depth_range_percent,
                stage1_conf_min_depth_range_percent,
                stage1_conf_mask_min_depth_range_meters,
                stage1_conf_min_depth_range_meters,
                stage1_conf_mask_depth_edges,
                stage1_conf_edge_rtol,
                stage1_conf_edge_atol,
                stage1_conf_edge_kernel_size,
                stage1_conf_mask_max_depth,
                stage1_conf_max_depth_rtol,
                stage1_conf_max_depth_atol,
                stage1_use_roma_matching,
                stage1_roma_version,
                stage1_roma_model,
                stage1_roma_num_samples,
                stage1_roma_certainty_threshold,
                stage1_roma_max_references,
                stage1_roma_reference_sampling,
                stage1_roma_loss_weight,
                stage1_roma_max_corr_dist,
                stage1_knn_backend,
                stage1_tensorboard,
                stage1_max_corr_dist,
                stage1_merge_voxel_size,
                stage1_icp_n_iter,
                stage1_icp_early_stopping_patience,
                stage1_icp_early_stopping_min_iters,
                stage1_icp_early_stopping_min_delta,
                stage1_icp_lr,
                stage1_icp_method,
                stage1_icp_local_twist_reg,
                stage1_icp_tv_reg,
                stage1_icp_tv_voxel_size,
                stage1_icp_tv_every_k,
                stage1_icp_tv_sample_ratio,
                stage1_icp_color_icp_weight,
                stage1_icp_color_icp_max_color_dist,
                stage1_icp_color_icp_k,
                stage1_save_intermediate_every,
                stage1_deform_log2_hashmap_size,
                stage1_deform_num_levels,
                stage1_deform_n_neurons,
                stage1_deform_n_hidden_layers,
                stage1_deform_min_res,
                stage1_deform_max_res,
                stage1_filter_points,
                stage1_filter_geom_sigma,
                stage1_filter_color_sigma,
                stage1_filter_worst_pct,
                stage1_filter_min_frames,
                stage1_filter_base_percentile,
            ]

            stage_reset_outputs = [
                stage_scene_root_selection,
                stage_scene_report_md,
                stage_scene_root_text,
                stage_run_name,
                stage_selected_run_dir_text,
                stage31_checkpoint_subdir,
                stage32_checkpoint_subdir,
                stage32_inverse_dir_name,
                stage32_original_images_dir,
                stage_next_step_md,
                run_stage0_button,
                run_stage1_button,
                run_stage2_button,
                run_stage31_button,
                run_stage32_button,
            ] + stage_parameter_outputs

            scene_choice_outputs = [
                stage_scene_report_md,
                stage_scene_root_text,
                stage_run_name,
                stage_selected_run_dir_text,
                stage31_checkpoint_subdir,
                stage32_checkpoint_subdir,
                stage32_inverse_dir_name,
                stage32_original_images_dir,
                stage_next_step_md,
                run_stage0_button,
                run_stage1_button,
                run_stage2_button,
                run_stage31_button,
                run_stage32_button,
            ]

            stage_scene_root_selection.change(
                fn=_refresh_stage_scene,
                inputs=[stage_scene_root_selection],
                outputs=scene_choice_outputs,
            )
            stage_scene_root_selection.change(
                fn=_stage_parameter_updates_for_scene,
                inputs=[stage_scene_root_selection, stage_run_name],
                outputs=stage_parameter_outputs,
            )
            stage0_source_mode.change(
                fn=_reset_stage_panel_for_new_source,
                inputs=[
                    stage0_source_mode,
                    stage0_uploaded_video_cached,
                    stage0_existing_video_selection,
                    stage0_existing_frames_dir,
                ],
                outputs=stage_reset_outputs,
            )
            stage0_existing_video_selection.change(
                fn=_reset_stage_panel_for_new_source,
                inputs=[
                    stage0_source_mode,
                    stage0_uploaded_video_cached,
                    stage0_existing_video_selection,
                    stage0_existing_frames_dir,
                ],
                outputs=stage_reset_outputs,
            )
            stage0_existing_frames_dir.change(
                fn=_reset_stage_panel_for_new_source,
                inputs=[
                    stage0_source_mode,
                    stage0_uploaded_video_cached,
                    stage0_existing_video_selection,
                    stage0_existing_frames_dir,
                ],
                outputs=stage_reset_outputs,
            )
            stage_run_name.change(
                fn=_refresh_stage_run,
                inputs=[stage_scene_root_selection, stage_run_name],
                outputs=[
                    stage_selected_run_dir_text,
                    stage31_checkpoint_subdir,
                    stage32_checkpoint_subdir,
                    stage32_inverse_dir_name,
                    stage32_original_images_dir,
                    stage_scene_report_md,
                    stage_next_step_md,
                    run_stage0_button,
                    run_stage1_button,
                    run_stage2_button,
                    run_stage31_button,
                    run_stage32_button,
                ],
            )
            stage_run_name.change(
                fn=_stage_parameter_updates_for_scene,
                inputs=[stage_scene_root_selection, stage_run_name],
                outputs=stage_parameter_outputs,
            )

        with gr.Tab("Inspect Existing Scene"):
            with gr.Row():
                inspect_scene_root_selection = gr.Dropdown(
                    label="Existing Scene Root",
                    choices=scene_choices,
                    value=default_scene,
                    info="Only scene roots with Stage 0 outputs are listed.",
                )
            inspect_report_md = gr.Markdown()

            with gr.Row():
                inspect_scene_root_out = gr.Textbox(label="Resolved Scene Root", interactive=False)
                inspect_latest_run_out = gr.Textbox(label="Latest Run Dir", interactive=False)

            with gr.Row():
                inspect_primary_preview = gr.Video(label="Primary Preview", interactive=False)
                inspect_secondary_preview = gr.Video(label="Secondary Preview", interactive=False)

            inspect_key_files = gr.Files(label="Key Files")
            inspect_notes = gr.Textbox(
                label="Notes",
                lines=2,
                interactive=False,
                value="This tab summarizes existing outputs. Live logs are shown on the pipeline and stage tabs.",
            )

            inspect_scene_root_selection.change(
                fn=_inspect_existing_scene,
                inputs=[inspect_scene_root_selection],
                outputs=[
                    inspect_report_md,
                    inspect_scene_root_out,
                    inspect_latest_run_out,
                    inspect_primary_preview,
                    inspect_secondary_preview,
                    inspect_key_files,
                    inspect_notes,
                ],
            )

            auto_sync_outputs = [
                existing_video_selection,
                existing_scene_root_selection,
                output_parent_selection,
                stage0_existing_video_selection,
                stage0_output_parent_selection,
                stage_scene_root_selection,
                inspect_scene_root_selection,
                stage_scene_report_md,
                stage_scene_root_text,
                stage_run_name,
                stage_selected_run_dir_text,
                stage31_checkpoint_subdir,
                stage32_checkpoint_subdir,
                stage32_inverse_dir_name,
                stage32_original_images_dir,
                stage_next_step_md,
                run_stage0_button,
                run_stage1_button,
                run_stage2_button,
                run_stage31_button,
                run_stage32_button,
                inspect_report_md,
                inspect_scene_root_out,
                inspect_latest_run_out,
                inspect_primary_preview,
                inspect_secondary_preview,
                inspect_key_files,
                inspect_notes,
            ]

            pipeline_sync_inputs = [
                pipeline_scene_root_text,
                existing_video_selection,
                existing_scene_root_selection,
                output_parent_selection,
                stage0_existing_video_selection,
                stage0_output_parent_selection,
                stage_scene_root_selection,
                inspect_scene_root_selection,
            ]
            stage_sync_inputs = [
                stage_run_state,
                existing_video_selection,
                existing_scene_root_selection,
                output_parent_selection,
                stage0_existing_video_selection,
                stage0_output_parent_selection,
                stage_scene_root_selection,
                inspect_scene_root_selection,
            ]
            stage_live_sync_inputs = [
                stage_live_scene_root,
                existing_video_selection,
                existing_scene_root_selection,
                output_parent_selection,
                stage0_existing_video_selection,
                stage0_output_parent_selection,
                stage_scene_root_selection,
                inspect_scene_root_selection,
            ]

            demo.load(
                fn=_sync_catalogs_and_scene_views,
                inputs=pipeline_sync_inputs,
                outputs=auto_sync_outputs,
            )
            pipeline_uploaded_event.then(
                fn=_sync_catalogs_and_scene_views,
                inputs=pipeline_sync_inputs,
                outputs=auto_sync_outputs,
            )
            stage0_uploaded_event.then(
                fn=_sync_catalogs_and_scene_views,
                inputs=stage_live_sync_inputs,
                outputs=auto_sync_outputs,
            ).then(
                fn=_reset_stage_panel_for_new_source,
                inputs=[
                    stage0_source_mode,
                    stage0_uploaded_video_cached,
                    stage0_existing_video_selection,
                    stage0_existing_frames_dir,
                ],
                outputs=stage_reset_outputs,
            )
            pipeline_scene_root_text.change(
                fn=_sync_catalogs_and_scene_views,
                inputs=pipeline_sync_inputs,
                outputs=auto_sync_outputs,
            )
            stage_live_scene_root.change(
                fn=_sync_catalogs_and_scene_views,
                inputs=stage_live_sync_inputs,
                outputs=auto_sync_outputs,
            )
            existing_scene_root_selection.change(
                fn=_sync_catalogs_and_scene_views,
                inputs=[
                    existing_scene_root_selection,
                    existing_video_selection,
                    existing_scene_root_selection,
                    output_parent_selection,
                    stage0_existing_video_selection,
                    stage0_output_parent_selection,
                    stage_scene_root_selection,
                    inspect_scene_root_selection,
                ],
                outputs=auto_sync_outputs,
            )
            stage_scene_root_selection.change(
                fn=_sync_catalogs_and_scene_views,
                inputs=[
                    stage_scene_root_selection,
                    existing_video_selection,
                    existing_scene_root_selection,
                    output_parent_selection,
                    stage0_existing_video_selection,
                    stage0_output_parent_selection,
                    stage_scene_root_selection,
                    inspect_scene_root_selection,
                ],
                outputs=auto_sync_outputs,
            )
            inspect_scene_root_selection.change(
                fn=_sync_catalogs_and_scene_views,
                inputs=[
                    inspect_scene_root_selection,
                    existing_video_selection,
                    existing_scene_root_selection,
                    output_parent_selection,
                    stage0_existing_video_selection,
                    stage0_output_parent_selection,
                    stage_scene_root_selection,
                    inspect_scene_root_selection,
                ],
                outputs=auto_sync_outputs,
            )
            pipeline_run_event.then(
                fn=_sync_catalogs_and_scene_views,
                inputs=pipeline_sync_inputs,
                outputs=auto_sync_outputs,
            )
            stage0_run_event.then(
                fn=_sync_catalogs_after_live_stage_run,
                inputs=stage_live_sync_inputs,
                outputs=auto_sync_outputs,
            )
            stage1_run_event.then(
                fn=_sync_catalogs_after_live_stage_run,
                inputs=stage_live_sync_inputs,
                outputs=auto_sync_outputs,
            )
            stage2_run_event.then(
                fn=_sync_catalogs_after_live_stage_run,
                inputs=stage_live_sync_inputs,
                outputs=auto_sync_outputs,
            )
            stage31_run_event.then(
                fn=_sync_catalogs_after_live_stage_run,
                inputs=stage_live_sync_inputs,
                outputs=auto_sync_outputs,
            )
            stage32_run_event.then(
                fn=_sync_catalogs_after_live_stage_run,
                inputs=stage_live_sync_inputs,
                outputs=auto_sync_outputs,
            )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the Gradio UI for video_to_world.")
    parser.add_argument("--server-name", default="0.0.0.0", help="Bind address for the Gradio server.")
    parser.add_argument("--server-port", type=int, default=7860, help="Port for the Gradio server.")
    parser.add_argument("--share", action="store_true", help="Enable a Gradio share link.")
    parser.add_argument("--inbrowser", action="store_true", help="Open the UI in a browser on launch.")
    args = parser.parse_args()

    _ensure_workspace_dirs()
    app = build_app()
    _, local_url, share_url = app.queue(default_concurrency_limit=2).launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        inbrowser=args.inbrowser,
        quiet=True,
        prevent_thread_lock=True,
    )
    print(f"* Running on local URL:  {local_url}")
    if args.server_name in {"0.0.0.0", "::"}:
        print(
            f"* Bound on all interfaces via {args.server_name}:{args.server_port}. "
            "Open localhost or this machine's LAN IP in a browser, not the bind address."
        )
    if share_url:
        print(f"* Running on public URL: {share_url}")
    else:
        print("* To create a public link, set `share=True` in `launch()`.")
    app.block_thread()


if __name__ == "__main__":
    main()
