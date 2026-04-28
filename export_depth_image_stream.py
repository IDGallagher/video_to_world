"""
Export Stage 0 DA3 outputs into an exact single-file frame stream for Unreal.

The stream is designed for GPU-side playback with DirectStorage + GDeflate:
    - exact BGRA8 color
    - exact G16-style depth codes (0 invalid, [1..65535] valid)
    - per-frame relative camera metadata
    - independent per-frame color/depth chunks for random access

Unlike the packed HAP path, this format does not repack depth into RGB or rely
on video codecs. It behaves like a video at runtime, but the on-disk payload is
structured frame data rather than media samples.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from export_stage0_kinect_video import _load_stage0_results, _relative_c2w_from_extrinsics


_MAGIC = b"DIVSTRM1"
_VERSION = 1
_CHUNK_ALIGNMENT = 4096
_D3D12_TEXTURE_DATA_PITCH_ALIGNMENT = 256

_COLOR_FORMAT_BGRA8 = 1
_DEPTH_FORMAT_G16 = 1

_HEADER_STRUCT = struct.Struct("<8sIIIIIIIIIdQQQ")
_FRAME_STRUCT = struct.Struct("<12f6fQIQI")

_DEPTH_U16_INVALID = np.uint16(0)
_DEPTH_U16_VALID_MIN = np.uint16(1)
_DEPTH_U16_VALID_MAX = np.uint16(65535)
_DEPTH_U16_VALID_RANGE = float(int(_DEPTH_U16_VALID_MAX) - int(_DEPTH_U16_VALID_MIN))
_DIRECTSTORAGE_TRANSLATION_SCALE = 100.0

_DEFAULT_OUTPUT_FILENAME = "depth_image_stream.divstream"
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class _FrameEntry:
    relative_c2w_video: np.ndarray
    near_cm: float
    far_cm: float
    focal_x: float
    focal_y: float
    principal_x: float
    principal_y: float
    color_offset: int
    color_size: int
    depth_offset: int
    depth_size: int


def _safe_output_component(raw: str) -> str:
    safe = _SAFE_NAME_RE.sub("_", str(raw).strip()).strip("._-")
    return safe or "scene"


def default_depth_image_stream_output_path(scene_root: str) -> str:
    scene_root_path = Path(scene_root).resolve()
    scene_name = _safe_output_component(scene_root_path.name)

    source_name = ""
    meta_path = scene_root_path / "preprocess_frames.json"
    if meta_path.exists():
        try:
            import json

            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
        source_input_path = str(meta.get("source_input_path") or "").strip()
        if source_input_path:
            source_path = Path(source_input_path)
            source_name = _safe_output_component(
                source_path.stem if source_path.suffix else source_path.name
            )

    base_name = source_name or scene_name

    return str(scene_root_path / "exports" / "depth_image_stream" / f"{base_name}.divstream")


def _align_up(value: int, alignment: int) -> int:
    return ((int(value) + alignment - 1) // alignment) * alignment


def _find_vs_installation() -> Path:
    if os.name == "nt":
        vswhere = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe")
    else:
        vswhere = Path("/mnt/c/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe")
    if not vswhere.exists():
        raise FileNotFoundError("vswhere.exe was not found. Visual Studio is required to build the GDeflate helper.")
    result = subprocess.run(
        [str(vswhere), "-products", "*", "-latest", "-property", "installationPath"],
        check=True,
        capture_output=True,
        text=True,
    )
    installation = result.stdout.strip()
    if not installation:
        raise FileNotFoundError("Visual Studio installation path could not be resolved.")
    if os.name == "nt":
        installation_path = Path(installation)
        if not installation_path.exists():
            raise FileNotFoundError("Visual Studio installation path could not be resolved.")
        return installation_path

    converted = subprocess.run(
        ["wslpath", installation],
        check=True,
        capture_output=True,
        text=True,
    )
    installation_path = Path(converted.stdout.strip())
    if not installation_path.exists():
        raise FileNotFoundError("Visual Studio installation path could not be resolved.")
    return installation_path


def _to_native_windows_path(path: str | Path) -> str:
    resolved = os.fspath(path)
    if os.name == "nt":
        return str(Path(resolved))
    result = subprocess.run(
        ["wslpath", "-w", resolved],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _run_windows_command(command_parts: list[str], *, cwd: Path) -> None:
    if os.name == "nt":
        subprocess.run(command_parts, check=True, cwd=str(cwd))
        return

    windows_cwd = _to_native_windows_path(cwd)
    if len(command_parts) >= 3 and command_parts[0].lower() == "cmd.exe" and command_parts[1].lower() == "/c":
        inner_command = command_parts[2]
    else:
        inner_command = " ".join(f'"{part}"' if " " in part else part for part in command_parts)
    subprocess.run(
        ["cmd.exe", "/c", f"cd /d {windows_cwd} && {inner_command}"],
        check=True,
    )


def _build_gdeflate_helper() -> Path:
    project_root = Path(__file__).resolve().parent
    output_dir = project_root / ".build" / "divstream_gdeflate"
    output_dir.mkdir(parents=True, exist_ok=True)
    exe_path = output_dir / "divstream_gdeflate_cli.exe"

    source_roots = [
        project_root / "tools" / "divstream_gdeflate_cli.cpp",
        project_root / "third_party" / "gdeflate_ref" / "GDeflate" / "GDeflateCompress.cpp",
        project_root / "third_party" / "gdeflate_ref" / "GDeflate" / "GDeflateDecompress.cpp",
        project_root / "third_party" / "gdeflate_ref" / "libdeflate" / "lib" / "adler32.c",
        project_root / "third_party" / "gdeflate_ref" / "libdeflate" / "lib" / "crc32.c",
        project_root / "third_party" / "gdeflate_ref" / "libdeflate" / "lib" / "deflate_compress.c",
        project_root / "third_party" / "gdeflate_ref" / "libdeflate" / "lib" / "deflate_decompress.c",
        project_root / "third_party" / "gdeflate_ref" / "libdeflate" / "lib" / "gdeflate_compress.c",
        project_root / "third_party" / "gdeflate_ref" / "libdeflate" / "lib" / "gdeflate_decompress.c",
        project_root / "third_party" / "gdeflate_ref" / "libdeflate" / "lib" / "gzip_compress.c",
        project_root / "third_party" / "gdeflate_ref" / "libdeflate" / "lib" / "gzip_decompress.c",
        project_root / "third_party" / "gdeflate_ref" / "libdeflate" / "lib" / "utils.c",
        project_root / "third_party" / "gdeflate_ref" / "libdeflate" / "lib" / "zlib_compress.c",
        project_root / "third_party" / "gdeflate_ref" / "libdeflate" / "lib" / "zlib_decompress.c",
        project_root / "third_party" / "gdeflate_ref" / "libdeflate" / "lib" / "x86" / "cpu_features.c",
    ]
    include_dirs = [
        project_root / "third_party" / "gdeflate_ref" / "GDeflate",
        project_root / "third_party" / "gdeflate_ref" / "libdeflate",
        project_root / "third_party" / "gdeflate_ref" / "libdeflate" / "lib",
    ]

    newest_input_mtime = max(path.stat().st_mtime for path in source_roots)
    if exe_path.exists() and exe_path.stat().st_mtime >= newest_input_mtime:
        return exe_path

    vs_install = _find_vs_installation()
    vcvars = vs_install / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    if not vcvars.exists():
        raise FileNotFoundError(f"vcvars64.bat not found at {vcvars}")

    compile_parts = [
        "call",
        f'"{_to_native_windows_path(vcvars)}"',
        "&&",
        "cl",
        "/nologo",
        "/std:c++17",
        "/O2",
        "/EHsc",
        "/MT",
        "/DNDEBUG",
        "/wd4244",
        "/wd4127",
        "/wd4267",
        "/wd4100",
        "/wd4245",
        "/wd4456",
        "/wd4018",
        "/wd4146",
        "/wd4310",
        "/D_CRT_SECURE_NO_WARNINGS",
    ]
    compile_parts.extend(f'/I"{_to_native_windows_path(path)}"' for path in include_dirs)
    compile_parts.extend(f'"{_to_native_windows_path(path)}"' for path in source_roots)
    compile_parts.extend(
        [
            f'/Fe:"{_to_native_windows_path(exe_path)}"',
        ]
    )

    command = " ".join(compile_parts)
    build_script = output_dir / "build_divstream_gdeflate.cmd"
    build_script.write_text(
        f"@echo off\r\ncd /d {_to_native_windows_path(output_dir)}\r\n{command}\r\n",
        encoding="utf-8",
    )
    _run_windows_command(["cmd.exe", "/c", _to_native_windows_path(build_script)], cwd=project_root)
    if not exe_path.exists():
        raise RuntimeError(f"Expected GDeflate helper executable was not produced: {exe_path}")
    return exe_path


def _compress_payload_with_helper(helper_path: Path, payload: bytes, *, level: int) -> bytes:
    with tempfile.TemporaryDirectory(prefix="divstream_gdeflate_") as temp_dir:
        temp_dir_path = Path(temp_dir)
        input_path = temp_dir_path / "input.bin"
        output_path = temp_dir_path / "output.gdeflate"
        input_path.write_bytes(payload)

        command = [
            _to_native_windows_path(helper_path),
            "compress",
            _to_native_windows_path(input_path),
            _to_native_windows_path(output_path),
            str(level),
        ]
        _run_windows_command(command, cwd=helper_path.parent)
        return output_path.read_bytes()


def _encode_depth_map_u16(depth_map_cm: np.ndarray, *, near_cm: float, far_cm: float) -> np.ndarray:
    if depth_map_cm.ndim != 2:
        raise ValueError(f"depth_map_cm must be 2-D, got shape {depth_map_cm.shape}")
    if not np.isfinite(near_cm) or not np.isfinite(far_cm) or far_cm <= near_cm:
        raise ValueError(f"Invalid depth range: near_cm={near_cm}, far_cm={far_cm}")

    valid_mask = np.isfinite(depth_map_cm) & (depth_map_cm > 0.0)
    encoded = np.zeros(depth_map_cm.shape, dtype=np.uint16)
    if not np.any(valid_mask):
        return encoded

    depth_norm = np.zeros(depth_map_cm.shape, dtype=np.float32)
    depth_norm[valid_mask] = (depth_map_cm[valid_mask] - near_cm) / (far_cm - near_cm)
    depth_norm[valid_mask] = np.clip(depth_norm[valid_mask], 0.0, 1.0)
    encoded_valid = np.rint(depth_norm[valid_mask] * _DEPTH_U16_VALID_RANGE + int(_DEPTH_U16_VALID_MIN))
    encoded[valid_mask] = encoded_valid.clip(
        int(_DEPTH_U16_VALID_MIN),
        int(_DEPTH_U16_VALID_MAX),
    ).astype(np.uint16)
    return encoded


def _copy_into_pitched_bgra(color_rgb: np.ndarray, *, row_pitch: int) -> bytes:
    height, width = color_rgb.shape[:2]
    packed = np.zeros((height, row_pitch), dtype=np.uint8)
    bgra = np.empty((height, width, 4), dtype=np.uint8)
    bgra[..., 0] = color_rgb[..., 2]
    bgra[..., 1] = color_rgb[..., 1]
    bgra[..., 2] = color_rgb[..., 0]
    bgra[..., 3] = 255
    packed[:, : width * 4] = bgra.reshape(height, width * 4)
    return packed.tobytes()


def _copy_into_pitched_g16(depth_u16: np.ndarray, *, row_pitch: int) -> bytes:
    height, width = depth_u16.shape
    packed = np.zeros((height, row_pitch), dtype=np.uint8)
    packed[:, : width * 2] = depth_u16.view(np.uint8).reshape(height, width * 2)
    return packed.tobytes()


def _frame_depth_range_cm(depth_map_m: np.ndarray, *, fallback_near_cm: float, fallback_far_cm: float) -> tuple[float, float]:
    valid = np.isfinite(depth_map_m) & (depth_map_m > 0.0)
    if not np.any(valid):
        return fallback_near_cm, fallback_far_cm

    near_cm = float(depth_map_m[valid].min()) * 100.0
    far_cm = float(depth_map_m[valid].max()) * 100.0
    if far_cm <= near_cm:
        far_cm = near_cm + 1.0e-3
    return near_cm, far_cm


def _write_zero_padding(stream, count: int) -> None:
    if count > 0:
        stream.write(b"\x00" * count)


def _relative_c2w_video_3x4(relative_c2w: np.ndarray) -> np.ndarray:
    if relative_c2w.shape != (4, 4):
        raise ValueError(f"relative_c2w must be 4x4, got {relative_c2w.shape}")
    relative_c2w_video = np.asarray(relative_c2w[:3, :4], dtype=np.float32).copy()
    relative_c2w_video[:3, 3] *= float(_DIRECTSTORAGE_TRANSLATION_SCALE)
    return relative_c2w_video


def export_depth_image_stream(
    *,
    scene_root: str,
    output_path: str | None = None,
    fps: int = 30,
    compression_level: int = 9,
    overwrite: bool = False,
) -> str:
    if fps < 1:
        raise ValueError("fps must be at least 1.")
    if compression_level < 1 or compression_level > 12:
        raise ValueError("compression_level must be between 1 and 12.")

    scene_root = os.path.abspath(scene_root)
    if output_path is None or str(output_path).strip() == "":
        output_path = default_depth_image_stream_output_path(scene_root)
    output_path = os.path.abspath(output_path)
    output_dir = os.path.dirname(output_path)

    if os.path.exists(output_path):
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_path}")
        os.remove(output_path)
    os.makedirs(output_dir, exist_ok=True)

    helper_path = _build_gdeflate_helper()
    images, depth_maps_m, extrinsics, intrinsics = _load_stage0_results(scene_root)
    relative_c2w = _relative_c2w_from_extrinsics(extrinsics)

    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"Expected images to be (N,H,W,3), got {images.shape}")
    if depth_maps_m.ndim != 3:
        raise ValueError(f"Expected depth_maps_m to be (N,H,W), got {depth_maps_m.shape}")

    num_frames, height, width, _ = images.shape
    if num_frames == 0:
        raise ValueError("results.npz contains zero frames.")
    if depth_maps_m.shape[0] != num_frames or intrinsics.shape[0] != num_frames or relative_c2w.shape[0] != num_frames:
        raise ValueError("Stage 0 arrays disagree on frame count.")

    color_row_pitch = _align_up(width * 4, _D3D12_TEXTURE_DATA_PITCH_ALIGNMENT)
    depth_row_pitch = _align_up(width * 2, _D3D12_TEXTURE_DATA_PITCH_ALIGNMENT)
    color_uncompressed_size = color_row_pitch * height
    depth_uncompressed_size = depth_row_pitch * height

    valid_depth = np.isfinite(depth_maps_m) & (depth_maps_m > 0.0)
    if not np.any(valid_depth):
        raise ValueError("No valid positive depths found in results.npz.")
    global_near_cm = float(depth_maps_m[valid_depth].min()) * 100.0
    global_far_cm = float(depth_maps_m[valid_depth].max()) * 100.0
    if global_far_cm <= global_near_cm:
        global_far_cm = global_near_cm + 1.0e-3

    frame_table_size = _FRAME_STRUCT.size * num_frames
    header_size = _HEADER_STRUCT.size
    payload_offset = _align_up(header_size + frame_table_size, _CHUNK_ALIGNMENT)

    frame_entries: list[_FrameEntry] = []
    chunk_blobs: list[tuple[int, bytes]] = []
    next_offset = payload_offset

    for frame_idx in range(num_frames):
        color_rgb = images[frame_idx]
        if color_rgb.dtype != np.uint8:
            color_rgb = np.clip(color_rgb, 0, 255).astype(np.uint8)

        near_cm, far_cm = _frame_depth_range_cm(
            depth_maps_m[frame_idx],
            fallback_near_cm=global_near_cm,
            fallback_far_cm=global_far_cm,
        )
        depth_u16 = _encode_depth_map_u16(depth_maps_m[frame_idx] * 100.0, near_cm=near_cm, far_cm=far_cm)

        color_payload = _copy_into_pitched_bgra(color_rgb, row_pitch=color_row_pitch)
        depth_payload = _copy_into_pitched_g16(depth_u16, row_pitch=depth_row_pitch)

        color_blob = _compress_payload_with_helper(helper_path, color_payload, level=compression_level)
        depth_blob = _compress_payload_with_helper(helper_path, depth_payload, level=compression_level)

        color_offset = next_offset
        next_offset += len(color_blob)
        next_offset = _align_up(next_offset, _CHUNK_ALIGNMENT)

        depth_offset = next_offset
        next_offset += len(depth_blob)
        next_offset = _align_up(next_offset, _CHUNK_ALIGNMENT)

        frame_entries.append(
            _FrameEntry(
                relative_c2w_video=_relative_c2w_video_3x4(relative_c2w[frame_idx]),
                near_cm=float(near_cm),
                far_cm=float(far_cm),
                focal_x=float(intrinsics[frame_idx, 0, 0]),
                focal_y=float(intrinsics[frame_idx, 1, 1]),
                principal_x=float(intrinsics[frame_idx, 0, 2]),
                principal_y=float(intrinsics[frame_idx, 1, 2]),
                color_offset=color_offset,
                color_size=len(color_blob),
                depth_offset=depth_offset,
                depth_size=len(depth_blob),
            )
        )
        chunk_blobs.append((color_offset, color_blob))
        chunk_blobs.append((depth_offset, depth_blob))

    with open(output_path, "wb") as stream:
        header = _HEADER_STRUCT.pack(
            _MAGIC,
            _VERSION,
            header_size,
            num_frames,
            width,
            height,
            _COLOR_FORMAT_BGRA8,
            _DEPTH_FORMAT_G16,
            color_row_pitch,
            depth_row_pitch,
            float(fps),
            header_size,
            frame_table_size,
            payload_offset,
        )
        stream.write(header)

        for frame in frame_entries:
            stream.write(
                _FRAME_STRUCT.pack(
                    *frame.relative_c2w_video.reshape(-1).tolist(),
                    frame.near_cm,
                    frame.far_cm,
                    frame.focal_x,
                    frame.focal_y,
                    frame.principal_x,
                    frame.principal_y,
                    frame.color_offset,
                    frame.color_size,
                    frame.depth_offset,
                    frame.depth_size,
                )
            )

        current = stream.tell()
        _write_zero_padding(stream, payload_offset - current)

        for chunk_offset, blob in chunk_blobs:
            current = stream.tell()
            if current > chunk_offset:
                raise RuntimeError(f"Chunk overlap while writing {output_path}")
            _write_zero_padding(stream, chunk_offset - current)
            stream.write(blob)

    return output_path
