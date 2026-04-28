"""
Export Stage 0 DA3 outputs into a per-frame DirectStorage BC7 stream.

Format goals:
    - GPU-native BC7 color blocks on disk without secondary compression
    - exact 16-bit depth on disk
    - per-frame metadata and offsets for simple random access
    - GDeflate-compressed depth payloads for DirectStorage-friendly runtime use
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from export_depth_image_stream import (
    _align_up,
    _build_gdeflate_helper,
    _compress_payload_with_helper,
    _encode_depth_map_u16,
    _find_vs_installation,
    _frame_depth_range_cm,
    _relative_c2w_video_3x4,
    _run_windows_command,
    _to_native_windows_path,
    _write_zero_padding,
)
from export_stage0_kinect_video import _load_stage0_results, _relative_c2w_from_extrinsics


_MAGIC = b"DIVBC7F1"
_VERSION = 1
_PAYLOAD_ALIGNMENT = 4096
_D3D12_TEXTURE_DATA_PITCH_ALIGNMENT = 256
_D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT = 512
_COLOR_FORMAT_BC7_RAW = 2
_DEPTH_FORMAT_G16_GDEFLATE = 5
_DEFAULT_OUTPUT_FILENAME = "depth_image_stream.divstream"

_HEADER_STRUCT = struct.Struct("<8sIIIIIIIIIdQQQ")
_FRAME_STRUCT = struct.Struct("<12f6fQIQI")

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


def default_depth_image_stream_bc7_output_path(scene_root: str) -> str:
    scene_root_path = Path(scene_root).resolve()
    scene_name = _safe_output_component(scene_root_path.name)

    source_name = ""
    meta_path = scene_root_path / "preprocess_frames.json"
    if meta_path.exists():
        try:
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


def _build_bc7_helper() -> Path:
    project_root = Path(__file__).resolve().parent
    output_dir = project_root / ".build" / "divstream_bc7"
    output_dir.mkdir(parents=True, exist_ok=True)
    exe_path = output_dir / "divstream_bc7_cli.exe"

    source_path = project_root / "tools" / "divstream_bc7_cli.cpp"
    ue_root = Path(r"C:\dev\UnrealEngine") if os.name == "nt" else Path("/mnt/c/dev/UnrealEngine")
    ispc_root = ue_root / "Engine" / "Source" / "ThirdParty" / "Intel" / "ISPCTexComp" / "ISPCTextureCompressor-14d998c"
    include_dir = ispc_root / "ispc_texcomp"
    library_path = ispc_root / "ISPC Texture Compressor" / "x64" / "Release" / "ispc_texcomp.lib"
    dll_path = ue_root / "Engine" / "Binaries" / "ThirdParty" / "Intel" / "ISPCTexComp" / "Win64-Release" / "ispc_texcomp.dll"

    for required_path in (source_path, include_dir, library_path, dll_path):
        if not required_path.exists():
            raise FileNotFoundError(f"Required BC7 helper input was not found: {required_path}")

    newest_input_mtime = max(path.stat().st_mtime for path in (source_path, library_path, dll_path))
    if exe_path.exists() and exe_path.stat().st_mtime >= newest_input_mtime:
        staged_dll = output_dir / "ispc_texcomp.dll"
        if not staged_dll.exists() or staged_dll.stat().st_mtime < dll_path.stat().st_mtime:
            shutil.copy2(dll_path, staged_dll)
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
        "/D_CRT_SECURE_NO_WARNINGS",
        f'/I"{_to_native_windows_path(include_dir)}"',
        f'"{_to_native_windows_path(source_path)}"',
        f'"{_to_native_windows_path(library_path)}"',
        f'/Fe:"{_to_native_windows_path(exe_path)}"',
    ]
    command = " ".join(compile_parts)
    build_script = output_dir / "build_divstream_bc7.cmd"
    build_script.write_text(
        f"@echo off\r\ncd /d {_to_native_windows_path(output_dir)}\r\n{command}\r\n",
        encoding="utf-8",
    )
    _run_windows_command(["cmd.exe", "/c", _to_native_windows_path(build_script)], cwd=project_root)
    if not exe_path.exists():
        raise RuntimeError(f"Expected BC7 helper executable was not produced: {exe_path}")
    shutil.copy2(dll_path, output_dir / "ispc_texcomp.dll")
    return exe_path


def _compress_bc7_rgba_with_helper(helper_path: Path, rgba_bytes: bytes, *, width: int, height: int, stride: int) -> bytes:
    with tempfile.TemporaryDirectory(prefix="divstream_bc7_") as temp_dir:
        temp_dir_path = Path(temp_dir)
        input_path = temp_dir_path / "input.rgba"
        output_path = temp_dir_path / "output.bc7"
        input_path.write_bytes(rgba_bytes)

        command = [
            _to_native_windows_path(helper_path),
            "compress",
            _to_native_windows_path(input_path),
            _to_native_windows_path(output_path),
            str(width),
            str(height),
            str(stride),
        ]
        _run_windows_command(command, cwd=helper_path.parent)
        return output_path.read_bytes()


def _pad_color_to_rgba(color_rgb: np.ndarray, *, padded_width: int, padded_height: int) -> np.ndarray:
    height, width = color_rgb.shape[:2]
    rgba = np.empty((padded_height, padded_width, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    rgba[:height, :width, :3] = color_rgb

    if padded_width > width:
        rgba[:height, width:padded_width, :3] = color_rgb[:, width - 1 : width, :3]
    if padded_height > height:
        rgba[height:padded_height, :, :3] = rgba[height - 1 : height, :, :3]

    return rgba


def _bc7_copyable_layout(width: int, height: int) -> tuple[int, int, int]:
    block_width = _align_up(width, 4) // 4
    block_height = _align_up(height, 4) // 4
    row_pitch = _align_up(block_width * 16, _D3D12_TEXTURE_DATA_PITCH_ALIGNMENT)
    total_bytes = _align_up(row_pitch * block_height, _D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT)
    return block_width, block_height, total_bytes


def _pack_bc7_to_pitched_rows(raw_bc7: bytes, *, block_width: int, block_height: int, row_pitch: int, frame_size: int) -> bytes:
    tight_row_pitch = block_width * 16
    expected_size = tight_row_pitch * block_height
    if len(raw_bc7) != expected_size:
        raise ValueError(
            f"Unexpected BC7 size: got {len(raw_bc7)} bytes, expected {expected_size} for "
            f"{block_width}x{block_height} blocks."
        )

    packed = np.zeros(frame_size, dtype=np.uint8)
    packed_rows = packed[: row_pitch * block_height].reshape(block_height, row_pitch)
    tight = np.frombuffer(raw_bc7, dtype=np.uint8).reshape(block_height, tight_row_pitch)
    packed_rows[:, :tight_row_pitch] = tight
    return packed.tobytes()


def _xor_depth_frame_bytes(current_frame_bytes: bytes, previous_frame_bytes: bytes) -> bytes:
    current_words = np.frombuffer(current_frame_bytes, dtype=np.uint16)
    previous_words = np.frombuffer(previous_frame_bytes, dtype=np.uint16)
    if current_words.shape != previous_words.shape:
        raise ValueError("Depth XOR frames must have identical sizes.")
    return np.bitwise_xor(current_words, previous_words).tobytes()


def _pack_tight_g16(depth_u16: np.ndarray) -> bytes:
    contiguous = np.ascontiguousarray(depth_u16, dtype=np.uint16)
    return contiguous.tobytes()


def _row_sub_filter_bytes(rows: np.ndarray) -> bytes:
    filtered = rows.copy()
    filtered[:, 1:] = ((rows[:, 1:].astype(np.int16) - rows[:, :-1].astype(np.int16)) & 0xFF).astype(np.uint8)
    return filtered.tobytes()


def _filter_depth_residual_frame_bytes(depth_frame_bytes: bytes, *, width: int, height: int) -> bytes:
    depth_words = np.frombuffer(depth_frame_bytes, dtype=np.uint16).reshape(height, width)
    low_plane = (depth_words & 0xFF).astype(np.uint8)
    high_plane = (depth_words >> 8).astype(np.uint8)
    return _row_sub_filter_bytes(low_plane) + _row_sub_filter_bytes(high_plane)


def _infer_scene_fps(scene_root: str) -> int:
    scene_root_path = Path(scene_root)
    sequence_info_candidates = (
        scene_root_path / "exports" / "kinect_rgbd_sequence" / "sequence_info.json",
        scene_root_path / "exports" / "kinect_rgbd_sequence_depth8" / "sequence_info.json",
        scene_root_path / "exports" / "kinect_rgbd_video" / "sequence_info.json",
    )

    for info_path in sequence_info_candidates:
        if not info_path.exists():
            continue
        try:
            payload = json.loads(info_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        frame_rate = payload.get("frame_rate")
        if frame_rate is None:
            frame_rate = payload.get("fps")
        if frame_rate is None:
            continue

        inferred_fps = int(round(float(frame_rate)))
        if inferred_fps >= 1:
            return inferred_fps

    return 30


def export_depth_image_stream_bc7(
    *,
    scene_root: str,
    output_path: str | None = None,
    fps: int | None = None,
    compression_level: int = 9,
    overwrite: bool = False,
) -> str:
    if fps is None:
        fps = _infer_scene_fps(scene_root)
    if fps < 1:
        raise ValueError("fps must be at least 1.")
    if compression_level < 1 or compression_level > 12:
        raise ValueError("compression_level must be between 1 and 12.")

    scene_root = os.path.abspath(scene_root)
    if output_path is None or str(output_path).strip() == "":
        output_path = default_depth_image_stream_bc7_output_path(scene_root)
    output_path = os.path.abspath(output_path)
    output_dir = os.path.dirname(output_path)

    if os.path.exists(output_path):
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_path}")
        os.remove(output_path)
    os.makedirs(output_dir, exist_ok=True)

    gdeflate_helper_path = _build_gdeflate_helper()
    bc7_helper_path = _build_bc7_helper()
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

    block_width, block_height, color_frame_size = _bc7_copyable_layout(width, height)
    color_row_pitch = _align_up(block_width * 16, _D3D12_TEXTURE_DATA_PITCH_ALIGNMENT)
    # Keep depth tightly packed on disk. The runtime uses the header row pitch
    # directly during GPU decode.
    depth_row_pitch = width * 2
    depth_frame_size = depth_row_pitch * height

    frame_table_size = _FRAME_STRUCT.size * num_frames
    header_size = _HEADER_STRUCT.size
    frame_table_offset = header_size
    payload_offset = _align_up(frame_table_offset + frame_table_size, _PAYLOAD_ALIGNMENT)

    frame_entries: list[_FrameEntry] = []
    frame_blobs: list[tuple[int, bytes]] = []
    next_offset = payload_offset

    valid_depth = np.isfinite(depth_maps_m) & (depth_maps_m > 0.0)
    if not np.any(valid_depth):
        raise ValueError("No valid positive depths found in results.npz.")
    global_near_cm = float(depth_maps_m[valid_depth].min()) * 100.0
    global_far_cm = float(depth_maps_m[valid_depth].max()) * 100.0
    if global_far_cm <= global_near_cm:
        global_far_cm = global_near_cm + 1.0e-3

    for frame_idx in range(num_frames):
        color_rgb = images[frame_idx]
        if color_rgb.dtype != np.uint8:
            color_rgb = np.clip(color_rgb, 0, 255).astype(np.uint8)
        color_rgba = _pad_color_to_rgba(color_rgb, padded_width=block_width * 4, padded_height=block_height * 4)
        color_raw_bc7 = _compress_bc7_rgba_with_helper(
            bc7_helper_path,
            color_rgba.tobytes(),
            width=block_width * 4,
            height=block_height * 4,
            stride=block_width * 4 * 4,
        )
        color_blob = _pack_bc7_to_pitched_rows(
            color_raw_bc7,
            block_width=block_width,
            block_height=block_height,
            row_pitch=color_row_pitch,
            frame_size=color_frame_size,
        )

        near_cm, far_cm = _frame_depth_range_cm(
            depth_maps_m[frame_idx],
            fallback_near_cm=global_near_cm,
            fallback_far_cm=global_far_cm,
        )
        depth_u16 = _encode_depth_map_u16(depth_maps_m[frame_idx] * 100.0, near_cm=near_cm, far_cm=far_cm)
        depth_blob = _compress_payload_with_helper(
            gdeflate_helper_path,
            _filter_depth_residual_frame_bytes(
                _pack_tight_g16(depth_u16),
                width=width,
                height=height,
            ),
            level=compression_level,
        )

        color_offset = next_offset
        next_offset += len(color_blob)
        next_offset = _align_up(next_offset, _PAYLOAD_ALIGNMENT)

        depth_offset = next_offset
        next_offset += len(depth_blob)
        next_offset = _align_up(next_offset, _PAYLOAD_ALIGNMENT)

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
        frame_blobs.append((color_offset, color_blob))
        frame_blobs.append((depth_offset, depth_blob))

    with open(output_path, "wb") as stream:
        header = _HEADER_STRUCT.pack(
            _MAGIC,
            _VERSION,
            header_size,
            num_frames,
            width,
            height,
            _COLOR_FORMAT_BC7_RAW,
            _DEPTH_FORMAT_G16_GDEFLATE,
            color_row_pitch,
            depth_row_pitch,
            float(fps),
            frame_table_offset,
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

        for blob_offset, blob in frame_blobs:
            current = stream.tell()
            if current > blob_offset:
                raise RuntimeError(f"Payload overlap while writing {output_path}")
            _write_zero_padding(stream, blob_offset - current)
            stream.write(blob)

    return output_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene_root", help="Stage 0 scene root containing exports/npz/results.npz")
    parser.add_argument(
        "--output",
        default="",
        help="Output path for the DIVBC7F1 DirectStorage stream. Defaults under exports/depth_image_stream/.",
    )
    parser.add_argument("--fps", type=int, default=None, help="Playback fps metadata for the stream. Defaults to scene metadata.")
    parser.add_argument(
        "--compression-level",
        type=int,
        default=9,
        help="GDeflate compression level [1..12] for per-frame depth payloads.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output file if present.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    output_path = export_depth_image_stream_bc7(
        scene_root=args.scene_root,
        output_path=args.output or None,
        fps=args.fps,
        compression_level=args.compression_level,
        overwrite=args.overwrite,
    )
    size_bytes = os.path.getsize(output_path)
    print(f"Exported: {output_path}")
    print(f"Size: {size_bytes} bytes ({size_bytes / (1024.0 * 1024.0):.2f} MiB)")


if __name__ == "__main__":
    main()
