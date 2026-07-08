"""Render VACE conditioning frames directly from a textured GLB.

The output is intentionally just guide/mask/depth frames and preview videos.
It does not call VACE. This renderer exists for GLB camera-motion checks where
the depth-volume cache is not the desired source of truth.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


COMPONENT_DTYPES = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}

TYPE_DIMS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _read_glb(path: Path) -> tuple[dict, bytes]:
    data = path.read_bytes()
    if data[:4] != b"glTF":
        raise ValueError(f"Not a binary GLB file: {path}")
    version, total_len = struct.unpack_from("<II", data, 4)
    if version != 2:
        raise ValueError(f"Only GLB v2 is supported, got v{version}: {path}")
    if total_len != len(data):
        raise ValueError(f"GLB length mismatch for {path}")

    offset = 12
    json_chunk: bytes | None = None
    bin_chunk: bytes | None = None
    while offset < len(data):
        chunk_len, chunk_type = struct.unpack_from("<I4s", data, offset)
        offset += 8
        chunk = data[offset : offset + chunk_len]
        offset += chunk_len
        if chunk_type == b"JSON":
            json_chunk = chunk.rstrip(b" \t\r\n\x00")
        elif chunk_type == b"BIN\x00":
            bin_chunk = chunk

    if json_chunk is None or bin_chunk is None:
        raise ValueError(f"GLB needs JSON and BIN chunks: {path}")
    return json.loads(json_chunk.decode("utf-8")), bin_chunk


def _accessor_array(gltf: dict, bin_chunk: bytes, accessor_index: int) -> np.ndarray:
    accessor = gltf["accessors"][accessor_index]
    view = gltf["bufferViews"][accessor["bufferView"]]
    dtype = np.dtype(COMPONENT_DTYPES[accessor["componentType"]])
    dims = TYPE_DIMS[accessor["type"]]
    count = int(accessor["count"])
    base_offset = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    stride = int(view.get("byteStride", dims * dtype.itemsize))

    if stride == dims * dtype.itemsize:
        arr = np.frombuffer(bin_chunk, dtype=dtype, count=count * dims, offset=base_offset)
        arr = arr.reshape((count, dims)) if dims > 1 else arr.reshape((count,))
    else:
        arr = np.ndarray(
            shape=(count, dims),
            dtype=dtype,
            buffer=bin_chunk,
            offset=base_offset,
            strides=(stride, dtype.itemsize),
        ).copy()

    if accessor.get("normalized"):
        if np.issubdtype(dtype, np.unsignedinteger):
            arr = arr.astype(np.float32) / float(np.iinfo(dtype).max)
        else:
            info = np.iinfo(dtype)
            arr = np.maximum(arr.astype(np.float32) / float(info.max), -1.0)
    return arr


def _extract_mesh(gltf: dict, bin_chunk: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mesh = gltf["meshes"][0]
    primitive = mesh["primitives"][0]
    attrs = primitive["attributes"]
    positions = _accessor_array(gltf, bin_chunk, attrs["POSITION"]).astype(np.float32, copy=False)
    uvs = _accessor_array(gltf, bin_chunk, attrs["TEXCOORD_0"]).astype(np.float32, copy=False)
    indices = _accessor_array(gltf, bin_chunk, primitive["indices"]).astype(np.int64, copy=False)
    faces = indices.reshape((-1, 3))

    material = gltf["materials"][primitive.get("material", 0)]
    pbr = material.get("pbrMetallicRoughness", {})
    texture_index = int(pbr.get("baseColorTexture", {}).get("index", 0))
    image_index = int(gltf["textures"][texture_index].get("source", 0))
    image = gltf["images"][image_index]
    image_view = gltf["bufferViews"][image["bufferView"]]
    start = int(image_view.get("byteOffset", 0))
    end = start + int(image_view["byteLength"])
    texture = np.asarray(Image.open(BytesIO(bin_chunk[start:end])).convert("RGB"), dtype=np.uint8)
    return positions, uvs, faces, texture


def _sample_texture(texture_rgb: np.ndarray, uv: np.ndarray, *, flip_v: bool) -> np.ndarray:
    h, w = texture_rgb.shape[:2]
    u = np.clip(uv[:, 0], 0.0, 1.0)
    v = np.clip(uv[:, 1], 0.0, 1.0)
    if flip_v:
        v = 1.0 - v
    x = np.rint(u * float(w - 1)).astype(np.int64)
    y = np.rint(v * float(h - 1)).astype(np.int64)
    return texture_rgb[y, x]


def _build_point_cache(
    positions: np.ndarray,
    uvs: np.ndarray,
    faces: np.ndarray,
    texture_rgb: np.ndarray,
    *,
    face_samples: int,
    uv_flip_v: bool,
) -> tuple[np.ndarray, np.ndarray]:
    point_chunks = [positions]
    uv_chunks = [uvs]
    tri_pos = positions[faces]
    tri_uv = uvs[faces]

    if face_samples >= 1:
        point_chunks.append(tri_pos.mean(axis=1))
        uv_chunks.append(tri_uv.mean(axis=1))
    if face_samples >= 4:
        bary = np.asarray(
            [
                [0.60, 0.20, 0.20],
                [0.20, 0.60, 0.20],
                [0.20, 0.20, 0.60],
            ],
            dtype=np.float32,
        )
        for weights in bary:
            point_chunks.append((tri_pos * weights.reshape(1, 3, 1)).sum(axis=1))
            uv_chunks.append((tri_uv * weights.reshape(1, 3, 1)).sum(axis=1))

    points = np.concatenate(point_chunks, axis=0).astype(np.float32, copy=False)
    point_uvs = np.concatenate(uv_chunks, axis=0).astype(np.float32, copy=False)
    colors = _sample_texture(texture_rgb, point_uvs, flip_v=uv_flip_v)
    return points, colors


def _normalize(vec: np.ndarray) -> np.ndarray:
    return vec / max(float(np.linalg.norm(vec)), 1.0e-12)


def _look_at_axes(
    center: np.ndarray,
    target: np.ndarray,
    *,
    convention: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = _normalize(target - center)
    up = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(forward, up))) > 0.95:
        up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)

    if convention == "opencv":
        right = _normalize(np.cross(forward, up))
        down_or_up = _normalize(np.cross(forward, right))
    elif convention == "camera_y_up":
        right = _normalize(np.cross(up, forward))
        down_or_up = _normalize(np.cross(forward, right))
    else:
        raise ValueError(f"Unsupported look_at convention: {convention}")
    return right, down_or_up, forward


def _camera_frames(
    *,
    camera_angle_x: float,
    distance: float,
    width: int,
    height: int,
    num_frames: int,
    truck_right: float,
    target_track: float,
    origin_z_sign: float,
    convention: str,
    source_distance_scale: float,
    target_down: float,
    square_pixels: bool,
    principal_x_offset_px: float,
    principal_y_offset_px: float,
) -> tuple[list[dict], dict]:
    effective_distance = float(distance) * float(source_distance_scale)
    origin0 = np.asarray([0.0, 0.0, float(origin_z_sign) * effective_distance], dtype=np.float64)
    target_base = np.zeros(3, dtype=np.float64)
    right_base, axis1_base, _ = _look_at_axes(origin0, target_base, convention=convention)
    target0 = target_base + axis1_base * float(target_down)
    right0, axis1_0, forward0 = _look_at_axes(origin0, target0, convention=convention)

    f = 1.0 / (2.0 * math.tan(float(camera_angle_x) * 0.5))
    fx = f * float(width)
    fy = fx if square_pixels else f * float(height)
    cx = float(width) * 0.5 + float(principal_x_offset_px)
    cy = float(height) * 0.5 + float(principal_y_offset_px)

    frames: list[dict] = []
    denom = max(1, int(num_frames) - 1)
    for idx in range(int(num_frames)):
        t = float(idx) / float(denom)
        ease = t * t * (3.0 - 2.0 * t)
        offset = right0 * float(truck_right) * ease
        center = origin0 + offset
        target = target0 + offset * float(target_track)
        right, axis1, forward = _look_at_axes(center, target, convention=convention)
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, 0] = right
        c2w[:3, 1] = axis1
        c2w[:3, 2] = forward
        c2w[:3, 3] = center
        frames.append(
            {
                "frame_index": idx,
                "transform_matrix": c2w.tolist(),
                "fl_x": fx,
                "fl_y": fy,
                "cx": cx,
                "cy": cy,
                "camera_center": center.tolist(),
                "look_target": target.tolist(),
                "path_t": t,
                "path_ease": ease,
            }
        )

    meta = {
        "origin0": origin0.tolist(),
        "target0": target0.tolist(),
        "source_right_axis": right0.tolist(),
        "source_axis1": axis1_0.tolist(),
        "source_forward_axis": forward0.tolist(),
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "camera_angle_x": float(camera_angle_x),
        "distance": float(distance),
        "effective_distance": float(effective_distance),
        "source_distance_scale": float(source_distance_scale),
        "target_down": float(target_down),
        "square_pixels": bool(square_pixels),
        "principal_x_offset_px": float(principal_x_offset_px),
        "principal_y_offset_px": float(principal_y_offset_px),
    }
    return frames, meta


def _render_points(
    points_world: np.ndarray,
    colors_rgb: np.ndarray,
    frame: dict,
    *,
    width: int,
    height: int,
    point_radius: int,
    background_rgb: tuple[int, int, int],
    convention: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c2w = np.asarray(frame["transform_matrix"], dtype=np.float64)
    center = c2w[:3, 3]
    axes = c2w[:3, :3]
    cam = (points_world.astype(np.float64) - center) @ axes
    z = cam[:, 2]
    valid = np.isfinite(cam).all(axis=1) & (z > 1.0e-5)
    image = np.full((height, width, 3), background_rgb, dtype=np.uint8)
    depth = np.zeros((height, width), dtype=np.float32)
    known = np.zeros((height, width), dtype=bool)
    if not np.any(valid):
        return image, depth, known

    cam = cam[valid]
    z = z[valid].astype(np.float32, copy=False)
    col = colors_rgb[valid]
    fx = float(frame["fl_x"])
    fy = float(frame["fl_y"])
    cx = float(frame["cx"])
    cy = float(frame["cy"])
    u = np.rint(fx * (cam[:, 0] / z) + cx).astype(np.int64)
    if convention == "opencv":
        v = np.rint(fy * (cam[:, 1] / z) + cy).astype(np.int64)
    else:
        v = np.rint(cy - fy * (cam[:, 1] / z)).astype(np.int64)

    zbuf = np.full(width * height, np.inf, dtype=np.float32)
    offsets = [(0, 0)]
    radius = max(0, int(point_radius))
    if radius > 0:
        offsets = [
            (dx, dy)
            for dy in range(-radius, radius + 1)
            for dx in range(-radius, radius + 1)
            if dx * dx + dy * dy <= radius * radius
        ]

    for dx, dy in offsets:
        uu = u + dx
        vv = v + dy
        inside = (uu >= 0) & (uu < width) & (vv >= 0) & (vv < height)
        if np.any(inside):
            linear = (vv[inside] * width + uu[inside]).astype(np.int64, copy=False)
            np.minimum.at(zbuf, linear, z[inside])

    flat_img = image.reshape((-1, 3))
    flat_depth = depth.reshape((-1,))
    flat_known = known.reshape((-1,))
    for dx, dy in offsets:
        uu = u + dx
        vv = v + dy
        inside = (uu >= 0) & (uu < width) & (vv >= 0) & (vv < height)
        if not np.any(inside):
            continue
        linear = (vv[inside] * width + uu[inside]).astype(np.int64, copy=False)
        zz = z[inside]
        chosen = zz <= (zbuf[linear] + 1.0e-6)
        if np.any(chosen):
            target = linear[chosen]
            flat_img[target] = col[inside][chosen]
            flat_depth[target] = zz[chosen]
            flat_known[target] = True

    return image, depth, known


def _mask_from_known(known: np.ndarray, *, feather_px: int) -> np.ndarray:
    missing = (~known).astype(np.uint8)
    if feather_px <= 0:
        return (missing * 255).astype(np.uint8)
    dist = cv2.distanceTransform(missing, cv2.DIST_L2, 3)
    alpha = np.clip(dist / float(feather_px), 0.0, 1.0)
    alpha[known] = 0.0
    return np.rint(alpha * 255.0).astype(np.uint8)


def _encode_depth_u16(depth: np.ndarray, *, near: float, far: float) -> np.ndarray:
    encoded = np.zeros(depth.shape, dtype=np.uint16)
    valid = np.isfinite(depth) & (depth > 0.0)
    if not np.any(valid):
        return encoded
    norm = (depth[valid] - near) / max(far - near, 1.0e-6)
    encoded[valid] = np.rint(np.clip(norm, 0.0, 1.0) * 65534.0 + 1.0).astype(np.uint16)
    return encoded


def _open_video(path: Path, *, width: int, height: int, fps: float, is_color: bool) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height), isColor=is_color)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {path}")
    return writer


def _write_contact_sheet(out_path: Path, images: list[np.ndarray], labels: list[str]) -> None:
    if not images:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    labeled = []
    for image, label in zip(images, labels):
        rgb = image.copy()
        panel = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.rectangle(panel, (0, 0), (min(panel.shape[1], 360), 28), (0, 0, 0), -1)
        cv2.putText(panel, label, (8, 20), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        labeled.append(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
    sheet = np.concatenate(labeled, axis=1)
    Image.fromarray(sheet).save(out_path)


def _write_source_alignment(
    out_dir: Path,
    source_image: Path,
    frame0_rgb: np.ndarray,
    *,
    width: int,
    height: int,
) -> dict:
    src = Image.open(source_image).convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
    src_rgb = np.asarray(src, dtype=np.uint8)
    diff = np.abs(src_rgb.astype(np.int16) - frame0_rgb.astype(np.int16)).astype(np.uint8)
    Image.fromarray(src_rgb).save(out_dir / "source_resized.png")
    Image.fromarray(diff).save(out_dir / "frame0_absdiff.png")
    _write_contact_sheet(
        out_dir / "frame0_alignment_sheet.png",
        [src_rgb, frame0_rgb, diff],
        ["source resized", "glb frame 0", "absolute diff"],
    )
    return {
        "source_image": str(source_image),
        "mean_abs_diff": float(diff.mean()),
        "median_abs_diff": float(np.median(diff)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Render GLB camera-motion conditioning frames for VACE inspection.")
    ap.add_argument("--glb", default=r"D:\archive\DeepRock1\DeepRock1.glb")
    ap.add_argument("--camera_json", default="")
    ap.add_argument("--source_image", default=r"D:\archive\DeepRock1\ian_101_dark_granite_rocks_at_the_bottom_of_the_ocean._No_wat_47f6e59d-04f6-4fc4-9783-163353d64c86_3.png")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--height", type=int, default=464)
    ap.add_argument("--frames", type=int, default=49)
    ap.add_argument("--fps", type=float, default=16.0)
    ap.add_argument("--truck_right", type=float, default=0.08)
    ap.add_argument("--target_track", type=float, default=1.0)
    ap.add_argument("--origin_z_sign", type=float, default=-1.0)
    ap.add_argument("--look_at_convention", choices=["opencv", "camera_y_up"], default="opencv")
    ap.add_argument("--source_distance_scale", type=float, default=1.0)
    ap.add_argument("--target_down", type=float, default=0.0)
    ap.add_argument("--square_pixels", action="store_true")
    ap.add_argument("--principal_x_offset_px", type=float, default=0.0)
    ap.add_argument("--principal_y_offset_px", type=float, default=0.0)
    ap.add_argument("--face_samples", type=int, default=1)
    ap.add_argument("--point_radius", type=int, default=1)
    ap.add_argument("--mask_feather_px", type=int, default=0)
    ap.add_argument("--background_rgb", default="127,127,127")
    ap.add_argument("--uv_flip_v", action="store_true")
    args = ap.parse_args()

    glb_path = Path(args.glb)
    camera_json = Path(args.camera_json) if args.camera_json else glb_path.with_suffix(".pixal3d_camera.json")
    source_image = Path(args.source_image)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    guide_dir = out_dir / "guide_frames"
    mask_dir = out_dir / "mask_frames"
    depth_dir = out_dir / "known_depth_png"
    for path in (guide_dir, mask_dir, depth_dir):
        path.mkdir(parents=True, exist_ok=True)

    background_rgb = tuple(int(x) for x in str(args.background_rgb).split(","))
    if len(background_rgb) != 3:
        raise ValueError("--background_rgb must be R,G,B")

    camera = _load_json(camera_json)
    gltf, bin_chunk = _read_glb(glb_path)
    positions, uvs, faces, texture_rgb = _extract_mesh(gltf, bin_chunk)
    print(f"mesh: vertices={positions.shape[0]} faces={faces.shape[0]} texture={texture_rgb.shape[1]}x{texture_rgb.shape[0]}")
    points, colors_rgb = _build_point_cache(
        positions,
        uvs,
        faces,
        texture_rgb,
        face_samples=int(args.face_samples),
        uv_flip_v=bool(args.uv_flip_v),
    )
    print(f"point cache: {points.shape[0]} points")

    frames, camera_meta = _camera_frames(
        camera_angle_x=float(camera["camera_angle_x"]),
        distance=float(camera["distance"]),
        width=int(args.width),
        height=int(args.height),
        num_frames=int(args.frames),
        truck_right=float(args.truck_right),
        target_track=float(args.target_track),
        origin_z_sign=float(args.origin_z_sign),
        convention=str(args.look_at_convention),
        source_distance_scale=float(args.source_distance_scale),
        target_down=float(args.target_down),
        square_pixels=bool(args.square_pixels),
        principal_x_offset_px=float(args.principal_x_offset_px),
        principal_y_offset_px=float(args.principal_y_offset_px),
    )

    guide_video = _open_video(out_dir / "video_guide.mp4", width=int(args.width), height=int(args.height), fps=float(args.fps), is_color=True)
    mask_video = _open_video(out_dir / "video_mask.mp4", width=int(args.width), height=int(args.height), fps=float(args.fps), is_color=False)

    preview_frames: list[np.ndarray] = []
    preview_labels: list[str] = []
    known_ratios: list[float] = []
    alignment: dict | None = None
    for idx, frame in enumerate(frames):
        image_rgb, depth, known = _render_points(
            points,
            colors_rgb,
            frame,
            width=int(args.width),
            height=int(args.height),
            point_radius=int(args.point_radius),
            background_rgb=background_rgb,
            convention=str(args.look_at_convention),
        )
        mask = _mask_from_known(known, feather_px=int(args.mask_feather_px))
        valid_depth = depth[depth > 0.0]
        if valid_depth.size:
            near = float(valid_depth.min())
            far = float(valid_depth.max())
            if far <= near:
                far = near + 1.0e-3
        else:
            near = 0.001
            far = 1.0

        frame["near"] = near
        frame["far"] = far
        frame["color_path"] = f"guide_frames/{idx:06d}.png"
        frame["mask_path"] = f"mask_frames/{idx:06d}.png"
        frame["depth_path"] = f"known_depth_png/depth_{idx:04d}.png"
        frame["known_ratio"] = float(known.mean())

        Image.fromarray(image_rgb).save(guide_dir / f"{idx:06d}.png")
        Image.fromarray(mask).save(mask_dir / f"{idx:06d}.png")
        Image.fromarray(_encode_depth_u16(depth, near=near, far=far)).save(depth_dir / f"depth_{idx:04d}.png")
        guide_video.write(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        mask_video.write(mask)
        known_ratios.append(float(known.mean()))

        if idx == 0 and source_image.exists():
            alignment = _write_source_alignment(out_dir, source_image, image_rgb, width=int(args.width), height=int(args.height))
        if idx in {0, max(0, int(args.frames) // 2), int(args.frames) - 1}:
            preview_frames.append(image_rgb)
            preview_labels.append(f"guide {idx:04d} known={known.mean():.2f}")
        print(f"rendered {idx:04d}: known_ratio={known.mean():.3f} near={near:.4f} far={far:.4f}")

    guide_video.release()
    mask_video.release()

    _write_contact_sheet(out_dir / "preview_contact_sheet.png", preview_frames, preview_labels)
    manifest = {
        "glb": str(glb_path),
        "camera_json": str(camera_json),
        "source_image": str(source_image),
        "output": {
            "width": int(args.width),
            "height": int(args.height),
            "frames": int(args.frames),
            "fps": float(args.fps),
            "guide_frames": "guide_frames",
            "mask_frames": "mask_frames",
            "known_depth_png": "known_depth_png",
            "video_guide": "video_guide.mp4",
            "video_mask": "video_mask.mp4",
            "preview_contact_sheet": "preview_contact_sheet.png",
        },
        "path": {
            "truck_right": float(args.truck_right),
            "target_track": float(args.target_track),
            "origin_z_sign": float(args.origin_z_sign),
            "look_at_convention": str(args.look_at_convention),
            "source_distance_scale": float(args.source_distance_scale),
            "target_down": float(args.target_down),
            "square_pixels": bool(args.square_pixels),
            "principal_x_offset_px": float(args.principal_x_offset_px),
            "principal_y_offset_px": float(args.principal_y_offset_px),
        },
        "render": {
            "face_samples": int(args.face_samples),
            "point_radius": int(args.point_radius),
            "mask_feather_px": int(args.mask_feather_px),
            "background_rgb": list(background_rgb),
            "uv_flip_v": bool(args.uv_flip_v),
            "known_ratio_min": float(min(known_ratios)) if known_ratios else 0.0,
            "known_ratio_max": float(max(known_ratios)) if known_ratios else 0.0,
        },
        "camera": camera_meta,
        "source_alignment": alignment,
        "frames": frames,
    }
    _write_json(out_dir / "conditioning_manifest.json", manifest)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
