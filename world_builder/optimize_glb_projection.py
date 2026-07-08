"""Optimize a single GLB projection against the original source image.

This is a local pose search around the current hand-fit GLB camera. It scores
only rendered GLB pixels so missing reconstruction/background does not dominate
the objective.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

import render_glb_conditioning as renderer


PARAM_NAMES = [
    "move_right",
    "move_down",
    "move_forward",
    "yaw_right_deg",
    "pitch_down_deg",
    "roll_clockwise_deg",
]


def _write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _rot_x(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, s],
            [0.0, -s, c],
        ],
        dtype=np.float64,
    )


def _rot_y(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.asarray(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float64,
    )


def _rot_z(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.asarray(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _frame_from_offsets(base_frame: dict, params: np.ndarray) -> dict:
    move_right, move_down, move_forward, yaw_deg, pitch_deg, roll_deg = params.astype(np.float64)
    base_c2w = np.asarray(base_frame["transform_matrix"], dtype=np.float64)
    base_axes = base_c2w[:3, :3]
    base_center = base_c2w[:3, 3]
    translation = base_axes @ np.asarray([move_right, move_down, move_forward], dtype=np.float64)

    yaw = math.radians(float(yaw_deg))
    pitch = math.radians(float(pitch_deg))
    roll = math.radians(float(roll_deg))
    local_rotation = _rot_y(yaw) @ _rot_x(pitch) @ _rot_z(roll)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = base_axes @ local_rotation
    c2w[:3, 3] = base_center + translation

    frame = dict(base_frame)
    frame["transform_matrix"] = c2w.tolist()
    frame["camera_center"] = c2w[:3, 3].tolist()
    frame["pose_offsets"] = {name: float(value) for name, value in zip(PARAM_NAMES, params)}
    return frame


def _bbox_stats(known: np.ndarray) -> dict:
    y, x = np.nonzero(known)
    if x.size == 0:
        h, w = known.shape
        return {
            "bbox": [0, 0, -1, -1],
            "margins": [w, h, w, h],
            "known_ratio": 0.0,
        }
    h, w = known.shape
    x0 = int(x.min())
    y0 = int(y.min())
    x1 = int(x.max())
    y1 = int(y.max())
    return {
        "bbox": [x0, y0, x1, y1],
        "margins": [x0, y0, w - 1 - x1, h - 1 - y1],
        "known_ratio": float(known.mean()),
    }


def _score_projection(image_rgb: np.ndarray, known: np.ndarray, target_rgb: np.ndarray) -> tuple[float, dict]:
    if float(known.mean()) < 0.20:
        return 1.0e6, {"known_ratio": float(known.mean()), "reason": "too few known pixels"}

    render_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    diff = np.abs(render_lab - target_lab)
    weights = np.asarray([0.65, 0.18, 0.17], dtype=np.float32)
    color_error = float(((diff * weights.reshape(1, 1, 3)).sum(axis=2)[known]).mean() / 255.0)

    render_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    target_gray = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    rgx = cv2.Sobel(render_gray, cv2.CV_32F, 1, 0, ksize=3)
    rgy = cv2.Sobel(render_gray, cv2.CV_32F, 0, 1, ksize=3)
    tgx = cv2.Sobel(target_gray, cv2.CV_32F, 1, 0, ksize=3)
    tgy = cv2.Sobel(target_gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_error = float((np.abs(rgx - tgx)[known] + np.abs(rgy - tgy)[known]).mean() * 0.5)

    stats = _bbox_stats(known)
    h, w = known.shape
    left, top, right, bottom = stats["margins"]
    margin_penalty = (
        (left / w) ** 2
        + (right / w) ** 2
        + (bottom / h) ** 2
        + max(0.0, (top - 10) / h) ** 2
    )
    coverage_penalty = max(0.0, 0.55 - float(known.mean())) ** 2
    score = color_error + 0.18 * grad_error + 2.8 * margin_penalty + 0.8 * coverage_penalty
    stats.update(
        {
            "score": float(score),
            "color_error": color_error,
            "grad_error": grad_error,
            "margin_penalty": float(margin_penalty),
            "coverage_penalty": float(coverage_penalty),
        }
    )
    return float(score), stats


def _render_eval(
    points: np.ndarray,
    colors: np.ndarray,
    base_frame: dict,
    params: np.ndarray,
    target_rgb: np.ndarray,
    *,
    width: int,
    height: int,
    point_radius: int,
) -> tuple[float, dict, np.ndarray, np.ndarray]:
    frame = _frame_from_offsets(base_frame, params)
    image, _, known = renderer._render_points(
        points,
        colors,
        frame,
        width=width,
        height=height,
        point_radius=point_radius,
        background_rgb=(127, 127, 127),
        convention="opencv",
    )
    score, stats = _score_projection(image, known, target_rgb)
    stats["params"] = {name: float(value) for name, value in zip(PARAM_NAMES, params)}
    return score, stats, image, known


def _source_rgb(path: Path, *, width: int, height: int) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB").resize((width, height), Image.Resampling.LANCZOS), dtype=np.uint8)


def _contact_sheet(path: Path, panels: list[tuple[str, np.ndarray]]) -> None:
    out = []
    for label, rgb in panels:
        panel = rgb.copy()
        bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
        cv2.rectangle(bgr, (0, 0), (min(420, bgr.shape[1]), 28), (0, 0, 0), -1)
        cv2.putText(bgr, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    Image.fromarray(np.concatenate(out, axis=1)).save(path)


def _save_projection(
    out_dir: Path,
    source_image: Path,
    points: np.ndarray,
    colors: np.ndarray,
    base_frame: dict,
    params: np.ndarray,
    stats: dict,
    *,
    width: int,
    height: int,
    point_radius: int,
    manifest_extra: dict,
) -> None:
    guide_dir = out_dir / "guide_frames"
    mask_dir = out_dir / "mask_frames"
    guide_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    target_rgb = _source_rgb(source_image, width=width, height=height)
    frame = _frame_from_offsets(base_frame, params)
    image, depth, known = renderer._render_points(
        points,
        colors,
        frame,
        width=width,
        height=height,
        point_radius=point_radius,
        background_rgb=(127, 127, 127),
        convention="opencv",
    )
    mask = renderer._mask_from_known(known, feather_px=0)
    Image.fromarray(image).save(guide_dir / "000000.png")
    Image.fromarray(mask).save(mask_dir / "000000.png")
    Image.fromarray(target_rgb).save(out_dir / "source_resized.png")
    diff = np.abs(target_rgb.astype(np.int16) - image.astype(np.int16)).astype(np.uint8)
    Image.fromarray(diff).save(out_dir / "frame0_absdiff.png")
    _contact_sheet(
        out_dir / "frame0_alignment_sheet.png",
        [
            ("source resized", target_rgb),
            ("optimized glb projection", image),
            ("absolute diff", diff),
        ],
    )
    _contact_sheet(
        out_dir / "mask_sheet.png",
        [
            ("mask white=inpaint", np.repeat(mask[:, :, None], 3, axis=2)),
        ],
    )
    final_score, final_stats = _score_projection(image, known, target_rgb)
    final_stats["score"] = final_score
    final_stats["params"] = {name: float(value) for name, value in zip(PARAM_NAMES, params)}
    final_stats["input_stats"] = stats
    valid_depth = depth[depth > 0.0]
    frame["near"] = float(valid_depth.min()) if valid_depth.size else 0.001
    frame["far"] = float(valid_depth.max()) if valid_depth.size else 1.0
    _write_json(
        out_dir / "optimization_manifest.json",
        {
            "frame": frame,
            "final_stats": final_stats,
            **manifest_extra,
        },
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Optimize a GLB projection around the current fitted camera.")
    ap.add_argument("--glb", default=r"D:\archive\DeepRock1\DeepRock1.glb")
    ap.add_argument("--camera_json", default=r"D:\archive\DeepRock1\DeepRock1.pixal3d_camera.json")
    ap.add_argument("--source_image", default=r"D:\archive\DeepRock1\ian_101_dark_granite_rocks_at_the_bottom_of_the_ocean._No_wat_47f6e59d-04f6-4fc4-9783-163353d64c86_3.png")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--opt_width", type=int, default=416)
    ap.add_argument("--opt_height", type=int, default=232)
    ap.add_argument("--final_width", type=int, default=832)
    ap.add_argument("--final_height", type=int, default=464)
    ap.add_argument("--base_distance_scale", type=float, default=0.95)
    ap.add_argument("--iterations", type=int, default=220)
    ap.add_argument("--point_stride", type=int, default=2)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    camera = renderer._load_json(Path(args.camera_json))
    gltf, bin_chunk = renderer._read_glb(Path(args.glb))
    positions, uvs, faces, texture = renderer._extract_mesh(gltf, bin_chunk)
    points, colors = renderer._build_point_cache(positions, uvs, faces, texture, face_samples=1, uv_flip_v=False)
    if int(args.point_stride) > 1:
        points_opt = points[:: int(args.point_stride)]
        colors_opt = colors[:: int(args.point_stride)]
    else:
        points_opt = points
        colors_opt = colors

    base_frames_opt, base_meta_opt = renderer._camera_frames(
        camera_angle_x=float(camera["camera_angle_x"]),
        distance=float(camera["distance"]),
        width=int(args.opt_width),
        height=int(args.opt_height),
        num_frames=1,
        truck_right=0.0,
        target_track=1.0,
        origin_z_sign=-1.0,
        convention="opencv",
        source_distance_scale=float(args.base_distance_scale),
        target_down=0.0,
        square_pixels=True,
        principal_x_offset_px=0.0,
        principal_y_offset_px=0.0,
    )
    target_opt = _source_rgb(Path(args.source_image), width=int(args.opt_width), height=int(args.opt_height))

    rng = np.random.default_rng(int(args.seed))
    bounds = np.asarray(
        [
            [-0.045, 0.045],
            [-0.035, 0.035],
            [-0.040, 0.030],
            [-8.0, 8.0],
            [-8.0, 8.0],
            [-3.0, 3.0],
        ],
        dtype=np.float64,
    )
    best_params = np.zeros(len(PARAM_NAMES), dtype=np.float64)
    best_score, best_stats, _, _ = _render_eval(
        points_opt,
        colors_opt,
        base_frames_opt[0],
        best_params,
        target_opt,
        width=int(args.opt_width),
        height=int(args.opt_height),
        point_radius=1,
    )
    trace = [{"iteration": 0, **best_stats}]
    print(f"initial score={best_score:.6f} stats={best_stats}")

    scales = np.asarray([0.018, 0.014, 0.015, 3.0, 3.0, 1.2], dtype=np.float64)
    for iteration in range(1, int(args.iterations) + 1):
        shrink = 0.35 + 0.65 * (1.0 - iteration / max(1.0, float(args.iterations)))
        if iteration % 7 == 0:
            candidate = bounds[:, 0] + rng.random(len(PARAM_NAMES)) * (bounds[:, 1] - bounds[:, 0])
        else:
            candidate = best_params + rng.normal(0.0, scales * shrink)
            candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])
        score, stats, _, _ = _render_eval(
            points_opt,
            colors_opt,
            base_frames_opt[0],
            candidate,
            target_opt,
            width=int(args.opt_width),
            height=int(args.opt_height),
            point_radius=1,
        )
        if score < best_score:
            best_score = score
            best_params = candidate
            best_stats = stats
            print(f"best {iteration:04d}: score={best_score:.6f} params={best_stats['params']} margins={best_stats['margins']}")
            trace.append({"iteration": iteration, **best_stats})

    # One deterministic coordinate pass after random search.
    steps = np.asarray([0.004, 0.003, 0.004, 0.8, 0.8, 0.35], dtype=np.float64)
    improved = True
    coordinate_round = 0
    while improved and coordinate_round < 4:
        improved = False
        coordinate_round += 1
        for dim in range(len(PARAM_NAMES)):
            for sign in (-1.0, 1.0):
                candidate = best_params.copy()
                candidate[dim] = np.clip(candidate[dim] + sign * steps[dim], bounds[dim, 0], bounds[dim, 1])
                score, stats, _, _ = _render_eval(
                    points_opt,
                    colors_opt,
                    base_frames_opt[0],
                    candidate,
                    target_opt,
                    width=int(args.opt_width),
                    height=int(args.opt_height),
                    point_radius=1,
                )
                if score < best_score:
                    best_score = score
                    best_params = candidate
                    best_stats = stats
                    improved = True
                    print(f"coord {coordinate_round}.{dim}: score={best_score:.6f} params={best_stats['params']}")
                    trace.append({"iteration": f"coord-{coordinate_round}-{dim}", **best_stats})
        steps *= 0.5

    _write_json(out_dir / "optimization_trace.json", {"best": best_stats, "trace": trace, "bounds": bounds.tolist()})

    base_frames_final, base_meta_final = renderer._camera_frames(
        camera_angle_x=float(camera["camera_angle_x"]),
        distance=float(camera["distance"]),
        width=int(args.final_width),
        height=int(args.final_height),
        num_frames=1,
        truck_right=0.0,
        target_track=1.0,
        origin_z_sign=-1.0,
        convention="opencv",
        source_distance_scale=float(args.base_distance_scale),
        target_down=0.0,
        square_pixels=True,
        principal_x_offset_px=0.0,
        principal_y_offset_px=0.0,
    )
    _save_projection(
        out_dir,
        Path(args.source_image),
        points,
        colors,
        base_frames_final[0],
        best_params,
        best_stats,
        width=int(args.final_width),
        height=int(args.final_height),
        point_radius=1,
        manifest_extra={
            "glb": str(Path(args.glb)),
            "camera_json": str(Path(args.camera_json)),
            "source_image": str(Path(args.source_image)),
            "base_camera": {
                "optimization": base_meta_opt,
                "final": base_meta_final,
            },
            "optimizer": {
                "iterations": int(args.iterations),
                "point_stride": int(args.point_stride),
                "seed": int(args.seed),
                "opt_width": int(args.opt_width),
                "opt_height": int(args.opt_height),
            },
        },
    )
    print(f"best score={best_score:.6f}")
    print(f"best params={best_stats['params']}")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
