"""Render orthographic check images of one or more depth volumes together.

Unprojects every frame of each volume (depth PNG + intrinsics + c2w) into a
shared world cloud and renders three orthographic views. Each volume can be
tinted for visual separation with --tint.
"""
import argparse
import json
import os

import cv2
import numpy as np


def load_volume(div_dir: str, max_pts_per_frame: int = 200_000):
    cams = json.load(open(os.path.join(div_dir, "cameras.json")))
    default_intr = {k: cams.get(k) for k in ("fl_x", "fl_y", "cx", "cy")}
    pts, cols = [], []
    for fr in cams["frames"]:
        dpath = os.path.join(div_dir, os.path.basename(fr["file_path"]))
        cpath = os.path.join(div_dir, os.path.basename(fr["color_path"]))
        code = cv2.imread(dpath, cv2.IMREAD_UNCHANGED)
        col = cv2.imread(cpath, cv2.IMREAD_COLOR)
        if code is None or col is None:
            continue
        near, far = fr["near"], fr["far"]
        d = near + ((code.astype(np.float64) - 1.0) / 65534.0) * (far - near)
        d[code == 0] = 0.0
        h, w = d.shape
        fx = fr.get("fl_x") or default_intr["fl_x"]
        fy = fr.get("fl_y") or default_intr["fl_y"]
        cx = fr.get("cx") or default_intr["cx"]
        cy = fr.get("cy") or default_intr["cy"]
        v, u = np.nonzero(d > 0)
        if len(v) == 0:
            continue
        if len(v) > max_pts_per_frame:
            sel = np.random.default_rng(0).choice(len(v), max_pts_per_frame, replace=False)
            v, u = v[sel], u[sel]
        z = d[v, u]
        x = (u - cx) / fx * z
        y = (v - cy) / fy * z
        cam_pts = np.stack([x, y, z, np.ones_like(z)], -1)
        c2w = np.array(fr["transform_matrix"], dtype=np.float64)
        wpts = (c2w @ cam_pts.T).T[:, :3]
        pts.append(wpts)
        cols.append(col[v, u][:, ::-1])  # BGR->RGB
    return np.concatenate(pts), np.concatenate(cols)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--volumes", nargs="+", required=True)
    ap.add_argument("--tint", nargs="*", default=[], help="per-volume tint like 1,1,1 or 1,0.6,0.6")
    ap.add_argument("--out_prefix", required=True)
    args = ap.parse_args()

    all_p, all_c = [], []
    for i, vd in enumerate(args.volumes):
        p, c = load_volume(vd)
        c = c.astype(np.float32)
        if i < len(args.tint):
            t = np.array([float(x) for x in args.tint[i].split(",")], np.float32)
            c = c * t
        print(f"{vd}: {len(p)} pts, extent {p.min(0).round(1)} .. {p.max(0).round(1)}")
        all_p.append(p)
        all_c.append(c)
    P = np.concatenate(all_p)
    C = np.clip(np.concatenate(all_c), 0, 255).astype(np.uint8)
    lo, hi = np.percentile(P, 0.5, 0), np.percentile(P, 99.5, 0)
    m = np.all((P > lo) & (P < hi), 1)
    P, C = P[m], C[m]
    for name, (a, b) in {"front_xy": (0, 1), "top_xz": (0, 2), "side_zy": (2, 1)}.items():
        W = H = 1000
        ra = P[:, a].max() - P[:, a].min()
        rb = P[:, b].max() - P[:, b].min()
        r = max(ra, rb) + 1e-9
        u = ((P[:, a] - P[:, a].min()) / r * (W - 1)).astype(int)
        v = ((P[:, b] - P[:, b].min()) / r * (H - 1)).astype(int)
        img = np.zeros((H, W, 3), np.uint8)
        img[np.clip(v, 0, H - 1), np.clip(u, 0, W - 1)] = C[:, ::-1]
        cv2.imwrite(f"{args.out_prefix}_{name}.png", img)
    print("wrote", args.out_prefix + "_{front_xy,top_xz,side_zy}.png")


if __name__ == "__main__":
    main()
