"""
Rotate DeepRock1_3's source camera N degrees about its own centre (pure yaw
about the world up vector) and reproject the ORIGINAL image directly into
that view via a planar homography - no mesh, no raycasting.

Why a homography is exact here: the new camera shares the same optical
centre as the original (pure rotation, zero translation). For any camera
pair related only by rotation, the mapping between their image planes is
depth-independent:

    pixel0_homog ~ H @ pixel1_homog,   H = K @ R0 @ R1^T @ K^-1

(K identical for both since only orientation changes; R0, R1 are the
world-to-cam rotations, same row-vector convention as the rest of this
project's camera code). Every destination pixel therefore has an exact,
gap-free source pixel wherever it falls inside the original frustum -
unlike the mesh-based reprojection (tmp_rot40_reproject.py), which had
holes/jagged edges wherever the reconstructed geometry was imperfect,
missing (fog crop), or mis-raycast.

Output: RGBA PNG, same resolution as the source image, alpha=1 wherever the
homography lands inside the original frame, alpha=0 elsewhere.
"""

import json
import os

import numpy as np
from PIL import Image

ROOT_A = "/mnt/d/archive/DeepRock1_3"
CAM = f"{ROOT_A}/DeepRock1_3.pixal3d_camera.json"
IMG = f"{ROOT_A}/ian_101_dark_granite_rocks_at_the_bottom_of_the_ocean._No_wat_47f6e59d-04f6-4fc4-9783-163353d64c86_3.png"

YAW_DEG = 20.0
OUT_DIR = f"/mnt/d/archive/DeepRock_rot{int(YAW_DEG)}"
SUPERSAMPLE = 2  # for anti-aliased boundary only; interior is exact either way


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cam = json.load(open(CAM))
    src = np.asarray(Image.open(IMG).convert("RGB"), dtype=np.float32)
    H, W = src.shape[:2]
    f = W / (2.0 * np.tan(cam["camera_angle_x"] / 2.0))
    cx, cy = 0.5 * W, 0.5 * H
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    Kinv = np.linalg.inv(K)

    # same camera-axis convention validated in phase A1 / tmp_rot40_reproject
    R0 = np.array([[-1.0, 0.0, 0.0],
                   [0.0, -1.0, 0.0],
                   [0.0, 0.0, 1.0]])
    a = np.radians(-YAW_DEG)
    Ry = np.array([[np.cos(a), 0.0, np.sin(a)],
                   [0.0, 1.0, 0.0],
                   [-np.sin(a), 0.0, np.cos(a)]])
    R1 = R0 @ Ry.T

    Hmat = K @ R0 @ R1.T @ Kinv
    print(f"[homog] yaw={YAW_DEG} deg, f={f:.1f}px")
    print(f"[homog] H =\n{Hmat}")

    Ws, Hs = W * SUPERSAMPLE, H * SUPERSAMPLE
    us, vs = np.meshgrid((np.arange(Ws) + 0.5) / SUPERSAMPLE,
                         (np.arange(Hs) + 0.5) / SUPERSAMPLE)
    ones = np.ones_like(us)
    p1 = np.stack([us, vs, ones], axis=-1).reshape(-1, 3)  # (N,3)

    p0 = p1 @ Hmat.T  # (N,3), row-vector form: p0_row = p1_row @ H^T  <=>  p0_col = H @ p1_col
    w0 = p0[:, 2]
    valid_front = w0 > 1e-9
    u0 = np.full(len(p0), -1.0)
    v0 = np.full(len(p0), -1.0)
    u0[valid_front] = p0[valid_front, 0] / w0[valid_front]
    v0[valid_front] = p0[valid_front, 1] / w0[valid_front]

    inb = valid_front & (u0 >= 0) & (u0 <= W - 1) & (v0 >= 0) & (v0 <= H - 1)
    print(f"[homog] {100*inb.mean():.1f}% of oversampled pixels fall inside the source frame")

    rgb = np.zeros((len(p0), 3), np.float32)
    idx = np.where(inb)[0]
    uu, vv = u0[idx], v0[idx]
    u0i, v0i = np.floor(uu).astype(int), np.floor(vv).astype(int)
    u1i = np.minimum(u0i + 1, W - 1)
    v1i = np.minimum(v0i + 1, H - 1)
    fu = (uu - u0i)[:, None]
    fv = (vv - v0i)[:, None]
    col = (src[v0i, u0i] * (1 - fu) * (1 - fv) + src[v0i, u1i] * fu * (1 - fv)
           + src[v1i, u0i] * (1 - fu) * fv + src[v1i, u1i] * fu * fv)
    rgb[idx] = col

    rgb = rgb.reshape(Hs, Ws, 3)
    alpha = inb.reshape(Hs, Ws).astype(np.float32)

    rgb = rgb.reshape(H, SUPERSAMPLE, W, SUPERSAMPLE, 3).mean(axis=(1, 3))
    alpha = alpha.reshape(H, SUPERSAMPLE, W, SUPERSAMPLE).mean(axis=(1, 3))
    nz = alpha > 0
    rgb[nz] /= alpha[nz, None]

    out = np.dstack([np.clip(rgb, 0, 255).astype(np.uint8),
                     (alpha * 255).round().astype(np.uint8)])
    Image.fromarray(out).save(f"{OUT_DIR}/rot{int(YAW_DEG)}_homography.png")
    prev = (rgb * alpha[..., None]).astype(np.uint8)
    Image.fromarray(prev).save(f"{OUT_DIR}/rot{int(YAW_DEG)}_homography_preview.jpg", quality=92)
    cov = float((alpha > 0.5).mean())
    print(f"[homog] coverage {100*cov:.1f}% | wrote rot{int(YAW_DEG)}_homography.png "
          f"({W}x{H} RGBA) and preview.jpg")


if __name__ == "__main__":
    main()
