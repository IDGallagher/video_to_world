"""
Rotate DeepRock1_3's source camera 40 degrees about its up vector (around the
focal point / camera centre) and re-render the image-determined geometry into
that view, textured directly from the source image.

For each pixel of the rotated view:
  - raycast the GLB mesh from the (unchanged) camera centre,
  - project the hit into the ORIGINAL camera; if it lands inside the source
    frame, bilinearly sample the source PNG there -> color, alpha 255,
  - otherwise (no hit, or outside the original frustum) -> black, alpha 0.

Because the rotation is about the camera centre, every new-view ray is also a
ray of the original camera, so first-hit geometry is exactly the surface the
source image determined - no separate occlusion test is needed.

Rendered at 2x and box-downsampled for anti-aliased edges.
Output: /mnt/d/archive/DeepRock_rot40/rot40_reproject.png (RGBA, 2944x1648)
        + rot40_preview.jpg (on black, for a quick look)
"""

import json
import os

import numpy as np
import trimesh
from PIL import Image

ROOT_A = "/mnt/d/archive/DeepRock1_3"
GLB = f"{ROOT_A}/DeepRock1_3.glb"
CAM = f"{ROOT_A}/DeepRock1_3.pixal3d_camera.json"
IMG = f"{ROOT_A}/ian_101_dark_granite_rocks_at_the_bottom_of_the_ocean._No_wat_47f6e59d-04f6-4fc4-9783-163353d64c86_3.png"
OUT_DIR = "/mnt/d/archive/DeepRock_rot20"

YAW_DEG = 20.0  # toward image-right (world -x), same side as the 1_2 extension
SUPERSAMPLE = 2
CHUNK = 2_000_000


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cam = json.load(open(CAM))
    src = np.asarray(Image.open(IMG).convert("RGB"), dtype=np.float32)
    H, W = src.shape[:2]
    f = W / (2.0 * np.tan(cam["camera_angle_x"] / 2.0))
    cx, cy = 0.5 * W, 0.5 * H
    origin = np.array([0.0, 0.0, -cam["distance"]])

    # original camera axes (rows of world-to-cam), validated in phase A1
    R0 = np.array([[-1.0, 0.0, 0.0],
                   [0.0, -1.0, 0.0],
                   [0.0, 0.0, 1.0]])
    # rotate the camera about the world up axis (+y): axes' = R_y(a) @ axes
    a = np.radians(-YAW_DEG)  # negative: view swings toward world -x = image right
    Ry = np.array([[np.cos(a), 0.0, np.sin(a)],
                   [0.0, 1.0, 0.0],
                   [-np.sin(a), 0.0, np.cos(a)]])
    R1 = R0 @ Ry.T  # new world-to-cam
    print(f"[rot40] new view dir (world): {R1[2]}")

    mesh = trimesh.load(GLB, force="mesh")
    print(f"[rot40] mesh {len(mesh.faces)} faces")

    # rays for the supersampled rotated view
    Ws, Hs = W * SUPERSAMPLE, H * SUPERSAMPLE
    us, vs = np.meshgrid((np.arange(Ws) + 0.5) / SUPERSAMPLE,
                         (np.arange(Hs) + 0.5) / SUPERSAMPLE)
    d_cam = np.stack([(us.ravel() - cx) / f, (vs.ravel() - cy) / f,
                      np.ones(Ws * Hs)], 1)
    dirs = d_cam @ R1
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    rgb = np.zeros((Hs * Ws, 3), np.float32)
    alpha = np.zeros(Hs * Ws, bool)
    n_hit = 0
    for i in range(0, len(dirs), CHUNK):
        d = dirs[i : i + CHUNK]
        locs, ray_idx, _ = mesh.ray.intersects_location(
            np.tile(origin, (len(d), 1)), d, multiple_hits=False)
        if not len(ray_idx):
            continue
        n_hit += len(ray_idx)
        # project hits into the ORIGINAL camera
        p = (locs - origin) @ R0.T
        u = f * p[:, 0] / p[:, 2] + cx
        v = f * p[:, 1] / p[:, 2] + cy
        ok = (p[:, 2] > 1e-6) & (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
        u, v = u[ok], v[ok]
        gidx = i + ray_idx[ok]
        # bilinear sample of the source image
        u0 = np.floor(u).astype(int)
        v0 = np.floor(v).astype(int)
        u1 = np.minimum(u0 + 1, W - 1)
        v1 = np.minimum(v0 + 1, H - 1)
        fu = (u - u0)[:, None]
        fv = (v - v0)[:, None]
        col = (src[v0, u0] * (1 - fu) * (1 - fv) + src[v0, u1] * fu * (1 - fv)
               + src[v1, u0] * (1 - fu) * fv + src[v1, u1] * fu * fv)
        rgb[gidx] = col
        alpha[gidx] = True
        print(f"[rot40] rays {i + len(d)}/{len(dirs)} hits so far {n_hit}", flush=True)

    rgb = rgb.reshape(Hs, Ws, 3)
    alpha = alpha.reshape(Hs, Ws).astype(np.float32)

    # box downsample to source resolution
    rgb = rgb.reshape(H, SUPERSAMPLE, W, SUPERSAMPLE, 3).mean(axis=(1, 3))
    alpha = alpha.reshape(H, SUPERSAMPLE, W, SUPERSAMPLE).mean(axis=(1, 3))
    # premultiplied average -> unpremultiply for clean edge colors
    nz = alpha > 0
    rgb[nz] /= alpha[nz, None]

    out = np.dstack([np.clip(rgb, 0, 255).astype(np.uint8),
                     (alpha * 255).round().astype(np.uint8)])
    Image.fromarray(out, "RGBA").save(f"{OUT_DIR}/rot20_reproject.png")
    prev = (rgb * alpha[..., None]).astype(np.uint8)
    Image.fromarray(prev).save(f"{OUT_DIR}/rot20_preview.jpg", quality=92)
    cov = float((alpha > 0.5).mean())
    print(f"[rot40] coverage {100*cov:.1f}% | wrote rot20_reproject.png ({W}x{H} RGBA) "
          f"and rot20_preview.jpg")


if __name__ == "__main__":
    main()
