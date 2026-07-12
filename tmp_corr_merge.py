"""
Merge DeepRock1_3 + DeepRock1_2 using the correspondence-driven TPS warp
(tmp_corr_warp_fit.py) instead of the ICP-based pipeline.

  1. B mesh -> A space via T0 (robust similarity from match pairs).
  2. Add the TPS displacement, blended to zero away from the anchor region.
  3. Seam-cut at mid-overlap (image-right is -x in GLB space).
  4. Export combined GLB + verification renders.

Outputs to /mnt/d/archive/DeepRock_join_23/corr_merge/.
"""

import json
import os

import numpy as np
import trimesh
from PIL import Image

ROOT_A = "/mnt/d/archive/DeepRock1_3"
ROOT_B = "/mnt/d/archive/DeepRock1_2"
WARP = "/mnt/d/archive/DeepRock_join_23/corr_warp/warp.npz"
OUT_DIR = "/mnt/d/archive/DeepRock_join_23/corr_merge"

BLEND_FULL = 0.05  # full TPS displacement within this distance of an anchor
BLEND_ZERO = 0.10  # zero past this
DISP_CLAMP = 0.08  # clamp |TPS displacement| (extrapolation guard), m
SEAM_FRACTION = 0.5


def tps_eval(x, centers, w, a, chunk=200_000):
    out = np.empty_like(x)
    for i in range(0, len(x), chunk):
        xi = x[i : i + chunk]
        K = np.linalg.norm(xi[:, None] - centers[None], axis=2)
        out[i : i + chunk] = K @ w + xi @ a[:3] + a[3]
    return out


def load_textured_mesh(path):
    scene = trimesh.load(path)
    if isinstance(scene, trimesh.Scene):
        meshes = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
        assert len(meshes) == 1
        return meshes[0]
    return scene


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    from scipy.spatial import cKDTree

    warp = np.load(WARP)
    T0, centers, w_tps, a_tps = warp["T0"], warp["centers"], warp["w"], warp["a"]

    mesh_a = load_textured_mesh(f"{ROOT_A}/DeepRock1_3.glb")
    mesh_b = load_textured_mesh(f"{ROOT_B}/DeepRock1_2.glb")

    vb = mesh_b.vertices @ T0[:3, :3].T + T0[:3, 3]
    d_anchor, _ = cKDTree(centers).query(vb, k=1, workers=-1)
    wgt = np.clip((BLEND_ZERO - d_anchor) / (BLEND_ZERO - BLEND_FULL), 0.0, 1.0)
    idx = np.where(wgt > 0)[0]
    disp = tps_eval(vb[idx], centers, w_tps, a_tps)
    mag = np.linalg.norm(disp, axis=1, keepdims=True)
    disp *= np.minimum(1.0, DISP_CLAMP / np.maximum(mag, 1e-9))
    vb[idx] += wgt[idx, None] * disp
    print(f"[merge] warped {len(idx)} B verts, median |disp| "
          f"{np.median(np.linalg.norm(disp,axis=1))*1000:.1f}mm")
    mesh_b.vertices = vb

    # seam: mid-range of anchor x (anchors live exactly in the overlap)
    x_lo, x_hi = np.percentile(centers[:, 0], [2, 98])
    x_seam = x_lo + SEAM_FRACTION * (x_hi - x_lo)
    print(f"[merge] anchor x range [{x_lo:.3f},{x_hi:.3f}] -> seam x={x_seam:.3f}")

    mesh_a.update_faces(mesh_a.triangles_center[:, 0] >= x_seam)
    mesh_a.remove_unreferenced_vertices()
    mesh_b.update_faces(mesh_b.triangles_center[:, 0] < x_seam)
    mesh_b.remove_unreferenced_vertices()
    print(f"[merge] after cut: A {len(mesh_a.faces)}f | B {len(mesh_b.faces)}f")

    scene = trimesh.Scene()
    scene.add_geometry(mesh_a, node_name="DeepRock1_3", geom_name="DeepRock1_3")
    scene.add_geometry(mesh_b, node_name="DeepRock1_2", geom_name="DeepRock1_2")
    out_glb = f"{OUT_DIR}/DeepRock1_23_joined_corrwarp.glb"
    scene.export(out_glb)
    print(f"[merge] wrote {out_glb}")

    json.dump(
        {
            "space": "DeepRock1_3 GLB space (meters)",
            "seam_x": float(x_seam),
            "warp": WARP,
            "blend_full_m": BLEND_FULL,
            "blend_zero_m": BLEND_ZERO,
        },
        open(f"{OUT_DIR}/join_metadata.json", "w"), indent=2)

    # verification render (normal-shaded raycast, wide view)
    combined = trimesh.util.concatenate(
        [trimesh.Trimesh(vertices=mesh_a.vertices, faces=mesh_a.faces, process=False),
         trimesh.Trimesh(vertices=mesh_b.vertices, faces=mesh_b.faces, process=False)])
    cam = json.load(open(f"{ROOT_A}/DeepRock1_3.pixal3d_camera.json"))
    w, h = 1400, 560
    f_norm = 1.0 / (2.0 * np.tan(cam["camera_angle_x"] / 2.0))
    K = np.array([[f_norm * w * 0.5, 0, w * 0.30], [0, f_norm * w * 0.5, h * 0.5],
                  [0, 0, 1]])
    origin = np.array([0.0, 0.05, -cam["distance"]])
    R = np.array([[-1.0, 0, 0], [0, -1.0, 0], [0, 0, 1.0]])
    us, vs = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    d_cam = np.stack([(us.ravel() - K[0, 2]) / K[0, 0],
                      (vs.ravel() - K[1, 2]) / K[1, 1], np.ones(w * h)], 1)
    dirs = d_cam @ R
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    locs, ray_idx, tri_idx = combined.ray.intersects_location(
        np.tile(origin, (len(dirs), 1)), dirs, multiple_hits=False)
    shade = np.zeros(len(dirs))
    n = combined.face_normals[tri_idx]
    shade[ray_idx] = np.abs((n * -dirs[ray_idx]).sum(axis=1))
    Image.fromarray((np.clip(shade.reshape(h, w), 0, 1) * 255).astype(np.uint8)).save(
        f"{OUT_DIR}/joined_render.jpg", quality=92)
    print("[merge] wrote joined_render.jpg")


if __name__ == "__main__":
    main()
