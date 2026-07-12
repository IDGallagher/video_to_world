"""
Phase A4 of the DeepRock1_3 + DeepRock1_2 join: merge meshes and export.

  1. Load both GLBs with textures intact.
  2. Map B into A space with the Phase A2 rigid+scale transform.
  3. Apply the Phase A3 deformation field to B's vertices, blended by
     NN-distance to A's cloud (full inside the overlap, zero away from it).
  4. Seam-cut: A keeps faces left of the seam plane, B keeps faces right.
  5. Export a combined GLB (two textured nodes) + verification renders.

Outputs to /mnt/d/archive/DeepRock_join_23/phaseA4/:
  - DeepRock1_23_joined.glb
  - joined_render.jpg (normal-shaded raycast panorama)
"""

import json
import os

import numpy as np
import torch
import trimesh
from PIL import Image

from models.deformation import DeformationGrid
from utils.geometry import se3_apply

ROOT_A = "/mnt/d/archive/DeepRock1_3"
ROOT_B = "/mnt/d/archive/DeepRock1_2"
GLB_A = f"{ROOT_A}/DeepRock1_3.glb"
GLB_B = f"{ROOT_B}/DeepRock1_2.glb"
XYZ_A = f"{ROOT_A}/DeepRock1_3 - Cloud.xyz"
CAM_A = f"{ROOT_A}/DeepRock1_3.pixal3d_camera.json"
A3_DIR = "/mnt/d/archive/DeepRock_join_23/phaseA3"
OUT_DIR = "/mnt/d/archive/DeepRock_join_23/phaseA4"

SEAM_FRACTION = 0.5  # position of the seam plane within the overlap x-range


def load_textured_mesh(path):
    scene = trimesh.load(path)
    if isinstance(scene, trimesh.Scene):
        meshes = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
        assert len(meshes) == 1, f"expected 1 mesh in {path}, got {len(meshes)}"
        return meshes[0]
    return scene


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    from scipy.spatial import cKDTree

    ck = torch.load(f"{A3_DIR}/deform_checkpoint.pt", weights_only=False)
    T = np.array(ck["T_b_to_a_rigid_4x4"])
    blend_full, blend_zero = ck["blend_full"], ck["blend_zero"]

    mesh_a = load_textured_mesh(GLB_A)
    mesh_b = load_textured_mesh(GLB_B)
    print(f"[A4] A: {len(mesh_a.vertices)}v {len(mesh_a.faces)}f | "
          f"B: {len(mesh_b.vertices)}v {len(mesh_b.faces)}f")

    # --- B -> A space (rigid + scale) ---
    vb = mesh_b.vertices @ T[:3, :3].T + T[:3, 3]

    # --- deformation, blended by NN distance to A's cloud ---
    pts_a = np.loadtxt(XYZ_A)[:, :3]
    tree_a = cKDTree(pts_a)
    d_b, _ = tree_a.query(vb, k=1, workers=-1)
    wgt = np.clip((blend_zero - d_b) / (blend_zero - blend_full), 0.0, 1.0)

    dp = ck["deform_params"]
    deform = DeformationGrid(
        ck["bbox_min"].cuda(), ck["bbox_max"].cuda(),
        min_res=dp["deform_min_res"], max_res=dp["deform_max_res"],
        num_levels=dp["deform_num_levels"],
        log2_hashmap_size=dp["deform_log2_hashmap_size"],
        n_neurons=dp["deform_n_neurons"], n_hidden_layers=dp["deform_n_hidden_layers"],
    ).cuda()
    deform.load_state_dict(ck["deform_state_dict"])
    deform.eval()
    xi_g = ck["xi_global"].cuda()

    vb_def = vb.copy()
    idx = np.where(wgt > 0)[0]
    with torch.no_grad():
        c = torch.from_numpy(vb[idx]).float().cuda()
        xi_l = deform(c)
        c1 = se3_apply(xi_l, c)
        c2 = se3_apply(xi_g.unsqueeze(0).expand(len(c1), 6), c1)
        moved = c2.cpu().numpy()
    vb_def[idx] = vb[idx] + wgt[idx, None] * (moved - vb[idx])
    disp = np.linalg.norm(vb_def[idx] - vb[idx], axis=1)
    print(f"[A4] deformed {len(idx)} B verts, median disp {np.median(disp)*1000:.1f}mm "
          f"max {disp.max()*1000:.1f}mm")
    mesh_b.vertices = vb_def

    # --- seam cut ---
    # Overlap x-range from B verts near A's surface
    near = d_b < ck["overlap_nn_dist"]
    # robust range: stray fog fragments can match A anywhere, so use percentiles
    x_lo, x_hi = np.percentile(vb[near, 0], [2, 98])
    x_seam = x_lo + SEAM_FRACTION * (x_hi - x_lo)
    print(f"[A4] overlap x [{x_lo:.3f},{x_hi:.3f}] -> seam at x={x_seam:.3f}")

    # Image-right is -x in GLB space (the export transform negates x), so A's
    # exclusive content is at +x and B's exclusive content at more-negative x.
    fc_a = mesh_a.triangles_center[:, 0]
    fc_b = mesh_b.triangles_center[:, 0]
    mesh_a.update_faces(fc_a >= x_seam)
    mesh_a.remove_unreferenced_vertices()
    mesh_b.update_faces(fc_b < x_seam)
    mesh_b.remove_unreferenced_vertices()
    print(f"[A4] after cut: A {len(mesh_a.faces)}f | B {len(mesh_b.faces)}f")

    scene = trimesh.Scene()
    scene.add_geometry(mesh_a, node_name="DeepRock1_3", geom_name="DeepRock1_3")
    scene.add_geometry(mesh_b, node_name="DeepRock1_2", geom_name="DeepRock1_2")
    out_glb = f"{OUT_DIR}/DeepRock1_23_joined.glb"
    scene.export(out_glb)
    print(f"[A4] wrote {out_glb}")

    with open(f"{OUT_DIR}/join_metadata.json", "w") as f:
        json.dump(
            {
                "space": "DeepRock1_3 GLB space (meters)",
                "seam_x": float(x_seam),
                "overlap_x": [float(x_lo), float(x_hi)],
                "T_b_to_a_rigid_4x4": T.tolist(),
                "deform_checkpoint": f"{A3_DIR}/deform_checkpoint.pt",
                "source_camera": json.load(open(CAM_A)),
            },
            f,
            indent=2,
        )

    # --- verification render: normal-shaded raycast panorama ---
    combined = trimesh.util.concatenate(
        [
            trimesh.Trimesh(vertices=mesh_a.vertices, faces=mesh_a.faces, process=False),
            trimesh.Trimesh(vertices=mesh_b.vertices, faces=mesh_b.faces, process=False),
        ]
    )
    cam = json.load(open(CAM_A))
    fov, dist = cam["camera_angle_x"], cam["distance"]
    w, h = 1400, 560
    f_norm = 1.0 / (2.0 * np.tan(fov / 2.0))
    K = np.array([[f_norm * w * 0.5, 0, w * 0.30], [0, f_norm * w * 0.5, h * 0.5], [0, 0, 1]])
    origin = np.array([0.0, 0.05, -dist])
    z = np.array([0.0, 0.0, 1.0])
    up = np.array([0.0, 1.0, 0.0])
    x = np.cross(z, up); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    us, vs = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    d_cam = np.stack(
        [(us.ravel() - K[0, 2]) / K[0, 0], (vs.ravel() - K[1, 2]) / K[1, 1], np.ones(w * h)],
        axis=1,
    )
    dirs = d_cam @ R
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    locs, ray_idx, tri_idx = combined.ray.intersects_location(
        np.tile(origin, (len(dirs), 1)), dirs, multiple_hits=False
    )
    shade = np.zeros(len(dirs))
    n = combined.face_normals[tri_idx]
    shade[ray_idx] = np.abs((n * -dirs[ray_idx]).sum(axis=1))
    img = (np.clip(shade.reshape(h, w), 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img).save(f"{OUT_DIR}/joined_render.jpg", quality=92)
    print("[A4] wrote joined_render.jpg")


if __name__ == "__main__":
    main()
