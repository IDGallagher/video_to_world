"""
Merge DeepRock1_3 + warped DeepRock1_4 into one GLB.

1_4's mesh vertices are placed by the closed-form rotation join
(tmp_join14_warp.py) plus the blended, clamped TPS residual repair.
The seam is ANGULAR - a vertical plane of constant azimuth through the
shared camera centre - which is the natural cut for a rotation join:
A keeps everything left of the seam azimuth, 1_4 everything right.

Outputs to /mnt/d/archive/DeepRock_join_34/merge/:
  DeepRock1_34_joined.glb, join_metadata.json, joined_render.jpg
"""

import json
import os

import numpy as np
import trimesh
from PIL import Image

ROOT_A = "/mnt/d/archive/DeepRock1_3"
ROOT_4 = "/mnt/d/archive/DeepRock1_4"
CAM_A = f"{ROOT_A}/DeepRock1_3.pixal3d_camera.json"
CAM_4 = f"{ROOT_4}/DeepRock1_4.pixal3d_camera.json"
WARP_DIR = "/mnt/d/archive/DeepRock_join_34/warp"
OUT_DIR = "/mnt/d/archive/DeepRock_join_34/merge"

YAW_DEG = 20.0
W_R20, H_R20 = 2944, 1648
SEAM_AZ_DEG = 30.0  # A's right frustum edge is at 39.9; keep A canonical up to 30
BLEND_FULL, BLEND_ZERO, DISP_CLAMP = 0.05, 0.10, 0.05

R0 = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])


def yawed_R(yaw_deg):
    a = np.radians(-yaw_deg)
    Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    return R0 @ Ry.T


def cam_from_json(path, w, h, R_w2c):
    cam = json.load(open(path))
    return {"f": w / (2.0 * np.tan(cam["camera_angle_x"] / 2.0)),
            "cx": 0.5 * w, "cy": 0.5 * h, "R_w2c": R_w2c,
            "origin": np.array([0.0, 0.0, -cam["distance"]]), "w": w, "h": h}


def load_textured_mesh(path):
    scene = trimesh.load(path)
    if isinstance(scene, trimesh.Scene):
        meshes = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
        assert len(meshes) == 1
        return meshes[0]
    return scene


def tps_eval(x, centers, w, a, chunk=200_000):
    out = np.empty_like(x)
    for i in range(0, len(x), chunk):
        xi = x[i : i + chunk]
        K = np.linalg.norm(xi[:, None] - centers[None], axis=2)
        out[i : i + chunk] = K @ w + xi @ a[:3] + a[3]
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    from scipy.spatial import cKDTree

    jp = json.load(open(f"{WARP_DIR}/join_params.json"))
    sigma, s2d, t2d = jp["sigma"], jp["s2d"], np.array(jp["t2d"])

    img4_path = json.load(open(CAM_4))["source_image_path"]
    w4, h4 = Image.open(img4_path).size
    cam4 = cam_from_json(CAM_4, w4, h4, R0)
    cam_true = cam_from_json(CAM_A, W_R20, H_R20, yawed_R(YAW_DEG))
    o_A = cam_true["origin"]

    mesh_a = load_textured_mesh(f"{ROOT_A}/DeepRock1_3.glb")
    mesh_4 = load_textured_mesh(f"{ROOT_4}/DeepRock1_4.glb")

    # --- closed-form placement of 1_4 verts ---
    v = mesh_4.vertices
    p = (v - cam4["origin"]) @ cam4["R_w2c"].T
    uv4 = np.stack([cam4["f"] * p[:, 0] / p[:, 2] + cam4["cx"],
                    cam4["f"] * p[:, 1] / p[:, 2] + cam4["cy"]], 1)
    t4 = np.linalg.norm(p, axis=1)
    uv_r = s2d * uv4 + t2d
    d = np.stack([(uv_r[:, 0] - cam_true["cx"]) / cam_true["f"],
                  (uv_r[:, 1] - cam_true["cy"]) / cam_true["f"],
                  np.ones(len(uv_r))], 1) @ cam_true["R_w2c"]
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    v_new = o_A + sigma * t4[:, None] * d

    # --- blended clamped TPS repair ---
    if jp["keep_tps"] and os.path.exists(f"{WARP_DIR}/tps.npz"):
        tps = np.load(f"{WARP_DIR}/tps.npz")
        dnn, _ = cKDTree(tps["centers"]).query(v_new, k=1, workers=-1)
        wgt = np.clip((BLEND_ZERO - dnn) / (BLEND_ZERO - BLEND_FULL), 0, 1)
        idx = np.where(wgt > 0)[0]
        disp = tps_eval(v_new[idx], tps["centers"], tps["w"], tps["a"])
        mag = np.linalg.norm(disp, axis=1, keepdims=True)
        disp *= np.minimum(1.0, DISP_CLAMP / np.maximum(mag, 1e-9))
        v_new[idx] += wgt[idx, None] * disp
        print(f"[j34-m] TPS applied to {len(idx)} verts")
    mesh_4.vertices = v_new

    # --- angular seam about the shared camera centre ---
    def azimuth_deg(pts):
        rel = pts - o_A
        return np.degrees(np.arctan2(-rel[:, 0], rel[:, 2]))  # + toward image right

    az_a = azimuth_deg(mesh_a.triangles_center)
    az_4 = azimuth_deg(mesh_4.triangles_center)
    mesh_a.update_faces(az_a <= SEAM_AZ_DEG)
    mesh_a.remove_unreferenced_vertices()
    mesh_4.update_faces(az_4 > SEAM_AZ_DEG)
    mesh_4.remove_unreferenced_vertices()
    print(f"[j34-m] after seam cut at {SEAM_AZ_DEG}deg: "
          f"A {len(mesh_a.faces)}f | 1_4 {len(mesh_4.faces)}f")

    scene = trimesh.Scene()
    scene.add_geometry(mesh_a, node_name="DeepRock1_3", geom_name="DeepRock1_3")
    scene.add_geometry(mesh_4, node_name="DeepRock1_4", geom_name="DeepRock1_4")
    out_glb = f"{OUT_DIR}/DeepRock1_34_joined.glb"
    scene.export(out_glb)
    print(f"[j34-m] wrote {out_glb}")

    json.dump(
        {
            "space": "DeepRock1_3 GLB space (meters)",
            "model": "rotation about shared camera centre",
            "yaw_deg": YAW_DEG, "sigma": sigma, "seam_azimuth_deg": SEAM_AZ_DEG,
            "camera_centre": o_A.tolist(),
            "warp_params": f"{WARP_DIR}/join_params.json",
        },
        open(f"{OUT_DIR}/join_metadata.json", "w"), indent=2)

    # --- verification render from the halfway view ---
    combined = trimesh.util.concatenate(
        [trimesh.Trimesh(vertices=mesh_a.vertices, faces=mesh_a.faces, process=False),
         trimesh.Trimesh(vertices=mesh_4.vertices, faces=mesh_4.faces, process=False)])
    w, h = 1400, 560
    fx = cam_true["f"] * w / W_R20 * 0.5
    K = np.array([[fx, 0, w * 0.5], [0, fx, h * 0.5], [0, 0, 1]])
    Rw = yawed_R(YAW_DEG / 2)
    origin = o_A + np.array([0.0, 0.05, 0.0])
    us, vs = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    d_cam = np.stack([(us.ravel() - K[0, 2]) / K[0, 0],
                      (vs.ravel() - K[1, 2]) / K[1, 1], np.ones(w * h)], 1)
    dirs = d_cam @ Rw
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    locs, ray_idx, tri_idx = combined.ray.intersects_location(
        np.tile(origin, (len(dirs), 1)), dirs, multiple_hits=False)
    shade = np.zeros(len(dirs))
    n = combined.face_normals[tri_idx]
    shade[ray_idx] = np.abs((n * -dirs[ray_idx]).sum(axis=1))
    Image.fromarray((np.clip(shade.reshape(h, w), 0, 1) * 255).astype(np.uint8)).save(
        f"{OUT_DIR}/joined_render.jpg", quality=92)
    print("[j34-m] wrote joined_render.jpg")


if __name__ == "__main__":
    main()
