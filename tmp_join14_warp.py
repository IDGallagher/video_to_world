"""
3D join of DeepRock1_4 (20-degree rotated Midjourney extension) onto
DeepRock1_3, using the rotation-about-camera-centre model.

Because 1_4's source image IS the view from A's camera yawed 20 degrees
about its own centre, the placement is closed-form:

    for each 1_4 point: project through Pixal3D's ASSUMED camera for 1_4
    (fov 75.0deg, centred) -> pixel + along-ray distance t4
    map pixel into the rot20 frame (uniform rescale, phase-0 fit)
    re-shoot through the TRUE camera (A's camera yawed 20deg, f_A, same centre)
    p_A = o_A + sigma * t4 * dir        sigma = median(t_A / t4) over matches

This also corrects Pixal3D's wrong FOV estimate (75.0 vs true 79.7 deg)
exactly, per vertex. A smoothed TPS is then fitted to the remaining
correspondence residuals and kept only if it improves held-out pairs.

Outputs to /mnt/d/archive/DeepRock_join_34/warp/:
  join_params.json, tps.npz (if kept), merged_reproj.jpg, metrics
"""

import json
import os

import numpy as np
import trimesh
from PIL import Image

ROOT_A = "/mnt/d/archive/DeepRock1_3"
ROOT_4 = "/mnt/d/archive/DeepRock1_4"
GLB_A = f"{ROOT_A}/DeepRock1_3.glb"
GLB_4 = f"{ROOT_4}/DeepRock1_4.glb"
CAM_A = f"{ROOT_A}/DeepRock1_3.pixal3d_camera.json"
CAM_4 = f"{ROOT_4}/DeepRock1_4.pixal3d_camera.json"
XYZ_A = f"{ROOT_A}/DeepRock1_3 - Cloud.xyz"
XYZ_4 = f"{ROOT_4}/DeepRock1_4 - Cloud.xyz"
P0_DIR = "/mnt/d/archive/DeepRock_join_34/phase0"
OUT_DIR = "/mnt/d/archive/DeepRock_join_34/warp"

YAW_DEG = 20.0
W_R20, H_R20 = 2944, 1648  # rot20 frame == A's image frame, camera yawed
VOXEL = 0.02
TPS_LAMBDA = 1e-3
HOLDOUT_FRAC = 0.2

R0 = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])


def yawed_R(yaw_deg):
    a = np.radians(-yaw_deg)
    Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    return R0 @ Ry.T


def cam_from_json(path, w, h, R_w2c):
    cam = json.load(open(path))
    return {
        "f": w / (2.0 * np.tan(cam["camera_angle_x"] / 2.0)),
        "cx": 0.5 * w, "cy": 0.5 * h,
        "R_w2c": R_w2c, "origin": np.array([0.0, 0.0, -cam["distance"]]),
        "w": w, "h": h,
    }


def rays(cam, uv):
    d = np.stack([(uv[:, 0] - cam["cx"]) / cam["f"],
                  (uv[:, 1] - cam["cy"]) / cam["f"], np.ones(len(uv))], 1)
    d = d @ cam["R_w2c"]
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def raycast_t(mesh, cam, uv):
    d = rays(cam, uv)
    o = np.tile(cam["origin"], (len(uv), 1))
    locs, ray_idx, _ = mesh.ray.intersects_location(o, d, multiple_hits=False)
    t = np.zeros(len(uv))
    hit = np.zeros(len(uv), bool)
    t[ray_idx] = np.linalg.norm(locs - cam["origin"], axis=1)
    hit[ray_idx] = True
    return t, hit


def project_t(cam, pts):
    p = (pts - cam["origin"]) @ cam["R_w2c"].T
    u = cam["f"] * p[:, 0] / p[:, 2] + cam["cx"]
    v = cam["f"] * p[:, 1] / p[:, 2] + cam["cy"]
    return np.stack([u, v], 1), np.linalg.norm(p, axis=1), p[:, 2]


def unproject_4_to_a(pts4, cam4, cam_true, s2d, t2d, sigma):
    uv4, t4, z4 = project_t(cam4, pts4)
    uv_r = s2d * uv4 + t2d
    d = rays(cam_true, uv_r)
    return cam_true["origin"] + sigma * t4[:, None] * d, z4


class TPS3D:
    def __init__(self, centers, values, lam):
        n = len(centers)
        K = np.linalg.norm(centers[:, None] - centers[None], axis=2)
        P = np.concatenate([centers, np.ones((n, 1))], 1)
        A = np.zeros((n + 4, n + 4))
        A[:n, :n] = K + lam * np.eye(n)
        A[:n, n:] = P
        A[n:, :n] = P.T
        sol = np.linalg.solve(A, np.concatenate([values, np.zeros((4, 3))]))
        self.centers, self.w, self.a = centers, sol[:n], sol[n:]

    def __call__(self, x, chunk=200_000):
        out = np.empty_like(x)
        for i in range(0, len(x), chunk):
            xi = x[i : i + chunk]
            K = np.linalg.norm(xi[:, None] - self.centers[None], axis=2)
            out[i : i + chunk] = K @ self.w + xi @ self.a[:3] + self.a[3]
        return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(0)

    m2d = json.load(open(f"{P0_DIR}/map_4_to_rot20.json"))
    s2d, t2d = m2d["s"], np.array(m2d["t"])

    img4 = Image.open(json.load(open(CAM_4))["source_image_path"].replace("/mnt/d", "/mnt/d"))
    w4, h4 = img4.size
    cam4 = cam_from_json(CAM_4, w4, h4, R0)
    cam_true = cam_from_json(CAM_A, W_R20, H_R20, yawed_R(YAW_DEG))
    print(f"[j34] cam4 assumed f={cam4['f']:.0f}px d={-cam4['origin'][2]:.3f} | "
          f"true f={cam_true['f']:.0f}px yaw {YAW_DEG}deg")

    mesh_a = trimesh.load(GLB_A, force="mesh")
    mesh_4 = trimesh.load(GLB_4, force="mesh")

    # --- sigma from matched rays ---
    m = np.load(f"{P0_DIR}/matches_known_region.npz")
    t_a, hit_a = raycast_t(mesh_a, cam_true, m["kpts_r20"])
    t_4, hit_4 = raycast_t(mesh_4, cam4, m["kpts_4"])
    both = hit_a & hit_4
    ratio = t_a[both] / t_4[both]
    sigma = float(np.median(ratio))
    print(f"[j34] sigma from {both.sum()} pairs: {sigma:.4f} "
          f"(iqr {np.percentile(ratio,25):.4f}..{np.percentile(ratio,75):.4f})")

    # --- correspondence residuals of the closed-form join ---
    pa = cam_true["origin"] + t_a[both, None] * rays(cam_true, m["kpts_r20"][both])
    d4 = rays(cam4, m["kpts_4"][both])
    p4 = cam4["origin"] + t_4[both, None] * d4
    p4_a, _ = unproject_4_to_a(p4, cam4, cam_true, s2d, t2d, sigma)
    resid = np.linalg.norm(pa - p4_a, axis=1)
    print(f"[j34] closed-form correspondence resid: median "
          f"{np.median(resid)*1000:.1f}mm p90 {np.percentile(resid,90)*1000:.1f}mm")

    # --- TPS on residuals, judged on held-out pairs ---
    hold = np.zeros(len(pa), bool)
    hold[rng.choice(len(pa), int(len(pa) * HOLDOUT_FRAC), replace=False)] = True
    disp = pa - p4_a
    keys = np.floor(p4_a[~hold] / VOXEL).astype(int)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    centers = np.array([np.median(p4_a[~hold][inv == g], axis=0) for g in range(inv.max() + 1)])
    values = np.array([np.median(disp[~hold][inv == g], axis=0) for g in range(inv.max() + 1)])
    tps = TPS3D(centers, values, TPS_LAMBDA)
    resid_tps = np.linalg.norm(pa[hold] - (p4_a[hold] + tps(p4_a[hold])), axis=1)
    print(f"[j34] +TPS ({len(centers)} anchors) held-out resid: median "
          f"{np.median(resid_tps)*1000:.1f}mm p90 {np.percentile(resid_tps,90)*1000:.1f}mm "
          f"(closed-form held-out: {np.median(resid[hold])*1000:.1f}mm)")
    keep_tps = np.median(resid_tps) < 0.85 * np.median(resid[hold])
    print(f"[j34] keep TPS: {keep_tps}")

    json.dump(
        {
            "sigma": sigma, "s2d": s2d, "t2d": t2d.tolist(), "yaw_deg": YAW_DEG,
            "closed_form_resid_median_mm": float(np.median(resid) * 1000),
            "closed_form_resid_p90_mm": float(np.percentile(resid, 90) * 1000),
            "tps_heldout_resid_median_mm": float(np.median(resid_tps) * 1000),
            "keep_tps": bool(keep_tps),
        },
        open(f"{OUT_DIR}/join_params.json", "w"), indent=2)
    if keep_tps:
        np.savez(f"{OUT_DIR}/tps.npz", centers=centers, w=tps.w, a=tps.a)

    # --- preview: warp full 1_4 cloud, splat with A ---
    da = np.loadtxt(XYZ_A)
    d4c = np.loadtxt(XYZ_4)
    pts_a, col_a = da[:, :3], da[:, 3:6] / 255.0
    col_4 = d4c[:, 3:6] / 255.0
    p4w, z4 = unproject_4_to_a(d4c[:, :3], cam4, cam_true, s2d, t2d, sigma)
    if keep_tps:
        dnn, _ = cKDTree(centers).query(p4w, k=1, workers=-1)
        wgt = np.clip((0.10 - dnn) / 0.05, 0, 1)[:, None]
        disp_full = tps(p4w)
        mag = np.linalg.norm(disp_full, axis=1, keepdims=True)
        disp_full *= np.minimum(1.0, 0.05 / np.maximum(mag, 1e-9))
        p4w = p4w + wgt * disp_full

    tree_a = cKDTree(pts_a)
    dnn, _ = tree_a.query(p4w, k=1, workers=-1)
    ov = dnn < 0.05
    print(f"[j34] overlap NN: median {np.median(dnn[ov])*1000:.1f}mm over {ov.sum()} pts")

    merged = np.concatenate([pts_a, p4w])
    mcol = np.concatenate([col_a, col_4])
    w, h = 1400, 520
    fx = cam_true["f"] * w / W_R20 * 0.5
    K2 = np.array([[fx, 0, w * 0.35], [0, fx, h * 0.5], [0, 0, 1]])
    Rw = yawed_R(YAW_DEG / 2)  # halfway view
    p_cam = (merged - cam_true["origin"]) @ Rw.T
    front = p_cam[:, 2] > 1e-6
    uvz = p_cam[front]
    u = (K2[0, 0] * uvz[:, 0] / uvz[:, 2] + K2[0, 2]).round().astype(int)
    v = (K2[1, 1] * uvz[:, 1] / uvz[:, 2] + K2[1, 2]).round().astype(int)
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    img = np.zeros((h, w, 3), np.float32)
    uu, vv, zz = u[ok], v[ok], uvz[ok, 2]
    cc = mcol[front][ok]
    order = np.argsort(-zz)
    img[vv[order], uu[order]] = cc[order]
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(
        f"{OUT_DIR}/merged_reproj.jpg", quality=92)
    print("[j34] wrote join_params.json / merged_reproj.jpg")


if __name__ == "__main__":
    main()
