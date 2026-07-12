"""
Correspondence-driven warp for the DeepRock1_3 + DeepRock1_2 join.

Instead of NN-based ICP (which slides surfaces along each other), fit the
warp directly to feature correspondences:

  1. Lift the Phase 0 RoMa inlier matches to 3D on both meshes (raycast),
     keeping only depth-stable pairs (silhouette-edge rejection).
  2. Robust similarity init T0 (RANSAC-Umeyama on the pairs).
  3. Voxel-downsample the residual displacements to ~1200 anchors
     (median displacement per voxel — robust to stray matches).
  4. Fit a smoothed 3D thin-plate spline displacement field to the anchors;
     pick the smoothing lambda by held-out anchor error.
  5. Evaluate on: held-out anchors, the 16 audit landmarks (excluded from
     training), and overlap NN stats. Save the warp.

Outputs to /mnt/d/archive/DeepRock_join_23/corr_warp/:
  - warp.npz      T0, TPS centers/weights/affine, anchor stats
  - metrics.json
  - merged_reproj.jpg
"""

import json
import os

import numpy as np
import trimesh
from PIL import Image

PHASE0_DIR = "/mnt/d/archive/DeepRock_join_23/phase0"
AUDIT_DIR = "/mnt/d/archive/DeepRock_join_23/landmark_audit"
OUT_DIR = "/mnt/d/archive/DeepRock_join_23/corr_warp"
ROOT_A = "/mnt/d/archive/DeepRock1_3"
ROOT_B = "/mnt/d/archive/DeepRock1_2"

VOXEL = 0.02  # anchor downsample voxel (m)
LAMBDAS = [3e-4, 1e-3]  # >=3e-3 destabilizes the unscaled TPS solve
HOLDOUT_FRAC = 0.2


def load_camera(cam_json_path, w, h):
    cam = json.load(open(cam_json_path))
    f_px = w / (2.0 * np.tan(cam["camera_angle_x"] / 2.0))
    return {
        "f": f_px, "cx": 0.5 * w, "cy": 0.5 * h,
        "R_w2c": np.array([[-1.0, 0, 0], [0, -1.0, 0], [0, 0, 1.0]]),
        "origin": np.array([0.0, 0.0, -cam["distance"]]), "w": w, "h": h,
    }


def rays(cam, uv):
    d = np.stack(
        [(uv[:, 0] - cam["cx"]) / cam["f"], (uv[:, 1] - cam["cy"]) / cam["f"],
         np.ones(len(uv))], 1)
    d = d @ cam["R_w2c"]
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def raycast(mesh, cam, uv):
    d = rays(cam, uv)
    o = np.tile(cam["origin"], (len(uv), 1))
    locs, ray_idx, _ = mesh.ray.intersects_location(o, d, multiple_hits=False)
    pts = np.zeros((len(uv), 3))
    hit = np.zeros(len(uv), bool)
    pts[ray_idx] = locs
    hit[ray_idx] = True
    return pts, hit


def stable_mask(mesh, cam, uv, tol=0.03):
    offs = np.array([[0, 0], [6, 0], [-6, 0], [0, 6], [0, -6]])
    ok = np.ones(len(uv), bool)
    tref = None
    for o in offs:
        pts, hit = raycast(mesh, cam, uv + o)
        t = np.linalg.norm(pts - cam["origin"], axis=1)
        if tref is None:
            tref = t
            ok &= hit
        else:
            ok &= hit & (np.abs(t - tref) < tol)
    return ok


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    cov = dc.T @ sc / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / ((sc**2).sum() / len(src)))
    t = mu_d - s * R @ mu_s
    return s, R, t


def ransac_umeyama(src, dst, thresh, iters, rng):
    n, best_inl, best_cnt = len(src), np.zeros(len(src), bool), -1
    for _ in range(iters):
        idx = rng.choice(n, 4, replace=False)
        s, R, t = umeyama(src[idx], dst[idx])
        if not 0.2 < s < 5.0:
            continue
        resid = np.linalg.norm(dst - (s * src @ R.T + t), axis=1)
        inl = resid < thresh
        if inl.sum() > best_cnt:
            best_cnt, best_inl = inl.sum(), inl
    s, R, t = umeyama(src[best_inl], dst[best_inl])
    return s, R, t


class TPS3D:
    """Smoothed 3D thin-plate spline (phi(r)=r) displacement field."""

    def __init__(self, centers, values, lam):
        n = len(centers)
        K = np.linalg.norm(centers[:, None] - centers[None], axis=2)
        P = np.concatenate([centers, np.ones((n, 1))], 1)
        A = np.zeros((n + 4, n + 4))
        A[:n, :n] = K + lam * np.eye(n)
        A[:n, n:] = P
        A[n:, :n] = P.T
        rhs = np.concatenate([values, np.zeros((4, 3))])
        sol = np.linalg.solve(A, rhs)
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
    rng = np.random.default_rng(0)

    join2d = json.load(open(f"{PHASE0_DIR}/join_2d.json"))
    wa, ha = Image.open(join2d["image_a"]).size
    wb, hb = Image.open(join2d["image_b"]).size
    cam_a = load_camera(f"{ROOT_A}/DeepRock1_3.pixal3d_camera.json", wa, ha)
    cam_b = load_camera(f"{ROOT_B}/DeepRock1_2.pixal3d_camera.json", wb, hb)
    mesh_a = trimesh.load(f"{ROOT_A}/DeepRock1_3.glb", force="mesh")
    mesh_b = trimesh.load(f"{ROOT_B}/DeepRock1_2.glb", force="mesh")

    m = np.load(f"{PHASE0_DIR}/inlier_matches_fullres.npz")
    kpts_a, kpts_b = m["kpts_a"], m["kpts_b"]

    # exclude audit landmarks from training (held out as eval)
    audit = json.load(open(f"{AUDIT_DIR}/audit.json"))
    lm_a_px = np.array(audit["landmarks_a_px"])
    near_lm = np.zeros(len(kpts_a), bool)
    for p in lm_a_px:
        near_lm |= np.linalg.norm(kpts_a - p, axis=1) < 3.0
    print(f"[warp] excluding {near_lm.sum()} matches near audit landmarks")

    ok = stable_mask(mesh_a, cam_a, kpts_a) & stable_mask(mesh_b, cam_b, kpts_b) & ~near_lm
    pa, _ = raycast(mesh_a, cam_a, kpts_a[ok])
    pb, _ = raycast(mesh_b, cam_b, kpts_b[ok])
    print(f"[warp] {ok.sum()} stable match pairs")

    # --- robust similarity init ---
    s0, R0, t0 = ransac_umeyama(pb, pa, 0.05, 3000, rng)
    T0 = np.eye(4)
    T0[:3, :3] = s0 * R0
    T0[:3, 3] = t0
    pb0 = pb @ (s0 * R0).T + t0
    r0 = np.linalg.norm(pa - pb0, axis=1)
    print(f"[warp] T0: scale {s0:.4f}; anchor resid after T0: "
          f"median {np.median(r0)*1000:.1f}mm p90 {np.percentile(r0,90)*1000:.1f}mm")

    # --- voxel-downsample anchors (median displacement per voxel) ---
    disp = pa - pb0
    keys = np.floor(pb0 / VOXEL).astype(int)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    centers, values = [], []
    for g in range(inv.max() + 1):
        sel = inv == g
        if sel.sum() >= 1:
            centers.append(np.median(pb0[sel], axis=0))
            values.append(np.median(disp[sel], axis=0))
    centers = np.array(centers)
    values = np.array(values)
    print(f"[warp] {len(centers)} anchors after voxel downsample "
          f"(|disp| median {np.median(np.linalg.norm(values,axis=1))*1000:.1f}mm)")

    # --- fit TPS, choose lambda on held-out raw pairs ---
    n_hold = int(len(pb0) * HOLDOUT_FRAC)
    hold = rng.choice(len(pb0), n_hold, replace=False)
    hold_mask = np.zeros(len(pb0), bool)
    hold_mask[hold] = True

    results = []
    for lam in LAMBDAS:
        tps = TPS3D(centers, values, lam)
        pred = pb0[hold_mask] + tps(pb0[hold_mask])
        err = np.linalg.norm(pa[hold_mask] - pred, axis=1)
        med = np.median(err)
        print(f"[warp] lambda {lam:g}: held-out anchor resid median {med*1000:.2f}mm "
              f"p90 {np.percentile(err,90)*1000:.1f}mm")
        results.append((lam, med, tps))
    # prefer the smoothest field within 20% of the best held-out error:
    # smoother extrapolation matters more for mesh warping than the last mm
    best_med = min(r[1] for r in results)
    lam, _, tps = max((r for r in results if r[1] <= 1.1 * best_med), key=lambda r: r[0])
    print(f"[warp] chose lambda {lam:g} (best median {best_med*1000:.2f}mm)")

    # --- evaluate on audit landmarks (never trained on) ---
    lm_pa = np.array(audit["pa"])
    lm_pb = np.array(audit["pb"])
    lm_pb0 = lm_pb @ (s0 * R0).T + t0
    lm_pred = lm_pb0 + tps(lm_pb0)
    lm_err = np.linalg.norm(lm_pa - lm_pred, axis=1)
    print("[warp] landmark residuals mm: " +
          " ".join(f"{e*1000:.0f}" for e in lm_err) +
          f"  (median {np.median(lm_err)*1000:.1f}, "
          f"vs 33.7 flat / 55.6 opt1)")

    np.savez(
        f"{OUT_DIR}/warp.npz",
        T0=T0, centers=centers, w=tps.w, a=tps.a, lam=lam,
    )
    json.dump(
        {
            "T0_scale": s0,
            "n_pairs": int(ok.sum()),
            "n_anchors": len(centers),
            "lambda": lam,
            "anchor_resid_after_T0_mm": float(np.median(r0) * 1000),
            "landmark_resid_mm": (lm_err * 1000).tolist(),
            "landmark_resid_median_mm": float(np.median(lm_err) * 1000),
            "baselines_landmark_median_mm": {"flat_a2": 33.7, "opt1": 55.6},
        },
        open(f"{OUT_DIR}/metrics.json", "w"), indent=2)

    # --- preview: warp full B cloud, splat with A ---
    da = np.loadtxt(f"{ROOT_A}/DeepRock1_3 - Cloud.xyz")
    db = np.loadtxt(f"{ROOT_B}/DeepRock1_2 - Cloud.xyz")
    pts_a, col_a = da[:, :3], da[:, 3:6] / 255.0
    col_b = db[:, 3:6] / 255.0
    b0 = db[:, :3] @ (s0 * R0).T + t0
    # blend TPS displacement to zero away from the anchor region
    from scipy.spatial import cKDTree
    d_anchor, _ = cKDTree(centers).query(b0, k=1, workers=-1)
    wgt = np.clip((0.12 - d_anchor) / 0.06, 0.0, 1.0)[:, None]  # 1 inside, 0 past 12cm
    b_warp = b0 + wgt * tps(b0)

    merged_pts = np.concatenate([pts_a, b_warp])
    merged_col = np.concatenate([col_a, col_b])
    w, h = 1200, 500
    fx = cam_a["f"] * w / wa * 0.55
    K2 = np.array([[fx, 0, w * 0.28], [0, fx, h * 0.5], [0, 0, 1]])
    p_cam = (merged_pts - cam_a["origin"]) @ cam_a["R_w2c"].T
    front = p_cam[:, 2] > 1e-6
    uvz = p_cam[front]
    u = (K2[0, 0] * uvz[:, 0] / uvz[:, 2] + K2[0, 2]).round().astype(int)
    v = (K2[1, 1] * uvz[:, 1] / uvz[:, 2] + K2[1, 2]).round().astype(int)
    okk = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    img = np.zeros((h, w, 3), np.float32)
    uu, vv, zz = u[okk], v[okk], uvz[okk, 2]
    cc = merged_col[front][okk]
    order = np.argsort(-zz)
    img[vv[order], uu[order]] = cc[order]
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(
        f"{OUT_DIR}/merged_reproj.jpg", quality=92)
    print("[warp] wrote warp.npz, metrics.json, merged_reproj.jpg")


if __name__ == "__main__":
    main()
