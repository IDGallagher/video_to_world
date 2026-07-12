"""
Option 1: analytic un-projection correction of DeepRock1_2.

Pixal3D reconstructed 1_2 assuming a centred principal point, but 1_2 is an
~40-degree off-axis crop of the shared outpaint panorama whose optical axis is
1_3's camera axis. This script undoes that assumption per-vertex:

  1. Project each B point into B's image with the ASSUMED (centred) camera,
     keeping its along-ray distance t_B.
  2. Map the pixel into panorama (= A image) coordinates via the Phase 0
     2D transform  p_a = (p_b - t2d) / s2d.
  3. Re-shoot through the TRUE camera: A's camera, pixel coords allowed to
     exceed A's image width.  p' = o_A + sigma * t_B * dir_A(pixel).
  4. sigma (single global depth scale) has a closed form: matched pixels lie
     on the same panorama ray, so sigma = median(t_A / t_B) over the Phase 0
     inlier matches lifted to 3D on each mesh.

The corrected B geometry lands directly in A's GLB space - no Umeyama, no
similarity fit. We then measure overlap NN residuals (vs. the 17mm rigid-ICP
plateau of the flat-join pipeline) and optionally refine with one short
colored ICP to see what systematic error remains.

Outputs to /mnt/d/archive/DeepRock_join_23/option1_unproject/.
"""

import json
import os

import numpy as np
import torch
import trimesh
from PIL import Image

from algos.icp import colored_icp_adam

ROOT_A = "/mnt/d/archive/DeepRock1_3"
ROOT_B = "/mnt/d/archive/DeepRock1_2"
GLB_A = f"{ROOT_A}/DeepRock1_3.glb"
GLB_B = f"{ROOT_B}/DeepRock1_2.glb"
CAM_A = f"{ROOT_A}/DeepRock1_3.pixal3d_camera.json"
CAM_B = f"{ROOT_B}/DeepRock1_2.pixal3d_camera.json"
XYZ_A = f"{ROOT_A}/DeepRock1_3 - Cloud.xyz"
XYZ_B = f"{ROOT_B}/DeepRock1_2 - Cloud.xyz"
PHASE0_DIR = "/mnt/d/archive/DeepRock_join_23/phase0"
OUT_DIR = "/mnt/d/archive/DeepRock_join_23/option1_unproject"

OVERLAP_NN_DIST = 0.05


def load_camera(cam_json_path, w, h):
    with open(cam_json_path) as f:
        cam = json.load(f)
    f_px = w / (2.0 * np.tan(cam["camera_angle_x"] / 2.0))
    origin = np.array([0.0, 0.0, -cam["distance"]])
    z = np.array([0.0, 0.0, 1.0])
    up = np.array([0.0, 1.0, 0.0])
    x = np.cross(z, up); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return {
        "f": f_px, "cx": 0.5 * w, "cy": 0.5 * h,
        "R_w2c": np.stack([x, y, z]), "origin": origin, "w": w, "h": h,
    }


def project(cam, pts):
    """world pts -> (pixels (N,2), along-ray distance t (N,), z (N,))"""
    p = (pts - cam["origin"]) @ cam["R_w2c"].T
    u = cam["f"] * p[:, 0] / p[:, 2] + cam["cx"]
    v = cam["f"] * p[:, 1] / p[:, 2] + cam["cy"]
    t = np.linalg.norm(p, axis=1)
    return np.stack([u, v], 1), t, p[:, 2]


def rays(cam, uv):
    """pixels -> unit world ray directions"""
    d = np.stack(
        [(uv[:, 0] - cam["cx"]) / cam["f"], (uv[:, 1] - cam["cy"]) / cam["f"],
         np.ones(len(uv))], 1)
    d = d @ cam["R_w2c"]
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def raycast_t(mesh, cam, uv):
    """along-ray hit distance for pixels; (t (N,), hit (N,))"""
    d = rays(cam, uv)
    o = np.tile(cam["origin"], (len(uv), 1))
    locs, ray_idx, _ = mesh.ray.intersects_location(o, d, multiple_hits=False)
    t = np.zeros(len(uv))
    hit = np.zeros(len(uv), bool)
    t[ray_idx] = np.linalg.norm(locs - cam["origin"], axis=1)
    hit[ray_idx] = True
    return t, hit


def unproject_b_to_a(pts_b, cam_b, cam_a, s2d, t2d, sigma, beta=0.0):
    uv_b, t_b, z_b = project(cam_b, pts_b)
    uv_a = (uv_b - t2d) / s2d
    d_a = rays(cam_a, uv_a)
    out = cam_a["origin"] + (sigma * t_b + beta)[:, None] * d_a
    return out, z_b


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    from scipy.spatial import cKDTree

    join2d = json.load(open(f"{PHASE0_DIR}/join_2d.json"))
    s2d = join2d["scale"]
    t2d = np.array(join2d["translation"])
    wa, ha = Image.open(join2d["image_a"]).size
    wb, hb = Image.open(join2d["image_b"]).size
    cam_a = load_camera(CAM_A, wa, ha)
    cam_b = load_camera(CAM_B, wb, hb)
    theta = np.degrees(np.arctan((((wb / 2) - t2d[0]) / s2d - cam_a["cx"]) / cam_a["f"]))
    print(f"[opt1] B centre ray is {theta:.1f} deg off A's axis; f_A={cam_a['f']:.0f}px")

    # --- sigma from lifted matches (same-ray property) ---
    mesh_a = trimesh.load(GLB_A, force="mesh")
    mesh_b = trimesh.load(GLB_B, force="mesh")
    m = np.load(f"{PHASE0_DIR}/inlier_matches_fullres.npz")
    t_a, hit_a = raycast_t(mesh_a, cam_a, m["kpts_a"])
    t_b, hit_b = raycast_t(mesh_b, cam_b, m["kpts_b"])
    both = hit_a & hit_b
    ratio = t_a[both] / t_b[both]
    sigma0 = float(np.median(ratio))
    print(f"[opt1] pure-scale sigma from {both.sum()} pairs: {sigma0:.4f} "
          f"(iqr {np.percentile(ratio,25):.4f}..{np.percentile(ratio,75):.4f})")

    # Affine depth transfer t_a ~= sigma * t_b + beta, robust (2 rounds of
    # least squares with residual trimming). Monocular depth is generally
    # only affine-consistent, and the wide ratio IQR shows pure scale is off.
    ta, tb = t_a[both], t_b[both]
    sel = np.ones(len(ta), bool)
    for _ in range(3):
        A_ls = np.stack([tb[sel], np.ones(sel.sum())], 1)
        (sigma, beta), *_ = np.linalg.lstsq(A_ls, ta[sel], rcond=None)
        resid = np.abs(sigma * tb + beta - ta)
        sel = resid < np.percentile(resid, 70)
    corr = np.corrcoef(tb, ta)[0, 1]
    print(f"[opt1] affine depth fit: sigma={sigma:.4f} beta={beta*1000:.1f}mm "
          f"(corr {corr:.3f}, inlier resid median {np.median(resid[sel])*1000:.1f}mm)")
    # Affine transfer measurably distorts near-field geometry (large beta
    # flattens close content) and scored worse after rigid refinement, so we
    # keep the pure-scale transfer and use the affine fit only as a diagnostic
    # of the depth noise floor between the two generations.
    sigma, beta = sigma0, 0.0

    # --- correct B's cloud and measure overlap residuals ---
    da = np.loadtxt(XYZ_A)
    db = np.loadtxt(XYZ_B)
    pts_a, col_a = da[:, :3], da[:, 3:6] / 255.0
    col_b = db[:, 3:6] / 255.0
    b_corr, z_b = unproject_b_to_a(db[:, :3], cam_b, cam_a, s2d, t2d, sigma, beta)
    keep = z_b > 1e-6
    print(f"[opt1] corrected {keep.sum()}/{len(b_corr)} B points (rest behind camera)")
    b_corr = b_corr[keep]
    col_bk = col_b[keep]

    tree_a = cKDTree(pts_a)
    d_b2a, _ = tree_a.query(b_corr, k=1, workers=-1)
    ov = d_b2a < OVERLAP_NN_DIST
    print(f"[opt1] overlap pts (NN<{OVERLAP_NN_DIST*100:.0f}cm): {ov.sum()}")
    print(f"[opt1] overlap median NN after unprojection alone: "
          f"{np.median(d_b2a[ov])*1000:.2f}mm (p90 {np.percentile(d_b2a[ov],90)*1000:.1f}mm)")

    # --- short rigid ICP on top to absorb any residual SE(3) ---
    rng = np.random.default_rng(0)
    nrm_a = da[:, 6:9]
    in_a_mask = tree_a.query_ball_point  # not used; keep simple crop below
    # crop ref to region near corrected B overlap
    tree_b = cKDTree(b_corr[ov]) if ov.sum() else None
    d_a2b, _ = tree_b.query(pts_a, k=1, workers=-1)
    ref_sel = np.where(d_a2b < OVERLAP_NN_DIST)[0]
    src_sel = np.where(ov)[0]
    if len(ref_sel) > 250_000:
        ref_sel = rng.choice(ref_sel, 250_000, replace=False)
    if len(src_sel) > 80_000:
        src_sel = rng.choice(src_sel, 80_000, replace=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, R_icp, t_icp = colored_icp_adam(
        torch.from_numpy(b_corr[src_sel]).float().to(device),
        torch.from_numpy(col_bk[src_sel]).float().to(device),
        torch.from_numpy(pts_a[ref_sel]).float().to(device),
        torch.from_numpy(col_a[ref_sel]).float().to(device),
        n_iter=40, lr=5e-3, knn_backend="cpu_kdtree",
        ref_normals=torch.from_numpy(nrm_a[ref_sel]).float().to(device),
        max_corr_dist=0.03, lambda_geometric=0.97,
    )
    R_icp = R_icp.detach().cpu().numpy()
    t_icp = t_icp.detach().cpu().numpy()
    b_ref = b_corr @ R_icp.T + t_icp
    d2, _ = tree_a.query(b_ref[ov], k=1, workers=-1)
    ang = np.degrees(np.arccos(np.clip((np.trace(R_icp) - 1) / 2, -1, 1)))
    print(f"[opt1] residual rigid: rot {ang:.2f} deg, trans {np.linalg.norm(t_icp)*1000:.1f}mm")
    print(f"[opt1] overlap median NN after +rigid: {np.median(d2)*1000:.2f}mm "
          f"(p90 {np.percentile(d2,90)*1000:.1f}mm)")

    json.dump(
        {
            "theta_deg": float(theta),
            "sigma": sigma,
            "median_nn_unproj_mm": float(np.median(d_b2a[ov]) * 1000),
            "p90_nn_unproj_mm": float(np.percentile(d_b2a[ov], 90) * 1000),
            "median_nn_unproj_rigid_mm": float(np.median(d2) * 1000),
            "p90_nn_unproj_rigid_mm": float(np.percentile(d2, 90) * 1000),
            "R_icp": R_icp.tolist(),
            "t_icp": t_icp.tolist(),
            "flat_join_baseline_mm": {"rigid_median": 17.3, "nonrigid_median": 4.2},
        },
        open(f"{OUT_DIR}/metrics.json", "w"), indent=2)

    # --- merged splat preview ---
    merged_pts = np.concatenate([pts_a, b_ref])
    merged_col = np.concatenate([col_a, col_bk])
    w, h = 1200, 500
    K2 = np.array([[cam_a["f"] * w / wa * 0.55, 0, w * 0.28],
                   [0, cam_a["f"] * w / wa * 0.55, h * 0.5], [0, 0, 1]])
    p_cam = (merged_pts - cam_a["origin"]) @ cam_a["R_w2c"].T
    front = p_cam[:, 2] > 1e-6
    uvz = p_cam[front]
    u = (K2[0, 0] * uvz[:, 0] / uvz[:, 2] + K2[0, 2]).round().astype(int)
    v = (K2[1, 1] * uvz[:, 1] / uvz[:, 2] + K2[1, 2]).round().astype(int)
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    img = np.zeros((h, w, 3), np.float32)
    uu, vv, zz = u[ok], v[ok], uvz[ok, 2]
    cc = merged_col[front][ok]
    order = np.argsort(-zz)
    img[vv[order], uu[order]] = cc[order]
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(
        f"{OUT_DIR}/merged_reproj.jpg", quality=92)
    print("[opt1] wrote metrics.json and merged_reproj.jpg")


if __name__ == "__main__":
    main()
