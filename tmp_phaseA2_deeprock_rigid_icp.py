"""
Phase A2 of the DeepRock1_3 + DeepRock1_2 join: rigid refinement.

Starts from the Phase A1 similarity (B -> A space), crops both .xyz clouds to
the overlap band, then alternates:

  1. colored_icp_adam (SE(3), geometry-weighted) refining B's placement
  2. similarity re-fit (Umeyama with scale) on tight nearest-neighbor pairs,
     to absorb residual scale error that rigid ICP cannot express

Outputs to /mnt/d/archive/DeepRock_join_23/phaseA2/:
  - transform_b_to_a_refined.json  (composed 4x4, B GLB space -> A GLB space)
  - merged_preview.ply / merged_reproj.jpg
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
XYZ_A = f"{ROOT_A}/DeepRock1_3 - Cloud.xyz"
XYZ_B = f"{ROOT_B}/DeepRock1_2 - Cloud.xyz"
CAM_A = f"{ROOT_A}/DeepRock1_3.pixal3d_camera.json"
A1_DIR = "/mnt/d/archive/DeepRock_join_23/phaseA1"
OUT_DIR = "/mnt/d/archive/DeepRock_join_23/phaseA2"

OVERLAP_MARGIN = 0.05  # m, padding around bbox intersection
SRC_MAX_PTS = 80_000
REF_MAX_PTS = 250_000
NN_INLIER_DIST = 0.015  # m, pairs used for similarity re-fit
ROUNDS = 2
ICP_ITERS = (60, 30)
LAMBDA_GEOMETRIC = 0.97


def load_xyz(path):
    d = np.loadtxt(path)
    return d[:, :3], d[:, 3:6] / 255.0, d[:, 6:9]


def apply_T(T, pts):
    return pts @ T[:3, :3].T + T[:3, 3]


def median_nn_dist(src, ref_tree):
    d, _ = ref_tree.query(src, k=1, workers=-1)
    return float(np.median(d))


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
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = t
    return T, s


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    from scipy.spatial import cKDTree

    device = "cuda" if torch.cuda.is_available() else "cpu"
    T_sim = np.array(json.load(open(f"{A1_DIR}/similarity_b_to_a.json"))["T_4x4"])

    pts_a, col_a, nrm_a = load_xyz(XYZ_A)
    pts_b, col_b, nrm_b = load_xyz(XYZ_B)
    print(f"[A2] A: {len(pts_a)} pts, B: {len(pts_b)} pts")

    T_total = T_sim.copy()
    rng = np.random.default_rng(0)
    log = {"rounds": []}

    for rnd in range(ROUNDS):
        b_now = apply_T(T_total, pts_b)

        # Overlap band: bbox intersection with margin
        lo = np.maximum(pts_a.min(0), b_now.min(0)) - OVERLAP_MARGIN
        hi = np.minimum(pts_a.max(0), b_now.max(0)) + OVERLAP_MARGIN
        in_a = np.all((pts_a >= lo) & (pts_a <= hi), axis=1)
        in_b = np.all((b_now >= lo) & (b_now <= hi), axis=1)
        print(f"[A2] round {rnd}: overlap pts A={in_a.sum()} B={in_b.sum()}")

        ref_idx = np.where(in_a)[0]
        src_idx = np.where(in_b)[0]
        if len(ref_idx) > REF_MAX_PTS:
            ref_idx = rng.choice(ref_idx, REF_MAX_PTS, replace=False)
        if len(src_idx) > SRC_MAX_PTS:
            src_idx = rng.choice(src_idx, SRC_MAX_PTS, replace=False)

        ref_p, ref_c = pts_a[ref_idx], col_a[ref_idx]
        ref_n = nrm_a[ref_idx]
        src_p, src_c = b_now[src_idx], col_b[src_idx]

        tree = cKDTree(ref_p)
        d_before = median_nn_dist(src_p, tree)

        src_aligned, R_icp, t_icp = colored_icp_adam(
            torch.from_numpy(src_p).float().to(device),
            torch.from_numpy(src_c).float().to(device),
            torch.from_numpy(ref_p).float().to(device),
            torch.from_numpy(ref_c).float().to(device),
            n_iter=ICP_ITERS[rnd],
            lr=1e-2,
            knn_backend="cpu_kdtree",
            ref_normals=torch.from_numpy(ref_n).float().to(device),
            max_corr_dist=0.04,
            lambda_geometric=LAMBDA_GEOMETRIC,
        )
        R_icp = R_icp.detach().cpu().numpy()
        t_icp = t_icp.detach().cpu().numpy()
        T_icp = np.eye(4)
        T_icp[:3, :3] = R_icp
        T_icp[:3, 3] = t_icp
        T_total = T_icp @ T_total

        src_p2 = apply_T(T_icp, src_p)
        d_after_icp = median_nn_dist(src_p2, tree)

        # Similarity re-fit on tight NN pairs (absorbs residual scale)
        d, j = tree.query(src_p2, k=1, workers=-1)
        m = d < NN_INLIER_DIST
        T_refit, s_refit = umeyama(src_p2[m], ref_p[j[m]])
        T_total = T_refit @ T_total
        d_after_refit = median_nn_dist(apply_T(T_refit, src_p2), tree)

        print(
            f"[A2] round {rnd}: median NN {d_before*1000:.2f}mm -> ICP "
            f"{d_after_icp*1000:.2f}mm -> refit(s={s_refit:.4f}) {d_after_refit*1000:.2f}mm "
            f"(refit pairs={m.sum()})"
        )
        log["rounds"].append(
            {
                "median_nn_before_m": d_before,
                "median_nn_after_icp_m": d_after_icp,
                "median_nn_after_refit_m": d_after_refit,
                "refit_scale": s_refit,
                "refit_pairs": int(m.sum()),
            }
        )

    log.update(
        {
            "description": "p_a = T[:3,:3] @ p_b + T[:3,3], B GLB space -> A GLB space (meters), includes A1 similarity",
            "T_4x4": T_total.tolist(),
            "scale_total": float(np.cbrt(np.linalg.det(T_total[:3, :3]))),
        }
    )
    with open(f"{OUT_DIR}/transform_b_to_a_refined.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"[A2] wrote transform_b_to_a_refined.json (total scale {log['scale_total']:.4f})")

    # Merged preview + reprojection
    b_final = apply_T(T_total, pts_b)
    merged_pts = np.concatenate([pts_a, b_final])
    merged_col = (np.concatenate([col_a, col_b]) * 255).astype(np.uint8)
    trimesh.PointCloud(merged_pts, colors=merged_col).export(f"{OUT_DIR}/merged_preview.ply")

    cam = json.load(open(CAM_A))
    fov, dist = cam["camera_angle_x"], cam["distance"]
    w, h = 1200, 500
    f_norm = 1.0 / (2.0 * np.tan(fov / 2.0))
    K2 = np.array(
        [[f_norm * w * 0.55, 0, w * 0.28], [0, f_norm * w * 0.55, h * 0.5], [0, 0, 1]]
    )
    origin = np.array([0.0, 0.0, -dist])
    z = -origin / np.linalg.norm(origin)
    up = np.array([0.0, 1.0, 0.0])
    x = np.cross(z, up); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    Rw = np.stack([x, y, z])
    p_cam = (merged_pts - origin) @ Rw.T
    front = p_cam[:, 2] > 1e-6
    uvz = p_cam[front]
    u = (K2[0, 0] * uvz[:, 0] / uvz[:, 2] + K2[0, 2]).round().astype(int)
    v = (K2[1, 1] * uvz[:, 1] / uvz[:, 2] + K2[1, 2]).round().astype(int)
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    img = np.zeros((h, w, 3), np.float32)
    uu, vv, zz = u[ok], v[ok], uvz[ok, 2]
    cc = np.concatenate([col_a, col_b])[front][ok]
    order = np.argsort(-zz)
    img[vv[order], uu[order]] = cc[order]
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(
        f"{OUT_DIR}/merged_reproj.jpg", quality=92
    )
    print("[A2] wrote merged_preview.ply and merged_reproj.jpg")


if __name__ == "__main__":
    main()
