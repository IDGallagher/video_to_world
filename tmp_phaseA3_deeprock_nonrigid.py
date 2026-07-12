"""
Phase A3 of the DeepRock1_3 + DeepRock1_2 join: non-rigid seam refinement.

Starts from the Phase A2 refined transform (B -> A GLB space), crops both
clouds to the overlap band, and trains algos.non_rigid_icp's DeformationGrid
to deform B's overlap geometry onto A's (A fixed / canonical).

Outputs to /mnt/d/archive/DeepRock_join_23/phaseA3/:
  - deform_checkpoint.pt      xi_global + DeformationGrid state + bbox + params
  - metrics.json              NN-distance stats before/after
  - overlap_error_{before,after}.ply   B overlap cloud colored by NN distance
  - merged_reproj.jpg         deformed-B + A splat through widened A camera
"""

import json
import os

import numpy as np
import torch
import trimesh
from PIL import Image

from algos.non_rigid_icp import non_rigid_icp
from utils.geometry import se3_apply

ROOT_A = "/mnt/d/archive/DeepRock1_3"
ROOT_B = "/mnt/d/archive/DeepRock1_2"
XYZ_A = f"{ROOT_A}/DeepRock1_3 - Cloud.xyz"
XYZ_B = f"{ROOT_B}/DeepRock1_2 - Cloud.xyz"
CAM_A = f"{ROOT_A}/DeepRock1_3.pixal3d_camera.json"
A2_DIR = "/mnt/d/archive/DeepRock_join_23/phaseA2"
OUT_DIR = "/mnt/d/archive/DeepRock_join_23/phaseA3"

OVERLAP_NN_DIST = 0.05  # B points closer than this to A's cloud count as overlap
BLEND_FULL = 0.05       # blend weight 1 below this NN distance ...
BLEND_ZERO = 0.10       # ... fading to 0 here (used for full-cloud preview)
SRC_MAX_PTS = 150_000
REF_MAX_PTS = 250_000
N_ITER = 200
LR = 0.01
MAX_CORR_DIST = 0.04
# The two assets are independent generations: the field only needs a smooth,
# low-frequency ~1-4cm correction. Keep it coarse and stiff so it cannot
# shuffle points tangentially.
DEFORM_PARAMS = dict(
    deform_log2_hashmap_size=19,
    deform_num_levels=8,
    deform_n_neurons=64,
    deform_n_hidden_layers=2,
    deform_min_res=8,
    deform_max_res=128,
)
LOCAL_TWIST_REG = 1e-2
TV_REG = 3e-3


def load_xyz(path):
    d = np.loadtxt(path)
    return d[:, :3], d[:, 3:6] / 255.0, d[:, 6:9]


def error_cloud(pts, dists, path, dmax=0.03):
    """Point cloud colored blue (0) -> red (dmax) by NN distance."""
    x = np.clip(dists / dmax, 0, 1)
    col = np.stack([x, 0.15 + 0 * x, 1 - x], axis=1)
    trimesh.PointCloud(pts, colors=(col * 255).astype(np.uint8)).export(path)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    from scipy.spatial import cKDTree

    device = "cuda"
    T = np.array(json.load(open(f"{A2_DIR}/transform_b_to_a_refined.json"))["T_4x4"])

    pts_a, col_a, nrm_a = load_xyz(XYZ_A)
    pts_b, col_b, _ = load_xyz(XYZ_B)
    b_now = pts_b @ T[:3, :3].T + T[:3, 3]

    # Data-driven overlap: mutual proximity of the two clouds
    tree_a_full = cKDTree(pts_a)
    d_b_to_a, _ = tree_a_full.query(b_now, k=1, workers=-1)
    in_b = d_b_to_a < OVERLAP_NN_DIST
    tree_b_full = cKDTree(b_now)
    d_a_to_b, _ = tree_b_full.query(pts_a, k=1, workers=-1)
    in_a = d_a_to_b < OVERLAP_NN_DIST
    print(f"[A3] overlap pts A={in_a.sum()} B={in_b.sum()}")

    rng = np.random.default_rng(0)
    ref_idx = np.where(in_a)[0]
    src_idx = np.where(in_b)[0]
    if len(ref_idx) > REF_MAX_PTS:
        ref_idx = rng.choice(ref_idx, REF_MAX_PTS, replace=False)
    if len(src_idx) > SRC_MAX_PTS:
        src_idx = rng.choice(src_idx, SRC_MAX_PTS, replace=False)

    ref_p = torch.from_numpy(pts_a[ref_idx]).float().to(device)
    ref_n = torch.from_numpy(nrm_a[ref_idx]).float().to(device)
    ref_c = torch.from_numpy(col_a[ref_idx]).float().to(device)
    src_p = torch.from_numpy(b_now[src_idx]).float().to(device)
    src_c = torch.from_numpy(col_b[src_idx]).float().to(device)

    tree = cKDTree(pts_a[in_a])
    d_before, _ = tree.query(b_now[src_idx], k=1, workers=-1)
    print(f"[A3] median NN before: {np.median(d_before)*1000:.2f}mm")

    metrics_out = {}
    src_final, xi_global, deform = non_rigid_icp(
        src_p.view(1, 1, -1, 3),
        ref_p.view(1, 1, -1, 3),
        n_iter=N_ITER,
        lr=LR,
        knn_backend="cpu_kdtree",
        method="point2point",
        ref_normals=ref_n,
        max_corr_dist=MAX_CORR_DIST,
        local_twist_reg=LOCAL_TWIST_REG,
        tv_reg=TV_REG,
        metrics_out=metrics_out,
        early_stopping_patience=15,
        early_stopping_min_iters=50,
        **DEFORM_PARAMS,
    )

    src_final_np = src_final.view(-1, 3).detach().cpu().numpy()
    d_after, _ = tree.query(src_final_np, k=1, workers=-1)
    print(
        f"[A3] median NN after: {np.median(d_after)*1000:.2f}mm "
        f"(p90 {np.percentile(d_before,90)*1000:.1f} -> {np.percentile(d_after,90)*1000:.1f}mm)"
    )

    error_cloud(b_now[src_idx], d_before, f"{OUT_DIR}/overlap_error_before.ply")
    error_cloud(src_final_np, d_after, f"{OUT_DIR}/overlap_error_after.ply")

    torch.save(
        {
            "xi_global": xi_global.detach().cpu(),
            "deform_state_dict": deform.state_dict(),
            "bbox_min": deform.bbox_min.cpu(),
            "bbox_max": deform.bbox_max.cpu(),
            "deform_params": DEFORM_PARAMS,
            "T_b_to_a_rigid_4x4": T,
            "overlap_nn_dist": OVERLAP_NN_DIST,
            "blend_full": BLEND_FULL,
            "blend_zero": BLEND_ZERO,
            "note": "apply: p1 = se3_apply(deform(p), p); p2 = se3_apply(xi_global, p1); "
            "input p is B cloud already mapped by T_b_to_a_rigid_4x4; "
            "blend displacement by NN-distance-to-A weight (1 below blend_full, 0 above blend_zero)",
        },
        f"{OUT_DIR}/deform_checkpoint.pt",
    )
    print(f"[A3] wrote deform_checkpoint.pt")

    with open(f"{OUT_DIR}/metrics.json", "w") as f:
        json.dump(
            {
                "median_nn_before_m": float(np.median(d_before)),
                "median_nn_after_m": float(np.median(d_after)),
                "p90_nn_before_m": float(np.percentile(d_before, 90)),
                "p90_nn_after_m": float(np.percentile(d_after, 90)),
                "n_src": int(len(src_idx)),
                "n_ref": int(len(ref_idx)),
                "n_iter": N_ITER,
            },
            f,
            indent=2,
        )

    # Reprojection preview: A + (B non-overlap rigid) + (B overlap deformed)
    # Only evaluate the deform field where the blend weight is nonzero — the
    # hashgrid is meaningless outside its training bbox.
    b_def = b_now.copy()
    eval_idx = np.where(d_b_to_a < BLEND_ZERO)[0]
    with torch.no_grad():
        b_eval = torch.from_numpy(b_now[eval_idx]).float().to(device)
        chunks = []
        for i in range(0, len(b_eval), 500_000):
            c = b_eval[i : i + 500_000]
            xi_l = deform(c)
            c1 = se3_apply(xi_l, c)
            c2 = se3_apply(xi_global.to(device).unsqueeze(0).expand(len(c1), 6), c1)
            chunks.append(c2.cpu().numpy())
        b_def[eval_idx] = np.concatenate(chunks)
    # blend displacement by NN-distance-to-A: 1 near A's surface, 0 far away
    wgt = np.clip((BLEND_ZERO - d_b_to_a) / (BLEND_ZERO - BLEND_FULL), 0.0, 1.0)
    b_prev = b_now + wgt[:, None] * (b_def - b_now)
    mag = np.linalg.norm((b_prev - b_now)[eval_idx], axis=1)
    print(
        f"[A3] blended disp mm: median {np.median(mag)*1000:.1f} "
        f"p90 {np.percentile(mag,90)*1000:.1f} max {mag.max()*1000:.1f}"
    )

    merged_pts = np.concatenate([pts_a, b_prev])
    merged_col = np.concatenate([col_a, col_b])

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
    cc = merged_col[front][ok]
    order = np.argsort(-zz)
    img[vv[order], uu[order]] = cc[order]
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(
        f"{OUT_DIR}/merged_reproj.jpg", quality=92
    )
    print("[A3] wrote merged_reproj.jpg")


if __name__ == "__main__":
    main()
