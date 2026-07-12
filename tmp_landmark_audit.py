"""
Landmark audit of the DeepRock1_3 / DeepRock1_2 join.

Picks ~12 well-spread, high-certainty RoMa matches in the image overlap,
verifies them visually (numbered markers on both source images), lifts each
to 3D on both meshes, and reports:

  - pairwise-distance ratio matrix stats  -> scale, with no transform assumed
  - Umeyama similarity on the landmarks   -> scale + rotation decomposed into
    pitch (x), yaw (y), roll (z = front axis) in the shared camera frame
  - the same decomposition for the transforms used so far (A2 rigid+sim,
    option-1 residual ICP) to check for accidental roll
  - per-landmark 3D residuals under each alignment

Outputs to /mnt/d/archive/DeepRock_join_23/landmark_audit/.
"""

import json
import os

import numpy as np
import trimesh
from PIL import Image

PHASE0_DIR = "/mnt/d/archive/DeepRock_join_23/phase0"
A2_DIR = "/mnt/d/archive/DeepRock_join_23/phaseA2"
OPT1_DIR = "/mnt/d/archive/DeepRock_join_23/option1_unproject"
OUT_DIR = "/mnt/d/archive/DeepRock_join_23/landmark_audit"
ROOT_A = "/mnt/d/archive/DeepRock1_3"
ROOT_B = "/mnt/d/archive/DeepRock1_2"

N_LANDMARKS = 16


def load_camera(cam_json_path, w, h):
    cam = json.load(open(cam_json_path))
    f_px = w / (2.0 * np.tan(cam["camera_angle_x"] / 2.0))
    origin = np.array([0.0, 0.0, -cam["distance"]])
    x = np.array([-1.0, 0.0, 0.0])
    y = np.array([0.0, -1.0, 0.0])
    z = np.array([0.0, 0.0, 1.0])
    return {"f": f_px, "cx": 0.5 * w, "cy": 0.5 * h,
            "R_w2c": np.stack([x, y, z]), "origin": origin, "w": w, "h": h}


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


def euler_xyz_deg(R):
    """Decompose R (world frame, camera looks along +z) into pitch/yaw/roll."""
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    pitch = np.degrees(np.arctan2(-R[2, 1], R[2, 2]) if sy > 1e-6 else 0)
    pitch = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    yaw = np.degrees(np.arctan2(-R[2, 0], sy))
    roll = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    return pitch, yaw, roll


def axis_angle(R):
    ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    n = np.linalg.norm(axis)
    return ang, axis / n if n > 1e-9 else axis


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


def annotate(img_path, kpts, out_path, half=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = Image.open(img_path)
    scale = 1200 / img.width
    img = img.resize((1200, round(img.height * scale)))
    fig, ax = plt.subplots(figsize=(14, 14 * img.height / img.width))
    ax.imshow(img)
    for i, (u, v) in enumerate(kpts):
        ax.plot(u * scale, v * scale, "o", ms=10, mfc="none", mec="lime", mew=2)
        ax.annotate(str(i), (u * scale + 12, v * scale - 8), color="lime",
                    fontsize=14, fontweight="bold")
    ax.axis("off")
    fig.savefig(out_path, dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    join2d = json.load(open(f"{PHASE0_DIR}/join_2d.json"))
    img_a_path, img_b_path = join2d["image_a"], join2d["image_b"]
    wa, ha = Image.open(img_a_path).size
    wb, hb = Image.open(img_b_path).size
    cam_a = load_camera(f"{ROOT_A}/DeepRock1_3.pixal3d_camera.json", wa, ha)
    cam_b = load_camera(f"{ROOT_B}/DeepRock1_2.pixal3d_camera.json", wb, hb)

    m = np.load(f"{PHASE0_DIR}/inlier_matches_fullres.npz")
    kpts_a, kpts_b, cert = m["kpts_a"], m["kpts_b"], m["certainty"]

    mesh_a = trimesh.load(f"{ROOT_A}/DeepRock1_3.glb", force="mesh")
    mesh_b = trimesh.load(f"{ROOT_B}/DeepRock1_2.glb", force="mesh")

    # --- select spread-out, depth-stable landmarks ---
    # bin A-side keypoints over a grid covering the match bounding box,
    # keep the 3 highest-certainty candidates per bin
    x0, x1 = kpts_a[:, 0].min(), kpts_a[:, 0].max()
    y0, y1 = kpts_a[:, 1].min(), kpts_a[:, 1].max()
    gx = np.clip(((kpts_a[:, 0] - x0) / (x1 - x0) * 8).astype(int), 0, 7)
    gy = np.clip(((kpts_a[:, 1] - y0) / (y1 - y0) * 5).astype(int), 0, 4)
    chosen = []
    for b in range(40):
        idx = np.where(gx + 8 * gy == b)[0]
        if len(idx):
            top = idx[np.argsort(cert[idx])[::-1][:3]]
            chosen.extend(top)
    chosen = np.array(chosen)

    # depth stability: raycast a small pixel cross around each candidate and
    # require consistent hit distance (rejects silhouette edges)
    def stable(mesh, cam, uv):
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
                ok &= hit & (np.abs(t - tref) < 0.03)
        return ok

    ok = stable(mesh_a, cam_a, kpts_a[chosen]) & stable(mesh_b, cam_b, kpts_b[chosen])
    chosen = chosen[ok]
    # thin to at most one per bin again, keep spatial spread
    seen, keep = set(), []
    for i in chosen:
        b = (gx[i], gy[i])
        if b not in seen:
            seen.add(b)
            keep.append(i)
    chosen = np.array(keep)[:N_LANDMARKS]
    print(f"[audit] {len(chosen)} stable landmarks")

    la, lb = kpts_a[chosen], kpts_b[chosen]
    pa, _ = raycast(mesh_a, cam_a, la)
    pb, _ = raycast(mesh_b, cam_b, lb)

    annotate(img_a_path, la, f"{OUT_DIR}/landmarks_a.jpg")
    annotate(img_b_path, lb, f"{OUT_DIR}/landmarks_b.jpg")

    # --- pairwise distance ratios (transform-free scale) ---
    n = len(chosen)
    ratios = []
    for i in range(n):
        for j in range(i + 1, n):
            da = np.linalg.norm(pa[i] - pa[j])
            db = np.linalg.norm(pb[i] - pb[j])
            if db > 0.02:
                ratios.append(da / db)
    ratios = np.array(ratios)
    print(f"[audit] pairwise |A|/|B| ratio: median {np.median(ratios):.4f} "
          f"iqr {np.percentile(ratios,25):.4f}..{np.percentile(ratios,75):.4f} "
          f"min {ratios.min():.3f} max {ratios.max():.3f}")

    # --- landmark Umeyama: scale + pitch/yaw/roll ---
    s, R, t = umeyama(pb, pa)
    pitch, yaw, roll = euler_xyz_deg(R)
    resid = np.linalg.norm(pa - (s * pb @ R.T + t), axis=1)
    print(f"[audit] landmark similarity B->A: scale {s:.4f}, "
          f"pitch {pitch:+.2f} yaw {yaw:+.2f} roll {roll:+.2f} deg")
    print(f"[audit] landmark residuals mm: " +
          " ".join(f"{r*1000:.0f}" for r in resid) +
          f" (median {np.median(resid)*1000:.1f})")

    # --- decompose the transforms we actually used ---
    T_a2 = np.array(json.load(open(f"{A2_DIR}/transform_b_to_a_refined.json"))["T_4x4"])
    s_a2 = np.cbrt(np.linalg.det(T_a2[:3, :3]))
    p2, y2, r2 = euler_xyz_deg(T_a2[:3, :3] / s_a2)
    print(f"[audit] A2 flat-join transform: scale {s_a2:.4f}, "
          f"pitch {p2:+.2f} yaw {y2:+.2f} roll {r2:+.2f} deg")

    opt1 = json.load(open(f"{OPT1_DIR}/metrics.json"))
    R_icp = np.array(opt1["R_icp"])
    p1, y1, r1 = euler_xyz_deg(R_icp)
    ang, ax = axis_angle(R_icp)
    print(f"[audit] opt1 residual ICP rotation: {ang:.2f} deg about axis "
          f"[{ax[0]:+.2f} {ax[1]:+.2f} {ax[2]:+.2f}] "
          f"(pitch {p1:+.2f} yaw {y1:+.2f} roll {r1:+.2f}) trans "
          f"{np.linalg.norm(opt1['t_icp'])*1000:.0f}mm")

    # --- per-landmark residuals under each alignment ---
    b_a2 = pb @ T_a2[:3, :3].T + T_a2[:3, 3]
    r_a2 = np.linalg.norm(pa - b_a2, axis=1)
    # option 1 mapping: along-ray transfer + residual ICP
    s2d, t2d = join2d["scale"], np.array(join2d["translation"])
    uv_a_from_b = (lb - t2d) / s2d
    t_b = np.linalg.norm(pb - cam_b["origin"], axis=1)
    d_a = rays(cam_a, uv_a_from_b)
    sigma = opt1["sigma"]
    b_o1 = cam_a["origin"] + sigma * t_b[:, None] * d_a
    b_o1 = b_o1 @ R_icp.T + np.array(opt1["t_icp"])
    r_o1 = np.linalg.norm(pa - b_o1, axis=1)
    print("[audit] per-landmark residual mm  (A2-flat | opt1):")
    for i in range(n):
        print(f"    #{i}: {r_a2[i]*1000:7.1f} | {r_o1[i]*1000:7.1f}")
    print(f"[audit] medians: A2 {np.median(r_a2)*1000:.1f}mm | "
          f"opt1 {np.median(r_o1)*1000:.1f}mm")

    json.dump(
        {
            "landmarks_a_px": la.tolist(),
            "landmarks_b_px": lb.tolist(),
            "pa": pa.tolist(), "pb": pb.tolist(),
            "pairwise_ratio_median": float(np.median(ratios)),
            "pairwise_ratio_iqr": [float(np.percentile(ratios, 25)),
                                   float(np.percentile(ratios, 75))],
            "landmark_similarity": {"scale": s, "pitch_deg": pitch,
                                    "yaw_deg": yaw, "roll_deg": roll},
            "residuals_mm": {"a2_flat": (r_a2 * 1000).tolist(),
                             "opt1": (r_o1 * 1000).tolist(),
                             "landmark_umeyama": (resid * 1000).tolist()},
        },
        open(f"{OUT_DIR}/audit.json", "w"), indent=2)
    print(f"[audit] wrote audit.json, landmarks_a.jpg, landmarks_b.jpg")


if __name__ == "__main__":
    main()
