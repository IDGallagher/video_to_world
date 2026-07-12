"""
Phase A1 of the DeepRock1_3 + DeepRock1_2 join: coarse 3D similarity alignment.

Uses the Phase 0 inlier 2D matches (tmp_phase0_deeprock_join_2d.py) and lifts
them to 3D on each asset's GLB by raycasting from the Pixal3D source camera:

  camera origin (0, 0, -distance) in GLB space, looking at the origin, up +Y,
  normalized intrinsics fx = fy = 1 / (2 tan(camera_angle_x / 2)) scaled by
  image width / height respectively (matches app.py
  camera_from_source_params_for_glb + render_utils.proj_camera_to_render_params).

Then RANSAC + Umeyama solves the 7-DoF similarity mapping DeepRock1_2 GLB
space into DeepRock1_3 GLB space.

Outputs to /mnt/d/archive/DeepRock_join_23/phaseA1/:
  - camera_check_{a,b}.jpg   raycast render vs source image (convention sanity check)
  - similarity_b_to_a.json   scale / R / t + diagnostics
  - merged_preview.ply       1_3 cloud + transformed 1_2 cloud (with colors)
  - merged_reproj.jpg        merged cloud splatted through A's source camera
"""

import json
import os

import numpy as np
import trimesh
from PIL import Image

ROOT_A = "/mnt/d/archive/DeepRock1_3"
ROOT_B = "/mnt/d/archive/DeepRock1_2"
GLB_A = f"{ROOT_A}/DeepRock1_3.glb"
GLB_B = f"{ROOT_B}/DeepRock1_2.glb"
CAM_A = f"{ROOT_A}/DeepRock1_3.pixal3d_camera.json"
CAM_B = f"{ROOT_B}/DeepRock1_2.pixal3d_camera.json"
XYZ_A = f"{ROOT_A}/DeepRock1_3 - Cloud.xyz"
XYZ_B = f"{ROOT_B}/DeepRock1_2 - Cloud.xyz"
PHASE0_DIR = "/mnt/d/archive/DeepRock_join_23/phase0"
OUT_DIR = "/mnt/d/archive/DeepRock_join_23/phaseA1"

CHECK_RENDER_W = 368
RANSAC_ITERS = 4000


def load_camera(cam_json_path: str, img_path: str):
    with open(cam_json_path) as f:
        cam = json.load(f)
    w, h = Image.open(img_path).size
    fov = cam["camera_angle_x"]
    dist = cam["distance"]
    f_norm = 1.0 / (2.0 * np.tan(fov / 2.0))
    # Square pixels: camera_angle_x is derived from a pixel focal length
    # (app.py: camera_angle_x = 2*atan(width/(2*fx))), so fy_px == fx_px.
    K = np.array(
        [
            [f_norm * w, 0.0, 0.5 * w],
            [0.0, f_norm * w, 0.5 * h],
            [0.0, 0.0, 1.0],
        ]
    )
    origin = np.array([0.0, 0.0, -dist])
    target = np.zeros(3)
    up = np.array([0.0, 1.0, 0.0])
    # OpenCV look-at: z forward to target, y down
    z = target - origin
    z = z / np.linalg.norm(z)
    x = np.cross(z, up)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    R_w2c = np.stack([x, y, z], axis=0)
    return {"K": K, "R_w2c": R_w2c, "origin": origin, "w": w, "h": h}


def pixel_rays(cam: dict, uv: np.ndarray) -> np.ndarray:
    """(N,2) pixel coords -> (N,3) world ray directions (unnormalized ok)."""
    K, R = cam["K"], cam["R_w2c"]
    d_cam = np.stack(
        [
            (uv[:, 0] - K[0, 2]) / K[0, 0],
            (uv[:, 1] - K[1, 2]) / K[1, 1],
            np.ones(len(uv)),
        ],
        axis=1,
    )
    return d_cam @ R  # R^T applied from the right


def raycast(mesh: trimesh.Trimesh, cam: dict, uv: np.ndarray):
    """Raycast pixels; returns (points (N,3), hit_mask (N,), tri_index (N,))."""
    dirs = pixel_rays(cam, uv)
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    origins = np.tile(cam["origin"], (len(uv), 1))
    locs, ray_idx, tri_idx = mesh.ray.intersects_location(
        origins, dirs, multiple_hits=False
    )
    pts = np.zeros((len(uv), 3))
    hit = np.zeros(len(uv), dtype=bool)
    tris = np.full(len(uv), -1, dtype=int)
    pts[ray_idx] = locs
    hit[ray_idx] = True
    tris[ray_idx] = tri_idx
    return pts, hit, tris


def camera_check_render(mesh: trimesh.Trimesh, cam: dict, src_img_path: str, out_path: str):
    """Low-res normal-shaded raycast next to the source image."""
    w = CHECK_RENDER_W
    h = round(cam["h"] * w / cam["w"])
    us, vs = np.meshgrid(
        (np.arange(w) + 0.5) * cam["w"] / w,
        (np.arange(h) + 0.5) * cam["h"] / h,
    )
    uv = np.stack([us.ravel(), vs.ravel()], axis=1)
    _, hit, tris = raycast(mesh, cam, uv)
    shade = np.zeros(len(uv))
    dirs = pixel_rays(cam, uv)
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    n = mesh.face_normals[tris[hit]]
    shade[hit] = np.abs((n * -dirs[hit]).sum(axis=1))
    render = shade.reshape(h, w)

    src = np.asarray(Image.open(src_img_path).convert("L").resize((w, h))) / 255.0
    combo = np.concatenate([src, render], axis=1)
    Image.fromarray((np.clip(combo, 0, 1) * 255).astype(np.uint8)).save(out_path)
    return hit.reshape(h, w).mean()


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """Least-squares similarity dst ~= s * R @ src + t."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    cov = dc.T @ sc / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (sc**2).sum() / len(src)
    s = float(np.trace(np.diag(D) @ S) / var_s) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def ransac_umeyama(src, dst, thresh, iters, rng):
    n = len(src)
    best_inl, best_cnt = np.zeros(n, bool), -1
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        try:
            s, R, t = umeyama(src[idx], dst[idx])
        except np.linalg.LinAlgError:
            continue
        if s <= 0.2 or s >= 5.0:
            continue
        resid = np.linalg.norm(dst - (s * (src @ R.T) + t), axis=1)
        inl = resid < thresh
        cnt = int(inl.sum())
        if cnt > best_cnt:
            best_cnt, best_inl = cnt, inl
    s, R, t = umeyama(src[best_inl], dst[best_inl])
    resid = np.linalg.norm(dst - (s * (src @ R.T) + t), axis=1)
    inl = resid < thresh
    s, R, t = umeyama(src[inl], dst[inl])
    resid = np.linalg.norm(dst - (s * (src @ R.T) + t), axis=1)
    return s, R, t, resid < thresh, resid


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cam_a = load_camera(CAM_A, json.load(open(f"{PHASE0_DIR}/join_2d.json"))["image_a"])
    cam_b = load_camera(CAM_B, json.load(open(f"{PHASE0_DIR}/join_2d.json"))["image_b"])
    print(f"[A1] cam A: {cam_a['w']}x{cam_a['h']} origin={cam_a['origin']}")
    print(f"[A1] cam B: {cam_b['w']}x{cam_b['h']} origin={cam_b['origin']}")

    mesh_a = trimesh.load(GLB_A, force="mesh")
    mesh_b = trimesh.load(GLB_B, force="mesh")
    print(f"[A1] mesh A: {len(mesh_a.vertices)} verts; mesh B: {len(mesh_b.vertices)} verts")

    src_a = json.load(open(f"{PHASE0_DIR}/join_2d.json"))["image_a"]
    src_b = json.load(open(f"{PHASE0_DIR}/join_2d.json"))["image_b"]
    cov_a = camera_check_render(mesh_a, cam_a, src_a, f"{OUT_DIR}/camera_check_a.jpg")
    cov_b = camera_check_render(mesh_b, cam_b, src_b, f"{OUT_DIR}/camera_check_b.jpg")
    print(f"[A1] camera check renders written (hit coverage A={cov_a:.2f} B={cov_b:.2f})")

    m = np.load(f"{PHASE0_DIR}/inlier_matches_fullres.npz")
    kpts_a, kpts_b = m["kpts_a"], m["kpts_b"]
    print(f"[A1] lifting {len(kpts_a)} matches")

    pts_a, hit_a, _ = raycast(mesh_a, cam_a, kpts_a)
    pts_b, hit_b, _ = raycast(mesh_b, cam_b, kpts_b)
    both = hit_a & hit_b
    print(f"[A1] hits: A={hit_a.mean():.2f} B={hit_b.mean():.2f} both={both.sum()}")

    pa, pb = pts_a[both], pts_b[both]
    diag_a = np.linalg.norm(mesh_a.bounds[1] - mesh_a.bounds[0])
    thresh = 0.02 * diag_a
    rng = np.random.default_rng(0)
    s, R, t, inl, resid = ransac_umeyama(pb, pa, thresh, RANSAC_ITERS, rng)
    print(
        f"[A1] similarity B->A: s={s:.4f} inliers={inl.sum()}/{len(pb)} "
        f"({100*inl.mean():.1f}%) thresh={thresh*1000:.1f}mm "
        f"median inlier resid={np.median(resid[inl])*1000:.1f}mm "
        f"median all resid={np.median(resid)*1000:.1f}mm"
    )
    euler = trimesh.transformations.euler_from_matrix(
        np.block([[R, np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]])
    )
    print(f"[A1] rotation (deg): {np.degrees(euler)}")
    print(f"[A1] translation (m): {t}")

    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = t
    out = {
        "description": "p_a = s * R @ p_b + t maps DeepRock1_2 GLB space into DeepRock1_3 GLB space (meters)",
        "scale": s,
        "R": R.tolist(),
        "t": t.tolist(),
        "T_4x4": T.tolist(),
        "rotation_euler_deg": list(np.degrees(euler)),
        "num_pairs": int(len(pb)),
        "num_inliers": int(inl.sum()),
        "inlier_thresh_m": thresh,
        "median_inlier_residual_m": float(np.median(resid[inl])),
        "median_all_residual_m": float(np.median(resid)),
    }
    with open(f"{OUT_DIR}/similarity_b_to_a.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[A1] wrote similarity_b_to_a.json")

    # Merged preview cloud: A as-is + B transformed
    xyz_a = np.loadtxt(XYZ_A)
    xyz_b = np.loadtxt(XYZ_B)
    pb_t = s * (xyz_b[:, :3] @ R.T) + t
    merged_pts = np.concatenate([xyz_a[:, :3], pb_t])
    merged_col = np.concatenate([xyz_a[:, 3:6], xyz_b[:, 3:6]]).astype(np.uint8)
    cloud = trimesh.PointCloud(merged_pts, colors=merged_col)
    cloud.export(f"{OUT_DIR}/merged_preview.ply")
    print(f"[A1] wrote merged_preview.ply ({len(merged_pts)} pts)")

    # Splat merged cloud through a widened version of A's camera (see both assets)
    w, h = 1200, 500
    K = cam_a["K"].copy()
    # widen: keep fy, set fx for ~2x horizontal FOV coverage, principal point right of center
    K2 = np.array(
        [
            [K[0, 0] * w / cam_a["w"] * 0.55, 0, w * 0.28],
            [0, K[1, 1] * h / cam_a["h"] * 0.55, h * 0.5],
            [0, 0, 1],
        ]
    )
    Rw, o = cam_a["R_w2c"], cam_a["origin"]
    p_cam = (merged_pts - o) @ Rw.T
    front = p_cam[:, 2] > 1e-6
    uvz = p_cam[front]
    u = (K2[0, 0] * uvz[:, 0] / uvz[:, 2] + K2[0, 2]).round().astype(int)
    v = (K2[1, 1] * uvz[:, 1] / uvz[:, 2] + K2[1, 2]).round().astype(int)
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    img = np.zeros((h, w, 3), np.float32)
    zbuf = np.full((h, w), np.inf)
    uu, vv, zz = u[ok], v[ok], uvz[ok, 2]
    cc = merged_col[front][ok] / 255.0
    order = np.argsort(-zz)  # far first, near overwrites
    img[vv[order], uu[order]] = cc[order]
    Image.fromarray((img * 255).astype(np.uint8)).save(f"{OUT_DIR}/merged_reproj.jpg", quality=92)
    print(f"[A1] wrote merged_reproj.jpg")


if __name__ == "__main__":
    main()
