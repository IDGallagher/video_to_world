"""Diagnostics for the A3 deformation: displacement stats + B-only splats."""

import json

import numpy as np
import torch

from models.deformation import DeformationGrid
from utils.geometry import se3_apply
from PIL import Image

A2_DIR = "/mnt/d/archive/DeepRock_join_23/phaseA2"
A3_DIR = "/mnt/d/archive/DeepRock_join_23/phaseA3"
XYZ_B = "/mnt/d/archive/DeepRock1_2/DeepRock1_2 - Cloud.xyz"
CAM_A = "/mnt/d/archive/DeepRock1_3/DeepRock1_3.pixal3d_camera.json"

ck = torch.load(f"{A3_DIR}/deform_checkpoint.pt", weights_only=False)
T = np.array(ck["T_b_to_a_rigid_4x4"])
d = np.loadtxt(XYZ_B)
pts_b, col_b = d[:, :3], d[:, 3:6] / 255.0
b_now = pts_b @ T[:3, :3].T + T[:3, 3]

dp = ck["deform_params"]
deform = DeformationGrid(
    ck["bbox_min"].cuda(), ck["bbox_max"].cuda(),
    min_res=dp["deform_min_res"], max_res=dp["deform_max_res"],
    num_levels=dp["deform_num_levels"], log2_hashmap_size=dp["deform_log2_hashmap_size"],
    n_neurons=dp["deform_n_neurons"], n_hidden_layers=dp["deform_n_hidden_layers"],
).cuda()
deform.load_state_dict(ck["deform_state_dict"])
deform.eval()
xi_g = ck["xi_global"].cuda()

inb = np.all(
    (b_now >= ck["bbox_min"].numpy()) & (b_now <= ck["bbox_max"].numpy()), axis=1
)
idx = np.where(inb)[0]
with torch.no_grad():
    c = torch.from_numpy(b_now[idx]).float().cuda()
    xi_l = deform(c)
    c1 = se3_apply(xi_l, c)
    c2 = se3_apply(xi_g.unsqueeze(0).expand(len(c1), 6), c1)
    disp = (c2 - c).cpu().numpy()
    omega = xi_l[:, :3].cpu().numpy()

mag = np.linalg.norm(disp, axis=1)
print(f"points in bbox: {len(idx)}")
print(f"disp mm: median {np.median(mag)*1000:.1f} p90 {np.percentile(mag,90)*1000:.1f} "
      f"p99 {np.percentile(mag,99)*1000:.1f} max {mag.max()*1000:.1f}")
print(f"|omega|: median {np.median(np.linalg.norm(omega,axis=1)):.4f} "
      f"p99 {np.percentile(np.linalg.norm(omega,axis=1),99):.4f}")
print(f"xi_global: {ck['xi_global'].numpy()}")

# B-only splat, rigid vs deformed+blended
b_def = b_now.copy()
b_def[idx] = c2.cpu().numpy()

cam = json.load(open(CAM_A))
fov, dist = cam["camera_angle_x"], cam["distance"]
w, h = 1200, 500
f_norm = 1.0 / (2.0 * np.tan(fov / 2.0))
K2 = np.array([[f_norm * w * 0.55, 0, w * 0.28], [0, f_norm * w * 0.55, h * 0.5], [0, 0, 1]])
origin = np.array([0.0, 0.0, -dist])
z = -origin / np.linalg.norm(origin); up = np.array([0.0, 1.0, 0.0])
x = np.cross(z, up); x /= np.linalg.norm(x); y = np.cross(z, x)
Rw = np.stack([x, y, z])


def splat(pts, cols, path):
    p_cam = (pts - origin) @ Rw.T
    front = p_cam[:, 2] > 1e-6
    uvz = p_cam[front]
    u = (K2[0, 0] * uvz[:, 0] / uvz[:, 2] + K2[0, 2]).round().astype(int)
    v = (K2[1, 1] * uvz[:, 1] / uvz[:, 2] + K2[1, 2]).round().astype(int)
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    img = np.zeros((h, w, 3), np.float32)
    uu, vv, zz = u[ok], v[ok], uvz[ok, 2]
    cc = cols[front][ok]
    order = np.argsort(-zz)
    img[vv[order], uu[order]] = cc[order]
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(path, quality=92)


splat(b_now, col_b, f"{A3_DIR}/diag_b_rigid.jpg")
splat(b_def, col_b, f"{A3_DIR}/diag_b_deformed.jpg")
print("wrote diag_b_rigid.jpg / diag_b_deformed.jpg")
