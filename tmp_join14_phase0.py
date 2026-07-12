"""
2D check for the DeepRock1_3 + DeepRock1_4 rotation join.

DeepRock1_4's source image is the Midjourney outpaint of our 20-degree
homography reprojection (rot20_homography.png). Verify with RoMa that the
known region survived generation, and fit the exact scale+translation mapping
1_4-image pixels -> rot20-frame pixels (2944x1648, where the true camera is
A's camera yawed 20 degrees).

Outputs to /mnt/d/archive/DeepRock_join_34/phase0/.
"""

import json
import os

import numpy as np
import torch
from PIL import Image

IMG_R20 = "/mnt/d/archive/DeepRock_rot20/rot20_homography.png"
IMG_4 = "/mnt/d/archive/DeepRock1_4/ian_101_dark_granite_rocks_at_the_bottom_of_the_ocean._No_water_c7c03b65-038e-4b51-bdff-7d92be9d188a.png"
OUT_DIR = "/mnt/d/archive/DeepRock_join_34/phase0"

MATCH_LONG_SIDE = 1472
NUM_SAMPLES = 10000
CERTAINTY_THRESHOLD = 0.1
RANSAC_ITERS = 5000
RANSAC_INLIER_PX = 8.0


def load_image_tensor(path, long_side):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = long_side / max(w, h)
    img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1), scale


def fit_scale_translation_ransac(kpts_a, kpts_b, n_iters, inlier_px, rng):
    n = kpts_a.shape[0]
    best_inliers, best_count = np.zeros(n, bool), -1
    for _ in range(n_iters):
        i, j = rng.choice(n, size=2, replace=False)
        da, db = kpts_a[j] - kpts_a[i], kpts_b[j] - kpts_b[i]
        denom = float(da @ da)
        if denom < 1e-6:
            continue
        s = float(da @ db) / denom
        if not 0.5 < s < 2.0:
            continue
        t = kpts_b[i] - s * kpts_a[i]
        resid = np.linalg.norm(kpts_b - (s * kpts_a + t), axis=1)
        inl = resid < inlier_px
        if inl.sum() > best_count:
            best_count, best_inliers = int(inl.sum()), inl
    a, b = kpts_a[best_inliers], kpts_b[best_inliers]
    am, bm = a.mean(0), b.mean(0)
    ac, bc = a - am, b - bm
    s = float((ac * bc).sum() / (ac * ac).sum())
    t = bm - s * am
    resid = np.linalg.norm(kpts_b - (s * kpts_a + t), axis=1)
    return s, t, resid < inlier_px, resid


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # match only within the known (alpha=1) region of the rot20 frame
    img_r20, scale_r = load_image_tensor(IMG_R20, MATCH_LONG_SIDE)
    img_4, scale_4 = load_image_tensor(IMG_4, MATCH_LONG_SIDE)
    alpha = np.asarray(Image.open(IMG_R20).split()[-1], dtype=np.float32) / 255.0

    from models.roma_matcher import RoMaMatcherWrapper
    matcher = RoMaMatcherWrapper(device=device, version="v2")
    kpts_r, kpts_4, cert = matcher.match_images(
        img_r20.to(device), img_4.to(device),
        num_samples=NUM_SAMPLES, certainty_threshold=CERTAINTY_THRESHOLD)
    kpts_r = kpts_r.cpu().numpy() / scale_r  # full-res rot20 frame coords
    kpts_4 = kpts_4.cpu().numpy() / scale_4  # full-res 1_4 image coords
    cert = cert.cpu().numpy()

    # keep matches whose rot20-side pixel is in the known region
    ai = alpha[np.clip(kpts_r[:, 1].round().astype(int), 0, alpha.shape[0] - 1),
               np.clip(kpts_r[:, 0].round().astype(int), 0, alpha.shape[1] - 1)]
    keep = ai > 0.9
    kpts_r, kpts_4, cert = kpts_r[keep], kpts_4[keep], cert[keep]
    print(f"[j34-p0] {len(kpts_r)} matches in known region")

    rng = np.random.default_rng(0)
    # fit p_r20 = s * p_4 + t (mapping 1_4 pixels into the rot20 frame)
    s, t, inl, resid = fit_scale_translation_ransac(
        kpts_4, kpts_r, RANSAC_ITERS, RANSAC_INLIER_PX * 0.5, rng)
    med = np.median(resid[inl]) if inl.sum() else np.inf
    print(f"[j34-p0] p_r20 = {s:.5f} * p_4 + ({t[0]:.1f},{t[1]:.1f})  "
          f"inliers {inl.sum()}/{len(kpts_4)} ({100*inl.mean():.1f}%) "
          f"median resid {med:.2f}px")
    print(f"[j34-p0] size-ratio prediction: {2944/2912:.5f} (x), {1648/1632:.5f} (y)")

    json.dump(
        {
            "mapping": "p_rot20frame = s * p_img4 + t (full-res px)",
            "s": s, "t": [float(t[0]), float(t[1])],
            "num_matches": int(len(kpts_4)),
            "inlier_ratio": float(inl.mean()),
            "median_inlier_resid_px": float(med),
        },
        open(f"{OUT_DIR}/map_4_to_rot20.json", "w"), indent=2)
    np.savez(f"{OUT_DIR}/matches_known_region.npz",
             kpts_r20=kpts_r[inl], kpts_4=kpts_4[inl], certainty=cert[inl])
    print(f"[j34-p0] wrote map_4_to_rot20.json / matches_known_region.npz")


if __name__ == "__main__":
    main()
