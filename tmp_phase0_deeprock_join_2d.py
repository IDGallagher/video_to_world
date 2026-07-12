"""
Phase 0 of the DeepRock1_3 + DeepRock1_2 join: measure the 2D relationship
between the two source images.

The images are expected to join side-by-side (1_2 to the right of 1_3) after
rescaling to a common size. We verify this and quantify it:

  1. Run RoMa dense matching between the two source images.
  2. RANSAC-fit a uniform scale + translation model mapping image A (1_3)
     pixels into image B (1_2) pixels:  p_b = s * p_a + t.
  3. Report the transform, inlier stats, and the overlap region in both
     images' pixel coordinates.
  4. Save a match visualization and an overlap composite for eyeballing.

Outputs go to /mnt/d/archive/DeepRock_join_23/phase0/.
"""

import json
import os

import numpy as np
import torch
from PIL import Image

IMG_A = "/mnt/d/archive/DeepRock1_3/ian_101_dark_granite_rocks_at_the_bottom_of_the_ocean._No_wat_47f6e59d-04f6-4fc4-9783-163353d64c86_3.png"
IMG_B = "/mnt/d/archive/DeepRock1_2/ian_101_dark_granite_rocks_at_the_bottom_of_the_ocean._No_water_bb4e9cc8-ea76-4677-adb8-0f08fff524ec.png"
OUT_DIR = "/mnt/d/archive/DeepRock_join_23/phase0"

MATCH_LONG_SIDE = 1472  # downscale for matching; results reported in full-res coords
NUM_SAMPLES = 10000
CERTAINTY_THRESHOLD = 0.1
RANSAC_ITERS = 5000
RANSAC_INLIER_PX = 8.0  # in matching-resolution pixels


def load_image_tensor(path: str, long_side: int) -> tuple[torch.Tensor, float]:
    """Load image as (3,H,W) float in [0,1], downscaled so max(H,W)==long_side.

    Returns the tensor and the scale factor from full-res to downscaled coords.
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = long_side / max(w, h)
    new_w, new_h = round(w * scale), round(h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1), scale


def fit_scale_translation_ransac(
    kpts_a: np.ndarray,  # (N,2)
    kpts_b: np.ndarray,  # (N,2)
    n_iters: int,
    inlier_px: float,
    rng: np.random.Generator,
) -> tuple[float, np.ndarray, np.ndarray]:
    """RANSAC fit of p_b = s * p_a + t (uniform scale, no rotation).

    Returns (s, t, inlier_mask).
    """
    n = kpts_a.shape[0]
    best_inliers = np.zeros(n, dtype=bool)
    best_count = -1

    for _ in range(n_iters):
        i, j = rng.choice(n, size=2, replace=False)
        da = kpts_a[j] - kpts_a[i]
        db = kpts_b[j] - kpts_b[i]
        denom = float(da @ da)
        if denom < 1e-6:
            continue
        # Least-squares scale for this pair under translation+scale model
        s = float(da @ db) / denom
        if s <= 0.1 or s >= 10.0:
            continue
        t = kpts_b[i] - s * kpts_a[i]
        resid = np.linalg.norm(kpts_b - (s * kpts_a + t), axis=1)
        inliers = resid < inlier_px
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers

    # Refine on inliers with least squares:
    # minimize sum || s*a + t - b ||^2  -> closed form
    a = kpts_a[best_inliers]
    b = kpts_b[best_inliers]
    a_mean = a.mean(axis=0)
    b_mean = b.mean(axis=0)
    a_c = a - a_mean
    b_c = b - b_mean
    s = float((a_c * b_c).sum() / (a_c * a_c).sum())
    t = b_mean - s * a_mean

    # One more inlier pass with refined model
    resid = np.linalg.norm(kpts_b - (s * kpts_a + t), axis=1)
    inliers = resid < inlier_px
    return s, t, inliers


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[phase0] device={device}")

    img_a, scale_a = load_image_tensor(IMG_A, MATCH_LONG_SIDE)
    img_b, scale_b = load_image_tensor(IMG_B, MATCH_LONG_SIDE)
    print(f"[phase0] A (1_3): {tuple(img_a.shape)} scale={scale_a:.4f}")
    print(f"[phase0] B (1_2): {tuple(img_b.shape)} scale={scale_b:.4f}")

    from models.roma_matcher import RoMaMatcherWrapper

    matcher = None
    for version, model_type in (("v2", "indoor"), ("v1", "outdoor")):
        try:
            matcher = RoMaMatcherWrapper(device=device, model_type=model_type, version=version)
            print(f"[phase0] using RoMa {version}")
            break
        except ImportError as e:
            print(f"[phase0] RoMa {version} unavailable: {e}")
    if matcher is None:
        raise SystemExit("No RoMa version available in this environment")

    kpts_a, kpts_b, certainty = matcher.match_images(
        img_a.to(device),
        img_b.to(device),
        num_samples=NUM_SAMPLES,
        certainty_threshold=CERTAINTY_THRESHOLD,
    )
    kpts_a = kpts_a.cpu().numpy()
    kpts_b = kpts_b.cpu().numpy()
    certainty = certainty.cpu().numpy()
    print(f"[phase0] {len(kpts_a)} matches after certainty >= {CERTAINTY_THRESHOLD}")
    if len(kpts_a) < 50:
        raise SystemExit("Too few matches - overlap may be tiny or images unrelated")

    rng = np.random.default_rng(0)
    s, t, inliers = fit_scale_translation_ransac(
        kpts_a, kpts_b, RANSAC_ITERS, RANSAC_INLIER_PX, rng
    )
    n_in = int(inliers.sum())
    resid = np.linalg.norm(kpts_b - (s * kpts_a + t), axis=1)
    print(
        f"[phase0] model (match-res): s={s:.4f} t=({t[0]:.1f},{t[1]:.1f}) "
        f"inliers={n_in}/{len(kpts_a)} ({100*n_in/len(kpts_a):.1f}%) "
        f"median inlier resid={np.median(resid[inliers]):.2f}px"
    )

    # Convert to full-resolution coordinates of each original image.
    # match coords: a_m = scale_a * a_full ; b_m = scale_b * b_full
    # b_m = s * a_m + t  ->  b_full = (s*scale_a/scale_b) * a_full + t/scale_b
    s_full = s * scale_a / scale_b
    t_full = t / scale_b

    wa, ha = Image.open(IMG_A).size
    wb, hb = Image.open(IMG_B).size

    # Footprint of image A mapped into B's pixel space
    corners_a = np.array([[0, 0], [wa, 0], [wa, ha], [0, ha]], dtype=np.float64)
    mapped = s_full * corners_a + t_full
    ax0, ay0 = mapped.min(axis=0)
    ax1, ay1 = mapped.max(axis=0)

    ox0, oy0 = max(ax0, 0), max(ay0, 0)
    ox1, oy1 = min(ax1, wb), min(ay1, hb)
    overlap_w = max(0.0, ox1 - ox0)
    overlap_h = max(0.0, oy1 - oy0)
    overlap_frac_b = (overlap_w * overlap_h) / (wb * hb)

    print(f"[phase0] A footprint in B: x[{ax0:.0f},{ax1:.0f}] y[{ay0:.0f},{ay1:.0f}]")
    print(
        f"[phase0] overlap in B coords: {overlap_w:.0f} x {overlap_h:.0f} px "
        f"({100*overlap_frac_b:.1f}% of B)"
    )

    result = {
        "image_a": IMG_A,
        "image_b": IMG_B,
        "model": "p_b = s * p_a + t (full-res pixel coords)",
        "scale": s_full,
        "translation": [float(t_full[0]), float(t_full[1])],
        "num_matches": int(len(kpts_a)),
        "num_inliers": n_in,
        "inlier_ratio": n_in / len(kpts_a),
        "median_inlier_residual_matchres_px": float(np.median(resid[inliers])),
        "a_footprint_in_b": [float(ax0), float(ay0), float(ax1), float(ay1)],
        "overlap_in_b": [float(ox0), float(oy0), float(ox1), float(oy1)],
        "overlap_fraction_of_b": float(overlap_frac_b),
        "match_long_side": MATCH_LONG_SIDE,
        "ransac_inlier_px": RANSAC_INLIER_PX,
    }
    with open(os.path.join(OUT_DIR, "join_2d.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"[phase0] wrote {OUT_DIR}/join_2d.json")

    # Save inlier matches for Phase A1 (full-res pixel coords in each image)
    np.savez(
        os.path.join(OUT_DIR, "inlier_matches_fullres.npz"),
        kpts_a=kpts_a[inliers] / scale_a,
        kpts_b=kpts_b[inliers] / scale_b,
        certainty=certainty[inliers],
    )
    print(f"[phase0] wrote {OUT_DIR}/inlier_matches_fullres.npz")

    # --- Visualization 1: side-by-side matches ---
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a_np = img_a.permute(1, 2, 0).numpy()
    b_np = img_b.permute(1, 2, 0).numpy()
    ha_m, wa_m = a_np.shape[:2]
    hb_m, wb_m = b_np.shape[:2]
    canvas = np.ones((max(ha_m, hb_m), wa_m + wb_m, 3), dtype=np.float32)
    canvas[:ha_m, :wa_m] = a_np
    canvas[:hb_m, wa_m:] = b_np

    fig, ax = plt.subplots(figsize=(24, 8))
    ax.imshow(canvas)
    idx = np.where(inliers)[0]
    show = idx[rng.choice(len(idx), size=min(150, len(idx)), replace=False)]
    for i in show:
        ax.plot(
            [kpts_a[i, 0], kpts_b[i, 0] + wa_m],
            [kpts_a[i, 1], kpts_b[i, 1]],
            "-", lw=0.5, alpha=0.6, color="lime",
        )
    ax.set_title(
        f"RoMa inlier matches (showing {len(show)}/{n_in}) | "
        f"s={s_full:.3f} t=({t_full[0]:.0f},{t_full[1]:.0f})px"
    )
    ax.axis("off")
    fig.savefig(os.path.join(OUT_DIR, "matches.jpg"), dpi=100, bbox_inches="tight")
    plt.close(fig)

    # --- Visualization 2: composite of A warped into B's canvas ---
    # Work at B's matching resolution for speed.
    s_m = s
    t_m = t
    comp = np.zeros((hb_m, wb_m, 3), dtype=np.float32)
    comp[:] = b_np * 0.5
    # Map every pixel of A into B space (forward splat at low res is fine here)
    ys, xs = np.mgrid[0:ha_m, 0:wa_m]
    xb = (s_m * xs + t_m[0]).round().astype(int)
    yb = (s_m * ys + t_m[1]).round().astype(int)
    valid = (xb >= 0) & (xb < wb_m) & (yb >= 0) & (yb < hb_m)
    comp[yb[valid], xb[valid]] += 0.5 * a_np[ys[valid], xs[valid]]
    Image.fromarray((np.clip(comp, 0, 1) * 255).astype(np.uint8)).save(
        os.path.join(OUT_DIR, "overlap_composite.jpg"), quality=92
    )
    print(f"[phase0] wrote matches.jpg and overlap_composite.jpg")


if __name__ == "__main__":
    main()
