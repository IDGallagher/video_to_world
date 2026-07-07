"""Anchor an exported reconstruction depth-volume into the seed asset's frame.

Inputs:
  --recon_div   folder written by export_depth_image_volume.py (cameras.json + pngs)
  --seed_meta   seed_meta.json written by seed_from_volume.py
  --out_div     output folder for the anchored sibling depth volume

Method (first-frame anchoring):
  1. The generated video's frame 0 IS the asset view recorded in seed_meta, so
     reconstruction frame 0 and the asset view see the same pixels.
  2. Scale: s = median(asset_depth / recon_depth) over pixels valid in both
     (asset depth resized to generation resolution, nearest-neighbor).
  3. Rescale recon world: all camera translations and near/far multiply by s.
     Depth PNG codes are unchanged because they encode relative to near/far.
  4. Rigid: T = C2W_asset_view @ inv(C2W_recon_frame0_scaled), applied to every
     recon camera, mapping the whole reconstruction into asset coordinates (cm).
"""
import argparse
import json
import os
import shutil

import cv2
import numpy as np


def decode_depth(path: str, near: float, far: float) -> np.ndarray:
    code = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    assert code is not None and code.dtype == np.uint16, path
    d = near + ((code.astype(np.float64) - 1.0) / 65534.0) * (far - near)
    d[code == 0] = 0.0
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon_div", required=True)
    ap.add_argument("--seed_meta", required=True)
    ap.add_argument("--out_div", required=True)
    args = ap.parse_args()

    meta = json.load(open(args.seed_meta))
    asset_dir = meta["volume_dir"]
    view = meta["view_frame"]
    gen_w, gen_h = meta["gen_size"]

    recon = json.load(open(os.path.join(args.recon_div, "cameras.json")))
    rframes = recon["frames"]
    r0 = rframes[0]

    # --- depths at generation resolution ---
    rd = decode_depth(os.path.join(args.recon_div, r0["file_path"]), r0["near"], r0["far"])
    if rd.shape[:2] != (gen_h, gen_w):
        rd = cv2.resize(rd, (gen_w, gen_h), interpolation=cv2.INTER_NEAREST)
    ad = decode_depth(os.path.join(asset_dir, view["file_path"]), view["near"], view["far"])
    ad = cv2.resize(ad, (gen_w, gen_h), interpolation=cv2.INTER_NEAREST)

    valid = (ad > 0) & (rd > 0)
    n_valid = int(valid.sum())
    assert n_valid > 1000, f"only {n_valid} shared valid depth pixels"
    ratios = ad[valid] / rd[valid]
    s = float(np.median(ratios))
    spread = float(np.percentile(ratios, 75) / np.percentile(ratios, 25))
    print(f"anchor: {n_valid} px, scale={s:.4f}, IQR ratio spread={spread:.3f}")

    # --- rigid alignment after scaling ---
    c2w_asset = np.array(view["transform_matrix"], dtype=np.float64)
    c2w_r0 = np.array(r0["transform_matrix"], dtype=np.float64)
    c2w_r0_scaled = c2w_r0.copy()
    c2w_r0_scaled[:3, 3] *= s
    T = c2w_asset @ np.linalg.inv(c2w_r0_scaled)

    # --- write anchored volume ---
    os.makedirs(args.out_div, exist_ok=True)
    out = dict(recon)
    out["coordinate_units"] = "centimeters"
    out["transform_translation_units"] = "centimeters"
    out["anchoring"] = {
        "asset_volume": asset_dir,
        "asset_view": meta["view"],
        "scale": s,
        "shared_valid_pixels": n_valid,
        "iqr_ratio_spread": spread,
    }
    new_frames = []
    for fr in rframes:
        c2w = np.array(fr["transform_matrix"], dtype=np.float64)
        c2w[:3, 3] *= s
        c2w = T @ c2w
        nf = dict(fr)
        nf["transform_matrix"] = c2w.tolist()
        nf["near"] = fr["near"] * s
        nf["far"] = fr["far"] * s
        new_frames.append(nf)
        for key in ("file_path", "color_path", "normal_path"):
            rel = fr.get(key)
            if rel:
                src = os.path.join(args.recon_div, rel)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(args.out_div, os.path.basename(rel)))
    out["frames"] = new_frames
    with open(os.path.join(args.out_div, "cameras.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("anchored volume written:", args.out_div)


if __name__ == "__main__":
    main()
