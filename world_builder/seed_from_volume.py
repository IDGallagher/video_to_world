"""Prepare a Wan i2v seed frame from a static depth-volume view.

Takes a DepthImageVolume asset folder (cameras.json + color/depth PNGs),
picks one view, resizes it to the requested generation resolution, and
writes a meta JSON recording everything anchoring needs later:
asset path, view index, original/generation resolutions, and the
intrinsics scale factors implied by the resize.
"""
import argparse
import json
import os

import cv2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume_dir", required=True, help="Depth volume folder with cameras.json")
    ap.add_argument("--view", type=int, default=0, help="View index to seed from")
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    cams = json.load(open(os.path.join(args.volume_dir, "cameras.json")))
    frame = cams["frames"][args.view]
    color_path = os.path.join(args.volume_dir, frame["color_path"])
    img = cv2.imread(color_path, cv2.IMREAD_COLOR)
    assert img is not None, color_path
    oh, ow = img.shape[:2]

    seed = cv2.resize(img, (args.width, args.height), interpolation=cv2.INTER_AREA)
    os.makedirs(args.out_dir, exist_ok=True)
    seed_path = os.path.join(args.out_dir, "seed.png")
    cv2.imwrite(seed_path, seed)

    meta = {
        "volume_dir": os.path.abspath(args.volume_dir),
        "view": args.view,
        "orig_size": [ow, oh],
        "gen_size": [args.width, args.height],
        "sx": args.width / ow,
        "sy": args.height / oh,
        "view_frame": frame,
        "volume_intrinsics": {k: cams[k] for k in ("w", "h", "fl_x", "fl_y", "cx", "cy") if k in cams},
    }
    with open(os.path.join(args.out_dir, "seed_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(seed_path)


if __name__ == "__main__":
    main()
