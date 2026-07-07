"""Prepare Wan2GP VACE v0.3 settings, download scripts, and launch scripts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.parse import urlparse

import cv2


PHASE_DEFAULTS = {
    "num_inference_steps": 8,
    "guidance_phases": 3,
    "guidance_scale": 3.5,
    "guidance2_scale": 1,
    "guidance3_scale": 1,
    "switch_threshold": 965,
    "switch_threshold2": 800,
    "model_switch_phase": 2,
    "flow_shift": 3,
    "sample_solver": "euler",
}


DEFAULT_PROMPT = (
    "A clean fog-free continuation of the same dark angular granite rock asset on the ocean floor. "
    "Sharp fractured black and charcoal stone, matching the existing geometry, material, scale, "
    "and neutral lighting. The new area continues naturally from the visible DeepRock formation. "
    "No rounded pebbles, no sand beach, no plants, no coral, no blue water tint."
)


DEFAULT_NEGATIVE = (
    "rounded pebbles, river stones, beach sand, coral, plants, seaweed, fish, submarine, "
    "blue fog, water tint, blur, distortion, warping, morphing, flicker, text, watermark"
)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _write_text_lf(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _to_wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    rest = resolved.as_posix()[3:] if resolved.as_posix()[1:3] == ":/" else resolved.as_posix()
    if drive:
        return f"/mnt/{drive}/{rest}"
    return rest


def _filename_from_url(url: str) -> str:
    return os.path.basename(urlparse(url).path)


def _select_url(urls: list[str], precision: str) -> str:
    matches = [url for url in urls if precision in _filename_from_url(url)]
    if not matches:
        raise ValueError(f"No URL matching precision '{precision}' in {urls}")
    return matches[0]


def _model_urls(wan2gp_dir: Path, precision: str) -> tuple[list[str], list[str]]:
    t2v = _load_json(wan2gp_dir / "defaults" / "t2v_2_2.json")
    vace = _load_json(wan2gp_dir / "defaults" / "vace_14B.json")
    vace_lightning = _load_json(wan2gp_dir / "defaults" / "vace_14B_lightning_3p_2_2.json")

    high_url = _select_url(t2v["model"]["URLs"], precision)
    low_url = _select_url(t2v["model"]["URLs2"], precision)
    module_url = _select_url(vace["model"]["modules"][0], precision)
    lora_urls = list(vace_lightning["model"]["loras"])
    return [high_url, low_url, module_url], lora_urls


def _copy_image_ref(src: Path, dst: Path, *, width: int, height: int) -> None:
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(src)
    resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(dst), resized)


def _prepare_image_refs(
    volume_dir: Path,
    cameras: dict,
    out_dir: Path,
    *,
    source_view: int,
    views: list[int] | None,
    width: int,
    height: int,
) -> list[Path]:
    image_ref_dir = out_dir / "image_refs"
    image_ref_dir.mkdir(parents=True, exist_ok=True)
    if views is None:
        candidates = [idx for idx in range(len(cameras["frames"])) if idx != source_view]
        views = candidates[:4]
    refs: list[Path] = []
    for view in views:
        frame = cameras["frames"][view]
        src = volume_dir / os.path.basename(frame["color_path"])
        dst = image_ref_dir / f"view_{view:04d}.png"
        _copy_image_ref(src, dst, width=width, height=height)
        refs.append(dst)
    return refs


def _write_download_script(
    path: Path,
    *,
    wan2gp_dir: Path,
    ckpt_urls: list[str],
    lora_urls: list[str],
) -> None:
    ckpt_dir = _to_wsl_path(wan2gp_dir / "ckpts")
    lora_dir = _to_wsl_path(wan2gp_dir / "loras")
    lines = [
        "#!/usr/bin/env bash",
        "set -u",
        f"mkdir -p {ckpt_dir!r} {lora_dir!r}",
        "download_one() {",
        "  local url=\"$1\"",
        "  local dir=\"$2\"",
        "  local name",
        "  name=\"$(basename \"${url%%\\?*}\")\"",
        "  cd \"$dir\" || exit 1",
        "  until wget -c --tries=0 --retry-connrefused --timeout=60 \"$url\"; do",
        "    echo \"retrying $name in 10s...\"",
        "    sleep 10",
        "  done",
        "}",
    ]
    for url in ckpt_urls:
        lines.append(f"download_one {url!r} {ckpt_dir!r}")
    for url in lora_urls:
        lines.append(f"download_one {url!r} {lora_dir!r}")
    lines.append("echo ALL_VACE_DOWNLOADS_COMPLETE")
    _write_text_lf(path, "\n".join(lines) + "\n")


def _write_download_launch_script(out_dir: Path, *, download_script: Path) -> None:
    launch_ps1 = out_dir / "launch_download_vace_wsl.ps1"
    download_wsl = _to_wsl_path(download_script)
    out_wsl = _to_wsl_path(out_dir)
    _write_text_lf(
        launch_ps1,
        (
            "$ErrorActionPreference = 'Stop'\n"
            "wsl -e bash -lc "
            f"'cd {out_wsl} && chmod +x {download_wsl} && "
            f"(setsid nohup {download_wsl} > download.log 2>&1 < /dev/null & "
            "echo $! > download.pid; disown); sleep 5; "
            "cat download.pid; ps -p $(cat download.pid) -o pid,stat,cmd --no-headers || true'\n"
        ),
    )


def _write_run_scripts(out_dir: Path, *, wan2gp_dir: Path, settings_path: Path) -> None:
    run_sh = out_dir / "run_vace.sh"
    out_wsl = _to_wsl_path(out_dir / "vace_out")
    log_wsl = _to_wsl_path(out_dir / "gen.log")
    settings_wsl = _to_wsl_path(settings_path)
    wan_wsl = _to_wsl_path(wan2gp_dir)
    _write_text_lf(
        run_sh,
        "\n".join(
            [
                "#!/usr/bin/env bash",
                f"cd {wan_wsl!r} || exit 1",
                "export PYTHONUNBUFFERED=1",
                "export TORCHDYNAMO_DISABLE=1",
                f"mkdir -p {out_wsl!r}",
                (
                    f"setsid nohup ./env_conda/bin/python3 wgp.py --process {settings_wsl!r} "
                    f"--output-dir {out_wsl!r} --profile 4 --perc-reserved-mem-max 0.25 "
                    f"--attention sage > {log_wsl!r} 2>&1 < /dev/null &"
                ),
                "pid=$!",
                "disown",
                "echo \"$pid\"",
            ]
        )
        + "\n",
    )
    launch_ps1 = out_dir / "launch_vace_wsl.ps1"
    _write_text_lf(
        launch_ps1,
        (
            "$ErrorActionPreference = 'Stop'\n"
            f"wsl -e bash -lc \"chmod +x '{_to_wsl_path(run_sh)}'; '{_to_wsl_path(run_sh)}'; sleep 5\"\n"
        ),
    )


def _inventory(wan2gp_dir: Path, urls: list[str], lora_urls: list[str]) -> list[dict]:
    rows: list[dict] = []
    for url in urls:
        path = wan2gp_dir / "ckpts" / _filename_from_url(url)
        rows.append({"kind": "ckpt", "path": str(path), "present": path.is_file(), "bytes": path.stat().st_size if path.is_file() else 0})
    for url in lora_urls:
        path = wan2gp_dir / "loras" / _filename_from_url(url)
        rows.append({"kind": "lora", "path": str(path), "present": path.is_file(), "bytes": path.stat().st_size if path.is_file() else 0})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare a Wan2GP VACE run from rendered conditioning assets.")
    ap.add_argument("--conditioning_dir", required=True)
    ap.add_argument("--volume_dir", default=r"D:\archive\DeepRock1\DeepRock1_depth_volume")
    ap.add_argument("--wan2gp_dir", default=r"D:\dev\Wan2GP")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--precision", default="quanto_mbf16_int8")
    ap.add_argument("--source_view", type=int, default=0)
    ap.add_argument("--image_ref_views", default=None, help="Comma-separated source color view indices. Defaults to four non-source views.")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--negative_prompt", default=DEFAULT_NEGATIVE)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    conditioning_dir = Path(args.conditioning_dir)
    volume_dir = Path(args.volume_dir)
    wan2gp_dir = Path(args.wan2gp_dir)
    out_dir = Path(args.out_dir) if args.out_dir else conditioning_dir / "vace"
    out_dir.mkdir(parents=True, exist_ok=True)

    cond_cameras = _load_json(conditioning_dir / "cameras.json")
    source_cameras = _load_json(volume_dir / "cameras.json")
    width = int(cond_cameras["w"])
    height = int(cond_cameras["h"])
    video_length = int(len(cond_cameras["frames"]))
    ref_views = None
    if args.image_ref_views:
        ref_views = [int(part.strip()) for part in args.image_ref_views.split(",") if part.strip()]
    refs = _prepare_image_refs(
        volume_dir,
        source_cameras,
        out_dir,
        source_view=int(args.source_view),
        views=ref_views,
        width=width,
        height=height,
    )

    ckpt_urls, lora_urls = _model_urls(wan2gp_dir, str(args.precision))
    settings = {
        "model_type": "vace_14B_lightning_3p_2_2",
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "video_prompt_type": "VAKI",
        "video_guide": _to_wsl_path(conditioning_dir / "video_guide.mp4"),
        "video_mask": _to_wsl_path(conditioning_dir / "video_mask.mp4"),
        "image_refs": [_to_wsl_path(path) for path in refs],
        "remove_background_images_ref": 0,
        "force_fps": "control",
        "mask_expand": 0,
        "resolution": f"{width}x{height}",
        "video_length": video_length,
        "seed": int(args.seed),
        **PHASE_DEFAULTS,
    }
    settings_path = out_dir / "vace_settings.json"
    _write_json(settings_path, settings)
    download_script = out_dir / "dl_vace_lightning.sh"
    _write_download_script(download_script, wan2gp_dir=wan2gp_dir, ckpt_urls=ckpt_urls, lora_urls=lora_urls)
    _write_download_launch_script(out_dir, download_script=download_script)
    _write_run_scripts(out_dir, wan2gp_dir=wan2gp_dir, settings_path=settings_path)
    manifest = {
        "conditioning_dir": str(conditioning_dir.resolve()),
        "volume_dir": str(volume_dir.resolve()),
        "wan2gp_dir": str(wan2gp_dir.resolve()),
        "settings": str(settings_path.resolve()),
        "download_script": str(download_script.resolve()),
        "download_launch_script": str((out_dir / "launch_download_vace_wsl.ps1").resolve()),
        "run_script": str((out_dir / "run_vace.sh").resolve()),
        "launch_script": str((out_dir / "launch_vace_wsl.ps1").resolve()),
        "selected_precision": str(args.precision),
        "downloads": _inventory(wan2gp_dir, ckpt_urls, lora_urls),
    }
    _write_json(out_dir / "vace_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
