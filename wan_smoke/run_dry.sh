#!/usr/bin/env bash
set -x
cd /mnt/d/dev/Wan2GP
exec ./env_conda/bin/python3 wgp.py --process /mnt/d/dev/video_to_world/wan_smoke/wan_settings.json --dry-run --output-dir /mnt/d/dev/video_to_world/wan_smoke/out
