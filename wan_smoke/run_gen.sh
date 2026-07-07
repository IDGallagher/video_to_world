#!/usr/bin/env bash
cd /mnt/d/dev/Wan2GP
export PYTHONUNBUFFERED=1
export TORCHDYNAMO_DISABLE=1
setsid nohup ./env_conda/bin/python3 wgp.py --process /mnt/d/dev/video_to_world/wan_smoke/wan_settings.json --output-dir /mnt/d/dev/video_to_world/wan_smoke/out --profile 4 --perc-reserved-mem-max 0.25 --attention sage > /mnt/d/dev/video_to_world/wan_smoke/gen.log 2>&1 < /dev/null &
disown
echo relaunched-optimized
