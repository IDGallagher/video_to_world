#!/usr/bin/env bash
cd /mnt/d/dev/video_to_world
export PYTHONUNBUFFERED=1
setsid nohup ~/miniforge3/envs/video_to_world/bin/python run_reconstruction.py --config.input-video /mnt/d/dev/video_to_world/wan_smoke/coral_arc_v2.mp4 > /mnt/d/dev/video_to_world/wan_smoke/recon.log 2>&1 < /dev/null &
disown
echo recon-launched
