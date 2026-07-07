#!/usr/bin/env bash
while true; do
  pid=$(pgrep -x python -a | grep run_reconstruction | head -1 | cut -d" " -f1)
  if [ -z "$pid" ]; then
    echo "RECONSTRUCTION ENDED"
    tail -c 2500 /mnt/d/dev/video_to_world/wan_smoke/recon.log | tr "\r" "\n" | grep -v "^$" | tail -12
    exit 0
  fi
  sleep 60
done
