#!/usr/bin/env bash
# Runs INSIDE WSL. Watches the wgp generation process without self-matching.
while true; do
  pid=$(pgrep -x python3 -a | grep "wgp.py" | head -1 | cut -d' ' -f1)
  if [ -z "$pid" ]; then
    echo "GENERATION PROCESS ENDED"
    echo "--- outputs ---"
    ls -la /mnt/d/dev/video_to_world/wan_smoke/out/ 2>/dev/null
    echo "--- last log ---"
    tail -c 2500 /mnt/d/dev/video_to_world/wan_smoke/gen.log | tr '\r' '\n' | grep -v '^$' | tail -12
    exit 0
  fi
  sleep 60
done
