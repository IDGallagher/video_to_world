#!/usr/bin/env bash
# Resumable overnight download of the Wan 2.2 Enhanced Lightning v2 i2v experts.
cd /mnt/d/dev/Wan2GP/ckpts || exit 1
BASE="https://huggingface.co/DeepBeepMeep/Wan2.2/resolve/main"
for f in wan22EnhancedLightning_v2I2VFP8HIGH.safetensors wan22EnhancedLightning_v2I2VFP8LOW.safetensors; do
  until wget -c --tries=0 --retry-connrefused --timeout=60 "$BASE/$f"; do
    echo "retrying $f in 10s..."
    sleep 10
  done
done
echo ALL_LIGHTNING_DOWNLOADS_COMPLETE
