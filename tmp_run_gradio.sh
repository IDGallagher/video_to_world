#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
exec /home/idgal/miniforge3/envs/video_to_world/bin/python -u "$SCRIPT_DIR/tmp_gradio_server.py" >> "$SCRIPT_DIR/gradio_app_runtime.log" 2>> "$SCRIPT_DIR/gradio_app_runtime.err"
