#!/usr/bin/env bash
set -eo pipefail

source ~/miniforge3/etc/profile.d/conda.sh
conda activate video_to_world
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SCENE="$SCRIPT_DIR/videos/ian_101_moving_forwards_underwater_clear_water_traveling_low__be00f3d5-6a4d-4fee-b498-2f6077b78586_3"
VIDEO=${SCENE}.mp4
RUN=frame_to_model_icp_50_2_offset0

ts() {
    date "+%Y-%m-%d %H:%M:%S %z"
}

printf "[%s] Pipeline start\n" "$(ts)"
printf "[%s] Stage 0 start: preprocess_video.py\n" "$(ts)"
python -u preprocess_video.py \
    --scene_root "$SCENE" \
    --model_name depth-anything/DA3NESTED-GIANT-LARGE \
    --image_ext png \
    --max_frames 100 \
    --max_stride 8 \
    --input_video "$VIDEO"

printf "[%s] Stage 1 start: frame_to_model_icp\n" "$(ts)"
python -u -m frame_to_model_icp \
    --config.root-path "$SCENE" \
    --config.icp-early-stopping-min-delta 5e-05

printf "[%s] Stage 3.1 start: train_inverse_deformation\n" "$(ts)"
python -u -m train_inverse_deformation \
    --config.root-path "$SCENE" \
    --config.run "$RUN" \
    --config.checkpoint-subdir after_non_rigid_icp \
    --config.n-epochs 15

printf "[%s] Stage 3.2 start: train_gs 3dgs 1000 steps\n" "$(ts)"
python -u -m train_gs \
    --config.root-path "$SCENE" \
    --config.run "$RUN" \
    --config.global-opt-subdir after_non_rigid_icp \
    --config.original-images-dir "$SCENE/frames_subsampled" \
    --config.inverse-deform-dir "$SCENE/$RUN/inverse_deformation" \
    --config.renderer 3dgs \
    --config.num-iters 1000 \
    --config.save-every 1000

printf "[%s] Pipeline done\n" "$(ts)"
