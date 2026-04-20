#!/usr/bin/env bash
set -eo pipefail

source ~/miniforge3/etc/profile.d/conda.sh
conda activate video_to_world
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SCENE="$SCRIPT_DIR/videos/ian_101_moving_forwards_underwater_clear_water_traveling_low__be00f3d5-6a4d-4fee-b498-2f6077b78586_3"
RUN=frame_to_model_icp_50_2_offset0
RUN_DIR="$SCENE/$RUN"
STAGE2_OUT=after_global_optimization
STAGE31_OUT="$RUN_DIR/inverse_deformation_after_global_optimization"

ts() {
    date "+%Y-%m-%d %H:%M:%S %z"
}

rm -rf "$RUN_DIR/$STAGE2_OUT" "$STAGE31_OUT"

printf "[%s] Stage 2 start: global_optimization\n" "$(ts)"
python -u -m global_optimization \
    --config.root-path "$SCENE" \
    --config.run "$RUN" \
    --config.checkpoint-subdir after_non_rigid_icp \
    --config.out-subdir "$STAGE2_OUT" \
    --config.knn-backend cpu_kdtree

printf "[%s] Stage 3.1 start: train_inverse_deformation\n" "$(ts)"
python -u -m train_inverse_deformation \
    --config.root-path "$SCENE" \
    --config.run "$RUN" \
    --config.checkpoint-subdir "$STAGE2_OUT" \
    --config.n-epochs 15 \
    --config.out-path "$STAGE31_OUT"

printf "[%s] Stage 3.1 done\n" "$(ts)"
