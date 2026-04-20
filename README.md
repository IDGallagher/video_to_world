# World Reconstruction From Inconsistent Views
Our method reconstructs 3D worlds from video diffusion models using non-rigid alignment to resolve inherent 3D inconsistencies in the generated sequences.

This is the official repository that contains source code for the paper *World Reconstruction From Inconsistent Views*.

[[arXiv](https://arxiv.org/abs/2603.16736)] [[Project Page](https://lukashoel.github.io/video_to_world/)] [[Video](https://www.youtube.com/watch?v=qXnUwhVmBzA)]

![Teaser](./assets/teaser.jpg)

If you find World Reconstruction From Inconsistent Views useful for your work please cite:
```
@misc{hoellein2026worldreconstructioninconsistentviews,
      title={World Reconstruction From Inconsistent Views}, 
      author={Lukas H{\"o}llein and Matthias Nie{\ss}ner},
      year={2026},
      eprint={2603.16736},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.16736}, 
}
```

## Prepare Environment

Clone this repository and create the conda environment:

```bash
git clone --branch main --single-branch https://github.com/lukasHoel/video_to_world
cd video_to_world

conda create -n video_to_world python=3.10
conda activate video_to_world

# Keep DA3-compatible numpy/opencv versions (numpy<2; opencv<4.12)
pip install "numpy<2" "opencv-python<4.12"
```

First, set up [DepthAnything-3](https://github.com/ByteDance-Seed/depth-anything-3):

```bash
mkdir -p third_party

# Clone DA3
git clone https://github.com/ByteDance-Seed/depth-anything-3 third_party/depth-anything-3
git -C third_party/depth-anything-3 checkout 2c21ea849ceec7b469a3e62ea0c0e270afc3281a

# Install DA3 + deps (minimal set for npz + gs_video)
pip install xformers torch\>=2 torchvision
pip install -e third_party/depth-anything-3

# Apply the trajectory-export patch
git -C third_party/depth-anything-3 apply ../../patches/da3-export-trajectory.patch
```

Install `gsplat`:

```bash
pip install --no-build-isolation \
  "git+https://github.com/nerfstudio-project/gsplat.git@v1.5.3"
```

Install `tinycudann`:

```bash
pip install setuptools==81.0.0
pip install "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch" --no-build-isolation
```

Then install the remaining dependencies:

```bash
pip install open3d scipy tyro tqdm tensorboard
pip install lpips viser nerfview romatch
```

Install RoMaV2 (patched to avoid a `dataclasses>=0.8` dependency-resolution issue, see [Parskatt/RoMaV2#26](https://github.com/Parskatt/RoMaV2/issues/26)):

```bash
# Clone RoMaV2
git clone https://github.com/Parskatt/RoMaV2 third_party/RoMaV2

# Patch dependency metadata (dataclasses>=0.8 -> dataclasses)
git -C third_party/RoMaV2 apply ../../patches/romav2-dataclasses.patch

# Install (optionally add `fused-local-corr` for the fused local correlation kernel)
pip install -e "third_party/RoMaV2[fused-local-corr]"
```

Optionally, install [torch_kdtree](https://github.com/thomgrand/torch_kdtree) for GPU-accelerated KD-tree nearest-neighbor queries:

```bash
export CUDA_HOME=/usr/local/cuda # point to a local installation of a corresponding cuda toolkit version
git clone https://github.com/thomgrand/torch_kdtree third_party/torch_kdtree
cd third_party/torch_kdtree
git submodule init && git submodule update
pip install -U cmake ninja
CPLUS_INCLUDE_PATH="$CUDA_HOME/include:${CPLUS_INCLUDE_PATH:-}" PATH="$CONDA_PREFIX/bin:$PATH" python -m pip install . --no-build-isolation
cd ../..
```

## Quickstart

Reconstruct a 3D world from a single MP4 (generated from a video model):

```bash
python run_reconstruction.py --config.input-video /path/to/video.mp4
```

### Gradio App

For a local UI around the same pipeline:

```bash
pip install gradio
python gradio_app.py --server-name 0.0.0.0 --server-port 7860
```

The app can:

- launch `run_reconstruction.py` from upload and dropdown-based inputs without typing paths
- browse existing videos and scene roots directly from the UI
- run Stage 0, Stage 1, Stage 2, Stage 3.1, or Stage 3.2 explicitly instead of only the full pipeline
- live-tail the pipeline log
- surface the latest preview videos and key artifacts discovered in the scene directory

Alternatively, run the full pipeline from a folder of frames:

```bash
python run_reconstruction.py --config.frames-dir /path/to/frames
```

### Presets: fast vs extensive

`run_reconstruction.py` supports two presets via `--config.mode`:

- **fast (default)**: skips global optimization, trains backward deformation for 15 epochs, terminates ICP with `icp_early_stopping_min_delta=5e-5`, trains 3DGS for 10k iterations.
- **extensive**: runs all stages, trains backward deformation for 30 epochs, terminates ICP with `icp_early_stopping_min_delta=5e-6`, trains both 2DGS and 3DGS for 15k iterations each.

Use `--config.renderer [2dgs,3dgs,both]` to select which type of Gaussian Splatting scene is optimized.

## Running Individual Stages

### Stage 0: DA3 preprocessing (video / frames → pointcloud)

```bash
python preprocess_video.py --input_video /path/to/video.mp4 --prepare_stage1_inputs --export_kinect_rgbd_video
```

This estimates per-frame pointclouds using DepthAnyting-3 and saves the results to `<scene_root> = /path/to/video` (overwrite via `--scene_root /path/to/da3_scene`).
With `--prepare_stage1_inputs`, Stage 0 also materializes the filtered `exports/ply/...` cache and writes `before_non_rigid_icp.ply` into the corresponding `frame_to_model_icp_<...>/` run directory. Add `--export_gs_video` if you also want the DA3 preview video and trajectory export. Add `--export_kinect_rgbd_video` to also write a packed KinectStreamer-style RGBD video under `exports/kinect_rgbd_video/`.

Subsampling of frames is controlled by `--max_frames` (default: 30) and `--max_stride` (default: 8).
The script extracts all frames to `<scene_root>/frames/`, then writes the selected subset (renumbered from `000000.*`) to `<scene_root>/frames_subsampled/` and runs DepthAnyting-3 on that folder.
This constrains memory of DA3 to the available budget (choose fewer frames for smaller GPUs).
Please consult the original repository for more information regarding memory.
If the scene contains much more frames, one can use [DA3-Streaming](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/da3_streaming/README.md) to predict per-frame pointclouds for all frames.

**Expected scene layout**:

```
<scene_root>/
  exports/
    npz/
      results.npz          # Contains: depth (N,H,W), conf (N,H,W),
                            #   extrinsics (N,3,4) w2c, intrinsics (N,3,3),
                            #   image (N,H,W,3) uint8
    kinect_rgbd_video/     # Optional packed Stage 0 export
      kinect_rgbd_hapq.mov
      sequence_info.json
      frames/
        000000.png         # Layout: [metadata_barcode|color|depth]
    kinect_rgbd_sequence/  # Optional exact packed frame-sequence export
      sequence_info.json
      frames/
        000000.png         # Layout: [metadata_barcode|color|depth]
    kinect_rgbd_sequence_depth8/  # Optional exact packed frame-sequence export with 8-bit inverse depth
      sequence_info.json
      frames/
        000000.png         # Layout: [metadata_barcode|color|depth]
    depth_image_stream/    # Optional exact DirectStorage stream export
      depth_image_stream.divstream
  frames/                   # extracted original frames
  frames_subsampled/         # renumbered subset used for DA3
```

If `--export_gs_video` is enabled, Stage 0 also writes:

```
  gs_video/
    *.mp4                  # flythrough video of naive DA3 reconstruction
    *_transforms.json      # exported camera trajectory (used later for evaluation)
```

If `--export_kinect_rgbd_video` is enabled, Stage 0 also writes:

```
  exports/
    kinect_rgbd_video/
      kinect_rgbd_hapq.mov
      sequence_info.json   # fps + metadata width + packed frame dimensions
      frames/
        000000.png         # [barcode|color|depth], first-frame-relative camera metadata
```

The Gradio app can also export:
- an exact packed frame sequence to `exports/kinect_rgbd_sequence/` using the current 16-bit inverse-depth RGB codebook
- an exact packed frame sequence to `exports/kinect_rgbd_sequence_depth8/` using 8-bit inverse-depth grayscale replicated across RGB
- an exact single-file DirectStorage stream to `exports/depth_image_stream/depth_image_stream.divstream`

The two PNG-sequence exports can be used with `ADepthImageVolumeImageSequenceActor` for side-by-side Unreal comparisons against the HAP path. The `.divstream` export is intended for `ADepthImageVolumeDirectStorageActor`.

Runtime HAP playback uses `kinect_rgbd_hapq.mov`. The accompanying `frames/` directory plus `sequence_info.json` are left behind so Unreal can also drive a separate exact PNG-sequence actor from the same packed frames.

#### DirectStorage stream format

- The `.divstream` export is not a video codec. It is a single-file frame stream designed for DirectStorage playback in Unreal.
- The current export format is `DIVBC7C3`.
- The export uses 3-frame chunks:
  - color is stored as raw `BC7` blocks on disk
  - depth is stored as exact `G16` bytes per frame
  - within each chunk, depth frame `0` is absolute and later depth frames are lossless XOR residuals against the previous frame
  - the chunked depth payload is then `GDeflate`-compressed
- Depth is stored as `uint16`, not float16:
  - `0` = invalid pixel
  - `[1, 65535]` = valid depth mapped linearly between that frame's `near` / `far` values in centimeters
- Per-frame metadata is stored in a binary frame table: first-frame-relative camera-to-world, `near`, `far`, `fx`, `fy`, `cx`, `cy`.
- `ADepthImageVolumeDirectStorageActor` only supports `DIVBC7C3`. Older `DIVSTRM1`, `DIVBC7C1`, and `DIVBC7C2` streams are rejected and should be re-exported.
- The Unreal runtime uses this data through `ADepthImageVolumeDirectStorageActor`, preserving the existing `DepthImageVolume` exact depth/color path while exposing video-style controls (`play`, `pause`, `seek`, `loop`, playback rate).

#### Packed Stage 0 video format

- Frame layout is `[metadata_barcode | color | depth]`.
- The left barcode is `64` pixels wide by default so each of the 13 float values spans 4-5 columns before HAP block compression.
- The left barcode stores 13 float32 values in this order: `px, py, pz, rx, ry, rz, near, far, fx, fy, cx, cy, hash`.
- Frame `000000` is the reference pose: its relative transform is identity. All later frames are stored relative to frame `000000`.
- `rx, ry, rz` are the vector part of a unit quaternion. The scalar `w` is recovered in the reader from `sqrt(max(0, 1 - x^2 - y^2 - z^2))`.
- `fx, fy, cx, cy` are per-frame intrinsics in pixels.
- The color and depth halves are center-padded as needed to HAP-compatible block dimensions. `cx/cy` are stored after padding so the camera model still matches the padded frame exactly.
- The packed depth half uses a lossless exact 16-bit inverse-depth codebook with the coarse byte in `G` and the low byte spread across a small `R/B` serpentine lattice, so it remains exact under lossless RGB storage and degrades more gracefully than a hue-wheel encoding if the frames are later transcoded lossily.
- The packed PNG frames are transcoded to `HAP Q` in `kinect_rgbd_hapq.mov` for Unreal playback. This is still lossy, but it avoids YUV conversion and temporal prediction, which makes it a much better fit than H.264 for the barcode and packed depth image.
- Depth normalization is global per exported clip: `u = (1/z - 1/far) / (1/near - 1/far)`, then `q = round(clamp(u, 0, 1) * 65535)`.
- The optional `kinect_rgbd_sequence_depth8/` export uses the same global inverse-depth normalization, but stores `q8 = round(clamp(u, 0, 1) * 254) + 1` as grayscale in the depth half, reserving `0` for invalid pixels.

The `results.npz` file is the primary input for all subsequent stages.

### Stage 1: Iterative Non-rigid Frame-to-model ICP

This runs non-rigid ICP on the Stage-0-prepared point-cloud cache and writes the aligned canonical point cloud plus per-frame deformation fields.

```bash
python -m frame_to_model_icp --config.root-path <scene_root> \
  --config.out-path <scene_root>/frame_to_model_icp_<N>_<stride>_offset<offset>
```

Point `--config.out-path` at the prepared Stage 0 run directory. Stage 1 loads the persisted Stage 0 prep config from that directory and only runs the non-rigid ICP stage.

#### Stage 0 pre-ICP sampling: `N`, `stride`, `offset`

Stage 0 can optionally prepare only a subset of frames from `exports/npz/results.npz` before Stage 1 starts. The run folder name encodes the chosen subset:

- **`--prepare_num_frames` (`N`)**: number of frames prepared for Stage 1 (default: 50).
- **`--prepare_stride`**: take every `stride`-th frame from the underlying sequence (default: 2).
- **`--prepare_offset`**: starting index into the underlying sequence (default: 0).

**Output**: `<scene_root>/frame_to_model_icp_<N>_<stride>_offset<offset>/` containing:
- `before_non_rigid_icp.ply` -- merged Stage-0-prepared point cloud before non-rigid ICP
- `after_non_rigid_icp/` -- per-frame SE(3) twists, deformation grids, merged point cloud
- `after_non_rigid_icp/config.json` -- run configuration

#### Stage 0 pre-ICP confidence filtering

Stage 0 now uses the DA3 per-pixel confidence map from `results.npz` when it builds the filtered point-cloud cache for Stage 1. The default config uses a mixed `voxel_or` mode that combines DA3 confidence with voxel-density heuristics.

For a simpler test that only uses DA3 confidence, run Stage 0 prep with one of these modes:

- `--prepare_conf_mode per_frame --prepare_conf_thresh_percentile 80`
- `--prepare_conf_mode global --prepare_conf_thresh_percentile 80`

The Gradio app exposes these under Stage 0 pre-ICP filtering, with `DA3 Only (Per Frame)` being the simplest direct comparison against Pi-Long-style confidence pruning.

### Stage 2: Global Optimization

This jointly refines all per-frame deformations in a single optimization to further sharpen and flatten the canonical point cloud.

```bash
python -m global_optimization --config.root-path <scene_root> \
    --config.run frame_to_model_icp_<N>_<stride>_offset<offset>
```

**Output**: `<align_run>/after_global_optimization/` containing refined deformations and canonical point clouds.

### Stage 3.1: Inverse Deformation Training

This trains an inverse deformation network that maps canonical-space points back into each frame’s camera space to enable deformation-aware rendering losses.

```bash
python -m train_inverse_deformation \
    --config.root-path <scene_root> \
    --config.run frame_to_model_icp_<N>_<stride>_offset<offset> \
    --config.checkpoint-subdir after_global_optimization
```

**Output**: `<align_run>/inverse_deformation/` containing `inverse_local.pt` and `config.pt`.

### Stage 3.2: Gaussian Splatting Training

This optimizes a 2DGS/3DGS scene initialized from the canonical point cloud while using the inverse deformation network to warp Gaussians per frame during training.

```bash
python -m train_gs \
    --config.root-path <scene_root> \
    --config.run frame_to_model_icp_<N>_<stride>_offset<offset> \
    --config.global-opt-subdir after_global_optimization \
    --config.inverse-deform-dir <align_run>/inverse_deformation \
    --config.original-images-dir <scene_root>/frames_subsampled
```

Use `--config.renderer 3dgs` for 3D Gaussian Splatting instead (default: 2DGS).

**Output**: `<align_run>/gs_<renderer>/` containing Gaussian checkpoint, rendered images, and evaluation metrics.

### Evaluation / Novel-View Rendering

This renders novel views from a trained GS checkpoint using the evaluation camera trajectory (e.g. the DA3-exported `_transforms.json`).

```bash
python -m eval_gs \
    --config.root-path <scene_root> \
    --config.run frame_to_model_icp_<N>_<stride>_offset<offset> \
    --config.checkpoint-dir <align_run>/gs_<renderer>
```

**Output**: `<align_run>/gs_<renderer>/gs_video_eval/` containing rendered images and MP4 videos along the evaluation camera path (override with `--config.out-dir`).

## Utilities

### Export a trained 3DGS checkpoint to PLY

```bash
python -m utils.export_checkpoint_to_ply \
    --config.root-path <scene_root> \
    --config.run frame_to_model_icp_<N>_<stride>_offset<offset> \
    --config.checkpoint-dir <align_run>/gs_<renderer>
```

**Output**: a 3DGS PLY file at `--config.out-ply` (default: `<align_run>/gs_3dgs/splats_3dgs.ply`).

### View a checkpoint (interactive)

```bash
python -m utils.view_checkpoint \
    --config.root-path <scene_root> \
    --config.run frame_to_model_icp_<N>_<stride>_offset<offset> \
    --config.checkpoint-dir <align_run>/gs_<renderer>
```

This launches an interactive viewer (Viser + nerfview) for both 2DGS and 3DGS checkpoints. By default it runs on `localhost:8080` (override with `--config.port`).

## Configuration

All hyperparameters live in dataclasses under `configs/`.
They can be modified via CLI parameters for detailed configuration of the individual stages.

| File | Stage | Description |
|------|-------|-------------|
| `configs/stage1_align.py` | 1 | Iterative Non-rigid Frame-to-model ICP (`FrameToModelICPConfig`) |
| `configs/stage2_global_optimization.py` | 2 | Global optimization |
| `configs/stage3_inverse_deformation.py` | 3.1 | Inverse deformation |
| `configs/stage3_gs.py` | 3.2 | Gaussian splatting (2DGS / 3DGS) |

## Acknowledgements

Our work builds on top of amazing open-source projects. We thank the authors for making their code available.

- [Depth Anything 3 (DA3)](https://github.com/DepthAnything/Depth-Anything-3): per-frame depth/point cloud prediction (Stage 0 input).
- [RoMa](https://github.com/Parskatt/RoMa): robust dense feature matching used for correspondences during alignment.
- [gsplat](https://github.com/nerfstudio-project/gsplat): Gaussian splatting rasterizer used for 2DGS/3DGS training and rendering.
- [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn): hash-grid encodings used by the deformation networks.
- [torch_kdtree](https://github.com/thomgrand/torch_kdtree): optional GPU-accelerated KD-tree for nearest-neighbor queries.

