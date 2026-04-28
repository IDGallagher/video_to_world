import json
import numpy as np
import torch
from pathlib import Path
from utils.third_party_bootstrap import prepend_local_third_party_paths
prepend_local_third_party_paths('depth_anything_3')
from depth_anything_3.api import DepthAnything3
from loop_utils.alignment_torch import robust_weighted_estimate_sim3_torch
import preprocess_video as pv

scene_root = Path('/mnt/d/dev/video_to_world/videos/_gradio_uploads/fish')
mask_path = scene_root / 'exports' / 'ply' / 'conf_voxel_or_vs1.0_g10.0_l10.0_p1.0_min50.0_edge_k3_r0.1000_a0.0000_sky_mindepth_pct10.000' / 'valid_pixel_indices.npz'
frames = sorted(str(p) for p in (scene_root / 'frames').glob('*.png'))
chunk_size = 20
overlap = 10
chunk_indices = pv._build_streaming_chunk_indices(len(frames), chunk_size, overlap)
model = DepthAnything3.from_pretrained('depth-anything/DA3NESTED-GIANT-LARGE').to(device=torch.device('cuda'))
model.eval()
chunks = []
for chunk_start, chunk_end in chunk_indices:
    preds = model.inference(
        image=frames[chunk_start:chunk_end],
        process_res=700,
        process_res_method='upper_bound_resize',
        infer_gs=False,
        use_ray_pose=False,
        ref_view_strategy='saddle_balanced',
        align_to_input_ext_scale=False,
    )
    chunks.append({
        'depth': np.asarray(preds.depth, dtype=np.float32),
        'conf': np.asarray(preds.conf, dtype=np.float32),
        'intrinsics': np.asarray(preds.intrinsics, dtype=np.float32),
        'extrinsics': np.asarray(preds.extrinsics, dtype=np.float32),
    })
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def estimate_dense(chunk_prev, chunk_cur):
    conf_prev = chunk_prev['conf'][-overlap:]
    conf_cur = chunk_cur['conf'][:overlap]
    point_prev = pv._depth_to_point_cloud_vectorized(chunk_prev['depth'][-overlap:], chunk_prev['intrinsics'][-overlap:], chunk_prev['extrinsics'][-overlap:])
    point_cur = pv._depth_to_point_cloud_vectorized(chunk_cur['depth'][:overlap], chunk_cur['intrinsics'][:overlap], chunk_cur['extrinsics'][:overlap])
    conf_threshold = min(float(np.median(conf_prev)), float(np.median(conf_cur))) * 0.1
    aligned_t = []
    aligned_s = []
    weights = []
    for i in range(min(point_prev.shape[0], point_cur.shape[0])):
        valid = (conf_prev[i] > conf_threshold) & (conf_cur[i] > conf_threshold)
        valid &= np.all(np.isfinite(point_prev[i]), axis=-1)
        valid &= np.all(np.isfinite(point_cur[i]), axis=-1)
        if not np.any(valid):
            continue
        aligned_t.append(point_prev[i][valid].astype(np.float32, copy=False))
        aligned_s.append(point_cur[i][valid].astype(np.float32, copy=False))
        weights.append(np.sqrt(conf_prev[i][valid] * conf_cur[i][valid]).astype(np.float32, copy=False))
    tgt = np.concatenate(aligned_t, axis=0)
    src = np.concatenate(aligned_s, axis=0)
    w = np.concatenate(weights, axis=0)
    s, R, t = robust_weighted_estimate_sim3_torch(src, tgt, w, delta=0.1, max_iters=5, tol=1e-9, align_method='se3')
    return float(s), np.asarray(R, dtype=np.float32), np.asarray(t, dtype=np.float32), conf_threshold

stored_extr = [None] * len(frames)
current_transform = (1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32))
scales = [1.0]
for chunk_idx, (chunk_start, chunk_end) in enumerate(chunk_indices):
    chunk = chunks[chunk_idx]
    if chunk_idx > 0:
        pair_s, pair_R, pair_t, _ = estimate_dense(chunks[chunk_idx-1], chunk)
        current_transform = pv._compose_sim3(current_transform, (pair_s, pair_R, pair_t))
        scales.append(pair_s)
    save_indices = pv._streaming_chunk_save_indices(chunk_index=chunk_idx, num_chunks=len(chunk_indices), chunk_len=chunk_end-chunk_start, overlap=overlap)
    depth_scale = float(current_transform[0])
    for local_idx in save_indices:
        global_idx = chunk_start + local_idx
        stored_extr[global_idx] = pv._transform_extrinsics_to_global(chunk['extrinsics'][local_idx], current_transform)
final_extr = pv._rebase_extrinsics_to_frame0_origin([e for e in stored_extr if e is not None])
# pose seam stats
H=[]
for ext in final_extr:
    M=np.eye(4,dtype=np.float64); M[:3,:4]=ext; H.append(M)
H=np.stack(H)
c2w=np.linalg.inv(H)
centers=c2w[:,:3,3]; rots=c2w[:,:3,:3]
steps=[]
for i in range(len(centers)-1):
    t=float(np.linalg.norm(centers[i+1]-centers[i]))
    dR=rots[i+1]@rots[i].T
    cosang=float(np.clip((np.trace(dR)-1.0)/2.0,-1.0,1.0))
    r=float(np.degrees(np.arccos(cosang)))
    steps.append({'pair':f'{i}->{i+1}','translation_m':t,'rotation_deg':r})
# masked extent continuity for first 50
npz = np.load(scene_root / 'exports' / 'npz' / 'results.npz')
depth = npz['depth']
intr = npz['intrinsics']
point_maps = pv._depth_to_point_cloud_vectorized(depth, intr, np.stack(final_extr, axis=0))
extent_rows=[]
with np.load(mask_path) as masks:
    prev_cent = None
    prev_extent_norm = None
    for i in range(50):
        idx = masks[f'frame_{i:05d}'].astype(np.int64)
        Hh,Ww = depth.shape[1:]
        yy = idx // Ww
        xx = idx % Ww
        pts = point_maps[i, yy, xx]
        valid = np.all(np.isfinite(pts), axis=1)
        pts = pts[valid]
        cent = pts.mean(axis=0)
        extent_norm = float(np.linalg.norm(pts.max(axis=0)-pts.min(axis=0)))
        row = {'frame': i, 'extent_norm': extent_norm}
        if prev_cent is not None:
            row['centroid_step_m'] = float(np.linalg.norm(cent-prev_cent))
            row['extent_delta'] = float(abs(extent_norm-prev_extent_norm))
        extent_rows.append(row)
        prev_cent = cent
        prev_extent_norm = extent_norm
# masked overlap residual first 4 seams
masked_rows=[]
with np.load(mask_path) as masks:
    for seam_idx in range(1,5):
        prev = chunks[seam_idx-1]
        cur = chunks[seam_idx]
        pair_s, pair_R, pair_t, _ = estimate_dense(prev, cur)
        point_prev = pv._depth_to_point_cloud_vectorized(prev['depth'][-overlap:], prev['intrinsics'][-overlap:], prev['extrinsics'][-overlap:])
        point_cur = pv._depth_to_point_cloud_vectorized(cur['depth'][:overlap], cur['intrinsics'][:overlap], cur['extrinsics'][:overlap])
        point_cur = pair_s * (point_cur @ pair_R.T) + pair_t
        for local_idx in range(overlap):
            global_idx = chunk_indices[seam_idx][0] + local_idx
            kept = masks[f'frame_{global_idx:05d}'].astype(np.int64)
            Hh,Ww = point_prev.shape[1:3]
            yy = kept // Ww
            xx = kept % Ww
            a = point_prev[local_idx, yy, xx]
            b = point_cur[local_idx, yy, xx]
            valid = np.all(np.isfinite(a), axis=1) & np.all(np.isfinite(b), axis=1)
            resid = np.linalg.norm(a[valid]-b[valid], axis=1)
            masked_rows.append(float(resid.mean()))
summary = {
    'pair_scales': scales,
    'max_boundary_translation_m': max(s['translation_m'] for s in steps if (int(s['pair'].split('->')[0])+1)%10==0),
    'max_boundary_rotation_deg': max(s['rotation_deg'] for s in steps if (int(s['pair'].split('->')[0])+1)%10==0),
    'max_centroid_step_m': max(r.get('centroid_step_m',0.0) for r in extent_rows),
    'max_extent_delta_m': max(r.get('extent_delta',0.0) for r in extent_rows),
    'mean_masked_overlap_residual_m': float(np.mean(masked_rows)),
}
print(json.dumps(summary, indent=2))
Path('/mnt/d/dev/video_to_world/videos/_gradio_uploads/fish_se3_streaming_probe.json').write_text(json.dumps(summary, indent=2))
