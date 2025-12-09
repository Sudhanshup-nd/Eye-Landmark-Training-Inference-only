#!/usr/bin/env python3
"""
Prediction-only evaluation for eye landmarks with dual overlays (face and eye-crop),
saving overlays organized by video_id. Also computes EAR and generates per-video
side-by-side videos (overlayed face image + EAR graph).

EAR definition:
    EAR = ( ||p2-p6|| + ||p3-p5|| ) / ( 2 * ||p1-p4|| )
With points ordered: p1=left corner, p2,p3=upper lid, p4=right corner, p5,p6=lower lid.

Two-eye support:
- Uses CSV column 'eye_side' directly ('left' or 'right') without guessing.
- EAR graph modes:
  - left: only left-eye EAR plotted, and only *_left.png images used in video.
  - right: only right-eye EAR plotted, and only *_right.png images used in video.
  - both: both EARs superimposed on the same graph (left-blue, right-red), legend shown.
          For each frame group, left image is shown then right, while incrementally adding points.

Video graph indexing:
- The graph's x-axis uses the numeric frame index x extracted from frame_key (e.g., frame_x_face00 -> x).
- Every plotted point is placed at x.
- The left/right image in the video is annotated with the exact frame index x.

IMPORTANT: Video building uses per-video EAR CSVs saved at:
  {output_dir}/ears_per_video/{video_id}.csv
Columns: frame_key, EAR_left, EAR_right

Usage:
python /inwdata2a/sudhanshu/landmarks_only_training/inference_pipeline/evaluate_predictions_no_gt.py \
  --checkpoint /inwdata2a/sudhanshu/landmarks_only_training/augmented-outputs_landmarks-randomized-padding/best_landmarks.pt \
  --config /inwdata2a/sudhanshu/landmarks_only_training/configs/default.yaml \
  --csv /inwdata2a/sudhanshu/landmarks_only_training/inference_pipeline/inference_pairs-sorted.csv \
  --output_dir outputs-augmented_landmarks-randomized-padding/eval_no_gt \
  --limit 2000 \
  --visualize \
  --video_fps 1 \
  --video_xaxis_frames 100 \
  --ear_graph_mode left \
  --no_augment

Outputs:
- Images:
    {output_dir}/overlays_face/{video_id}/{frame_key}_{eye_side}.png
    {output_dir}/overlays_eye/{video_id}/{frame_key}_{eye_side}.png
- CSV (global per-sample):
    {output_dir}/predicted_landmarks.csv  (includes EAR per sample)
- CSVs (per-video time series aligned by frame_key):
    {output_dir}/ears_per_video/{video_id}.csv with columns: frame_key, EAR_left, EAR_right
- Videos (per video_id, depending on ear_graph_mode), built using per-video CSV:
    {output_dir}/videos/{video_id}_{mode}.mp4
    Each frame combines:
      left: overlayed face image with "Frame: x" text
      right: EAR graph with fixed x-axis (0..max_x_frames-1) using true frame index x
"""

import argparse
import os
from pathlib import Path
import sys
import re
import io
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from PIL import Image
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

import cv2
import torchvision.transforms as T

# Project path setup
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if __package__ in (None, ""):
    __package__ = "src"

# Internal imports
from src.utils import load_config, load_checkpoint
from src.transforms import build_val_transforms
from src.model import EyeLandmarkModel  # landmark-only

def robust_parse_visibility(raw):
    if raw is None:
        return 1
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, (int, float)):
        try:
            if isinstance(raw, float) and torch.isnan(torch.tensor(raw)):
                return 1
        except Exception:
            pass
        return 1 if int(raw) != 0 else 0
    if not isinstance(raw, str):
        return 1
    t = raw.strip().lower().strip("\"' ")
    if t in ("true", "1", "yes", "y", "visible"):
        return 1
    if t in ("false", "0", "no", "n", "invisible", "nan", "none", "null", ""):
        return 0
    try:
        return 1 if int(t) != 0 else 0
    except ValueError:
        return 1

def parse_bbox(bbox_str):
    if not isinstance(bbox_str, str):
        return None
    s = bbox_str.strip()
    if s.lower() in ("", "nan", "none", "null"):
        return None
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        return None
    try:
        x1, y1, x2, y2 = map(float, parts)
    except ValueError:
        return None
    return (x1, y1, x2, y2)

def crop_eye_image_with_transform(face_img: Image.Image, bbox, target_size: int):
    if bbox is None:
        return None, None
    w_face, h_face = face_img.size
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), w_face - 1))
    x2 = max(0, min(int(x2), w_face - 1))
    y1 = max(0, min(int(y1), h_face - 1))
    y2 = max(0, min(int(y2), h_face - 1))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    crop = face_img.crop((x1, y1, x2 + 1, y2 + 1))  # inclusive
    cw, ch = crop.size
    pad_w = pad_h = 0
    offset_x = offset_y = 0
    side = max(cw, ch)
    if cw != ch:
        pad_w = side - cw
        pad_h = side - ch
        offset_x = pad_w // 2
        offset_y = pad_h // 2
        padded = Image.new("RGB", (side, side), (0, 0, 0))
        padded.paste(crop, (offset_x, offset_y))
        crop = padded
    if crop.size != (target_size, target_size):
        crop = crop.resize((target_size, target_size), Image.BILINEAR)
    scale = (target_size / float(side)) if side > 0 else 1.0
    transform = {
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "cw": cw, "ch": ch,
        "side": side,
        "pad_w": pad_w, "pad_h": pad_h,
        "offset_x": offset_x, "offset_y": offset_y,
        "scale": scale,
        "target_size": target_size,
    }
    return crop, transform

def map_face_pts_to_eye_crop(pred_abs_pts, transform):
    if transform is None:
        return []
    x1 = transform["x1"]; y1 = transform["y1"]
    offset_x = transform["offset_x"]; offset_y = transform["offset_y"]
    scale = transform["scale"]
    eye_pts = []
    for (x, y) in pred_abs_pts:
        lx = x - x1
        ly = y - y1
        px = lx + offset_x
        py = ly + offset_y
        ex = px * scale
        ey = py * scale
        eye_pts.append((ex, ey))
    return eye_pts

def denormalize_predictions(pred_tensor: torch.Tensor, bbox, cfg):
    if bbox is None:
        return []
    x1, y1, x2, y2 = bbox
    bw = max(1.0, (x2 - x1 + 1))
    bh = max(1.0, (y2 - y1 + 1))
    normalized = cfg['data'].get('normalize_landmarks', False)
    local = cfg['data'].get('landmarks_local_coords', True)
    pts = []
    for i in range(pred_tensor.shape[0]):
        xn, yn = pred_tensor[i, 0].item(), pred_tensor[i, 1].item()
        if normalized and local:
            x_abs = xn * bw + x1
            y_abs = yn * bh + y1
        elif local and not normalized:
            x_abs = xn + x1
            y_abs = yn + y1
        else:
            x_abs = xn
            y_abs = yn
        pts.append((x_abs, y_abs))
    return pts

def visualize_face_overlay(pil_img, pred_abs, out_png):
    fig = plt.figure(figsize=(3, 3))
    plt.imshow(pil_img)
    plt.axis('off')
    if pred_abs:
        for (x, y) in pred_abs:
            plt.scatter([x], [y], c='lime', s=4, marker='o',
                        edgecolors='black', linewidths=0.5)
    plt.title("Predicted Landmarks (Face)")
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches='tight', dpi=150)
    plt.close(fig)

def visualize_eye_overlay(eye_img, eye_pts, out_png):
    fig = plt.figure(figsize=(3, 3))
    plt.imshow(eye_img)
    plt.axis('off')
    if eye_pts:
        for (x, y) in eye_pts:
            plt.scatter([x], [y], c='cyan', s=10, marker='o',
                        edgecolors='black', linewidths=0.5)
    plt.title("Predicted Landmarks (Eye Crop)")
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches='tight', dpi=150)
    plt.close(fig)

def compute_ear(pred_pts_abs):
    if pred_pts_abs is None or len(pred_pts_abs) < 6:
        return np.nan
    p1 = np.array(pred_pts_abs[0])
    p2 = np.array(pred_pts_abs[1])
    p3 = np.array(pred_pts_abs[2])
    p4 = np.array(pred_pts_abs[3])
    p5 = np.array(pred_pts_abs[4])
    p6 = np.array(pred_pts_abs[5])
    def dist(a, b):
        return float(np.linalg.norm(a - b))
    denom = 2.0 * dist(p1, p4)
    if denom <= 1e-6:
        return np.nan
    ear = (dist(p2, p6) + (dist(p3, p5))) / denom
    return ear

# Graph rendering that uses true frame index x for plotting
def render_ear_graph_image_dual_by_x(ears_left_by_x, ears_right_by_x,
                                     max_x_frames=500, fig_size=(5, 5), y_limits=None,
                                     mode="both", highlight_x=None):
    xs_left = sorted(ears_left_by_x.keys())
    ys_left = [ears_left_by_x[x] for x in xs_left]
    xs_right = sorted(ears_right_by_x.keys())
    ys_right = [ears_right_by_x[x] for x in xs_right]

    if y_limits is None:
        vals = []
        if mode in ("left", "both"):
            vals += [v for v in ys_left if not np.isnan(v)]
        if mode in ("right", "both"):
            vals += [v for v in ys_right if not np.isnan(v)]
        if len(vals) > 0:
            ymin = float(np.min(vals)); ymax = float(np.max(vals))
            if abs(ymax - ymin) < 1e-6:
                ymin -= 0.1; ymax += 0.1
            else:
                pad = 0.1 * (ymax - ymin)
                ymin -= pad; ymax += pad
        else:
            ymin, ymax = 0.0, 1.0
    else:
        ymin, ymax = y_limits

    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(1, 1, 1)
    if mode in ("left", "both") and len(xs_left) > 0:
        ax.plot(xs_left, ys_left, color='blue', linewidth=2, label='Left EAR')
        if highlight_x and ("left" in highlight_x) and (highlight_x["left"] is not None):
            hx = highlight_x["left"]
            if hx in ears_left_by_x and not np.isnan(ears_left_by_x[hx]):
                # Dot color changed to red for visibility against blue line
                ax.scatter([hx], [ears_left_by_x[hx]], color='red', s=36, marker='o',
                           edgecolors='black', linewidths=0.6, zorder=5)
    if mode in ("right", "both") and len(xs_right) > 0:
        ax.plot(xs_right, ys_right, color='red', linewidth=2, label='Right EAR')
        if highlight_x and ("right" in highlight_x) and (highlight_x["right"] is not None):
            hx = highlight_x["right"]
            if hx in ears_right_by_x and not np.isnan(ears_right_by_x[hx]):
                # Dot also red to match the request (graph is red already)
                ax.scatter([hx], [ears_right_by_x[hx]], color='red', s=36, marker='o',
                           edgecolors='black', linewidths=0.6, zorder=5)

    ax.set_title("EAR over Frames (x)")
    ax.set_xlabel("Frame index (x)")
    ax.set_ylabel("EAR")
    ax.set_xlim(0, max_x_frames - 1)
    ax.set_ylim(ymin, ymax)
    ax.grid(True, alpha=0.3)
    if mode == "both":
        ax.legend(loc='upper right')

    buf = io.BytesIO()
    plt.tight_layout()
    fig.canvas.draw()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    graph_img = Image.open(buf).convert("RGB")
    graph_np = np.array(graph_img)
    return graph_np

def stack_side_by_side(left_bgr, right_bgr, target_height=None):
    h1, w1 = left_bgr.shape[:2]
    h2, w2 = right_bgr.shape[:2]
    if target_height is None:
        target_height = max(h1, h2)
    def resize_by_height(img, target_h):
        h, w = img.shape[:2]
        scale = target_h / float(h)
        new_w = int(round(w * scale))
        return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)
    left_r = resize_by_height(left_bgr, target_height)
    right_r = resize_by_height(right_bgr, target_height)
    combined = np.concatenate([left_r, right_r], axis=1)
    return combined

def parse_frame_group_from_filename(fname: str):
    base = os.path.splitext(os.path.basename(fname))[0]
    parts = base.split("_")
    if len(parts) < 3:
        return base, "unknown"
    eye_side = parts[-1].lower()
    group_key = "_".join(parts[:-1])  # drop only the eye_side suffix
    return group_key, eye_side

def extract_frame_index(frame_key: str) -> int:
    if not isinstance(frame_key, str):
        return -1
    m = re.search(r'frame[_-]?(\d+)', frame_key)
    if m:
        return int(m.group(1))
    m2 = re.search(r'(\d+)', frame_key)
    return int(m2.group(1)) if m2 else -1

def collect_overlay_sequence_grouped(face_overlay_dir):
    """
    Collect PNG overlay file paths and group them by frame group key.
    Returns an ordered list of (group_key, files_for_group_dict), sorted by numeric x.
    files_for_group_dict has keys 'left' and/or 'right' with file paths.
    """
    if not os.path.isdir(face_overlay_dir):
        return []
    files = [f for f in os.listdir(face_overlay_dir) if f.lower().endswith(".png")]
    groups = {}
    for f in files:
        fp = os.path.join(face_overlay_dir, f)
        g, side = parse_frame_group_from_filename(f)
        if g not in groups:
            groups[g] = {}
        groups[g][side] = fp
    # Sort by numeric frame index x extracted from group_key (not lexicographic)
    ordered = sorted(groups.items(), key=lambda kv: extract_frame_index(kv[0]))
    return ordered

def annotate_frame_number_on_bgr(bgr_img, x_value):
    img = bgr_img.copy()
    text = f"Frame: {x_value}" if (isinstance(x_value, int) and x_value >= 0) else "Frame: ?"
    cv2.putText(img, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    return img

def build_video_per_id_dual_with_csv(video_id, overlays_face_root, per_video_csv_path,
                                     output_video_path, fps=25, max_x_frames=500, mode="both"):
    """
    Build the video using EAR values from the per-video CSV:
      per_video_csv_path: {output_dir}/ears_per_video/{video_id}.csv
    Columns: frame_key, EAR_left, EAR_right

    The graph indexes points using true frame index x extracted from frame_key.
    Images are annotated with 'Frame: x'. In 'left'/'right' modes only the corresponding side and images are used.
    In 'both' mode, we render left then right for each group where available.
    """
    # Load per-video CSV
    if not os.path.isfile(per_video_csv_path):
        print(f"Per-video EAR CSV not found: {per_video_csv_path}")
        return False
    dfv = pd.read_csv(per_video_csv_path)
    if 'frame_key' not in dfv.columns:
        print(f"Invalid per-video EAR CSV (missing frame_key): {per_video_csv_path}")
        return False

    # Build dicts by x
    ears_left_by_x = {}
    ears_right_by_x = {}
    for _, r in dfv.iterrows():
        fk = str(r['frame_key'])
        x = extract_frame_index(fk)
        # Only set values if present (avoid NaN overwrites)
        if 'EAR_left' in dfv.columns:
            val = r.get('EAR_left', np.nan)
            if not pd.isna(val):
                ears_left_by_x[x] = float(val)
        if 'EAR_right' in dfv.columns:
            val = r.get('EAR_right', np.nan)
            if not pd.isna(val):
                ears_right_by_x[x] = float(val)

    # Collect overlay frames grouped by frame group (to find images for plotting), sorted by numeric x
    face_overlay_dir = os.path.join(overlays_face_root, video_id)
    grouped = collect_overlay_sequence_grouped(face_overlay_dir)
    if not grouped:
        print(f"No overlay images for video_id={video_id} in {face_overlay_dir}")
        return False

    writer = None

    if mode == "left":
        for group_key, files_dict in grouped:
            if "left" not in files_dict:
                continue
            left_img_bgr = cv2.imread(files_dict["left"])
            if left_img_bgr is None:
                continue
            x_val = extract_frame_index(group_key)

            graph_rgb = render_ear_graph_image_dual_by_x(
                ears_left_by_x, {},
                max_x_frames=max_x_frames,
                fig_size=(5, 5),
                y_limits=None,
                mode="left",
                highlight_x={"left": x_val}
            )
            right_img_bgr = cv2.cvtColor(graph_rgb, cv2.COLOR_RGB2BGR)

            left_img_bgr_annot = annotate_frame_number_on_bgr(left_img_bgr, x_val)
            combined = stack_side_by_side(left_img_bgr_annot, right_img_bgr, target_height=left_img_bgr.shape[0])

            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                h, w = combined.shape[:2]
                os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
                writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
            writer.write(combined)

    elif mode == "right":
        for group_key, files_dict in grouped:
            if "right" not in files_dict:
                continue
            right_overlay_bgr = cv2.imread(files_dict["right"])
            if right_overlay_bgr is None:
                continue
            x_val = extract_frame_index(group_key)

            graph_rgb = render_ear_graph_image_dual_by_x(
                {}, ears_right_by_x,
                max_x_frames=max_x_frames,
                fig_size=(5, 5),
                y_limits=None,
                mode="right",
                highlight_x={"right": x_val}
            )
            graph_bgr = cv2.cvtColor(graph_rgb, cv2.COLOR_RGB2BGR)

            right_overlay_bgr_annot = annotate_frame_number_on_bgr(right_overlay_bgr, x_val)
            combined = stack_side_by_side(right_overlay_bgr_annot, graph_bgr, target_height=right_overlay_bgr.shape[0])

            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                h, w = combined.shape[:2]
                os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
                writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
            writer.write(combined)

    else:  # mode == "both"
        for group_key, files_dict in grouped:
            x_val = extract_frame_index(group_key)

            if "left" in files_dict:
                left_img_bgr = cv2.imread(files_dict["left"])
                if left_img_bgr is not None:
                    graph_rgb = render_ear_graph_image_dual_by_x(
                        ears_left_by_x, ears_right_by_x,
                        max_x_frames=max_x_frames,
                        fig_size=(5, 5),
                        y_limits=None,
                        mode="both",
                        highlight_x={"left": x_val, "right": None}
                    )
                    right_img_bgr = cv2.cvtColor(graph_rgb, cv2.COLOR_RGB2BGR)
                    left_img_bgr_annot = annotate_frame_number_on_bgr(left_img_bgr, x_val)
                    combined = stack_side_by_side(left_img_bgr_annot, right_img_bgr, target_height=left_img_bgr.shape[0])

                    if writer is None:
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        h, w = combined.shape[:2]
                        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
                        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
                    writer.write(combined)

            if "right" in files_dict:
                right_overlay_bgr = cv2.imread(files_dict["right"])
                if right_overlay_bgr is not None:
                    graph_rgb = render_ear_graph_image_dual_by_x(
                        ears_left_by_x, ears_right_by_x,
                        max_x_frames=max_x_frames,
                        fig_size=(5, 5),
                        y_limits=None,
                        mode="both",
                        highlight_x={"left": None, "right": x_val}
                    )
                    graph_bgr = cv2.cvtColor(graph_rgb, cv2.COLOR_RGB2BGR)
                    right_overlay_bgr_annot = annotate_frame_number_on_bgr(right_overlay_bgr, x_val)
                    combined = stack_side_by_side(right_overlay_bgr_annot, graph_bgr, target_height=right_overlay_bgr.shape[0])

                    if writer is None:
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        h, w = combined.shape[:2]
                        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
                        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
                    writer.write(combined)

    if writer is not None:
        writer.release()
        return True
    return False

def build_inference_transforms(cfg):
    image_size = int(cfg['data'].get('image_size', 128))
    mean = cfg['data'].get('mean', [0.485, 0.456, 0.406])
    std = cfg['data'].get('std', [0.229, 0.224, 0.225])
    return T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR, antialias=True),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--visible_only", action="store_true")
    ap.add_argument("--visualize", action="store_true")
    ap.add_argument("--video_fps", type=int, default=25, help="FPS for output videos")
    ap.add_argument("--video_xaxis_frames", type=int, default=500, help="Fixed x-axis length for EAR graph")
    ap.add_argument("--ear_graph_mode", choices=["left", "right", "both"], default="both", help="Which EAR to plot on graph")
    ap.add_argument("--no_augment", action="store_true", help="Use deterministic inference transforms (no random ops)")
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    os.makedirs(args.output_dir, exist_ok=True)
    overlays_face_root = os.path.join(args.output_dir, "overlays_face")
    overlays_eye_root = os.path.join(args.output_dir, "overlays_eye")
    videos_root = os.path.join(args.output_dir, "videos")
    ears_root = os.path.join(args.output_dir, "ears_per_video")
    if args.visualize:
        os.makedirs(overlays_face_root, exist_ok=True)
        os.makedirs(overlays_eye_root, exist_ok=True)
    os.makedirs(videos_root, exist_ok=True)
    os.makedirs(ears_root, exist_ok=True)

    per_sample_csv = os.path.join(args.output_dir, "predicted_landmarks.csv")

    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    model = EyeLandmarkModel(
        backbone_name=cfg['model']['backbone'],
        pretrained=False,
        hidden_landmarks=cfg['model']['hidden_landmarks'],
        dropout=cfg['model']['dropout'],
        num_landmarks=cfg['data']['num_landmarks']
    )
    model.load_state_dict(ckpt['model'], strict=False)
    model.to(device)
    model.eval()

    df = pd.read_csv(args.csv)
    if args.limit is not None:
        df = df.head(args.limit)

    # Sort input rows by video_id, numeric frame index (x), then left before right
    def extract_frame_index_local(frame_key: str) -> int:
        if not isinstance(frame_key, str):
            return -1
        m = re.search(r'frame[_-]?(\d+)', frame_key)
        if m:
            return int(m.group(1))
        m2 = re.search(r'(\d+)', frame_key)
        return int(m2.group(1)) if m2 else -1

    df['frame_index'] = df['frame_key'].apply(extract_frame_index_local)
    df['eye_side_norm'] = df['eye_side'].astype(str).str.strip().str.lower()
    df['eye_order'] = df['eye_side_norm'].map({'left': 0, 'right': 1}).fillna(2)
    df = df.sort_values(by=['video_id', 'frame_index', 'eye_order', 'frame_key'])

    val_tf = build_inference_transforms(cfg) if args.no_augment else build_val_transforms(cfg)
    target_size = int(cfg['data'].get('image_size', 128))

    rows_out = []
    samples_used = 0
    skipped = 0

    # Collect per-video EAR time series for CSV building
    ears_by_video_frame = defaultdict(lambda: defaultdict(lambda: {"left": np.nan, "right": np.nan}))

    for idx, row in enumerate(tqdm(df.itertuples(index=False), total=len(df), desc="Predicting landmarks")):
        img_path = getattr(row, "path_to_dataset", None)
        raw_vis = getattr(row, "eye_visibility", "true")
        video_id = getattr(row, "video_id", "unknown")
        frame_key = getattr(row, "frame_key", f"sample_{idx}")
        eye_side = getattr(row, "eye_side", "unknown")

        gt_vis = robust_parse_visibility(raw_vis)
        if args.visible_only and gt_vis == 0:
            continue

        bbox = parse_bbox(getattr(row, "eye_bbox_face", ""))

        if not isinstance(img_path, str) or not os.path.exists(img_path):
            skipped += 1
            continue
        if bbox is None:
            skipped += 1
            continue

        face_img = Image.open(img_path).convert("RGB")
        eye_img, transform = crop_eye_image_with_transform(face_img, bbox, target_size)
        if eye_img is None or transform is None:
            skipped += 1
            continue

        tensor_img = val_tf(eye_img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(tensor_img)
            pred_lm = out['landmarks'][0].cpu()

        pred_abs = denormalize_predictions(pred_lm, bbox, cfg)
        eye_pts = map_face_pts_to_eye_crop(pred_abs, transform)

        ear_value = compute_ear(pred_abs)

        pts_str = ";".join([f"{x:.2f},{y:.2f}" for (x, y) in pred_abs])
        rows_out.append({
            "index": idx,
            "video_id": video_id,
            "frame_key": frame_key,
            "eye_side": eye_side,
            "img_path": img_path,
            "eye_visibility": gt_vis,
            "num_pred_points": len(pred_abs),
            "pred_landmarks_abs": pts_str,
            "EAR": ear_value,
        })
        samples_used += 1

        if args.visualize:
            face_dir = os.path.join(overlays_face_root, video_id)
            eye_dir = os.path.join(overlays_eye_root, video_id)
            os.makedirs(face_dir, exist_ok=True)
            os.makedirs(eye_dir, exist_ok=True)
            out_png_face = os.path.join(face_dir, f"{frame_key}_{eye_side}.png")
            out_png_eye = os.path.join(eye_dir, f"{frame_key}_{eye_side}.png")
            visualize_face_overlay(face_img, pred_abs, out_png_face)
            visualize_eye_overlay(eye_img, eye_pts, out_png_eye)

        side_norm = str(eye_side).strip().lower()
        if side_norm in ("left", "right"):
            ears_by_video_frame[video_id][frame_key][side_norm] = ear_value

    # Global per-sample CSV (includes EAR)
    pd.DataFrame(rows_out).to_csv(per_sample_csv, index=False)

    # Per-video EAR time-series CSVs aligned by frame_key
    for video_id, frames_dict in ears_by_video_frame.items():
        records = []
        def sort_key_fk(fk): return extract_frame_index(fk)
        for frame_key in sorted(frames_dict.keys(), key=sort_key_fk):
            ed = frames_dict[frame_key]
            records.append({
                "frame_key": frame_key,
                "EAR_left": ed.get("left", np.nan),
                "EAR_right": ed.get("right", np.nan),
            })
        out_csv_video = os.path.join(ears_root, f"{video_id}.csv")
        pd.DataFrame(records).to_csv(out_csv_video, index=False)

    # Build videos per video_id using per-video CSV values (not per-sample rows)
    if args.visualize:
        for video_id in ears_by_video_frame.keys():
            mode = args.ear_graph_mode
            video_path = os.path.join(videos_root, f"{video_id}_{mode}.mp4")
            per_video_csv_path = os.path.join(ears_root, f"{video_id}.csv")
            ok = build_video_per_id_dual_with_csv(
                video_id=video_id,
                overlays_face_root=overlays_face_root,
                per_video_csv_path=per_video_csv_path,
                output_video_path=video_path,
                fps=args.video_fps,
                max_x_frames=args.video_xaxis_frames,
                mode=mode
            )
            if not ok:
                print(f"Warning: Could not build video for video_id={video_id} (no frames or IO error)")

    print("\n=== Prediction-only run (x-indexed graphs using per-video CSV) ===")
    print(f"Samples used: {samples_used}")
    print(f"Skipped samples: {skipped}")
    print("Per-sample predictions CSV:", per_sample_csv)
    print("Per-video EAR CSV root:", ears_root)
    if args.visualize:
        print("Face overlays root:", overlays_face_root)
        print("Eye-crop overlays root:", overlays_eye_root)
        print("Videos root:", videos_root)
        print(f"EAR graph mode: {args.ear_graph_mode}")

if __name__ == "__main__":
    main()