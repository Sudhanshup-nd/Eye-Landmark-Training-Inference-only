#!/usr/bin/env python3
"""
Debug script: run model, crop eyes from face using bbox, and save ONE image per sample
with predicted landmarks drawn and numbered (1..N) in the order the model outputs.

Inputs:
- --checkpoint PATH (required)
- --config PATH (optional, default: configs/default.yaml)
- --csv PATH (required): CSV with columns:
    video_id,frame_key,eye_side,eye_visibility,path_to_dataset,eye_bbox_face
- --output_dir DIR (required)
- --limit N (optional)
- --visible_only (optional)

Usage:
 python /inwdata2a/sudhanshu/landmarks_only_training/inference_pipeline/debug_landmrks_numbers.py \
--checkpoint /inwdata2a/sudhanshu/landmarks_only_training/augmented-outputs_landmarks-randomized-padding/best_landmarks.pt \
--config /inwdata2a/sudhanshu/landmarks_only_training/configs/default.yaml \
--csv /inwdata2a/sudhanshu/landmarks_only_training/inference_pipeline/inference_pairs.csv \
--output_dir outputs-augmented_landmarks-randomized-padding/debug_numbers \
--limit 2 \
--visible_only

Output:
- One PNG per sample:
    {output_dir}/{video_id}/{frame_key}_{eye_side}_indexed.png
  The image is the eye crop (square) with numbered landmarks.

Usage example:
python debug_landmark_numbers.py \
  --checkpoint /path/to/best_landmarks.pt \
  --config configs/default.yaml \
  --csv /path/to/inference_pairs.csv \
  --output_dir /path/to/output_debug_numbers \
  --limit 100 \
  --visible_only
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

# Project path setup (assumes this file is under inference_pipeline/ and code is under src/)
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if __package__ in (None, ""):
    __package__ = "src"

# Internal imports from your project
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
    """
    Crop eye region from face using inclusive bbox, pad to square, resize to target_size.
    Returns:
      crop_img (PIL.Image), transform dict
    """
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

def denormalize_predictions(pred_tensor: torch.Tensor, bbox, cfg):
    """
    Convert predicted normalized local coords back to absolute face coords.
    """
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

def map_face_pts_to_eye_crop(pred_abs_pts, transform):
    """
    Map absolute face coordinates to eye-crop coordinates after padding and resizing.
    """
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

def visualize_eye_overlay_numbered(eye_img, eye_pts, out_png):
    """
    Save the eye crop image with predicted landmarks and their index labels (1-based).
    Only ONE output image per sample.
    """
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(eye_img)
    ax.axis('off')

    if eye_pts:
        for i, (x, y) in enumerate(eye_pts):
            ax.scatter([x], [y], c='orange', s=40, marker='o',
                       edgecolors='black', linewidths=0.8, zorder=5)
            ax.text(x + 2, y - 2, str(i + 1),
                    color='white', fontsize=10, weight='bold',
                    bbox=dict(facecolor='black', alpha=0.5, pad=1, edgecolor='none'),
                    zorder=6)

    ax.set_title("Predicted Landmarks (Eye Crop) indices 1..N")
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches='tight', dpi=150)
    plt.close(fig)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--visible_only", action="store_true")
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)

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

    val_tf = build_val_transforms(cfg)
    target_size = int(cfg['data'].get('image_size', 128))

    saved = 0
    skipped = 0

    for idx, row in enumerate(tqdm(df.itertuples(index=False), total=len(df), desc="Debug: numbering landmarks")):
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

        # Model inference on eye crop
        tensor_img = val_tf(eye_img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(tensor_img)
            pred_lm = out['landmarks'][0].cpu()  # shape: (num_landmarks, 2)

        # Map predictions to eye crop coords (for clearer visualization)
        # First transform predictions back to absolute face coords
        pred_abs = denormalize_predictions(pred_lm, bbox, cfg)
        # Then map into eye crop space
        eye_pts = map_face_pts_to_eye_crop(pred_abs, transform)

        # Output path: one image per sample
        out_dir = os.path.join(args.output_dir, str(video_id))
        os.makedirs(out_dir, exist_ok=True)
        out_png = os.path.join(out_dir, f"{frame_key}_{eye_side}_indexed.png")

        visualize_eye_overlay_numbered(eye_img, eye_pts, out_png)
        saved += 1

    print("\n=== Debug run: numbered landmark overlays ===")
    print(f"Saved images: {saved}")
    print(f"Skipped samples: {skipped}")
    print(f"Output root: {args.output_dir}")

if __name__ == "__main__":
    main()