

import os
import json
from pathlib import Path
from typing import List, Optional, Tuple
import csv
import cv2
import numpy as np
from tqdm import tqdm

# ------------------------------------------------------------------
# EDIT THESE PATHS AND PARAMETERS
# ------------------------------------------------------------------
FRAMES_ROOT = Path("/inwdata2a/sudhanshu/landmarks_only_training/inference_pipeline/video_frames")          # e.g. Path("/inwdata2a/.../extracted-frames-phase-2")
JSON_ROOT   = Path("/inwdata2a/sudhanshu/landmarks_only_training/inference_pipeline/face-eye-bbox")            # e.g. Path("/inwdata2a/.../phase-2-face-crops")
OUTPUT_ROOT = Path("/inwdata2a/sudhanshu/landmarks_only_training/inference_pipeline/face-crops")          # e.g. Path("/inwdata2a/.../face-crops-output")

JSON_NAME         = "face_detections.json"   # Name of the json file in each video folder
IMAGE_EXT         = ".jpg"                   # Frame extension if needed
SUMMARY_CSV_NAME  = "face_crops_summary.csv"

MIN_CONF          = 0.67  # Minimum confidence to keep a face bbox
MIN_SIDE          = 0.0      # Skip crops smaller than this (width or height)
PAD_RATIO         = 0.0    # 0.05 adds 5% padding on each side
CONVERT_RGB       = False  # If True, save in RGB layout (OpenCV writes BGR by default anyway)
SKIP_IF_EXISTS    = False  # If True, do not overwrite existing crop files
VERBOSE           = True   # Extra printing
# ------------------------------------------------------------------


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_json(json_path: Path) -> Optional[List[dict]]:
    if not json_path.exists():
        return None
    try:
        with json_path.open("r") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return None
        return data
    except Exception as e:
        print(f"[WARN] Failed to load {json_path}: {e}")
        return None


def clamp_bbox(x1, y1, x2, y2, w, h) -> Tuple[int, int, int, int]:
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def apply_padding(x1, y1, x2, y2, w, h, pad_ratio: float):
    if pad_ratio <= 0:
        return x1, y1, x2, y2
    bw = x2 - x1 + 1
    bh = y2 - y1 + 1
    pad_w = int(bw * pad_ratio)
    pad_h = int(bh * pad_ratio)
    x1 -= pad_w
    y1 -= pad_h
    x2 += pad_w
    y2 += pad_h
    return clamp_bbox(x1, y1, x2, y2, w, h)


def extract_crops_for_video(video_id: str,
                            rows_out: List[dict]):
    json_path = JSON_ROOT / video_id / JSON_NAME
    data = load_json(json_path)
    if data is None:
        if VERBOSE:
            print(f"[INFO] No or invalid JSON for video {video_id}, skipping.")
        return

    out_subdir = OUTPUT_ROOT / video_id
    ensure_dir(out_subdir)

    frames_dir = FRAMES_ROOT / video_id

    for entry in data:
        img_name = entry.get("img_name")
        if not img_name:
            continue

        frame_path = frames_dir / img_name
        if not frame_path.exists():
            if VERBOSE:
                print(f"[WARN] Missing frame {frame_path}")
            continue

        img = cv2.imread(str(frame_path))
        if img is None:
            if VERBOSE:
                print(f"[WARN] Failed to read image {frame_path}")
            continue
        h, w = img.shape[:2]

        face_bboxes_nested = entry.get("face_bboxes", [])
        box_confs_nested = entry.get("box_confs", [])

        if len(face_bboxes_nested) == 1 and isinstance(face_bboxes_nested[0], list):
            face_bboxes = face_bboxes_nested[0]
        else:
            face_bboxes = face_bboxes_nested

        if len(box_confs_nested) == 1 and isinstance(box_confs_nested[0], list):
            box_confs = box_confs_nested[0]
        else:
            box_confs = box_confs_nested

        if len(box_confs) != len(face_bboxes):
            box_confs = [1.0] * len(face_bboxes)

        for idx_bbox, bbox in enumerate(face_bboxes):
            if not (isinstance(bbox, list) and len(bbox) == 4):
                continue
            x1, y1, x2, y2 = bbox
            conf = float(box_confs[idx_bbox]) if idx_bbox < len(box_confs) else 1.0
            if conf < MIN_CONF:
                continue

            x1, y1, x2, y2 = clamp_bbox(x1, y1, x2, y2, w, h)
            x1, y1, x2, y2 = apply_padding(x1, y1, x2, y2, w, h, PAD_RATIO)

            bw = x2 - x1 + 1
            bh = y2 - y1 + 1
            if bw < MIN_SIDE or bh < MIN_SIDE:
                continue

            crop = img[y1:y2+1, x1:x2+1]
            if CONVERT_RGB:
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            crop_name = f"{Path(img_name).stem}_face{idx_bbox:02d}.jpg"
            crop_path = out_subdir / crop_name

            if SKIP_IF_EXISTS and crop_path.exists():
                pass
            else:
                cv2.imwrite(str(crop_path), crop)

            rows_out.append({
                "video_id": video_id,
                "frame_name": img_name,
                "frame_path": str(frame_path),
                "crop_path": str(crop_path),
                "bbox_x1": x1,
                "bbox_y1": y1,
                "bbox_x2": x2,
                "bbox_y2": y2,
                "width": bw,
                "height": bh,
                "confidence": conf,
                "pad_ratio": PAD_RATIO
            })


def main():
    # Validate roots
    if not FRAMES_ROOT.exists():
        print(f"[ERROR] FRAMES_ROOT does not exist: {FRAMES_ROOT}")
        return
    if not JSON_ROOT.exists():
        print(f"[ERROR] JSON_ROOT does not exist: {JSON_ROOT}")
        return
    ensure_dir(OUTPUT_ROOT)

    # Collect video IDs from JSON root
    video_ids = []
    for child in sorted(JSON_ROOT.iterdir()):
        if child.is_dir():
            json_file = child / JSON_NAME
            if json_file.exists():
                video_ids.append(child.name)

    if not video_ids:
        print("[ERROR] No video folders with JSON found. Check paths.")
        return

    rows: List[dict] = []
    for vid in tqdm(video_ids, desc="Processing videos"):
        extract_crops_for_video(vid, rows_out=rows)

    # Write summary CSV
    csv_path = OUTPUT_ROOT / SUMMARY_CSV_NAME
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video_id", "frame_name", "frame_path", "crop_path",
                "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
                "width", "height", "confidence", "pad_ratio"
            ]
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"[DONE] Extracted {len(rows)} face crops across {len(video_ids)} videos.")
    print(f"Summary CSV: {csv_path}")


if __name__ == "__main__":
    main()