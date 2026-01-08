"""
Batch convert eye crops from original-image coordinates to face-crop coordinates across multiple folders.

Assumptions:
- Each target folder contains:
  - face_detections.json
  - gaze_eye_crops-final.json
- You use RAW face crops (no resizing, no padding). We only translate and clamp to face-crop bounds.

Behavior:
- Scans all immediate subfolders of --root_dir (and optionally the root itself) and for each folder
  that has both JSON files, produces gaze_eye_crops-face-relative.json alongside them.

Usage:
  python /inwdata2a/sudhanshu/landmarks_only_training/inference_pipeline/convert_eye_crops_to_face_space.py \
    --root_dir /inwdata2a/sudhanshu/landmarks_only_training/inference_pipeline/face-eye-bbox

Examples matching your folders:
- Root with these children:
    /inwdata2a/sudhanshu/landmarks_only_training/inference_pipeline/face-eye-bbox/8_gaze_p_ew2_ebc_frame_wise_combined_01
    /inwdata2a/sudhanshu/landmarks_only_training/inference_pipeline/face-eye-bbox/perclos_valid_drowsy_15_26_4658573554_clip_clip_2

This script will detect both folders and create:
    gaze_eye_crops-face-relative.json
in each of them.
"""

import os
import json
import argparse
from typing import Dict, Tuple, List

FACE_JSON_NAME = "face_detections.json"
EYE_JSON_NAME = "gaze_eye_crops-final.json"
OUT_JSON_NAME = "gaze_eye_crops-face-relative.json"


def load_face_detections(face_json_path: str) -> Dict[str, Tuple[int, int, int, int]]:
    """
    Load face_detections.json and build a map: img_name -> primary face bbox (Fx1, Fy1, Fx2, Fy2).
    The input 'face_bboxes' can be nested; we take the first bbox if present.
    """
    with open(face_json_path, "r") as f:
        data = json.load(f)

    img_to_face = {}
    for entry in data:
        img_name = entry.get("img_name")
        face_bboxes = entry.get("face_bboxes", [])
        bbox = None

        def first_bbox_from_nested(nested):
            if isinstance(nested, list) and len(nested) > 0:
                # Case 1: [[x1,y1,x2,y2], ...]
                if isinstance(nested[0], list) and len(nested[0]) == 4 and all(isinstance(v, (int, float)) for v in nested[0]):
                    return nested[0]
                # Dive deeper if further nested
                for item in nested:
                    if isinstance(item, list):
                        fb = first_bbox_from_nested(item)
                        if fb is not None:
                            return fb
            return None

        bbox = first_bbox_from_nested(face_bboxes)
        if bbox is None or img_name is None:
            continue

        Fx1, Fy1, Fx2, Fy2 = [int(round(v)) for v in bbox]
        img_to_face[img_name] = (Fx1, Fy1, Fx2, Fy2)

    return img_to_face


def clamp_bbox(x1, y1, x2, y2, w, h):
    """
    Clamp a bbox to image bounds [0, w) x [0, h).
    """
    nx1 = max(0, min(x1, w - 1))
    ny1 = max(0, min(y1, h - 1))
    nx2 = max(0, min(x2, w - 1))
    ny2 = max(0, min(y2, h - 1))
    if nx2 < nx1: nx1, nx2 = nx2, nx1
    if ny2 < ny1: ny1, ny2 = ny2, ny1
    return nx1, ny1, nx2, ny2


def convert_folder(folder_path: str) -> bool:
    """
    Convert eye crops for a single folder if both required JSON files exist.
    Returns True if conversion succeeded, False otherwise.
    """
    face_json_path = os.path.join(folder_path, FACE_JSON_NAME)
    eye_json_path = os.path.join(folder_path, EYE_JSON_NAME)
    out_json_path = os.path.join(folder_path, OUT_JSON_NAME)

    if not (os.path.isfile(face_json_path) and os.path.isfile(eye_json_path)):
        return False

    try:
        img_to_face = load_face_detections(face_json_path)
        with open(eye_json_path, "r") as f:
            eye_map = json.load(f)
    except Exception as e:
        print(f"[{folder_path}] Failed to load inputs: {e}")
        return False

    out_map = {}
    total_images = 0
    converted_images = 0

    for img_name, eyes in eye_map.items():
        total_images += 1
        if img_name not in img_to_face:
            continue

        Fx1, Fy1, Fx2, Fy2 = img_to_face[img_name]
        Fw = max(1, Fx2 - Fx1)
        Fh = max(1, Fy2 - Fy1)

        face_relative_eyes = []
        ok = True
        for bbox in eyes:
            if not isinstance(bbox, list) or len(bbox) != 4:
                ok = False
                break
            Ex1, Ey1, Ex2, Ey2 = bbox

            # Translate to face coordinates (no scaling, no padding)
            x1_face = int(round(Ex1 - Fx1))
            y1_face = int(round(Ey1 - Fy1))
            x2_face = int(round(Ex2 - Fx1))
            y2_face = int(round(Ey2 - Fy1))

            # Clamp inside face-crop bounds
            x1_face, y1_face, x2_face, y2_face = clamp_bbox(x1_face, y1_face, x2_face, y2_face, Fw, Fh)
            face_relative_eyes.append([x1_face, y1_face, x2_face, y2_face])

        if ok and len(face_relative_eyes) == 2:
            out_map[img_name] = face_relative_eyes
            converted_images += 1

    try:
        with open(out_json_path, "w") as f:
            json.dump(out_map, f, indent=2)
        print(f"[{folder_path}] Saved {converted_images}/{total_images} images to {OUT_JSON_NAME}")
        return True
    except Exception as e:
        print(f"[{folder_path}] Failed to write output: {e}")
        return False


def find_target_folders(root_dir: str, include_root_if_has_files: bool = True) -> List[str]:
    """
    Return list of folders under root_dir that contain the required JSON files.
    Optionally include root_dir itself if it has the files.
    """
    targets = []
    if include_root_if_has_files:
        if os.path.isfile(os.path.join(root_dir, FACE_JSON_NAME)) and os.path.isfile(os.path.join(root_dir, EYE_JSON_NAME)):
            targets.append(root_dir)

    # Scan immediate subfolders
    for name in os.listdir(root_dir):
        sub = os.path.join(root_dir, name)
        if not os.path.isdir(sub):
            continue
        face_ok = os.path.isfile(os.path.join(sub, FACE_JSON_NAME))
        eye_ok = os.path.isfile(os.path.join(sub, EYE_JSON_NAME))
        if face_ok and eye_ok:
            targets.append(sub)

    return targets


def main():
    parser = argparse.ArgumentParser(description="Batch convert eye crops to face-crop coordinates across folders.")
    parser.add_argument("--root_dir", type=str, required=True,
                        help="Root directory containing one or more folders with face_detections.json and gaze_eye_crops-final.json")
    parser.add_argument("--include_root", action="store_true",
                        help="Include the root_dir itself for conversion if it contains the JSON files")
    args = parser.parse_args()

    root_dir = args.root_dir
    if not os.path.isdir(root_dir):
        raise NotADirectoryError(f"Root directory not found: {root_dir}")

    targets = find_target_folders(root_dir, include_root_if_has_files=args.include_root)
    if not targets:
        print("No folders found with required JSON files.")
        return

    print(f"Found {len(targets)} folder(s) to convert.")
    success = 0
    for folder in targets:
        ok = convert_folder(folder)
        if ok:
            success += 1

    print(f"Conversion complete. Successful: {success}/{len(targets)}")


if __name__ == "__main__":
    main()