"""
Build an inference CSV from:
- A root folder containing per-video face crop images (each video_id is a subfolder with frames as JPG/PNG)
- A root folder containing per-video eye bbox JSON files (each video_id is a subfolder with gaze_eye_crops-face-relative.json)

Output CSV columns:
  video_id,frame_key,eye_side,eye_visibility,path_to_dataset,eye_bbox_face

Updates for your case:
- Face crop filenames may contain an extra "00" (e.g., frame_536_face00.jpg) instead of the usual "_face_0".
- Eye JSON keys may be "frame_536.jpg".
- This script implements flexible matching to resolve such differences.

Matching strategy:
1) Exact match: {face_dir}/{frame_name}
2) Face00 variants:
   - base + "_face00" with .jpg/.png/.jpeg
3) Common face variants:
   - base + "_face_0" or "_face_1" with .jpg/.png/.jpeg
4) Same base without extension:
   - base + .jpg/.png/.jpeg
5) Glob fallback:
   - files starting with base and containing "face" (preferring jpg, then face index/order)

Usage:
  python /inwdata2a/sudhanshu/landmarks_only_training/inference_pipeline/build_inference_csv_from_face_and_eye_json.py \
    --face_crops_root /inwdata2a/sudhanshu/landmarks_only_training/inference_pipeline/face-crops \
    --eye_bbox_root /inwdata2a/sudhanshu/landmarks_only_training/inference_pipeline/face-eye-bbox \
    --out_csv /inwdata2a/sudhanshu/landmarks_only_training/inference_pipeline/inference_pairs.csv \
    --debug
"""

import os
import json
import argparse
import csv
import glob
from typing import List, Dict, Optional

EYE_JSON_NAME = "gaze_eye_crops-face-relative.json"
VALID_EXTS = (".jpg", ".jpeg", ".png")

def list_video_ids(root_dir: str, debug: bool=False) -> List[str]:
    ids = []
    try:
        for name in os.listdir(root_dir):
            sub = os.path.join(root_dir, name)
            if os.path.isdir(sub):
                ids.append(name)
    except Exception as e:
        print(f"[ERROR] Cannot list directory {root_dir}: {e}")
        return []
    ids_sorted = sorted(ids)
    if debug:
        print(f"[DEBUG] list_video_ids('{root_dir}') -> {len(ids_sorted)} subfolders")
        for i, vid in enumerate(ids_sorted[:10]):
            print(f"  [DEBUG] video_id[{i}]: {vid}")
        if len(ids_sorted) > 10:
            print("  [DEBUG] ...")
    return ids_sorted

def format_bbox(b: List[int]) -> str:
    if not isinstance(b, (list, tuple)) or len(b) != 4:
        return ""
    return f"{int(b[0])},{int(b[1])},{int(b[2])},{int(b[3])}"

def find_face_image_path(face_dir: str, frame_name: str, debug: bool=False) -> Optional[str]:
    """
    Resolve face image path for a given JSON frame key using flexible matching, including 'face00' variant.

    Priority:
      1) Exact match
      2) base + '_face00' + ext
      3) base + '_face_0' or '_face_1' + ext
      4) base + ext
      5) Glob: base*face*
    """
    exact_path = os.path.join(face_dir, frame_name)
    if os.path.isfile(exact_path):
        if debug:
            print(f"[DEBUG] Exact match: {exact_path}")
        return exact_path

    base, _ext = os.path.splitext(frame_name)
    candidates = []

    # 2) face00 variant
    for e in (".jpg", ".png", ".jpeg"):
        candidates.append(os.path.join(face_dir, f"{base}_face00{e}"))

    # 3) common face variants
    for idx in (0, 1):
        for e in (".jpg", ".png", ".jpeg"):
            candidates.append(os.path.join(face_dir, f"{base}_face_{idx}{e}"))

    # 4) same base with other extensions
    for e in (".jpg", ".png", ".jpeg"):
        candidates.append(os.path.join(face_dir, f"{base}{e}"))

    for c in candidates:
        if os.path.isfile(c):
            if debug:
                print(f"[DEBUG] Variant match: {c}")
            return c

    # 5) glob fallback
    pattern = os.path.join(face_dir, f"{base}*face*")
    matches = [m for m in glob.glob(pattern) if os.path.isfile(m) and os.path.splitext(m)[1].lower() in VALID_EXTS]
    if matches:
        def sort_key(p):
            ext_rank = 0 if p.lower().endswith((".jpg", ".jpeg")) else 1
            name = os.path.basename(p)
            # Prefer 'face00' over others, then try to parse index
            if "_face00" in name:
                face_rank = -1
            else:
                face_rank = 999
                for idx in (0, 1, 2, 3):
                    if f"_face_{idx}" in name:
                        face_rank = idx
                        break
            return (ext_rank, face_rank, len(name))
        matches.sort(key=sort_key)
        best = matches[0]
        if debug:
            print(f"[DEBUG] Glob match: {best}")
        return best

    if debug:
        print(f"[DEBUG] No match for base '{base}' in '{face_dir}'")
    return None

def build_rows_for_video(
    video_id: str,
    face_crops_root: str,
    eye_bbox_root: str,
    default_visibility: bool = True,
    debug: bool=False
) -> List[Dict[str, str]]:
    rows = []
    face_dir = os.path.join(face_crops_root, video_id)
    eye_dir = os.path.join(eye_bbox_root, video_id)
    eye_json_path = os.path.join(eye_dir, EYE_JSON_NAME)

    if debug:
        print(f"[DEBUG] Video '{video_id}' face_dir exists={os.path.isdir(face_dir)} eye_json exists={os.path.isfile(eye_json_path)}")

    if not os.path.isdir(face_dir):
        print(f"[WARN] Face crops folder missing for video_id: {video_id} at {face_dir}")
        return rows
    if not os.path.isfile(eye_json_path):
        print(f"[WARN] Eye JSON missing for video_id: {video_id} at {eye_json_path}")
        return rows

    try:
        with open(eye_json_path, "r") as f:
            eye_map = json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to read JSON for {video_id}: {e}")
        return rows

    if debug:
        print(f"[DEBUG] '{video_id}' JSON frames: {len(eye_map)}")

    missing_images = 0
    malformed = 0
    added = 0

    for frame_name, eyes in eye_map.items():
        if not isinstance(eyes, list) or len(eyes) != 2:
            malformed += 1
            if debug:
                print(f"[DEBUG] Skip '{frame_name}': eyes malformed ({type(eyes)})")
            continue

        img_path = find_face_image_path(face_dir, frame_name, debug=debug)
        if img_path is None:
            missing_images += 1
            continue

        frame_key = os.path.splitext(os.path.basename(img_path))[0]  # e.g., frame_536_face00

        left_bbox, right_bbox = eyes
        if isinstance(left_bbox, list) and len(left_bbox) == 4:
            rows.append({
                "video_id": video_id,
                "frame_key": frame_key,
                "eye_side": "left",
                "eye_visibility": "True" if default_visibility else "False",
                "path_to_dataset": img_path,
                "eye_bbox_face": format_bbox(left_bbox),
            })
            added += 1
        if isinstance(right_bbox, list) and len(right_bbox) == 4:
            rows.append({
                "video_id": video_id,
                "frame_key": frame_key,
                "eye_side": "right",
                "eye_visibility": "True" if default_visibility else "False",
                "path_to_dataset": img_path,
                "eye_bbox_face": format_bbox(right_bbox),
            })
            added += 1

    print(f"[INFO] Video '{video_id}': frames={len(eye_map)} rows_added={added} missing_images={missing_images} malformed={malformed}")
    if added == 0 and debug:
        print(f"[DEBUG] Example unresolved key for '{video_id}': {next(iter(eye_map.keys()), 'N/A')}")

    return rows

def main():
    parser = argparse.ArgumentParser(description="Build inference CSV from face crops and eye bbox JSONs (flexible matching incl. face00).")
    parser.add_argument("--face_crops_root", type=str, required=True)
    parser.add_argument("--eye_bbox_root", type=str, required=True)
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--visibility", type=str, default="True",
                        help="Default eye_visibility for all rows (True/False)")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug prints")
    args = parser.parse_args()

    face_crops_root = args.face_crops_root
    eye_bbox_root = args.eye_bbox_root
    out_csv = args.out_csv

    print(f"[INFO] face_crops_root: {face_crops_root}")
    print(f"[INFO] eye_bbox_root:   {eye_bbox_root}")
    print(f"[INFO] out_csv:         {out_csv}")

    if not os.path.isdir(face_crops_root):
        raise NotADirectoryError(f"Face crops root not found: {face_crops_root}")
    if not os.path.isdir(eye_bbox_root):
        raise NotADirectoryError(f"Eye bbox root not found: {eye_bbox_root}")

    default_visibility = str(args.visibility).strip().lower() in ("true", "1", "yes", "y")

    face_ids = set(list_video_ids(face_crops_root, debug=args.debug))
    eye_ids = set(list_video_ids(eye_bbox_root, debug=args.debug))
    common_ids = sorted(face_ids.intersection(eye_ids))
    if not common_ids:
        print("[ERROR] No common video_ids found between roots.")
        if args.debug:
            print(f"[DEBUG] face_ids={sorted(face_ids)}")
            print(f"[DEBUG] eye_ids={sorted(eye_ids)}")
        return

    print(f"[INFO] Common video_id(s): {', '.join(common_ids)}")
    print("[INFO] Building rows...")

    all_rows: List[Dict[str, str]] = []
    for vid in common_ids:
        rows = build_rows_for_video(
            video_id=vid,
            face_crops_root=face_crops_root,
            eye_bbox_root=eye_bbox_root,
            default_visibility=default_visibility,
            debug=args.debug
        )
        print(f"[INFO] [{vid}] rows collected: {len(rows)}")
        all_rows.extend(rows)

    if not all_rows:
        print("[ERROR] No rows generated. Nothing to write.")
        return

    fieldnames = ["video_id", "frame_key", "eye_side", "eye_visibility", "path_to_dataset", "eye_bbox_face"]
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)

    print(f"[INFO] CSV written: {out_csv}")
    print(f"[INFO] Total rows: {len(all_rows)}")

if __name__ == "__main__":
    main()