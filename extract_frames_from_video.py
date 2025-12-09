"""
Extract frames from an MP4 (or any video supported by OpenCV) and save them into a folder
named after the video file (without extension). Frames are saved as frame_1.jpg, frame_2.jpg, ...

Usage:
  python /inwdata2a/sudhanshu/landmarks_only_training/inference_pipeline/extract_frames_from_video.py \
   --video /inwdata2a/sudhanshu/landmarks_only_training/inference_pipeline/videos/perclos_valid_drowsy_15_26_4658573554_clip_clip_2.mp4 \
   --output_root /inwdata2a/sudhanshu/landmarks_only_training/inference_pipeline/video_frames

Behavior:
- Creates an output folder: {output_root}/{video_stem}
- Saves frames as JPEG: frame_1.jpg, frame_2.jpg, ...
- Preserves the original frame rate; does not skip frames.
- Prints basic statistics at the end.

Dependencies:
- OpenCV (cv2)
- Python 3.7+

"""

import os
import sys
import argparse
import cv2

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def extract_frames(video_path: str, output_root: str):
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.join(output_root, video_stem)
    ensure_dir(out_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {video_path}")
    print(f"Output folder: {out_dir}")
    print(f"Resolution: {width}x{height}, FPS: {fps:.2f}, Total frames reported: {total_frames}")

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        out_name = f"frame_{count}.jpg"
        out_path = os.path.join(out_dir, out_name)

        # Save as JPEG with reasonable quality
        ok = cv2.imwrite(out_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            print(f"Warning: failed to write frame {count} to {out_path}", file=sys.stderr)

        if count % 100 == 0:
            print(f"Saved {count} frames...")

    cap.release()

    print(f"Done. Saved {count} frames to {out_dir}.")


def main():
    parser = argparse.ArgumentParser(description="Extract frames from a video into a folder as frame_1.jpg, frame_2.jpg, ...")
    parser.add_argument("--video", type=str, required=True, help="Path to the input video file (e.g., .mp4)")
    parser.add_argument("--output_root", type=str, required=True, help="Root directory where the frame folder will be created")
    args = parser.parse_args()

    extract_frames(args.video, args.output_root)


if __name__ == "__main__":
    main()