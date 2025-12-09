
# Code to make video from saved images 
import cv2
import os
from glob import glob

def frames_to_video(frames_dir, output_path, fps=0.8):
    # Get all image files and sort them (by name)
    frame_files = sorted(glob(os.path.join(frames_dir, "*.png")))
    print(f"Found {len(frame_files)} frames in {frames_dir}")

    if not frame_files:
        print("No frames found!")
        return

    # Read first frame to get size
    first_frame = cv2.imread(frame_files[0])
    if first_frame is None:
        print(f"Could not read the first frame: {frame_files[0]}")
        return
    height, width = first_frame.shape[:2]
    print(f"Frame size: width={width}, height={height}")

    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    for fname in frame_files:
        frame = cv2.imread(fname)
        if frame is None:
            print(f"Warning: Could not read {fname}")
            continue
        if frame.shape[0] != height or frame.shape[1] != width:
            print(f"Warning: Frame {fname} has different size {frame.shape[1]}x{frame.shape[0]}, resizing.")
            frame = cv2.resize(frame, (width, height))
        out.write(frame)
        frame_count += 1
        print(f"Wrote frame {frame_count}: {fname}")

    out.release()
    print(f"Video saved to {output_path}")
    print(f"Total frames written: {frame_count}")
    if frame_count == 0:
        print("Warning: No frames were written to the video. Check your input images.")

if __name__ == "__main__":
    frames_dir = "/inwdata2a/sudhanshu/landmarks_only_training/outputs-augmented_landmarks-randomized-padding/eval_no_gt/overlays_face/perclos_valid_drowsy_15_26_4658573554_clip_clip_2"  # Change if needed
    output_path = "/inwdata2a/sudhanshu/landmarks_only_training/outputs-augmented_landmarks-randomized-padding/eval_no_gt/overlays_face/perclos_valid_drowsy_15_26_4658573554_clip_clip_2.mp4"  # Change if needed
    frames_to_video(frames_dir, output_path, fps=0.8)







# # Code to convert any video to powerpoint friendly format (mp4 with h264 and aac)
# import ffmpeg
# import sys

# def convert_to_powerpoint_friendly(input_path, output_path, trim_one_minute=False):
#     """
#     Converts any video to MP4 with H.264 video codec and AAC audio codec.
#     Optionally trims the video to the first 1 minute.
#     Suitable for PowerPoint embedding.
#     """
#     try:
#         stream = ffmpeg.input(input_path)
        
#         if trim_one_minute:
#             # Trim video to the first 60 seconds
#             stream = ffmpeg.input(input_path, t=60)
        
#         (
#             stream
#             .output(output_path, vcodec='libx264', acodec='aac', strict='experimental', movflags='faststart')
#             .overwrite_output()
#             .run()
#         )
#         print(f"Conversion complete. Saved as: {output_path}")
#     except ffmpeg.Error as e:
#         print("Error during conversion:", e)
#         sys.exit(1)

# # ------------------ Example Usage ------------------
# input_video = "/inwdata2a/sudhanshu/eyelid-training/landmarks_video (2).mp4"
# output_video = "/inwdata2a/sudhanshu/eyelid-training/converted_video.mp4"

# # Ask user if they want only 1 min
# take_one_min = input("Do you want to trim the video to 1 minute? (y/n): ").strip().lower()
# convert_to_powerpoint_friendly(input_video, output_video, trim_one_minute=(take_one_min == 'y'))

