1. use /inwdata2a/sudhanshu/landmarks_only_training/inference_pipeline/extract_frames_from_video.py which takes videos and saves the frames in desired output.

2. once we have the farmes now run /inwdata2a/sudhanshu/nd_data_processing_scripts/gaze_estimation_mydata.py
 when you run this script, it will save face_detections.json and gaze_eye_crops-final.json in each folder inside output_root

3. since our eyecrops in json file is image relative, but we want them to be relative to face crops, so run
/inwdata2a/sudhanshu/landmarks_only_training/inference_pipeline/convert_eye_crops_to_face_space.py, which takes "face-eye-bbox" folder generated in "2" and gives gaze_eye_crops-face-relative.json


4. now we need face crops in order to create the final training csv which requires the face-crop images path, so run 
/inwdata2a/sudhanshu/landmarks_only_training/inference_pipeline/extract-face-crops.py which takes the video frames folder, face-eye-bbox folder (generated in 2) and saves the face crops in output directory (face-crops)


5. Now run python /inwdata2a/sudhanshu/landmarks_only_training/inference_pipeline/build_inference_csv_from_face_and_eye_json.py
which takes the face-eye-bbox folder (containing face crops, eyecrops wrt image, eye crops wrt face crops in json file for each video id) 
and gives the final csv which is used for train, val and test 

6. The csv doesnt contain sorted frames so run /inwdata2a/sudhanshu/landmarks_only_training/inference_pipeline/sort-csv.py 
it saves the csv which contains the frames in sorted order (temporal 1,2,3..)

7. Now that we have the inference-pairs-sorted.csv run
/inwdata2a/sudhanshu/landmarks_only_training/inference_pipeline/evaluate_predictions_no_gt.py
which predicts eye landmarks with dual overlays (face and eye-crop),
saving overlays organized by video_id. Also computes EAR and generates per-video
side-by-side videos (overlayed face image + EAR graph)