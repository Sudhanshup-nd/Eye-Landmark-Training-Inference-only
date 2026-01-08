'''
 This code takes as input:
A root folder containing subfolders of images (frames extracted from videos)
A CSV file listing specific frames to process (with columns: image_path) (generated from annotation.xml file of cvat export)
gaze_estimation_mydata.py contains the main pipeline that calls this script to generate face and eye crops for all frames of each folder.
but during labelling data creation we randomly sampled 50 frames out of 1800 frames per folder, so to get the face and eye crops of only those 50 frames,
we modified this script to read the CSV file and only process those frames listed in the CSV 
'''





import warnings  
warnings.filterwarnings('ignore') 
import os  
import subprocess
import sys
import torch
import torchvision.transforms as transforms
sys.path.append(os.path.abspath('../..')) 
from D3DFA import mobilenet_v1
import numpy as np
import cv2
import dlib
from utils.ddfa import ToTensorGjz, NormalizeGjz, str2bool
import scipy.io as sio
from utils.inference import get_suffix, parse_roi_box_from_landmark, crop_img, predict_68pts, dump_to_ply, dump_vertex, \
    draw_landmarks, predict_dense, parse_roi_box_from_bbox, get_colors, write_obj_with_colors
from utils.cv_plot import plot_pose_box
import tqdm
from utils.estimate_pose import parse_pose
from utils.render import get_depths_image, cget_depths_image, cpncc
from utils.paf import gen_img_paf
import argparse
import torch.backends.cudnn as cudnn
import json
import pandas as pd
from yoloface.face_detector import YoloDetector
from PIL import Image
from collections import Counter
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from pathlib import Path
import onnxruntime as ort
from scipy.ndimage import gaussian_filter1d
from models import *

STD_SIZE = 120

# ------------------------------------------------------------------------------------------------------
# 1️⃣ --- FUNCTIONS (unchanged except each_uuid which now supports CSV filtering) ---
# ------------------------------------------------------------------------------------------------------

def generate_face_crop(src_dir, json_path):
    model = YoloDetector(target_size=None, device="cuda:0", min_face=220)
    json_file_name = json_path + '.json'
    os.makedirs(os.path.dirname(json_file_name), exist_ok=True)
    data = []

    for file_name in os.listdir(src_dir):
        if not file_name.endswith('.jpg'):
            continue
        image_path = os.path.join(src_dir, file_name)
        _orgimg = np.array(Image.open(image_path))
        orgimg = _orgimg
        bboxes, points, box_confs = model.predict(orgimg, conf_thres=0.3, iou_thres=0.5)
        if len(bboxes)==0 or len(bboxes[0])==0:
            continue
        print("Found face. Saving...")
        d = {'img_name': file_name, 'face_bboxes': bboxes, 'keypoints': points, 'box_confs': box_confs}
        data.append(d)

    with open(json_file_name, 'w') as f:
        json.dump(data, f)
    print(f"✅ Saved {len(data)} detections to {json_file_name}")


def get_eye_patch(txt_file_path, img_path=None, out_path=None):
    with open(txt_file_path, 'r') as f:
        lines = f.readlines()
    txt_file = []
    txt_file.append([float(x) for x in lines[0].strip().split(' ')])
    txt_file.append([float(x) for x in lines[1].strip().split(' ')])

    low_x_l, low_y_l, high_x_l, high_y_l = 100000, 100000, -1, -1
    low_x_r, low_y_r, high_x_r, high_y_r = 100000, 100000, -1, -1

    for i in range(len(txt_file[0])):
        if i > 35 and i < 42:
            if txt_file[0][i] < low_x_l: low_x_l = txt_file[0][i]
            if txt_file[1][i] < low_y_l: low_y_l = txt_file[1][i]
            if txt_file[0][i] > high_x_l: high_x_l = txt_file[0][i]
            if txt_file[1][i] > high_y_l: high_y_l = txt_file[1][i]
        if i > 41 and i < 48:
            if txt_file[0][i] < low_x_r: low_x_r = txt_file[0][i]
            if txt_file[1][i] < low_y_r: low_y_r = txt_file[1][i]
            if txt_file[0][i] > high_x_r: high_x_r = txt_file[0][i]
            if txt_file[1][i] > high_y_r: high_y_r = txt_file[1][i]

    dx_left = high_x_l - low_x_l
    dx_right = high_x_r - low_x_r
    _fraction = 0.5
    offset_x_l, offset_y_l = int(dx_left * _fraction), int(dx_left * _fraction)
    offset_x_r, offset_y_r = int(dx_right * _fraction), int(dx_right * _fraction)
    bb_list = []
    bb_list.append([int(low_x_l) - offset_x_l, int(low_y_l) - offset_y_l,
                    int(high_x_l) + offset_x_l, int(high_y_l)+offset_y_l])
    bb_list.append([int(low_x_r) - offset_x_r, int(low_y_r) - offset_y_r,
                    int(high_x_r) + offset_x_r, int(high_y_r)+offset_y_r])
    return bb_list


def process_to_df(json_data: list):
    sample_ = json_data[0]
    cols = sample_.keys()
    vals = []
    for d in json_data:
        l = []
        for col in cols:
            l.append(d[col])
        vals.append(l)
    df = pd.DataFrame(vals, columns=cols)
    return df


# ✅ UPDATED each_uuid() - now respects CSV frame filtering
def each_uuid(args, img_dir, out_dir):
    use_direct_yoloface_predictions = True
    if use_direct_yoloface_predictions:
        __path = os.path.join(out_dir, "face_detections.json")
        with open(__path, 'r') as f:
            y = json.load(f)
            _d = []
            for _frame in y:
                _frame['img_name'] = _frame['img_name']
                _d.append(_frame)
        yoloface_df = process_to_df(_d)
        crops_dir = os.path.join(out_dir, "crops")
        os.makedirs(crops_dir, exist_ok=True)
        yoloface_df.to_csv(os.path.join(crops_dir, "gaze_json_eye_check.csv"))

    outdir = out_dir
    checkpoint_fp = '/inwdata2a/sudhanshu/nd_data_processing_scripts/D3DFA/models/phase1_wpdc_vdc.pth.tar'
    arch = 'mobilenet_1'
    checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
    model = getattr(mobilenet_v1, arch)(num_classes=62)
    model_dict = model.state_dict()
    for k in checkpoint.keys():
        model_dict[k.replace('module.', '')] = checkpoint[k]
    model.load_state_dict(model_dict)
    if args.mode == 'gpu':
        cudnn.benchmark = True
        model = model.cuda()
    model.eval()

    if args.dlib_landmark:
        dlib_landmark_model = '/inwdata2a/sudhanshu/nd_data_processing_scripts/D3DFA/models/shape_predictor_68_face_landmarks.dat'
        face_regressor = dlib.shape_predictor(dlib_landmark_model)
    if args.dlib_bbox:
        face_detector = dlib.get_frontal_face_detector()

    tri = sio.loadmat('/inwdata2a/sudhanshu/nd_data_processing_scripts/D3DFA/visualize/tri.mat')['tri']
    transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])

    # ✅ Only use filtered frames if provided
    if hasattr(args, "filtered_frames") and args.filtered_frames:
        img_files = [os.path.join(img_dir, x) for x in args.filtered_frames if os.path.exists(os.path.join(img_dir, x))]
        print(f"➡️ Using {len(img_files)} filtered frames from CSV.")
    else:
        img_files = [os.path.join(img_dir, x) for x in os.listdir(img_dir) if x.endswith('.jpg')]
        print(f"⚠️ No filter list provided, processing all {len(img_files)} frames.")

    count_face_missing = 0
    os.makedirs(os.path.join(out_dir, "txt"), exist_ok=True)
    for img_fp in tqdm.tqdm(img_files, desc=f"Processing {os.path.basename(img_dir)}"):
        img_ori = cv2.imread(img_fp)
        if img_ori is None:
            continue

        image_name = os.path.basename(img_fp)
        filtered = yoloface_df[yoloface_df['img_name'] == image_name]
        if filtered.empty:
            continue
        rects = filtered['face_bboxes'].values[0][0]
        if not len(rects):
            continue

        # --- rest of your original code continues here unchanged ---
        suffix = get_suffix(img_fp)
        roi_box = parse_roi_box_from_bbox(rects[0])
        img = crop_img(img_ori, roi_box)
        img = cv2.resize(img, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
        input = transform(img).unsqueeze(0)
        with torch.no_grad():
            if args.mode == 'gpu':
                input = input.cuda()
            param = model(input)
            param = param.squeeze().cpu().numpy().flatten().astype(np.float32)
        pts68 = predict_68pts(param, roi_box)
        wfp = '{}_{}.txt'.format(os.path.join(outdir,'txt', img_fp.replace(suffix, '').split('/')[-1]), 0)
        np.savetxt(wfp, pts68, fmt='%.3f')
    print(f"✅ Finished processing {len(img_files)} filtered frames for {os.path.basename(img_dir)}.")


# ------------------------------------------------------------------------------------------------------
# 2️⃣ --- MAIN PIPELINE (only folder/frame loading modified to use CSV) ---
# ------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3DDFA inference pipeline')
    parser.add_argument('-m', '--mode', default='cpu', type=str)
    parser.add_argument('--show_flg', default='false', type=str2bool)
    parser.add_argument('--bbox_init', default='one', type=str)
    parser.add_argument('--dump_res', default='false', type=str2bool)
    parser.add_argument('--dump_vertex', default='false', type=str2bool)
    parser.add_argument('--dump_pts', default='true', type=str2bool)
    parser.add_argument('--dump_roi_box', default='false', type=str2bool)
    parser.add_argument('--dump_pose', default='false', type=str2bool)
    parser.add_argument('--dump_depth', default='false', type=str2bool)
    parser.add_argument('--dump_pncc', default='false', type=str2bool)
    parser.add_argument('--dump_paf', default='false', type=str2bool)
    parser.add_argument('--paf_size', default=3, type=int)
    parser.add_argument('--dump_obj', default='false', type=str2bool)
    parser.add_argument('--dlib_bbox', default='true', type=str2bool)
    parser.add_argument('--dlib_landmark', default='False', type=str2bool)
    args = parser.parse_args()

    input_root = "/inwdata2a/sudhanshu/video-path-creation/extracted_frames-final"
    output_root = "/inwdata2a/sudhanshu/eyelid-training/gaze_eyecrops4"
    csv_file = "/inwdata2a/sudhanshu/data-and-labels/annotation-without-skipping-any-folder-name.csv"

    # --- Load CSV and group by folder ---
    csv_df = pd.read_csv(csv_file)
    csv_df['folder'] = csv_df['image_path'].apply(lambda x: x.split('/')[1])
    csv_df['frame'] = csv_df['image_path'].apply(lambda x: x.split('/')[-1].replace('_face_0.jpg', '.jpg'))
    folder_to_frames = csv_df.groupby('folder')['frame'].apply(list).to_dict()
    print(f"✅ Loaded CSV with {len(folder_to_frames)} unique folders.")

    # --- Process only CSV-listed frames ---
    for folder, frame_list in tqdm.tqdm(folder_to_frames.items(), desc="Processing filtered folders"):
        img_dir = os.path.join(input_root, folder)
        out_dir = os.path.join(output_root, folder)
        json_path = os.path.join(out_dir, "face_detections")
        os.makedirs(out_dir, exist_ok=True)

        # Step 1: Face detection for filtered frames
        model = YoloDetector(target_size=None, device="cuda:0", min_face=220)
        json_file_name = json_path + '.json'
        data = []
        for file_name in tqdm.tqdm(frame_list, desc=f"Detecting faces in {folder}", leave=False):
            img_path = os.path.join(img_dir, file_name)
            if not os.path.exists(img_path):
                continue
            _orgimg = np.array(Image.open(img_path))
            orgimg = _orgimg
            bboxes, points, box_confs = model.predict(orgimg, conf_thres=0.3, iou_thres=0.5)
            if len(bboxes) == 0 or len(bboxes[0]) == 0:
                continue
            d = {'img_name': file_name, 'face_bboxes': bboxes, 'keypoints': points, 'box_confs': box_confs}
            data.append(d)

        with open(json_file_name, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✅ Saved {len(data)} face detections for {folder}")

        if len(data) == 0:
            continue

        # Step 2: Landmark + eye crop generation
        args.filtered_frames = frame_list
        each_uuid(args, img_dir, out_dir)

        save_dict = {}
        _model_txt_outputs_dir = os.path.join(out_dir, "txt")
        os.makedirs(_model_txt_outputs_dir, exist_ok=True)
        all_txt_files = os.listdir(_model_txt_outputs_dir)
        for _txt_file in tqdm.tqdm(all_txt_files, desc=f"Extracting eyes for {folder}", leave=False):
            _txt_file_path = os.path.join(_model_txt_outputs_dir, _txt_file)
            bb_list = get_eye_patch(_txt_file_path, img_path=None)
            save_dict[_txt_file.replace('_0.txt', '.jpg')] = bb_list

        out_path = os.path.join(out_dir, 'gaze_eye_crops-final.json')
        with open(out_path, 'w') as f:
            json.dump(save_dict, f)
        print(f"✅ Finished gaze crops for {folder}")

    print("\n🎯 Finished processing all CSV-listed frames successfully.")
