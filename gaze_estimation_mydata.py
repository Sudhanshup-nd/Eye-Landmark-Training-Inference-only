
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
# import matplotlib.pyplot as plt
# import mplcursors
# import plotly.tools as tls
# import plotly.graph_objects as go

from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable

from pathlib import Path
# import onnx
import onnxruntime as ort
from scipy.ndimage import gaussian_filter1d
from models import *


STD_SIZE = 120

def generate_face_crop(src_dir, json_path):

    model = YoloDetector(target_size=None, device="cuda:0", min_face=220)

    uuid_cnt = 0
    uuid_dir_path = src_dir
    
    json_file_name = json_path + '.json'
    os.makedirs(os.path.dirname(json_file_name), exist_ok=True)
    
    data = []
    uuid_cnt +=1
    count = 0
   # uuid_frames_path = os.path.join(uuid_dir_path, "extracted_frames")
    uuid_frames_path = uuid_dir_path
    for file_name in os.listdir(uuid_frames_path):
        
        if not file_name.endswith('.jpg'):
            continue
        image_path = os.path.join(uuid_frames_path, file_name)
        _orgimg = np.array(Image.open(image_path))
        # orgimg = np.stack([_orgimg, _orgimg, _orgimg], axis=-1)
        orgimg = _orgimg # for 1600x1200 image size
        bboxes, points, box_confs = model.predict(orgimg, conf_thres=0.3, iou_thres = 0.5)

        if len(bboxes)==0:
            print("Couldn't find a face")
            continue
        if len(bboxes[0])==0:
            print("Couldn't find a face")
            continue

        print("Found face. Saving...")

        d = {}
        d['img_name']       = file_name
        d['face_bboxes']    = bboxes
        d['keypoints']      = points
        d['box_confs']      = box_confs
        data.append(d)
        count+=1
    with open(json_file_name, 'w') as f:
        json.dump(data, f)

def get_eye_patch(txt_file_path, img_path=None, out_path=None):
    # txt_file_path = '/data5/ishita/teyed_detections/ip_documentation/outcrops/txt/frame_0001_0.txt'
    # img_path = '/data5/ishita/teyed_detections/ip_documentation/extracted_frames/frame_0001.png'

    with open(txt_file_path, 'r') as f:
        lines = f.readlines()

    txt_file = []
    txt_file.append([float(x) for x in lines[0].strip().split(' ')])
    txt_file.append([float(x) for x in lines[1].strip().split(' ')])


    # img = cv2.imread(img_path)

    low_x_l, low_y_l, high_x_l, high_y_l = 100000, 100000, -1, -1
    low_x_r, low_y_r, high_x_r, high_y_r = 100000, 100000, -1, -1

    for i in range(len(txt_file[0])):

        if i > 35 and i < 42:
            #cv2.circle(img, (int(txt_file[0][i]), int(txt_file[1][i])), 1, (255,0,0), 2)
            if txt_file[0][i] < low_x_l:
                low_x_l = txt_file[0][i]
            if txt_file[1][i] < low_y_l:
                low_y_l = txt_file[1][i]
            if txt_file[0][i] > high_x_l:
                high_x_l = txt_file[0][i]
            if txt_file[1][i] > high_y_l:
                high_y_l = txt_file[1][i]

        if i > 41 and i < 48:
            #cv2.circle(img, (int(txt_file[0][i]), int(txt_file[1][i])), 1, (255,0,0), 2)
            if txt_file[0][i] < low_x_r:
                low_x_r = txt_file[0][i]
            if txt_file[1][i] < low_y_r:
                low_y_r = txt_file[1][i]
            if txt_file[0][i] > high_x_r:
                high_x_r = txt_file[0][i]
            if txt_file[1][i] > high_y_r:
                high_y_r = txt_file[1][i]

    dx_left = high_x_l - low_x_l
    dy_left = high_y_l - low_y_l
    dx_right = high_x_r - low_x_r
    dy_right = high_y_r - low_y_r

    _fraction = 0.5

    offset_x_l, offset_y_l = int(dx_left * _fraction),  int(dx_left * _fraction) #int(dy_left * 0.6)
    offset_x_r, offset_y_r = int(dx_right * _fraction), int(dx_right * _fraction) #int(dy_right * 0.6)

    bb_list = []
    bb_list.append([int(low_x_l) - offset_x_l, int(low_y_l) - offset_y_l, int(high_x_l) + offset_x_l, int(high_y_l)+offset_y_l])
    bb_list.append([int(low_x_r) - offset_x_r, int(low_y_r) - offset_y_r, int(high_x_r) + offset_x_r, int(high_y_r)+offset_y_r])

    # cv2.imwrite("check_eye_crop.png", img)

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


def each_uuid(args, img_dir, out_dir):

    use_direct_yoloface_predictions = True

    if use_direct_yoloface_predictions:
        __path = os.path.join(out_dir, "face_detections.json")

        #yoloface_path = '/inwdata2/datasets/dms_field_videos/dms_field_videos_prod/gt_model_predictions/yoloface_predictions/images_orij_extraplanes_y_png_all_frames.json'
        with open(__path, 'r') as f:
            y = json.load(f) # [{}, {}]
            _d = []
            for _frame in y:
                _frame['img_name'] = _frame['img_name']
                _d.append(_frame)
            
        yoloface_df = process_to_df(_d)
        crops_dir = os.path.join(out_dir, "crops")
        os.makedirs(crops_dir, exist_ok=True)
        yoloface_df.to_csv(os.path.join(crops_dir, "gaze_json_eye_check.csv"))

    outdir = out_dir
    # 1. load pre-tained model
    checkpoint_fp = '/inwdata2a/sudhanshu/nd_data_processing_scripts/D3DFA/models/phase1_wpdc_vdc.pth.tar'
    arch = 'mobilenet_1'

    checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
    model = getattr(mobilenet_v1, arch)(num_classes=62)  # 62 = 12(pose) + 40(shape) +10(expression)

    model_dict = model.state_dict()
    # because the model is trained by multiple gpus, prefix module should be removed
    for k in checkpoint.keys():
        model_dict[k.replace('module.', '')] = checkpoint[k]
    model.load_state_dict(model_dict)
    if args.mode == 'gpu':
        cudnn.benchmark = True
        model = model.cuda()
    model.eval()

    # 2. load dlib model for face detection and landmark used for face cropping
    if args.dlib_landmark:
        dlib_landmark_model = '/inwdata2a/sudhanshu/nd_data_processing_scripts/D3DFA/models/shape_predictor_68_face_landmarks.dat'
        face_regressor = dlib.shape_predictor(dlib_landmark_model)
    if args.dlib_bbox:
        face_detector = dlib.get_frontal_face_detector()

    # 3. forward
    tri = sio.loadmat('/inwdata2a/sudhanshu/nd_data_processing_scripts/D3DFA/visualize/tri.mat')['tri']
    transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])
    #for img_fp in args.files:
    #img_dir = '/inwdata/datasets/dms_internal_dataset2/images/val/'
    # img_dir = '/inwdata2/datasets/dms_field_videos/dms_field_videos_prod/images/images_orij_extraplanes_y_png/all_frames'
    img_files = [os.path.join(img_dir, x) for x in os.listdir(img_dir)] # if x.split('_')[0]=='Aashish']
    count_face_missing = 0
    for img_fp in tqdm.tqdm(img_files):
        img_ori = cv2.imread(img_fp)


        if use_direct_yoloface_predictions:  # get facecrop from yoloface predictions
            image_name = img_fp.split('/')[-1]
            filtered = yoloface_df[yoloface_df['img_name'] == image_name]
            if filtered.empty:
                print(f"No face detection found for {image_name}, skipping.")
                rects = [[0, 0, 0, 0]]
            else:
                rects = filtered['face_bboxes'].values[0][0]
            if not len(rects):
                rects = [[0, 0, 0, 0]]

        #need this condition for old data as the primary box is 'person' instead of 'face'...
        if sum(rects[0]) == 0:
            # no face in in labels...
            print('image_name : ', img_fp, '; no face present in labels.')
            count_face_missing += 1
            continue


        if len(rects) ==0:
            print('rects is empty.')
            import sys
            sys.exit(1)


        pts_res = []
        Ps = []  # Camera matrix collection
        poses = []  # pose collection, [todo: validate it]
        vertices_lst = []  # store multiple face vertices
        ind = 0
        suffix = get_suffix(img_fp)
        for rect in rects:
            # whether use dlib landmark to crop image, if not, use only face bbox to calc roi bbox for cropping
            if args.dlib_landmark:
                # - use landmark for cropping
                pts = face_regressor(img_ori, rect).parts()
                pts = np.array([[pt.x, pt.y] for pt in pts]).T
                roi_box = parse_roi_box_from_landmark(pts)
            else:
                # - use detected face bbox
                #bbox = [rect.left(), rect.top(), rect.right(), rect.bottom()]
                bbox =rect
                roi_box = parse_roi_box_from_bbox(bbox)

            img = crop_img(img_ori, roi_box)

            # forward: one step
            img = cv2.resize(img, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
            input = transform(img).unsqueeze(0)
            with torch.no_grad():
                if args.mode == 'gpu':
                    input = input.cuda()
                param = model(input)
                param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

            # 68 pts
            pts68 = predict_68pts(param, roi_box)
            print("PTS 68 (1) ###########################")

            # two-step for more accurate bbox to crop face
            if args.bbox_init == 'two':
                roi_box = parse_roi_box_from_landmark(pts68)
                img_step2 = crop_img(img_ori, roi_box)
                img_step2 = cv2.resize(img_step2, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
                input = transform(img_step2).unsqueeze(0)
                with torch.no_grad():
                    if args.mode == 'gpu':
                        input = input.cuda()
                    param = model(input)
                    param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

                pts68 = predict_68pts(param, roi_box)
                print("PTS 68 (2) ###########################")

            pts_res.append(pts68)
            P, pose = parse_pose(param)
            Ps.append(P)
            poses.append(pose)

            # dense face 3d vertices
            if args.dump_ply or args.dump_vertex or args.dump_depth or args.dump_pncc or args.dump_obj:
                vertices = predict_dense(param, roi_box)
                vertices_lst.append(vertices)
            if args.dump_ply:
                dump_to_ply(vertices, tri, '{}_{}.ply'.format(os.path.join(outdir,'ply', img_fp.replace(suffix, '').split('/')[-1]), ind))
            if args.dump_vertex:
                dump_vertex(vertices, '{}_{}.mat'.format(os.path.join(outdir,img_fp.replace(suffix, '').split('/')[-1]), ind))
            if args.dump_pts:
                wfp = '{}_{}.txt'.format(os.path.join(outdir,'txt', img_fp.replace(suffix, '').split('/')[-1]), ind)
                np.savetxt(wfp, pts68, fmt='%.3f')
                print('Save 68 3d landmarks to {}'.format(wfp))
            if args.dump_roi_box:
                wfp = '{}_{}.roibox'.format(os.path.join(outdir,img_fp.replace(suffix, '').split('/')[-1]), ind)
                np.savetxt(wfp, roi_box, fmt='%.3f')
                ##print('Save roi box to {}'.format(wfp))
            if args.dump_paf:
                wfp_paf = '{}_{}_paf.jpg'.format(os.path.join(outdir, img_fp.replace(suffix, '').split('/')[-1]), ind)
                wfp_crop = '{}_{}_crop.jpg'.format(os.path.join(outdir, img_fp.replace(suffix, '').split('/')[-1]), ind)
                paf_feature = gen_img_paf(img_crop=img, param=param, kernel_size=args.paf_size)

                cv2.imwrite(wfp_paf, paf_feature)
                cv2.imwrite(wfp_crop, img)
                ##print('Dump to {} and {}'.format(wfp_crop, wfp_paf))
            if args.dump_obj:
                wfp = '{}_{}.obj'.format(os.path.join(outdir,'obj',img_fp.replace(suffix, '').split('/')[-1]), ind)
                colors = get_colors(img_ori, vertices)
                write_obj_with_colors(wfp, vertices, tri, colors)
                ##print('Dump obj with sampled texture to {}'.format(wfp))
            ind += 1

        if args.dump_pose:
            # P, pose = parse_pose(param)  # Camera matrix (without scale), and pose (yaw, pitch, roll, to verify)
            img_pose = plot_pose_box(img_ori, Ps, pts_res)
            wfp = os.path.join(outdir, img_fp.replace(suffix, '_pose.jpg').split('/')[-1])
            cv2.imwrite(wfp, img_pose)
            ##print('Dump to {}'.format(wfp))
        if args.dump_depth:
            #wfp = img_fp.replace(suffix, '_depth.png')
            wfp = os.path.join(outdir, img_fp.replace(suffix, '_depth.jpg').split('/')[-1])
            # depths_img = get_depths_image(img_ori, vertices_lst, tri-1)  # python version
            depths_img = cget_depths_image(img_ori, vertices_lst, tri - 1)  # cython version
            cv2.imwrite(wfp, depths_img)
            ##print('Dump to {}'.format(wfp))
        if args.dump_pncc:
            #wfp = img_fp.replace(suffix, '_pncc.png')
            wfp = os.path.join(outdir, img_fp.replace(suffix, '_pncc.jpg').split('/')[-1])
            pncc_feature = cpncc(img_ori, vertices_lst, tri - 1)  # cython version
            cv2.imwrite(wfp, pncc_feature[:, :, ::-1])  # cv2.imwrite will swap RGB -> BGR
            ##print('Dump to {}'.format(wfp))
        if args.dump_res:
            #draw_landmarks(img_ori, pts_res, wfp=img_fp.replace(suffix, '_3DDFA.jpg'), show_flg=args.show_flg)
            draw_landmarks(img_ori, pts_res, wfp=os.path.join(outdir, '3DDFA', img_fp.replace(suffix, '_3DDFA.jpg').split('/')[-1]), show_flg=args.show_flg)
    print('face missing in labels - ', count_face_missing, '/', len(img_files))
    




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='3DDFA inference pipeline')
    parser.add_argument('-f', '--files', nargs='+',
                        help='image files paths fed into network, single or multiple images')
    parser.add_argument('-m', '--mode', default='cpu', type=str, help='gpu or cpu mode')
    parser.add_argument('--show_flg', default='false', type=str2bool, help='whether show the visualization result')
    parser.add_argument('--bbox_init', default='one', type=str,
                        help='one|two: one-step bbox initialization or two-step')
    parser.add_argument('--dump_res', default='false', type=str2bool, help='whether write out the visualization image')
    parser.add_argument('--dump_vertex', default='false', type=str2bool,
                        help='whether write out the dense face vertices to mat')
    parser.add_argument('--dump_ply', default='false', type=str2bool)
    parser.add_argument('--dump_pts', default='true', type=str2bool)
    parser.add_argument('--dump_roi_box', default='false', type=str2bool)
    parser.add_argument('--dump_pose', default='false', type=str2bool)
    parser.add_argument('--dump_depth', default='false', type=str2bool)
    parser.add_argument('--dump_pncc', default='false', type=str2bool)
    parser.add_argument('--dump_paf', default='false', type=str2bool)
    parser.add_argument('--paf_size', default=3, type=int, help='PAF feature kernel size')
    parser.add_argument('--dump_obj', default='false', type=str2bool)
    parser.add_argument('--dlib_bbox', default='true', type=str2bool, help='whether use dlib to predict bbox')
    parser.add_argument('--dlib_landmark', default='False', type=str2bool,
                        help='whether use dlib landmark to crop image')

    args = parser.parse_args()
        


import os
import json
import numpy as np
from PIL import Image

STD_SIZE = 120
def generate_face_crop1(src_dir, json_path, num_samples=1):
    model = YoloDetector(target_size=None, device="cuda:0", min_face=220)

    json_file_name = json_path + '.json'
    os.makedirs(os.path.dirname(json_file_name), exist_ok=True)

    data = []
    uuid_frames_path = src_dir

    files = sorted([f for f in os.listdir(uuid_frames_path) if f.endswith('.jpg')])

    if len(files) == 0:
        print(f"⚠️ No images found in {src_dir}")
        return

    total_frames = len(files)

    if total_frames <= num_samples:
        sampled_files = files
    else:
        # RANDOMLY select num_samples indices (no replacement)
        indices = np.random.choice(total_frames, num_samples, replace=False)
        sampled_files = [files[i] for i in indices]

    for file_name in tqdm.tqdm(sampled_files, desc=f"Processing {os.path.basename(src_dir)} ({len(sampled_files)} frames)"):
        image_path = os.path.join(uuid_frames_path, file_name)
        _orgimg = np.array(Image.open(image_path))
        orgimg = _orgimg  # for 1600x1200 image size

        bboxes, points, box_confs = model.predict(orgimg, conf_thres=0.3, iou_thres=0.5)

        if len(bboxes) == 0 or len(bboxes[0]) == 0:
            continue

        d = {
            'img_name': file_name,
            'face_bboxes': bboxes,
            'keypoints': points,
            'box_confs': box_confs
        }
        data.append(d)

    with open(json_file_name, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"✅ Saved {len(data)} detections to {json_file_name}")








""" 1. when you run this script, it will save face_detections.json and gaze_eye_crops-final.json in each folder inside output_root
    2. generate_face_crop1 gives you option to choose how many folders to process and how many samples to take from each folder,
       while generate_face_crop processes all folders and all frames in each folder 
      """

# -----------------------------
# Configuration
# -----------------------------
folders_txt = "/inwdata2a/sudhanshu/eyelid-training/splits/temp.txt"
input_root = "/inwdata2a/sudhanshu/landmarks_only_training/inference_pipeline/video_frames"
output_root = "/inwdata2a/sudhanshu/landmarks_only_training/inference_pipeline/face-eye-bbox"

# -----------------------------
# Read folder names
# -----------------------------
with open(folders_txt, 'r') as f:
    folder_names = [line.strip() for line in f.readlines()]

# Take only first 150 folders
# folder_names = folder_names[:5]   # how many folders to process

print(f"Processing {len(folder_names)} folders...")



for folder in tqdm.tqdm(folder_names, desc="Processing folders"):
    img_dir = os.path.join(input_root, folder)
    out_dir = os.path.join(output_root, folder)
    json_path = os.path.join(out_dir, "face_detections")
    
    os.makedirs(out_dir, exist_ok=True)
  #  os.makedirs(out_dir1, exist_ok=True)

    # Step 1: Face detection (uncomment if you want to run it)
    generate_face_crop(img_dir, json_path)
   # generate_face_crop1(img_dir, json_path, num_samples=100)  # how many samples to take from each folder

    # Step 2: Check face_detections.json for presence and non-empty data
    face_json_file = json_path + ".json"
    proceed = True
    if not os.path.isfile(face_json_file):
        print(f"Skipping {folder}: no face_detections json file available.")
        proceed = False
    else:
        with open(face_json_file, "r") as f:
            try:
                detections = json.load(f)
                if len(detections) == 0:
                    print(f"Skipping {folder}: face_detections json file is empty.")
                    proceed = False
            except json.JSONDecodeError:
                print(f"Skipping {folder}: face_detections json file is not valid JSON.")
                proceed = False

    if not proceed:
        continue

    # Step 3: Landmark/model processing
    print(f"Generating eyecrops for {folder}...")

    _model_txt_outputs_dir = os.path.join(out_dir, "txt")
    os.makedirs(_model_txt_outputs_dir, exist_ok=True)
    each_uuid(args, img_dir, out_dir)

    # Step 4: Eye crop extraction
    save_dict = {}
    all_txt_files = os.listdir(_model_txt_outputs_dir)
    for _txt_file in tqdm.tqdm(all_txt_files):
        _txt_file_path = os.path.join(_model_txt_outputs_dir, _txt_file)
        bb_list = get_eye_patch(_txt_file_path, img_path=None)
        save_dict[_txt_file.replace('_0.txt', '.jpg')] = bb_list

    out_path = os.path.join(out_dir, 'gaze_eye_crops-final.json')
    with open(out_path, 'w') as f:
        json.dump(save_dict, f)
    print(f"Finished Generating eyecrops for {folder}")

print("\n🎯 Finished processing all folders.")