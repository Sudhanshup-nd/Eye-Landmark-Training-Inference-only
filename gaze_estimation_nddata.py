"""
Steps to process one video for pupil detection
1. Generate frames from the video [dms_png_generation.py]
2. Generating facecrops [dms_eb_data_face_crop_generation.py]
3. Generating eye crops [batch_custom_main_copy2.py, generate_eye_labels.py]
4. Break the eye crop file into left and right eye crops
5. Testing the model on the eye crops
"""
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

    model = YoloDetector(target_size=None, device="cuda:2", min_face=220)

    uuid_cnt = 0
    uuid_dir_path = src_dir
    
    json_file_name = json_path + '.json'
    
    data = []
    uuid_cnt +=1
    count = 0
    uuid_frames_path = os.path.join(uuid_dir_path, "extracted_frames")
    for file_name in os.listdir(uuid_frames_path):
        
        if not file_name.endswith('.png'):
            continue
        image_path = os.path.join(uuid_frames_path, file_name)
        _orgimg = np.array(Image.open(image_path))
        # orgimg = np.stack([_orgimg, _orgimg, _orgimg], axis=-1)
        orgimg = _orgimg # for 1600x1200 image size
        # print(image_path)
        # print(orgimg.shape)
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

    # cv2.rectangle(img, (int(low_x_l) - offset_x_l, int(low_y_l) - offset_y_l), (int(high_x_l) + offset_x_l, int(high_y_l)+offset_y_l), (0, 255, 0), 2 )
    # cv2.rectangle(img, (int(low_x_r) - offset_x_r, int(low_y_r) - offset_y_r), (int(high_x_r) + offset_x_r, int(high_y_r)+offset_y_r), (0, 255, 0), 2 )

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
        __path = "/data5/ishita/teyed_detections/ip_documentation/8_gaze_faces.json"

        #yoloface_path = '/inwdata2/datasets/dms_field_videos/dms_field_videos_prod/gt_model_predictions/yoloface_predictions/images_orij_extraplanes_y_png_all_frames.json'
        with open(__path, 'r') as f:
            y = json.load(f) # [{}, {}]
            _d = []
            for _frame in y:
                _frame['img_name'] = _frame['img_name']
                _d.append(_frame)
            
        yoloface_df = process_to_df(_d)

        yoloface_df.to_csv("/data5/ishita/teyed_detections/ip_documentation/gaze_json_eye_check.csv")

        # import sys
        # sys.exit(1)

    # outdir = 'out_dir_fraction_0.5_dms_field_videos_prod_extraplanes'
    outdir = out_dir
    # 1. load pre-tained model
    checkpoint_fp = '/data5/ishita/teyed_detections/D3DFA/models/phase1_wpdc_vdc.pth.tar'
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
        dlib_landmark_model = '/data5/ishita/teyed_detections/D3DFA/models/shape_predictor_68_face_landmarks.dat'
        face_regressor = dlib.shape_predictor(dlib_landmark_model)
    if args.dlib_bbox:
        face_detector = dlib.get_frontal_face_detector()

    # 3. forward
    tri = sio.loadmat('/data5/ishita/teyed_detections/D3DFA/visualize/tri.mat')['tri']
    transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])
    #for img_fp in args.files:
    #img_dir = '/inwdata/datasets/dms_internal_dataset2/images/val/'
    # img_dir = '/inwdata2/datasets/dms_field_videos/dms_field_videos_prod/images/images_orij_extraplanes_y_png/all_frames'
    img_files = [os.path.join(img_dir, x) for x in os.listdir(img_dir)] # if x.split('_')[0]=='Aashish']
    count_face_missing = 0
    for img_fp in tqdm.tqdm(img_files):
        img_ori = cv2.imread(img_fp)
        if False: #args.dlib_bbox:
            rects = face_detector(img_ori, 1)
            print('orig: ', rects)
        if False: # get facecrop from labels file
            #rects = []
            label_file = img_fp.replace('images', 'labels').replace('png', 'txt')
            if not os.path.isfile(label_file):
                continue
            with open(label_file, 'r') as f:
                _label = f.readlines()[0].split(' ')
                #rects_normalised = [float(x) for x in _label[1+4:1+8]]
                rects_normalised = [float(x) for x in _label[1+8:1+12]] # for old data

                #denormalise
                rects = [[]]
                width = 1600
                height = 1200
                x1 = (rects_normalised[0] - rects_normalised[2]/2)*width
                y1 = (rects_normalised[1] - rects_normalised[3]/2)*height
                x2 = (rects_normalised[0] + rects_normalised[2]/2)*width
                y2 = (rects_normalised[1] + rects_normalised[3]/2)*height
                #rects[0].append( (round(x1), round(y1)) )
                #rects[0].append( ( round(x2), round(y2) ) )
                rects[0].append( round(x1))
                rects[0].append(round(y1))
                rects[0].append(round(x2))
                rects[0].append(round(y2))
                #print('custom: ', rects)

        if use_direct_yoloface_predictions: # get facecrop from yoloface predictions
            image_name = img_fp.split('/')[-1]
            rects = yoloface_df[yoloface_df['img_name'] == image_name]['face_bboxes'].values[0][0]
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
        if False: #len(rects) == 0:
            rects = dlib.rectangles()
            rect_fp = img_fp + '.bbox'
            lines = open(rect_fp).read().strip().split('\n')[1:]
            for l in lines:
                l, r, t, b = [int(_) for _ in l.split(' ')[1:]]
                rect = dlib.rectangle(l, r, t, b)
                rects.append(rect)

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
    

def process_video_file(folder_path, video_path):

    fps = 10

    output_dir = os.path.join(folder_path, "gaze_extracted_frames")
    os.makedirs(output_dir, exist_ok=True)

    subprocess.run(['ffmpeg', '-i', video_path, '-vf', f'fps={fps}', os.path.join(output_dir, 'frame_%04d.png')])

def draw_bboxes_preds_only(image, pupil_predictions, iris_predictions, save_path):
    
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    if np.max(image) <= 1.0:
        image = (image * 255).astype(np.uint8)
        
    # if image.shape[2] == 3:
    #     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # elif image.shape[2] == 4:
    #     image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)  # Handle RGBA if necessary
    # else:
    #     raise ValueError("Image does not have 3 or 4 channels.")
        
    height, width = image.shape[0], image.shape[1]
    
    print(height, width)

    x1, y1, x2, y2 = pupil_predictions[0], pupil_predictions[1], pupil_predictions[2], pupil_predictions[3]
    x1 = (x1 + 1) * width / 2
    y1 = (1 - y1) * height / 2
    x2 = (x2 + 1) * width / 2
    y2 = (1 - y2) * height / 2
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 1)

    x1, y1, x2, y2 = iris_predictions[0], iris_predictions[1], iris_predictions[2], iris_predictions[3]
    x1 = (x1 + 1) * width / 2
    y1 = (1 - y1) * height / 2
    x2 = (x2 + 1) * width / 2
    y2 = (1 - y2) * height / 2
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 1)

    cv2.imwrite(save_path, image)

def find_eye_ball_centre(eye_region_frame):

    try:

        contours, _ = cv2.findContours(eye_region_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        # print("calculated contours")
        if not contours:
            # print("No contours found.")
            return None, None, None, None
        # print(len(contours[0]))
        if len(contours[0]) >= 5:
            # print("in ellipse find")
            ellipse = cv2.fitEllipse(contours[0])
            (x, y), (major, minor), angle = ellipse
            area = np.pi * (major / 2) * (minor / 2)
            eye_ball_centre = (int(x), int(y))
            leftmost = (0, int(y))
            rightmost = (eye_region_frame.shape[1], int(y))

            # print("ret try")

            return eye_ball_centre, leftmost, rightmost, ellipse
        
        return None, None, None, None

    except:
        # print("exc try")
        return None, None, None, None


def calculate_gaze_vector(eyeball_center, pupil_center):
    x_e, y_e = eyeball_center
    x_p, y_p = pupil_center
    
    g_x = x_p - x_e
    g_y = y_p - y_e
    
    gaze_vector = np.array([g_x, g_y])
    
    magnitude = np.linalg.norm(gaze_vector)
    
    if magnitude != 0:
        normalized_gaze_vector = gaze_vector / magnitude
    else:
        normalized_gaze_vector = np.array([0, 0])
    
    return gaze_vector, normalized_gaze_vector, magnitude

def calculate_angle(vector1, vector2): 
    unit_vector1 = vector1 / np.linalg.norm(vector1)  
    unit_vector2 = vector2 / np.linalg.norm(vector2)  
    dot_product = np.dot(unit_vector1, unit_vector2)  
    angle = np.arccos(dot_product)
    angle = round(np.degrees(angle), 2)
    cross_product = np.cross(unit_vector1, unit_vector2)  
    signed_angle = angle if cross_product >= 0 else 0-angle
    return signed_angle

def find_gaze_direction_pupil_eye_width_2(magnitude, angle, pupil_radius, eye_left, eye_right):

    eye_width = eye_right[0] - eye_left[0]

    if angle <=-15 and angle >=-165:
        return "down"
    elif magnitude<min(0.1*eye_width, pupil_radius):
        return "straight"
    elif angle>-15 and angle <= 30:
        return "right"
    elif angle>30 and angle<=120:
        return "up"
    else:
        return "left"
    
def find_gaze_direction_pupil_eye_width_3(magnitude, angle, pupil_radius, eye_left, eye_right):

    eye_width = eye_right[0] - eye_left[0]

    if angle <=-15 and angle >=-165:
        return "down"
    elif magnitude<min(0.1*eye_width, pupil_radius):
        return "straight"
    elif angle>-15 and angle <= 50:
        return "right"
    elif angle>50 and angle<=100:
        return "up"
    else:
        return "left"
    
def find_gaze_direction_pupil_eye_width_4(magnitude, angle, pupil_radius, eye_left, eye_right):

    eye_width = eye_right[0] - eye_left[0]

    if angle <=-15 and angle >=-165:
        return "down"
    elif (angle>100 and angle<=180) or angle<-165:
        return "left"
    elif angle>-15 and angle <= 50:
        return "right"
    elif angle>50 and angle<=100:
        return "up"
    elif magnitude<min(0.1*eye_width, pupil_radius):
        return "straight"
    
def update_eye_ball_centre_stats(new_centre, prev_stats, alpha=0.2):
    
    mean, median, mode, moving_avg, history, counter = prev_stats
    
    history.append(new_centre)
    # if len(history) > 1000:  # Keep only the last 1000 entries
    #     history.pop(0)
    
    history_array = np.array(history, dtype=np.float32)
    
    new_mean = mean + (new_centre - mean) / len(history)
    
    new_median = np.median(history_array, axis=0)
    
    counter[tuple(new_centre)] += 1
    new_mode = max(counter, key=counter.get)
    
    new_moving_avg = alpha * new_centre + (1 - alpha) * moving_avg

    return new_mean, new_median, new_mode, new_moving_avg, history, counter

def get_final_gaze_direction(prev, curr, average_angle):
    if prev==curr:
        return prev
    elif prev == "close" or prev == "straight":
        return curr
    elif curr == "close" or curr == "straight":
        return prev
    elif prev == "straight":
        return curr
    elif curr == "straight":
        return curr
    else:
        if average_angle <=-15 and average_angle >=-165:
            return "down"
        elif average_angle>-15 and average_angle <= 30:
            return "right"
        elif average_angle>30 and average_angle<=120:
            return "up"
        else:
            return "left"
    return None

def eye_part_seg_pupil_centre(f_mask_image):
    f_mask_image = (f_mask_image * 255).astype(np.uint8)
    contours, hierarchy = cv2.findContours(f_mask_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  
      
    if len(contours) > 0:  
        (f_center, f_radius) = cv2.minEnclosingCircle(contours[0])  
    else:   
        f_center = (-1.0, -1.0) 
  
    return f_center


def test_unlabelled(model, eye_ball_session, test_loader, criterion, device, batch_size):

    eye_close_classifier = CEyeClassifier()

    model.eval()

    num_frames = 0

    frame_images = []
    frames_w_closed_eyes = []
    gaze_classification = []

    eye_ball_centre_stats = (np.array([0, 0], dtype=np.float32),  # Mean
                          np.array([0, 0], dtype=np.float32),  # Median
                          np.array([0, 0], dtype=np.float32),  # Mode
                          np.array([0, 0], dtype=np.float32),  # Moving Average
                          [],  # History
                          Counter())

    # for i, (image_cv, image, image_c, full_image) in tqdm.tqdm(enumerate(test_loader)):
    #     print(full_image.shape)
    #     exit(1)

    output_video = cv2.VideoWriter("/data5/ishita/teyed_detections/ip_documentation/8_gaze_b_p_ew4_ebc_dy_dy_moving_avg_combined_01.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 10, (1296, 1296))

    prev_gaze = "straight"
    prev_angle = 0

    with torch.no_grad():

        save_index = 1

        # print("test loader len: ", len(test_loader))

        left_angles, left_directions, right_angles, right_directions, average_angles, final_gaze_directions, left_image_locations, right_image_locations = [], [], [], [], [], [], [], []

        for i, (image_to_save, image_cv, image, image_c, full_image_resized, full_image, is_left_eye) in tqdm.tqdm(enumerate(test_loader)):

            # print(image_cv.shape) #, full_image_resized.shape)
            full_image = np.array(full_image)[0]
            image_eye_ball = Variable(image_cv) 

            image, image_eye_ball = image.to(device), image_eye_ball.to(device).float()
            image_eye_ball = image_eye_ball.to('cuda') 

            # print(image_eye_ball.shape, image.shape) # torch.Size([1, 1, 64, 64]) torch.Size([1, 3, 64, 64])
            # cv2.imwrite("eye_ball_image.png", image_eye_ball[0].permute(1, 2, 0).detach().cpu().numpy()) # black image
            # cv2.imwrite("pupil_image.png", image[0].permute(1, 2, 0).detach().cpu().numpy()) # correct image
            # exit(1)
            
            predictions = model(image)
            # print(predictions.shape)
            
            input_name = eye_ball_session.get_inputs()[0].name  
            input_data_numpy = image_eye_ball.detach().cpu().numpy()   
            
            # assert input_data_numpy.shape == (1, 1, 64, 64), "Input data is not the correct shape."  
            
            input_data = {input_name: input_data_numpy}    

                
            # Perform inference    
            outputs = eye_ball_session.run(None, input_data) 

            outputs = ort_session.run(None, input_data)      
            # output1_tensor = torch.from_numpy(outputs[0])    
            output2_tensor = torch.from_numpy(outputs[1])
            output2_prob = output2_tensor
            output2_pred = (output2_prob >= 0.5).float() 
            output2_pred_np = output2_pred.detach().cpu().numpy().squeeze()

            output1_tensor = torch.from_numpy(outputs[0])

            output1_test_cpu = output1_tensor.cpu()
            unique_values = np.unique(output1_test_cpu.detach().numpy())

                    
            output1_test_cpu = output1_test_cpu.squeeze(0) 

            pupil_area_mask = (output1_test_cpu == 3)
            pupil_pixels = torch.nonzero(pupil_area_mask)
            # # pupil_pixels = np.argwhere(pupil_area_mask)

            if pupil_pixels.numel() == 0:
                if is_left_eye:
                    left_angles.append(404)
                    left_directions.append(404)

                    left_image_locations.append(404)
                else:
                    right_angles.append(404)
                    right_directions.append(404)

                    right_image_locations.append(404)

                    average_angles.append(404)
                    final_gaze_directions.append(404)

                frame_images.append(full_image)
                # print("continue 1")
                continue

            pupil_pixels = pupil_pixels.float()
            pupil_center_y, pupil_center_x = pupil_pixels.mean(axis=0)  # mean along axis 0 (rows and columns)
            pupil_centre = (pupil_center_x.item(), pupil_center_y.item())
            f_centre = eye_part_seg_pupil_centre(pupil_area_mask.detach().cpu().numpy())
            print("Pupil centre: ", pupil_centre)
            print("f centre: ", f_centre)
            distances = torch.norm(pupil_pixels - torch.tensor([pupil_center_y, pupil_center_x]), dim=1)
            pupil_radius = distances.max().item()
            # print(unique_values, output1_test_cpu.shape)
            # exit(1)

            # output1, output2, _ = eye_ball_model(image_eye_ball)

            # output1_test = torch.argmax(output1,dim=1) 
            # output1_test_cpu = output1_test.cpu()  
            # unique_values = np.unique(output1_test_cpu.detach().numpy())       
            # output1_test_cpu = output1_test_cpu.squeeze(0)

            # pupil_area_mask = (output1_test_cpu == 3)
            # pupil_pixels = torch.nonzero(pupil_area_mask)
            # # pupil_pixels = np.argwhere(pupil_area_mask)
            # pupil_pixels = pupil_pixels.float()
            # pupil_center_y, pupil_center_x = pupil_pixels.mean(axis=0)  # mean along axis 0 (rows and columns)
            # pupil_center = (pupil_center_x.item(), pupil_center_y.item())
            
            # print("Eye ball centre: ", eye_ball_centre)

            # print(image.shape)

            eye_preds = torch.argmax(eye_close_classifier.detect(image_c)).item()
            # print(eye_preds)
            # exit(1)
            # print(predictions.shape)

            # pupil_predictions = predictions[:, :4]

            # print(pupil_predictions.shape)
            # print(iris_predictions.shape)

            # pupil_predictions_c = pupil_predictions.cpu().numpy()
            
            

            num_frames+=1

            # im = image_cv[j].permute(1, 2, 0).numpy()

            # frame_images.append(full_image[j])

            height, width = image_to_save[0].shape[0], image_to_save[0].shape[1]
            # print(image_to_save.shape, height, width)

            # x1, y1, x2, y2 = pupil_predictions_c[0][0], pupil_predictions_c[0][1], pupil_predictions_c[0][2], pupil_predictions_c[0][3]
            # # print(x1, y1, x2, y2)
            # x1 = (x1 + 1) * width / 2
            # y1 = (1 - y1) * height / 2
            # x2 = (x2 + 1) * width / 2
            # y2 = (1 - y2) * height / 2
            # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # print(x1, y1, x2, y2)

            # pupil_radius = np.sqrt(pow((x1-x2), 2)+pow((y1-y2), 2))/2
            # pupil_centre = ((x1+x2)//2, (y1+y2)//2)
            # print(full_image.shape)
            # exit(1)

            mask = (output2_pred_np * 255).astype(np.uint8)
            # print(im.shape)
            # cv2.imwrite("check_mask.png", mask)
            # cv2.imwrite("check_eye_patch.png", cv2.cvtColor(input_data_numpy[0, 0], cv2.COLOR_GRAY2BGR))
            # cv2.imwrite("check_eye_patch_2.png", im)
            # exit(1)
            # print(type(mask), mask.shape)

            if mask is None or mask.size == 0:
                if is_left_eye:
                    left_angles.append(404)
                    left_directions.append(404)

                    left_image_locations.append(404)
                else:
                    right_angles.append(404)
                    right_directions.append(404)

                    right_image_locations.append(404)

                    average_angles.append(404)
                    final_gaze_directions.append(404)

                frame_images.append(full_image)
                # print("continue 1")
                continue

            eye_ball_centre, leftmost, rightmost, ellipse = find_eye_ball_centre(mask)  

            if not eye_ball_centre or not pupil_centre:
                if is_left_eye:
                    left_angles.append(404)
                    left_directions.append(404)

                    left_image_locations.append(404)
                else:
                    right_angles.append(404)
                    right_directions.append(404)

                    right_image_locations.append(404)

                    average_angles.append(404)
                    final_gaze_directions.append(404)

                frame_images.append(full_image)
                # print("continue 2")
                continue

            eye_ball_centre_stats = update_eye_ball_centre_stats(np.array(eye_ball_centre, dtype=np.float32), eye_ball_centre_stats, alpha=0.2)

            eye_ball_centre_mean, eye_ball_centre_median, eye_ball_centre_mode, eye_ball_centre_moving_avg, _, _ = eye_ball_centre_stats

            gaze_vector, normalized_gaze_vector, gaze_vector_magnitude = calculate_gaze_vector(eye_ball_centre_moving_avg, pupil_centre)
            leftmost = (0, int(eye_ball_centre_moving_avg[1]))
            rightmost = (full_image.shape[1], int(eye_ball_centre_moving_avg[1]))
            eye_ball_vector, _, _ = calculate_gaze_vector(leftmost, rightmost)

            angle = calculate_angle(gaze_vector, eye_ball_vector)

            # print("Angle: ", angle)

            # gaze_direction = find_gaze_direction_pupil(gaze_vector_magnitude, angle, pupil_radius)
            # gaze_direction = find_gaze_direction_pupil_eye_width(gaze_vector_magnitude, angle, pupil_radius, leftmost, rightmost)
            # gaze_direction = find_gaze_direction_pupil_eye_width_2(gaze_vector_magnitude, angle, pupil_radius, leftmost, rightmost)
            # gaze_direction = find_gaze_direction_pupil_eye_width_3(gaze_vector_magnitude, angle, pupil_radius, leftmost, rightmost)
            gaze_direction = find_gaze_direction_pupil_eye_width_4(gaze_vector_magnitude, angle, pupil_radius, leftmost, rightmost)


            if eye_preds == 0:
                gaze_direction = "close"

            # print(gaze_direction)
            # print("appeneded normally")

            # print(image.shape)
            image_to_save = np.array(image_to_save[0])
            # print(image_to_save.shape)
            
            
            # for i in range(image_to_save.shape[0]):    
            #     for j in range(image_to_save.shape[1]):    
            #         if output2_pred_np[i, j] == 1:    
            #             image_to_save[i, j, 2] = 255 
            #         if output1_test_cpu[i, j] == 3:
            #             image_to_save[i, j, 0] = 255

            # print(type(image_to_save), pupil_centre)
            # image_to_save = (image_to_save*255).astype(np.uint8)
            
            # print(gaze_direction)

            if is_left_eye:
                prev_gaze = gaze_direction
                prev_angle = angle

                left_angles.append(angle)
                left_directions.append(gaze_direction)

                loc = "/data5/ishita/teyed_detections/gaze_b_8_video_eyecrop_ops_ew4_dy_dy_moving_avg_combined/left_"+str(save_index)+".png"
                left_image_locations.append(loc)

                cv2.circle(image_to_save, (int(pupil_centre[0]), int(pupil_centre[1])), 1, (255, 0, 0), 1)
                cv2.circle(image_to_save, (int(f_centre[0]), int(f_centre[1])), 1, (0, 255, 0), 1)
                # cv2.circle(image_to_save, (int(eye_ball_centre_moving_avg[0]), int(eye_ball_centre_moving_avg[1])), 1, (255, 255, 0), 1)
                # cv2.ellipse(image_to_save, ellipse, (255, 255, 0), 1)
                # cv2.arrowedLine(image_to_save, (int(eye_ball_centre_moving_avg[0]), int(eye_ball_centre_moving_avg[1])), (int(pupil_centre[0]), int(pupil_centre[1])), (0, 255, 0), 1)
                # cv2.rectangle(image_to_save, (x1, y1), (x2, y2), (255, 0, 0), 1)
                # cv2.rectangle(image_to_save, (int(pupil_centre[0]-pupil_radius), int(pupil_centre[1]-pupil_radius)), (int(pupil_centre[0]+pupil_radius), int(pupil_centre[1]+pupil_radius)), (255, 0, 0), 1)
                # cv2.putText(image_to_save, str(angle), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                #     0.5, (255, 0, 0), 1)
                cv2.imwrite(loc, image_to_save)
                
            else:
                right_angles.append(angle)
                right_directions.append(gaze_direction)

                loc = "/data5/ishita/teyed_detections/gaze_b_8_video_eyecrop_ops_ew4_dy_dy_moving_avg_combined/right_"+str(save_index)+".png"
                right_image_locations.append(loc)

                average_angle = (prev_angle+angle)/2

                # final_gaze_direction = get_final_gaze_direction(prev_gaze, gaze_direction, average_angle)
                cv2.circle(image_to_save, (int(pupil_centre[0]), int(pupil_centre[1])), 1, (255, 0, 0), 1)
                cv2.circle(image_to_save, (int(f_centre[0]), int(f_centre[1])), 1, (0, 255, 0), 1)
                # cv2.circle(image_to_save, (int(eye_ball_centre[0]), int(eye_ball_centre[1])), 1, (255, 255, 0), 1)
                # cv2.ellipse(image_to_save, ellipse, (255, 255, 0), 1)
                # cv2.arrowedLine(image_to_save, (int(eye_ball_centre[0]), int(eye_ball_centre[1])), (int(pupil_centre[0]), int(pupil_centre[1])), (0, 255, 0), 1)
                # cv2.rectangle(image_to_save, (x1, y1), (x2, y2), (255, 0, 0), 1)
                # cv2.rectangle(image_to_save, (int(pupil_centre[0]-pupil_radius), int(pupil_centre[1]-pupil_radius)), (int(pupil_centre[0]+pupil_radius), int(pupil_centre[1]+pupil_radius)), (255, 0, 0), 1)
                # cv2.putText(image_to_save, str(angle), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                    # 0.5, (255, 0, 0), 1)
                cv2.imwrite(loc, image_to_save)

                # cv2.putText(full_image, final_gaze_direction, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                    # 2, (255, 0, 0), 1)

                average_angles.append(average_angle)
                final_gaze_directions.append(final_gaze_direction)

                frame_images.append(full_image)

            save_index+=1

    print("Writing output video: ")
    for i in range(len(frame_images)):
        
        frame_i = frame_images[i]
        # frame_i = frame_i.numpy()

        # Write frame to video
        output_video.write(frame_i)


    output_video.release()
    # print(len(left_angles), len(left_directions), len(right_angles), len(right_directions), len(average_angles), len(final_gaze_directions), len(left_image_locations), len(right_image_locations))

    # df = pd.DataFrame({
    #     "Left_Angle": left_angles,
    #     "Left_Direction": left_directions,
    #     "Right_Angle": right_angles,
    #     "Right_Direction": right_directions,
    #     "Average_Angle": average_angles,
    #     "Final_Gaze_Direction": final_gaze_directions,
    #     "Left_Image": left_image_locations,
    #     "Right_Image": right_image_locations
    # })

    # df.to_csv("/data5/ishita/teyed_detections/ip_documentation/gaze_b_8_video_eyecrop_ops_ew4_dy_dy_moving_avg_combined.csv", index=False)
    # print("csv_file_saved.")

class OneVideo(Dataset):
    def __init__(self, dataset_path, images_path, transform, data_transform):

        with open(dataset_path, 'r') as file:
            self.data = json.load(file)

        self.images_path = images_path

        self.transform = transform
        self.data_transform = data_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        is_left_eye = (index % 2 == 0)

        img_name = self.data[index]['frame']
        eye_crop_coordinates = self.data[index]['coordinates']

        # print(img_name, eye_crop_coordinates)

        full_image = cv2.imread(self.images_path+img_name)

        full_image_copy = full_image

        im_cv = cv2.imread(self.images_path+img_name)
        
        if im_cv is None:
            im_cv = np.zeros((64, 64, 3), dtype=np.uint8)
        if full_image is None:
            full_image = np.zeros((64, 64, 3), dtype=np.uint8)
        # image_path = self.images_path+uuid+"/"+img_name
        # image = Image.open(image_path)
        # print(image.size)
        im_cv2 = cv2.cvtColor(im_cv, cv2.COLOR_BGR2GRAY)

        image = Image.fromarray(im_cv.astype('uint8'), 'RGB')

        x1, y1, x2, y2 = eye_crop_coordinates[0], eye_crop_coordinates[1], eye_crop_coordinates[2], eye_crop_coordinates[3]

        im_cv = im_cv[y1:y2, x1:x2]

        im_cv2 = im_cv2[y1:y2, x1:x2]
        im_cv2 = cv2.resize(im_cv2, (64, 64))

        # cv2.imwrite("eye_ball_image_before.png", im_cv2)
        # exit(1)

        pil_image = Image.fromarray(im_cv2)

        # print(pil_image.size, np.array(pil_image).shape)
        cv2.imwrite("eye_ball_image_before.png", np.array(pil_image))

        eb_input_tensor = self.data_transform(pil_image) 

        eb_input_tensor = eb_input_tensor.numpy()*255
        eb_input_tensor = torch.from_numpy(eb_input_tensor)

        # print(eb_input_tensor.permute(1, 2, 0).numpy().shape)

        # cv2.imwrite("eye_ball_image_before_sending.png", eb_input_tensor.permute(1, 2, 0).numpy())
        # exit(1)

        if im_cv.shape[0] == 0 or im_cv.shape[1] == 0:
            im_cv = np.zeros((64, 64, 3), dtype=np.uint8)

        if im_cv2.shape[0] == 0 or im_cv2.shape[1] == 0:
            im_cv2 = np.zeros((64, 64, 1), dtype=np.uint8)

        if full_image.shape[0] == 0 or full_image.shape[1] == 0:
            full_image = np.zeros((64, 64, 3), dtype=np.uint8)

        full_image = cv2.resize(full_image, (64, 64))
        
        eye_crop = image.crop((x1, y1, x2, y2))
        # eye_crop = im_cv[y1:y2, x1:x2]

        transform_c = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((64, 64)),
            # transforms.Resize((288, 384)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # for Res152 that takes 3 channel input
        ])

        eye_cropc = transform_c(eye_crop)
        
        if self.transform:
            eye_crop = self.transform(eye_crop)

        im_cv = cv2.resize(im_cv, (64, 64))
        # im_cv2 = cv2.resize(im_cv2, (64, 64))
        # print(im_cv2.shape)
        # im_cv2 = im_cv2.unsqueeze(0) 

        eye_cropt = eye_crop.numpy() * 255
        eye_cropt = torch.from_numpy(eye_cropt)

        eye_croptc = eye_cropc.numpy() * 255
        eye_croptc = torch.from_numpy(eye_croptc)
        
        return im_cv, eb_input_tensor, eye_cropt, eye_croptc, full_image, full_image_copy, is_left_eye


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

class CEyeClassifier:
    
    def __init__(self):
        """
        Face detection class constructor. Face detection model using key point estimation.
        The model detect face, person, left eye, right eye, mouth and 5 key points such
        as left and right shoulder, left and right ear and nose. The model is developed on
        Pytorch and YOLO architecture.
        """
        
        self.model = ort.InferenceSession(
            '/data5/ishita/teyed_detections/ip_documentation/eye_classifier_bs1.onnx', providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    

    def detect(self, f_image):
        # print("Face detection input tensor name = {0}".format(self.model.get_inputs()[0].name))
        ort_inputs = {self.model.get_inputs()[0].name: to_numpy(f_image)}
        pred = self.model.run(None, ort_inputs)[0]
        pred = np.array(pred)
        pred = torch.from_numpy(pred)
        return pred


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
        
        
    # # STEP 1 : Generating frames

    print("Generating frames...")

    folder_path = "/data5/ishita/teyed_detections/ip_documentation"
    video_path = "/data5/ishita/teyed_detections/gaze_check_vid.mp4"
    process_video_file(folder_path, video_path)

    print("Finished Generating frames")

    # STEP 2 : Generating face crops

    print("Generating face crops...")

    print("Running script")

    dirp = "/data5/ishita/teyed_detections/ip_documentation"
    json_path = "/data5/ishita/teyed_detections/ip_documentation/8_gaze_faces"
    generate_face_crop(dirp, json_path)

    print("Finished Generating face crops.")

    STEP 3: Generating eye crops

    print("Generating eyecrops...")

    img_dir = folder_path+"/gaze_extracted_frames"
    out_dir = folder_path+"/gaze_outcrops"
    os.makedirs(out_dir, exist_ok=True)

    _model_txt_outputs_dir = '/data5/ishita/teyed_detections/ip_documentation/gaze_outcrops/txt'
    os.makedirs(_model_txt_outputs_dir, exist_ok=True)

    each_uuid(args, img_dir, out_dir)

    label_imgs = pd.read_csv("/data5/ishita/teyed_detections/ip_documentation/gaze_json_eye_check.csv")
    all_image_names = list(label_imgs['img_name'])
    _model_txt_outputs_dir = '/data5/ishita/teyed_detections/ip_documentation/gaze_outcrops/txt'
    out_dir = '/data5/ishita/teyed_detections/ip_documentation'
    # img_dir = '/inwdata2/datasets/dms_field_videos/dms_field_videos_prod/images/images_orij_direct_mp4_png/all_frames'

    save_dict = {}
    all_txt_files = os.listdir(os.path.join(_model_txt_outputs_dir))
    for _txt_file in tqdm.tqdm(all_txt_files):
        _txt_file_path = os.path.join(_model_txt_outputs_dir, _txt_file)
        bb_list = get_eye_patch(_txt_file_path, img_path=None)
        save_dict[_txt_file.replace('_0.txt', '.png')]= bb_list

    # all_image_names = [x.split('.')[0] + '_0.txt' for x in os.listdir(img_dir)]
    # for _image_name in tqdm.tqdm(all_image_names):
    #     if _image_name not in all_txt_files:
    #         save_dict[_image_name.replace('_0.txt', '.png')] = [[0,0,0,0], [0,0,0,0]]

    out_path = os.path.join(out_dir, 'gaze_images_orij_direct_mp4_png_eye_fraction_0.5.json')
    with open(out_path, 'w') as f:
        json.dump(save_dict, f)

    print("Finished Generating eyecrops")

    STEP 4: Breaking the contents to left and right patches for ease in step 5

    with open(out_path, 'r') as file:
        data = json.load(file)

    transformed_data = []
    for frame, coordinates_list in data.items():
        for coordinates in coordinates_list:
            transformed_data.append({"frame": frame, "coordinates": coordinates})

    output_file = "gaze_final_eye_crops.json"
    with open(output_file, "w") as f:
        json.dump(transformed_data, f, indent=4)


    # STEP 5: Checking the model outputs on the eye crops

    print("Running model now...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device used: ", device)

    batch_size = 1
    seed = 42

    torch.manual_seed(42)
    torch.cuda.manual_seed(seed)

    pupil_model = MNetV2Backlow().to(device)
    pupil_model.load_state_dict(torch.load("/data5/ishita/teyed_detections/model_checkpoints/MNetV2low_64/best_mse_epoch28.pth"))

    eye_ball_model_path = '/data5/ishita/teyed_detections/ip_documentation/gaze_estimation/EYESEG_slicing_softmax_sigmoid_batch1.onnx' 
    ort_session = ort.InferenceSession(eye_ball_model_path)

    # print("Number of parameters of the model: ", sum(p.numel() for p in pupil_model.parameters()))

    criterion = nn.MSELoss()

    image_size = 64
    transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            # transforms.Resize((288, 384)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # for Res152 that takes 3 channel input
        ])
    
    data_transform = transforms.Compose([transforms.Resize(size=(64, 64)),  
                                    transforms.ToTensor(),  
                                    transforms.Normalize(mean=[0.485], std=[0.229])])  

    test_dataset = OneVideo("/data5/ishita/teyed_detections/gaze_final_eye_crops.json", "/data5/ishita/teyed_detections/ip_documentation/gaze_extracted_frames/", transform, data_transform)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False)

    print("Dataset Loaded.")

    test_unlabelled(pupil_model, ort_session, test_loader, criterion, device, batch_size)

    print("Finished finding outputs and saving images.")