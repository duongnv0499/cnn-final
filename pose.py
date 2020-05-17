import cv2
import numpy as np
import sys

import torch

from datasets.lip import LipValDataset

from models.single_person_pose_with_mobilenet import SinglePersonPoseEstimationWithMobileNet

from modules.calc_pckh import calc_pckh
from modules.load_state import load_state

import time

def predict(net, img):
    net = net.eval()
    base_height = 256
    scales = [1]
    stride = 8
    #start = time.time()
    avg_heatmaps = infer(net, img, scales, base_height, stride)
    all_keypoints = []
    ##end = time.time()
    #print('image process take : ', end - start )
    for kpt_idx in range(16):
        all_keypoints.append(extract_keypoints(avg_heatmaps[:, :, kpt_idx]))

    return all_keypoints

def extract_keypoints(heatmap, min_confidence=-100):
    ind = np.unravel_index(np.argmax(heatmap, axis=None), heatmap.shape)
    if heatmap[ind] < min_confidence:
        ind = (-1, -1)
    else:
        ind = (int(ind[1]), int(ind[0]))
    return ind




def infer(net, img, scales, base_height, stride, img_mean=[128, 128, 128], img_scale=1/256):
    
    height, width, _ = img.shape
    scales_ratios = [scale * base_height / max(height, width) for scale in scales]
    avg_heatmaps = np.zeros((height, width, 17), dtype=np.float32)

    for ratio in scales_ratios:
        
        resized_img = cv2.resize(img, dsize=None, fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)
        max_side = max(resized_img.shape[0], resized_img.shape[1])
        padded_img = np.ones((max_side, max_side, 3), dtype=np.uint8) * img_mean
        x_offset = (padded_img.shape[1] - resized_img.shape[1]) // 2
        y_offset = (padded_img.shape[0] - resized_img.shape[0]) // 2
        padded_img[y_offset:y_offset + resized_img.shape[0], x_offset:x_offset + resized_img.shape[1], :] = resized_img
        padded_img = normalize(padded_img, img_mean, img_scale)
        pad = [y_offset, x_offset,
               padded_img.shape[0] - resized_img.shape[0] - y_offset,
               padded_img.shape[1] - resized_img.shape[1] - x_offset]

        tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
        stages_output = net(tensor_img)

        heatmaps = np.transpose(stages_output[-1].squeeze().cpu().data.numpy(), (1, 2, 0))
        heatmaps = cv2.resize(heatmaps, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
        heatmaps = heatmaps[pad[0]:heatmaps.shape[0] - pad[2], pad[1]:heatmaps.shape[1] - pad[3]:, :]
        heatmaps = cv2.resize(heatmaps, (width, height), interpolation=cv2.INTER_CUBIC)
        avg_heatmaps = avg_heatmaps + heatmaps / len(scales_ratios)

    return avg_heatmaps

def normalize(img, img_mean, img_scale):
    img = np.array(img, dtype=np.float32)
    img = (img - img_mean) * img_scale
    return img
