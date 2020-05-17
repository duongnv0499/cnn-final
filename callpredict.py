import cv2
import numpy as np
import sys

import torch

from datasets.lip import LipValDataset
from models.single_person_pose_with_mobilenet import SinglePersonPoseEstimationWithMobileNet
from modules.calc_pckh import calc_pckh
from modules.load_state import load_state

from pose import predict

import time
net = SinglePersonPoseEstimationWithMobileNet(num_refinement_stages=1)
checkpoint = torch.load('./checkpoint_epoch_70.pth', map_location='cpu')
load_state(net, checkpoint)


def predict_img(img):
    image = cv2.imread(img)
    start = time.time()
    a = predict(net, image)
    end = time.time()
    print(end-start, 'second')

    print(a)
    return a 

predict_img("./36981_218953.jpg")