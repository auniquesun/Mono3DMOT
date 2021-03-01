from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import os.path as osp
import cv2
import logging
import argparse
import motmetrics as mm
import numpy as np
import torch

from tracker.multitracker import JDETracker
from tracking_utils import visualization_test as vis
from tracking_utils.log import logger
from tracking_utils.timer import Timer
from tracking_utils.evaluation import Evaluator
import datasets.dataset.jde as datasets

from tracking_utils.utils import mkdir_if_missing
from loc3d_utils import Loc3DParams
from opts import opts


def letterbox(img, height=608, width=1088,
              color=(127.5, 127.5, 127.5)):  # resize a rectangular image to a padded rectangular
    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(float(height) / shape[0], float(width) / shape[1])
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
    dw = (width - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
    return img, ratio, dw, dh

    
def compute_location(tlwh, l3dp):
    x1, y1, w, h = tlwh
    # center of pedestrian bbox
    x_j, y_j = x1 + w/2, y1 + h/2
    # center of image
    x_i, y_i = l3dp.img_width/2, l3dp.img_height/2

    # pedestrian 3d location in camera coordinate system
    x_c = l3dp.fy / l3dp.fx * l3dp.H * (x_j - x_i) / h
    y_c = l3dp.H * (y_j - y_i) / h
    z_c = l3dp.H * l3dp.fy / h

    return (x_c, y_c, z_c)


def run(opt):
    cap = cv2.VideoCapture(opt.access_token)

    if not cap.isOpened():
        print('camera is not opened !!!')
        exit()

    tracker = JDETracker(opt)
    timer = Timer()
    l3dp = Loc3DParams(opt.camera_fx, opt.camera_fy, img_size=opt.img_size, H=opt.pedes_height)
    frame_id = 0

    while True:
        ret, img0 = self.cap.read()
        if not ret:
            print('There is no frame !!!')
            break

        if frame_id % 100 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        # Resize img0    
        img0 = cv2.resize(img0, (1920, 1080))
        # Padded resize
        img, _, _, _ = letterbox(img0, height=1088, width=608)
        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        # run tracking
        timer.tic()
        blob = torch.from_numpy(img).cuda().unsqueeze(0)
        online_targets = tracker.update(blob, img0)
        online_tlwhs = []
        online_ids = []
        # save 3d locations
        online_locs = []

        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                loc = compute_location(tlwh, l3dp)
                online_locs.append(loc)
        timer.toc()
        
        online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, online_locs, frame_id=frame_id,
                                        fps=1. / timer.average_time)
        cv2.imshow('online_im', online_im)
        
        frame_id += 1
        if cv.waitKey(1) == ord('q'):
            break
    

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init()
    run(opt)
    