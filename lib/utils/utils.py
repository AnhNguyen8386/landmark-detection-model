# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Ke Sun (sunk@mail.ustc.edu.cn), Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.optim as optim


def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.DATASET
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / dataset / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
                        (cfg_name + '_' + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)


def get_optimizer(cfg, model):
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR
        )
    elif cfg.TRAIN.OPTIMIZER == 'rmsprop':
        optimizer = optim.RMSprop(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            alpha=cfg.TRAIN.RMSPROP_ALPHA,
            centered=cfg.TRAIN.RMSPROP_CENTERED
        )

    return optimizer


def save_checkpoint(states, predictions, is_best,
                    output_dir, filename='checkpoint.pth'):
    preds = predictions.cpu().data.numpy()
    torch.save(states, os.path.join(output_dir, filename))
    torch.save(preds, os.path.join(output_dir, 'current_pred.pth'))

    latest_path = os.path.join(output_dir, 'latest.pth')
    if os.path.islink(latest_path):
        os.remove(latest_path)
    os.symlink(os.path.join(output_dir, filename), latest_path)

    if is_best and 'state_dict' in states.keys():
        torch.save(states['state_dict'].module, os.path.join(output_dir, 'model_best.pth'))

def get_max_preds(batch_heatmaps):
    assert isinstance(batch_heatmaps, np.ndarray), "batch_heatmaps must be numpy"
    assert batch_heatmaps.ndim == 4
    batch, joints, h, w = batch_heatmaps.shape
    heatmaps_reshaped = batch_heatmaps.reshape((batch, joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.max(heatmaps_reshaped, 2)
    maxvals = maxvals.reshape(batch, joints, 1)
    idx = idx.reshape(batch, joints, 1)
    preds = np.tile(idx, (1,1,2)).astype(np.float32)
    preds[...,0] = preds[...,0] % w
    preds[...,1] = preds[...,1] // w
    pred_mask = preds[...,0:1] >= 0
    preds *= pred_mask
    return preds, maxvals

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.])
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    return np.array([
        src_point[0]*cs - src_point[1]*sn,
        src_point[0]*sn + src_point[1]*cs
    ], dtype=np.float32)

def get_3rd_point(a, b):
    return b + np.array([-(b[1]-a[1]), b[0]-a[0]], dtype=np.float32)

def get_affine_transform(center, scale, rot, output_size, shift=np.array([0,0], dtype=np.float32), inv=False):
    if isinstance(scale, (float, int)):
        scale = np.array([scale, scale], dtype=np.float32)
    scale = np.array(scale, dtype=np.float32)
    scale_temp = scale * 200.0
    src_w = scale_temp[0]
    dst_w, dst_h = output_size[0], output_size[1]
    rot_rad = np.pi*rot/180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], dtype=np.float32)
    src = np.zeros((3,2), dtype=np.float32)
    dst = np.zeros((3,2), dtype=np.float32)
    src[0,:] = center + scale_temp * shift
    src[1,:] = center + src_dir + scale_temp * shift
    src[2,:] = get_3rd_point(src[0,:], src[1,:])
    dst[0,:] = [dst_w*0.5, dst_h*0.5]
    dst[1,:] = dst[0,:] + dst_dir
    dst[2,:] = get_3rd_point(dst[0,:], dst[1,:])
    if inv:
        trans = cv2.getAffineTransform(dst, src)
    else:
        trans = cv2.getAffineTransform(src, dst)
    return trans

def transform_preds(coords, center, scale, output_size):
    t = get_affine_transform(center, scale, 0, output_size, inv=True)
    target = np.zeros_like(coords)
    for i in range(coords.shape[0]):
        target[i,:] = affine_transform(coords[i,:], t)
    return target

def get_final_preds(batch_heatmaps, centers, scales):
    if isinstance(batch_heatmaps, torch.Tensor):
        batch_heatmaps = batch_heatmaps.numpy()
    coords, _ = get_max_preds(batch_heatmaps)
    preds = np.zeros_like(coords)
    batch = coords.shape[0]
    h, w = batch_heatmaps.shape[2], batch_heatmaps.shape[3]
    for i in range(batch):
        preds[i] = transform_preds(coords[i], centers[i], scales[i], [w,h])
    return preds, None