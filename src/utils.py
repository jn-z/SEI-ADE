import pdb

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import cv2
from PIL import Image, ImageDraw
import math
from dataset import ctDataset
import logging
import os
logger = logging.getLogger("zjn-cennet")
def setup_logs(save_dir, run_name):
    # initialize logger
    logger = logging.getLogger("cdc")
    logger.setLevel(logging.INFO)

    # create the logging file handler
    log_file = os.path.join(save_dir, run_name + ".log")
    fh = logging.FileHandler(log_file)

    # create the logging console handler
    ch = logging.StreamHandler()

    # format
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    fh.setFormatter(formatter)

    # add handlers to logger object
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
def snapshot(dir_path, run_name, state):
    snapshot_file = os.path.join(dir_path,
                                 run_name + '-model_best.pth')

    torch.save(state, snapshot_file)
    logger.info("Snapshot saved to {}\n".format(snapshot_file))
def plot_heapmap(heatmap):
    ''' Plot the predicted heatmap

        Args:
            heatmap ([h, w]) - the heatmap output from keypoint estimator
    '''
    fig, ax = plt.subplots()
    im = ax.imshow(heatmap)
    ax.set_title("Prediction Heatmap")
    fig.tight_layout()
    plt.show()

def generate_gt_data(index):
    ''' Generate GT data as a detection result for testing
    '''

    my_dataset = ctDataset()
    gt_res = my_dataset.__getitem__(index)
    for key in gt_res:
        gt_res[key]  = torch.from_numpy(gt_res[key])
    wh = torch.zeros((1, 2, 128, 128))
    reg = torch.zeros((1, 2, 128, 128))
    hm = gt_res['hm'].reshape(1, 1, 128, 128)

    for i in range(128):

        if gt_res['reg_mask'][i] == 0:
            continue
        else:
            ind = gt_res['ind'][i]
            height_idx = int(ind // 128)
            width_idx = int(ind % 128)
            wh[0, 0, height_idx, width_idx] = gt_res['wh'][i, 0]
            wh[0, 1, height_idx, width_idx] = gt_res['wh'][i, 1]

            reg[0, 0, height_idx, width_idx] = gt_res['reg'][i, 0]
            reg[0, 1, height_idx, width_idx] = gt_res['reg'][i, 1]


    return hm, wh, reg
def gather_feat_(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat
def gather_feat_clses(feat, ind, mask=None):
    #pdb.set_trace() #feat=[1,6,256,256],ind =[1,6,100]
    #feat = feat.permute(0, 2, 1, 3).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind) #torch.Size([1, 100, 256])
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat
def gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    ind_s = ind[:,::2, :]
    ind_e = ind[:,1::2, :]
    #ind  = ind.unsqueeze(3).expand(ind.size(0), ind.size(1),ind.size(2), dim)
    feat_s = feat.gather(1, ind_s)
    feat_e = feat.gather(1, ind_e)
    feat = (feat_s + feat_e) / 2 # optimize in process
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _gather_feat_predict(feat, ind, mask=None):
    dim  = feat.size(2)
    # feat.shape =[2,256,256,2]
    #pdb.set_trace() # ind = torch.Size([2, 128, 2]), feat = [2,256*256,2]
    ind  = ind.unsqueeze(3).expand(ind.size(0), ind.size(1),ind.size(2), dim) # ind = [2,128,2,2]
    #ind = ind.permute(0, 1, 3, 2).contiguous() # ind = [2,128,6,256]
    feat_s = feat.gather(1, ind[:, :, 0, :]) # feat = [2,256*256,2],# ind = [2,128,2]
    feat_e = feat.gather(1, ind[:, :, 1, :])
    feat = (feat_s + feat_e) / 2 # optimize in process
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat
def transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat_predict(feat, ind)
    return feat