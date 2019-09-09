from __future__ import division

from models import *
from utils2.utils1 import *
from utils2.datasets import *
from utils2.parse_config import *

import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim


def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model.eval()

    # Get dataloader
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = [] # List of tuples (TP, confs, pred)
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc='Detecting objects')): # tqdm.tqdm(a)会观察a的迭代情况，并且绘制出进度条。

        # Extract labels
        print('\n\nevaluate')
        print('imgs.shape', imgs.shape)
        print('targets.shape', targets.shape)
        labels += targets[:, 1].tolist() # 物体类型
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size # 把bbox还原为 绝对大小

        imgs = Variable(imgs.type(Tensor), requires_grad=False)
        print('targets.shape', targets.shape)
        print(iou_thres)
        with torch.no_grad():
            outputs = model(imgs) # outputs.shape = [batch_size, 10000+,
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres) # outputs是一个list 长度为batch_size  元素为[obj_num,7] x1, y1, x2, y2, obj_conf, cls_conf, cls
        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres) # 返回batch_size个元素的list  每个元素包含每张图片的[TP, 预测分数， 预测类别]

    # Concatenate sample statistics

    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class
