from __future__ import division

from models import *
from utils2.utils1 import *
from utils2.datasets import *

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

if __name__ == '__main__': # 判断这个模块是否是 主模块  也就是说 程序是否是从这个模块开始的
    parser = argparse.ArgumentParser() # 从终端输入一些参数，并定义为parser.parse_args()的属性
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 如果有GPU就用GPU 否则 CPU

    os.makedirs("output", exist_ok=True) # 创建一个output的文件夹  后面会用来存放检测好的图像

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device) # 创建模型


    if opt.weights_path.endswith('.weights'):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path) # 读取模型参数
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval() # 模型切换为预测模式

    dataloader = DataLoader( # 把batch_size个图像打包为一个batch
        ImageFolder(opt.image_folder, img_size=opt.img_size), # 一个包含 __getitem__() 和 __len__()方法的类，服务与DataLoader
        batch_size=opt.batch_size,
        shuffle=False, # 是否打乱顺序
        num_workers=opt.n_cpu # cpu工作数量
    )

    classes = load_classes(opt.class_path) # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = [] # Stores image paths
    img_detections = [] # Stores detections for each image index

    print('\n执行物体检测')
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader): # img_paths 是包含一个batch所有图像的路径  input_imgs就是batch [batch_size, 3, 416, 416]
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad(): # 就算是模型设置了model.eval() 这样模型不会更新weights   但是 设置torch.no_grad()可以不保存梯度 节约内存
            detections = model(input_imgs) # 获取网络检测结果
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres) # 非最大值抑制
            print(detections)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time) # 这里是返回 处理一个batch所用的时间
        prev_time = current_time
        print('\t+ Batch %d, Inference Time: %s' % (batch_i, inference_time))

        # Save image and detections
        imgs.extend(img_paths) # img_paths储存图像路径的tuple   .entend 把这个tuple的元素 添加到imgs 这个list中
        img_detections.extend(detections) # 把detections (tensor) 添加到 img_detections (list)中

    # Bounding-box colors
    cmap = plt.get_cmap('tab20b') # 获取一种调色板
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]  # np.linspace(0, 1, 20) 在 (0, 1) 中间均匀的取20个点

    print('\n保存图像：')
    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        print('(%d) Image: "%s"' % (img_i, path))

        # Create plot
        img = np.array(Image.open(path)) # Image.open   打开的图像 通道数在最后   而opencv 打开的图像在最前
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, opt.img_size, img.shape[:2]) # 把在416*416的图像上检测的bbox 还原到原图上的bbox
            unique_labels = detections[:, -1].cpu().unique() # 返回物体类别的种类
            n_cls_preds = len(unique_labels) # 返回物体类别的种类数目
            bbox_colors = random.sample(colors, n_cls_preds) # 在colors调色板中随机不放回抽取 n_cls_preds 个颜色
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                print('\t+ Label: %s, Conf: %.5f' % (classes[int(cls_pred)], cls_conf.item()))

                box_w = x2 - x1
                box_h = y2 - y1

                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])] #  np.where 返回满足条件的索引((index),)
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor='none') # bbox 的绘图补丁
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                plt.text(
                    x1,
                    y1,
                    s=classes[int(cls_pred)],
                    color='white',
                    verticalalignment='top', # 对齐线在上方
                    bbox={'color': color, 'pad': 0},
                )

        # Save generated image with detections
        plt.axis('off')
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        filename = path.split("/")[-1].split(".")[0]
        plt.savefig(f"output/{filename}.png", bbox_inches="tight", pad_inches=0.0)
        plt.close()








