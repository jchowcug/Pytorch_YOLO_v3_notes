import glob
import random
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from utils2.augmentations import horisontal_flip
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def pad_to_square(img, pad_value): # 补充图像为正方形
    c, h, w, = img.shape
    dim_diff = np.abs(h - w) # 记录高宽的差值
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding 如果高小于等于宽 则在上下补充  否则在左右补充
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)   # 分别为左右上下
    # Add padding
    img = F.pad(img, pad, 'constant', value=pad_value) # functional.pad   按照pad 用value 补充图像四周

    return img, pad


def resize(image, size):

    image = F.interpolate(image.unsqueeze(0), size=size, mode='nearest').squeeze(0) # 差值放缩 由于functional.interpolate 只能对四维的input 处理[batch, channels, width, height] 所以这里 在第0维unsqueeze
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0] # 在288，488之间等份32个，并且随机取样1个
    images = F.interpolate(images, size=new_size, mode='nearest') # 用上面随机取样的size 对图像 放缩
    return images


class ImageFolder(Dataset): # 在pytorch中，通常是通过集成torch.utils.data.Dataset类来实现数据输入 需要该类的 __getitem__() 和 ）__len__() 方法 实现后会将自雷的实例作为DataLoader的参数，来构建生成batch的实例对象   这里一般会包括简单的数据处理
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob('%s/*.*' % folder_path)) # glob.glob 返回所有满足格式 '%s/*.*' 的文件名  好用
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)] # 这里有个取余的操作  猜测是为了防止越界
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path)) # 把一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray，转换成形状为[C,H,W]，取值范围是[0,1.0]的torch.FloadTensor   PIL格式的图像为[H, W, C] np(opencv)图像格式为[C,H,W]
        # Pad to square resolution
        img, _ = pad_to_square(img, 0) # 把图像补充为正方形
        # Resize
        img = resize(img, self.img_size) # 把图像放缩为 416*416 大小

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, augment=True, multiscale=True, normalized_labels=True):
        with open(list_path, 'r') as file:
            self.img_files = file.readlines()

        self.label_files = [
            path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt') # str.replace('abc', 'qwe') 返回把str中的'abc'替换为'qwe’的字符串 但是不会改变str，只是返回值改变了而已
            for path in self.img_files
        ]
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_babels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __getitem__(self, index):





        img_path = self.img_files[index % len(self.img_files)].rstrip()

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:])) # expand 可以直接改变目标的shape 这里改变了第一个通通道

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_babels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape





        label_path = self.label_files[index % len(self.img_files)].rstrip()

        targets = None
        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            # Extract coordinates for unpaddedd + unscaled image
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2) # 由于标签的bbox 为相对于图像大小的相对值 x, y, w, h   其中x,y为bbox坐上坐标
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2) # 这里还原为 x1, y1, x2, y2 的形式 并且 坐标为绝对坐标
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            # Adjust for added padding
            x1 += pad[0] # 修改bbox坐标为 pad 后的坐标
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            # Returns (x, y, w, h)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w # 还原为相对坐标 并且为 x, y, w, h 形式   其中 x, y 为bbox中心点坐标
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)

        return img_path, img, targets

    def collate_fn(self, batch): # 一般collate_fn的任务是把一个batch的多个__getitem__的输出 打包在一起，这里多两个任务，一个是把标签tensor添加batch图像序号，一个是随机放缩图像（数据增强）
        # 输入的是一个list 里面包含batch_size个tuple  tuple包含3个元素img_path, img, targets  collate_fn详情件DataLoader的注释
        paths, imgs, targets = list(zip(*batch)) #     *相当于打开括号   list(zip())相当于添加括号，不过不是*的逆过程，而是 对应维度提取出来，然后添加括号 比如 list(zip(['a', 1],['b', 2])) == [('a', 'b'), (1, 2)]  zip()过后 需要list()一下才能显示  返回3个tuple
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None] # 顺便把tuple 变为 list  里面包含batch_size个tensor tensor shape为 [num_bbox, 6] 后面5个为cls, x, y ,w ,h
        # Add sample index to targets
        for i, boxes in enumerate(targets): # 给[num_bbox, 6]矩阵的第1列 标上batch中图像的索引
            boxes[:, 0] = i
        targets = torch.cat(targets, 0) # 把batch_size 个 [num_bbox, 6] 按照0维度叠加起来为一个大tensor
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0: # 每10个batch 随机放缩一次图像
            self.img_size = random.choice(range(self.min_size, self.max_size+1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets  # 返回的是tuple(路径）, tensor().shape=[8, 3, w, h], tensor([batch_box, 6])

    def __len__(self):
        return len(self.img_files)