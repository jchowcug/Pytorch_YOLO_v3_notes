from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from utils2.parse_config import *
from utils2.utils1 import build_targets, to_cpu

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def create_modules(module_defs):

    """ Constructs module list of layer blocks from module configuration in module_defs """

    hyperparams = module_defs.pop(0) # .pop(0) 移除列表第一个元素，并返回该元素， 这里返回描述网络整体信息的字典
    output_filters = [int(hyperparams['channels'])] # output_filters 记录各个layer后的 channels ，这里先记录输入图像的channels
    module_list = nn.ModuleList()
    for module_i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def['type'] == 'convolutional': # 如果是卷积层
            bn = int(module_def['batch_normalize'])
            filters = int(module_def['filters']) # 就是out_channels
            kernel_size = int(module_def['size'])
            pad = (kernel_size -1) // 2
            modules.add_module( # 添加layer（'layer_name', layer)
                f'conv_{module_i}', # layer 名, 例如， conv_1
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def['stride']),
                    padding=pad,
                    bias=not bn, # 如果有batch_norm 就不用bias
                ),
            )
            if bn: # 如果需要 batch normalize
                modules.add_module(f'batch_norm_{module_i}', nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5))
            if module_def['activation'] == 'leaky': # 激活 LeakyReLU 与 ReLU 的区别在于 当 x < 0 时 y(leakyrelu)=ax, a 为很小的常数
                modules.add_module(f'leaky_{module_i}', nn.LeakyReLU(0.1))

        elif module_def['type'] == 'maxpool':
            kernel_size = int(module_def['size'])
            stride = int(module_def['stride'])
            if kernel_size == 2 and stride == 1: ############为什么需要这么做？
                modules.add_module(f'_debug_padding_{module_i}', nn.ZeroPad2d((0, 1, 0, 1))) # nn.ZeorPad2d((a, b, c, d)) 分别在图像左右上下填充a, b, c, d层0
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module(f'maxpool_{module_i}', maxpool)

        elif module_def['type'] == 'upsample':
            upsample = Upsample(scale_factor=int(module_def['stride']), mode='nearest') # 用nn.functional.interpolate 实现上采样
            modules.add_module(f'upsample_{module_i}', upsample)

        elif module_def['type'] == 'route':
            layers = [int(x) for x in module_def['layers'].split(',')]
            filters = sum([output_filters[1:][i] for i in layers])
            modules.add_module(f'route_{module_i}', EmptyLayer()) # 先用一个空层，占个位置，后面再处理

        elif module_def['type'] == 'shortcut':
            filters = output_filters[1:][int(module_def['from'])]
            modules.add_module(f'shortcut_{module_i}', EmptyLayer())

        elif module_def['type'] == 'yolo':
            anchor_idxs = [int(x) for x in module_def['mask'].split(',')] # 选择用哪几个anchor
            # Extract anchors
            anchors = [int(x) for x in module_def['anchors'].split(',')]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def['classes'])
            img_size = int(hyperparams['height'])
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes, img_size)
            modules.add_module(f'yolo_{module_i}', yolo_layer)
        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list


class Upsample(nn.Module):


    def __init__(self, scale_factor, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class EmptyLayer(nn.Module):
    """ Placeholder for 'route' and 'shortcut' layers """

    def __init__(self):
        super(EmptyLayer, self).__init__()


class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes, img_dim=416):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors # [ [], [], [] ]
        self.num_anchors = len(anchors)
        self.num_classes = num_classes # object 种类
        self.ignore_thres = 0.5 # 分类器阈值
        self.mse_loss = nn.MSELoss() # 均方损失
        self.bce_loss = nn.BCELoss() # 交叉熵损失
        self.obj_scale = 1 # 权重
        self.noobj_scale = 100 # 权重
        self.metrics = {}
        self.img_dim = img_dim # 图像一列的像素
        self.grid_size = 0  # 一列的grid数量

    def compute_grid_offsets(self, grid_size, cuda=True):
        self.grid_size = grid_size
        g = self.grid_size
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride = self.img_dim / self.grid_size # 一个grid一列有多少个像素
        # Calculate offsets for each grid
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor) # arange(3)生成[0, 1, 2]  arange(3).repeat(3, 1)生成[[0, 1, 2], [0, 1,2], [0, 1, 2]] 最终生成shape[1, 1, grid_size, grid_size]
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)  # 这里grid_x, grid_y 的作用是 便于后面计算相对于grid的坐标值
        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors]) # anchor 相对于grid的宽高
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    def forward(self, x, targets=None, img_dim=None):

        # x:[1, 255, grid_size, grid_size]
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        self.img_dim = img_dim
        num_samples = x.size(0) # batch size
        grid_size = x.size(2)

        prediction = (
            x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size) # [1， 3， 85， grid_size, grid_size]
            .permute(0, 1, 3, 4, 2) # 改变维度顺序 [1, 3, grid_size, grid_size, 85]
            .contiguous() # copy 一份数据，这份数据与原数据无关，即使原数据改变，contiguous()后的数据不变
        )
        print('\nin yolo prediction\n', prediction.shape)

        # Get outputs
        x = torch.sigmoid(prediction[..., 0]) # grid预测的bbox中心点 [1, 3, 13, 13, 1] ... 相当于 :, :, : 多个
        y = torch.sigmoid(prediction[..., 1])
        w = prediction[..., 2] # anchor 宽，高 放缩系数 [1, 3, 13, 13, 1]  anchor 变化为bbox用
        h = prediction[..., 3]
        pred_conf = torch.sigmoid(prediction[..., 4]) # grid 是否有物体的 confidence [1, 3, 13, 13,1]
        pred_cls = torch.sigmoid(prediction[..., 5:]) # grid 对物体属于某个类型的分数 【1，3， 13, 13， 80】

        # If grid size does not match current we compute new offsets
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + self.grid_x # bbox添加补偿，获得相对于grid_size的坐标   对anchor平移
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w # 用放缩系数 × anchor 尺寸  以达到获取精确的bbox尺寸  对anchor 放缩
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        output = torch.cat( # cat 把数据按最后一个维度（-1）拼接起来
            (
                pred_boxes.view(num_samples, -1, 4) * self.stride, # 还原为图像坐标 改变维度[1, 3 * grid**2, 4]
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1,
        )

        if targets is None:
            return output, 0
        else:
            # 预测bbox与label bbox的iou[b, best_anchor, x, y]  ， 类型预测正确的mask[]与物体数相同 正确预测物体为True
            # 有物体的mask[b,best,gj,gi], 没有物体的mask[b,best,gj,gi]，
            # tx[b, best_n, gj, gi] ty[b, best_n, gj, gi]相对于gird的GT偏移
            # tw[b, best_n, gj, gi] th[b, best_n, gj, gi] 相对于anchor的GT放缩
            # tcls[b, best_n, gj, gi, target_labels] GT类型  tconf[b, best_n, gj, gi] GT自信度 confidence
            print('\n\nin yolo train module')
            print('pred_boxes.shape', pred_boxes.shape)
            print('pred_cls.shape', pred_cls.shape)
            print('targets.shape', targets.shape)
            print('scaled_anchors.shape', self.scaled_anchors.shape)
            print('self.ignore_thres', self.ignore_thres)
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target=targets,
                anchors=self.scaled_anchors,
                ignore_thres=self.ignore_thres,
            )

            # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
            # 计算
            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj ################### 查看论文
            loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            # Metrics
            cls_acc = 100 * class_mask[obj_mask].mean()
            conf_obj = pred_conf[obj_mask].mean()
            conf_noobj = pred_conf[noobj_mask].mean()
            conf50 = (pred_conf > 0.5).float() # conf > .5 认为有物体 ###################### 为什么要float 一下
            iou50 = (iou_scores > 0.5).float() # iou > 50% 的mask
            iou75 = (iou_scores > 0.75).float() # iou75 > 75% 的mask
            detected_mask = conf50 * class_mask * tconf # 认为有物体并正确预测类型，而且确实有物体的mask 即，正确预测有物体，类型的mask（不考虑iou）（这被认为是全部预测的）
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16) # 精确度 （认为iou > .5 即正确) 预测正确的 / 全部预测的
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16) # 召回率50% 预测正确的 / 全部物体
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16) # 召回率75% 预测正确的 / 全部物体

            self.metrics = {
                "loss": to_cpu(total_loss).item(), # 总损失 item 总只有一个的tensor中获取python number
                "x": to_cpu(loss_x).item(), # x 损失
                "y": to_cpu(loss_y).item(),
                "w": to_cpu(loss_w).item(),
                "h": to_cpu(loss_h).item(),
                "conf": to_cpu(loss_conf).item(), # 自信度损失
                "cls": to_cpu(loss_cls).item(), # 类型损失
                "cls_acc": to_cpu(cls_acc).item(), # 正确预测类型的精度
                "recall50": to_cpu(recall50).item(), # 召回率
                "recall75": to_cpu(recall75).item(),
                "precision": to_cpu(precision).item(), # 精确率
                "conf_obj": to_cpu(conf_obj).item(), # 预测正确有物体的平均自信度
                "conf_noobj": to_cpu(conf_noobj).item(), # 预测正确无物体的平均自信度
                "grid_size": grid_size,
            }
            return output, total_loss


class Darknet(nn.Module):
    """ YOLOv3 object detection model """

    def __init__(self, config_path, img_size=416):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(config_path) #一个list 这个list 储存的是描述网络结构的 字典 dict 第一个字典描述网络整体信息
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.yolo_layers = [layer[0] for layer in self.module_list if hasattr(layer[0], 'metrics')] # 如果layer中有metric这个属性（这是一个包含损失和精度等信息的字典），那么它是yolo layer
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)

    def forward(self, x, targets=None):
        # x的shape[batch_size, channels, width, height]
        img_dim = x.shape[2]

        print('\n at the moment in model img.shape \n', x.shape)

        loss = 0
        layer_outputs, yolo_outputs = [], [] # 记录所有的feature map    记录所有yolo 之后的feature map
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)): # zip将两个iterable对应元素打包 a = [1, 2, 3] b = [4, 5, 6] c = zip(a, b) c = [(1, 4), (2, 5), (3, 6)]
            if module_def['type'] in ['convolutional', 'upsample', 'maxpool']: # 如果是 卷积层 上采样层 池化层 ，直接把x输入进去
                x = module(x)
            elif module_def['type'] == 'route': # route 如果只有一个参数 如 -4 则输出第-4个feature map 如果有两个参数如 61, -1 则是把第61和-1个feature map 在channels维度上拼接起来
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def['layers'].split(',')], 1) # 对图像在channels维度上叠加
            elif module_def['type'] == 'shortcut': # 把当前feature map 和 倒数第‘from'个feature map 的 x 的每个数值对应相加
                layer_i = int(module_def['from'])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def['type'] == 'yolo': # 有3个yolo层 在detect 中 分别有grid_size 13 26 52
                print('\n in model before yolo img.shape \n', x.shape)
                x, layer_loss = module[0](x, targets, img_dim)
                loss += layer_loss # 损失为所有的yolo损失之和
                yolo_outputs.append(x) # 记录所有yolo 之后的feature map
            layer_outputs.append(x)
        yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1)) # 把所有的yolo之后的feiture map 在channels 维度上 叠加
        return yolo_outputs if targets is None else (loss, yolo_outputs)

    def load_darknet_weights(self, weights_path):
        """ Parses and loads the weights stored in 'weights_path' """

        # Open the weights file
        with open(weights_path, 'rb') as f: # 'rb' 以二进制格式打开一个文件用于只读
            header = np.fromfile(f, dtype=np.int32, count=5) # First five are header values
            self.header_info = header
            self.seen = header[3]
            weights = np.fromfile(f, dtype=np.float32)


        # Establish cutoff for loading backbone weights
        cutoff = None
        if 'darknet53.conv.74' in weights_path:
            cutoff = 75

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if i == cutoff:
                break
            if module_def['type'] == 'convolutional':
                conv_layer = module[0] # module 分为3层 conv,   batch_norm,   leaky_relu
                if module_def['batch_normalize']:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1] # 这里获取bn的模型参数 包括 bias(beta) weight(gamma) running_mean(E(x)) running_var(Var(x))   计算公式为  y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta 其中 gamma、beta是通过反向传播求得的， E(x) Var(x) 是通过 公式\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t 更新的 E(x) Var(x) 是batch为单位，各个feature map单独计算的 比如batch_size=8 feature map = 32 则 要计算32个这些值
                    num_b = bn_layer.bias.numel()  # numel 返回tensor中元素个数   这里返回bn 中 bias的元素个数
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.bias) # view_as 返回与给定tensor相同大小的原tensor
                    bn_layer.bias.data.copy_(bn_b) # data 令数值不在求梯度
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.weight) ########数据储存方式 要搞清楚
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)# 卷积核大小为k x k x input_channels  卷output_channels 的次数 所以卷积的权重大小为[output_channels, input_channels, kernel_width, kernel_height]
                ptr += num_w

    def save_darknet_weights(self, path, cutoff=-1):
        """
        :param path:     - path of the new weights file
        :param cutoff:   - save layers bewteen 0 and cutoff (cutoff = -1 -> all are saved)
        """
        fp = open(path, 'wb')
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def['type'] == 'convolutional':
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def['batch_normalize']:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()




