from __future__ import division

from models import *
from utils2.logger import *
from utils2.utils1 import *
from utils2.datasets import *
from utils2.parse_config import *
from test import evaluate

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3-custom.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/custom.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    opt = parser.parse_args()
    print(opt)

    logger = Logger('log')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs('output', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(opt.data_config) # 储存class的数量 训练集名路径，测试集名路径 类型名路径
    train_path = data_config['train']
    valid_path = data_config['valid']
    class_names = load_classes(data_config['names']) # 返回一个储存类型名的list

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    model.apply(weights_init_normal) # apply递归的网络的所有 submodule 和 自身 对每个部分都调用weights_init函数，以submodule作为参数 输入到weights_init_normal 函数中

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith('.path'):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)

    # Get dataloader
    dataset = ListDataset(train_path, augment=True, multiscale=opt.multiscale_training)
    dataloader = torch.utils.data.DataLoader( # 返回的是一个 list 里面包含 num_batch 个 list  里面的元素是 collate_fn的返回值(会把tuple转为list)
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True, # pin_memory 为 True 时运行更快，但是更加占用内存
        collate_fn=dataset.collate_fn, # collate_fn 是把一张一张图像及其label 打包为 batch 的函数，
    )

    optimizer = torch.optim.Adam(model.parameters()) # model.parameters()是一个generator 用list可以看到 里面保存着模型的各个参数

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    for epoch in range(opt.epochs):
        model.train()
        start_time = time.time()
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i # len(dataloader) 就是 batch的数量

            imgs = Variable(imgs.to(device)) # 转为variable 用于构建图谱
            targets = Variable(targets.to(device), requires_grad=False)

            print('\n before model imgs.shape \n', imgs.shape)
            loss, outputs = model(imgs, targets) # 用模型计算 损失 和 输出   (imgs.shape=[8, 3, 352, 325], targets.shape=[45, 6]
            loss.backward() # 调用backward的量可以是标量，也可以是矩阵，但是如果是矩阵后面()中要包含另一个描述权重的矩阵，两个矩阵的shape要相同 权重矩阵建议用torch.Tensor([])定义    torch.Tensor 等价于 torch.FloatTensor   torch.tensor 会根据后面data的类型创建
                            # loss.backward 会根据这个 loss 这个节点  反向求所有节点的梯度
                            # backward() 有个参数retain_variables=False  它会在求完梯度后，清空计算图，如果True 就不会清空，保留图谱。  注：只有有图谱，才能反向传播计算梯度
            if batches_done % opt.gradient_accumulations: # 如果不及时把Variable的梯度清空，Variable的梯度会累加
                # Accumulates gradient before each step
                optimizer.step() # 根据梯度更新模型的各个参数 weight  bias 等     注：这里的梯度不是x的梯度，而是模型参数的梯度（因为优化的是模型参数，而不是x）
                optimizer.zero_grad() # 把定义optimizer时model.parameters()的所有 Variable的梯度归零，如果不归零，下次backward的梯度就会直接加在原梯度上

            #
            #
            #

            log_str = '\n--- [Epoch %d/%d, Batch %d/%d] ----\n' % (epoch, opt.epochs, batch_i, len(dataloader))

            metrics_table = [['Metrics', *[f'YOLO Layer {i}' for i in range(len(model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: '%.6f' for m in metrics} # 定义formats为一个字典
                formats['grid_size'] = '%2d'
                formats['cls_acc'] = '%.2f%%' # 在字符串中要打出% 需要 %%
                # 这里 formats[metric[ 是一个字符串'%2d'  而 '%2d' % 1   的含义是 把1转化为%2d的字符串
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers] # yolo.metrics是yolo层的metrics属性，是一个字典， get(key, default=None)是字典的方法，key是要查找的键，如果不存在，返回default值
                metrics_table += [[metric, *row_metrics]]

                # Tensorboard logging
                tensorboard_log = []
                for j, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items(): # items 以列表返回可遍历的(键, 值) 元组数组  可以方便迭代dict
                        if name != 'grid_size':
                            tensorboard_log += [(f'{name}_{j+1}', metric)]

                tensorboard_log += [('loss', loss.item())]

                logger.list_of_scalars_summary(tensorboard_log, batches_done)

            log_str += AsciiTable(metrics_table).table
            log_str += f'\nTotal loss {loss.item()}'

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            print(log_str)


            model.seen += imgs.size(0)

        if epoch % opt.evaluation_interval == 0:
            print('\n---- Evaluating Model ----')
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=8,
            )