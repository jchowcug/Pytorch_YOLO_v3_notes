import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def to_cpu(tensor):
    return tensor.detach().cpu()


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, 'r')
    names = fp.read().split('\n')[:-1] # fp 只能调用一次之久就会清空
    return names


def weights_init_normal(m):
    classname = m.__class__.__name__ # submodule 的名称
    if classname.find('Conv') != -1: # find函数 如果存在 返回0   不存在返回-1
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02) # torch.nn.init.normal_ 对tensor 或者 float  正态分布N(0,0.02)初始化
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02) # N(1, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0) # 常数初始化


def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves."""
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i] # 按分数 从大到小排列

    # Find unique classes
    unique_classes = np.unique(target_cls) # 去除多余类型

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in tqdm.tqdm(unique_classes, desc='Computing AP'):
        i =  pred_cls == c # 获取这个类型的索引
        n_gt = (target_cls == c).sum() # 标签中这个类型的数目
        n_p = i.sum() # prediction中这个类型的数目

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum() # cumsum() 累加 [1, 2, 3].cumsum() = [1, 3, 6]
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

        # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2) # 这种计算方式只适合两个anchor包含，或者承十字交叉型
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


def rescale_boxes(boxes, current_dim, original_shape):
    """ Rescales bounding boxes to the original shape """
    orig_h, orig_w = original_shape
    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h

    return boxes



def bbox_iou(box1, box2, x1y1x2y2=True):
    """ Returns the IoU of two bounding boxes """
    # 这里的box1, box2 的值都是相对于grid 大小的
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp( # 为什么要 +1 因为要计算一行像素点的个数，而不是简单的求两个坐标的距离
        inter_rect_y2 - inter_rect_y1 + 1, min=0 # 比如 一个box的左上坐标为（3,4），另一个box的左上坐标为（5,6） 对于x轴，如果直接相减，得到5 - 3 = 2 ,但实际上 应该是 第3，4,5个坐标点都算，所以是3
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and preforms
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4]) # prediction shape [batch_size, num_anchor*(3个grid_size), 85]
    output = [None for _ in range(len(prediction))] # len(prediction) = batch_size 用于储存输出结果 这个操作的好处是后面可以用索引
    for image_i, image_pred in enumerate(prediction): # 一张一张的迭代batch中的图像 image_pred shape [num_anchor*(3个grid_size), 85] 在detect中为 [10647, 85]
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= conf_thres] # 获取自信度大于阈值的prediction 在detect中 [9, 85]
        # If none are remaining => process next image
        if not image_pred.size(0): # 如果图像中没有自信度大于阈值的 就跳过 .size(0) 返回第0维度的数量
            continue
        # Object confidence times class confidence
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0] # 是否有物体的自信度 × 类型的自信度   .max(1)在第1维度比较大小 并返回最大值 和最大值的索引 detect 中 [0.9206, 0.8981, 0.9290, 0.9899, 0.9927, 0.9873, 0.9927, 0.9389, 0.8004]

        # Sort by it
        image_pred = image_pred[(-score).argsort()] # .argsort() 默认返回 最后一个维度的升序排列的索引  整理过后 image_pred的排列为自上向下分数由高到低

        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True) # 获取类别分数最高的分值  和  类型  keepdim = True 输出为[[1],[2],[3]]  keepdim = False 输出为[1, 2, 3]
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1) # 去掉后面的80个类型分数，取而代之的是 类型分数最大值和 分类的结果 shape为[obj_num, 7] 7代表 x1, y1, x2, y2, obj_conf, cls_conf, cls
        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres # 总分数最高的bbox与所有bbox求iou 并 返回iou > 阈值的mask
            label_match = detections[0, -1] == detections[:, -1] # 返回类型与最高分bbox物体类型相同的mask

            # Indices of boxes with lower confidence socres, large IOUs and matching labels
            invalid = large_overlap & label_match # 返回类型与最高分bbox物体类型相同的mask 且 iou 大的 mask  这认为应该是同一个bbox
            weights = detections[invalid, 4:5] # 返回上面mask 对应的bbox的 obj_conf

            # Merge overlapping bboxes by order of confidence
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()  # 把重叠的同类的bbox的坐标 用 obj_conf作为权重 加权平均 得到 融合后的bbox
            keep_boxes += [detections[0]] # 非最大值抑制后  该保留的bbox
            detections = detections[~invalid] # 去掉上面融合过的bbox剩下的作为新的detections
        if keep_boxes: # 如果有bbox保留  如果是 if []:   这个会返回 False
            output[image_i] = torch.stack(keep_boxes) # 把list[tensor] 叠加起来为一个大tensor

    return output


def get_batch_statistics(outputs, targets, iou_threshold):
    """ compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
        pred_boxes = outputs[:, :4]
        pred_scores = outputs[:, 4]
        pred_labels = outputs[:, -1]

        ture_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[targets[:, 0] == sample_i][:, 1:] # 标签中，被正确预测出来的记为annotations

        target_labels = annotations[:, 0] if len(annotations) else [] # 正确预测出来的物体类型
        if len(annotations): # 如果有正确物体类型
            detected_boxes = []
            targets_boxes = annotations[:, 1:]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(annotations): # 如果annotations全部在被outputs中找出来了  结束循环
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels: # 如果 检测的类别   标签中没有，跳过
                    continue

                iou, box_index = bbox_iou(pred_box.unsqueeze(0), targets_boxes).max(0)
                if iou >= iou_threshold and box_index not in detected_boxes:
                    ture_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([ture_positives, pred_scores, pred_labels])
    return batch_metrics




def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):
    """

    :param pred_boxes: bbox 的 x, y, w, h (相对于 grid 的坐标) [1, 3, 13, 13, 4]
    :param pred_cls:  bbox的类别分值 [1, 3, 13, 13,80]
    :param target:  标签 二维  这个矩阵包含一个batch的所有labels   shape = [ all_batch_obj, 6]   行数代表一个batch中所有obj的数量，列分别为[图像序号, 物体类型, bbox_x, y, w, h]
    :param anchors:  anchor 尺寸 (相对于 grid 的坐标)
    :param ignore_thres:  阈值
    :return:
    """
    print('\n\nbuild_targets\n')
    ByteTensor = torch.cuda.ByteTensor if torch.cuda.is_available() else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    nB = pred_boxes.size(0) # batch_size
    nA = pred_boxes.size(1) # anchor数量
    nC = pred_cls.size(-1) # class 类别数量
    nG = pred_boxes.size(2) # 图像一列的grid数量

    # Output tensor
    obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0) # 创建shape为[1, 3, grid_size, grid_size]的mask 默认为没有物体
    noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1) # 同样默认为没有物体
    class_mask = FloatTensor(nB, nA, nG, nG).fill_(0) # 默认为类型0
    iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0) # 这个是为了储存后面计算出来的anchor的iou得分
    tx = FloatTensor(nB, nA, nG, nG).fill_(0) # bbox的 x, y, w, h, class
    ty = FloatTensor(nB, nA, nG, nG).fill_(0)
    tw = FloatTensor(nB, nA, nG, nG).fill_(0)
    th = FloatTensor(nB, nA, nG, nG).fill_(0)
    tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)
    print('obj_mask.shape', obj_mask.shape)

    # Convert to position relative to box

    target_boxes = target[:, 2:6] * nG # 标签bbox的 x, y, w, h 相对于gird_size大小
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]
    # Get anchors with best iou
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors]) # 括号里面的是一个list包含3个tensor torch.stack 把这3个tensor 整合为一个大的tensor 这样相当于使一个list 转为了tensor
    best_ious, best_n = ious.max(0) # 在tensor 的第0个维度 比较，并返回    最大值，最大值的索引  # best_n 表示最适合用于检测物体的anchor
    print('best_n.shape', best_n.shape)
    # Separate target values
    b, target_labels = target[:, :2].long().t() # b代表batch图像的顺序     target_labels 物体的类型        行数代表batch中物体的数量
    gx, gy = gxy.t() # 获取标签bbox 的x, y  这里有个小细节， gx, gy = gxy 这样操作是不对的，用了一个t() 转置一下就对了
    gw, gh = gwh.t()
    gi, gj = gxy.long().t() # 向下取整
    # Set masks
    obj_mask[b, best_n, gj, gi] = 1 # 设置 有物体，最佳anchor 的mask   注意：这里是有物体，而没有说有几个物体
    noobj_mask[b, best_n, gj, gi] = 0 # 设置没有物体的mask

    # Set noobj mask to zero where iou exceeds ignore threshold
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0 # 就算不是最佳的anchor 但是，只要 iou大于 阈值，那么这种anchor任然认为具有价值的 (比如在两个物体 有重合的情况 )

    # Coordinates
    tx[b, best_n, gj, gi] = gx - gx.floor() # floor() 向下取整 获取label的x的小数 GT平移系数
    ty[b, best_n, gj, gi] = gy - gy.floor()
    print('tx.shape',tx.shape)
    # width and height
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)  # 宽 高 GT放缩系数    预测的W 而计算得到的w   W = exp(w)*scaled_anchor   所以这里用log
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)  # + 一个1e-16 是为了防止log(0)的情况出现

    # One-hot encoding of label
    tcls[b, best_n, gj, gi, target_labels] = 1
    # Compute label correctness and iou at best anchor
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float() # 判断80个类型中最大的分数的类型 是否 和标签一致 一致的标记为1
    iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)
    print('class_mask.shape', class_mask.shape)
    print('iou_scores', iou_scores.shape)


    tconf = obj_mask.float()
    # 预测bbox与label bbox的iou[b, best_anchor, x, y]  ， 类型预测正确的mask[]与物体数相同 正确预测物体为True
    # 有物体的mask[b,best,gj,gi], 没有物体的mask[b,best,gj,gi]，
    # tx[b, best_n, gj, gi] ty[b, best_n, gj, gi]相对于gird的GT偏移
    # tw[b, best_n, gj, gi] th[b, best_n, gj, gi] 相对于anchor的GT放缩
    # tcls[b, best_n, gj, gi, target_labels] GT类型  tconf[b, best_n, gj, gi] GT自信度 confidence
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf


