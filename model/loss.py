import torch
import torch.nn.functional as F
import torch.nn as nn


def bce_iou_loss(pred, mask):
    weight = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')

    pred = torch.sigmoid(pred)
    inter = pred * mask
    union = pred + mask
    iou = 1 - (inter + 1) / (union - inter + 1)

    weighted_bce = (weight * bce).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))
    weighted_iou = (weight * iou).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))

    return (weighted_bce + weighted_iou).mean()


def dice_bce_loss(pred, mask):
    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')

    pred = torch.sigmoid(pred)
    inter = pred * mask
    union = pred + mask
    iou = 1 - (2. * inter + 1) / (union + 1)

    return (bce + iou).mean()


def tversky_loss(pred, mask, alpha=0.5, beta=0.5, gamma=2):
    pred = torch.sigmoid(pred)

    # flatten label and prediction tensors
    pred = pred.view(-1)
    mask = mask.view(-1)

    # True Positives, False Positives & False Negatives
    TP = (pred * mask).sum()
    FP = ((1 - mask) * pred).sum()
    FN = (mask * (1 - pred)).sum()

    Tversky = (TP + 1) / (TP + alpha * FP + beta * FN + 1)

    return (1 - Tversky) ** gamma


def tversky_bce_loss(pred, mask, alpha=0.5, beta=0.5, gamma=2):
    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')

    pred = torch.sigmoid(pred)

    # flatten label and prediction tensors
    pred = pred.view(-1)
    mask = mask.view(-1)

    # True Positives, False Positives & False Negatives
    TP = (pred * mask).sum()
    FP = ((1 - mask) * pred).sum()
    FN = (mask * (1 - pred)).sum()

    Tversky = (TP + 1) / (TP + alpha * FP + beta * FN + 1)

    return bce + (1 - Tversky) ** gamma


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DiceLoss(nn.Module):
    def __init__(self, weight=None):
        super(DiceLoss, self).__init__()
        if weight is not None:
            weight = torch.Tensor(weight)
            self.weight = weight / torch.sum(weight)  # Normalized weight
        self.smooth = 1e-5

    def forward(self, predict, target):
        N, C = predict.size()[:2]
        predict = predict.view(N, C, -1)  # (N, C, *)
        target = target.view(N, 1, -1)  # (N, 1, *)

        predict = F.softmax(predict, dim=1)  # (N, C, *) ==> (N, C, *)
        ## convert target(N, 1, *) into one hot vector (N, C, *)
        target_onehot = torch.zeros(predict.size()).cuda()  # (N, 1, *) ==> (N, C, *)
        target_onehot.scatter_(1, target, 1)  # (N, C, *)

        intersection = torch.sum(predict * target_onehot, dim=2)  # (N, C)
        union = torch.sum(predict.pow(2), dim=2) + torch.sum(target_onehot, dim=2)  # (N, C)
        ## p^2 + t^2 >= 2*p*t, target_onehot^2 == target_onehot
        dice_coef = (2 * intersection + self.smooth) / (union + self.smooth)  # (N, C)

        if hasattr(self, 'weight'):
            if self.weight.type() != predict.type():
                self.weight = self.weight.type_as(predict)
                dice_coef = dice_coef * self.weight * C  # (N, C)
        dice_loss = 1 - torch.mean(dice_coef)  # 1

        return dice_loss


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask) #计算池化后的掩膜和原始掩膜之间的绝对差异，使得较大的差异区域在损失计算中占有更大的权重
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none') # 返回的是每个像素的损失
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3)) # 使用 weit 权重来加权交叉熵损失 wbce 图像的高度和宽度求和

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3)) # 计算加权的交集
    union = ((pred + mask) * weit).sum(dim=(2, 3)) # 计算加权的并集
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def irstd_loss(pred, mask, max_pos_weight=30.0):
    """Class-balanced BCE plus soft IoU for sparse infrared targets."""
    spatial_dims = tuple(range(2, mask.ndim))
    positive = mask.sum(dim=spatial_dims, keepdim=True)
    negative = mask[0, 0].numel() - positive
    pos_weight = (negative / positive.clamp_min(1.0)).clamp(1.0, max_pos_weight)
    pixel_weight = torch.where(mask > 0.5, pos_weight, torch.ones_like(mask))
    bce = F.binary_cross_entropy_with_logits(pred, mask, weight=pixel_weight, reduction="sum")
    bce = bce / pixel_weight.sum().clamp_min(1.0)

    probability = pred.sigmoid()
    intersection = (probability * mask).sum(dim=spatial_dims)
    union = (probability + mask - probability * mask).sum(dim=spatial_dims)
    soft_iou = 1 - ((intersection + 1.0) / (union + 1.0)).mean()
    return bce + soft_iou


def irstd_v2_loss(
    pred,
    mask,
    max_pos_weight=100.0,
    alpha=0.3,
    beta=0.7,
    gamma=1.5,
    iou_weight=0.5,
):
    """Per-image balanced BCE, focal Tversky and soft IoU for tiny targets."""
    spatial_dims = tuple(range(2, mask.ndim))
    positive = mask.sum(dim=spatial_dims, keepdim=True)
    pixels_per_image = mask[0, 0].numel()
    negative = pixels_per_image - positive
    positive_weight = (negative / positive.clamp_min(1.0)).clamp(1.0, max_pos_weight)
    pixel_weight = torch.where(mask > 0.5, positive_weight, torch.ones_like(mask))
    bce = F.binary_cross_entropy_with_logits(pred, mask, weight=pixel_weight, reduction="none")
    bce = (bce.sum(dim=spatial_dims) / pixel_weight.sum(dim=spatial_dims).clamp_min(1.0)).mean()

    probability = pred.sigmoid()
    true_positive = (probability * mask).sum(dim=spatial_dims)
    false_positive = (probability * (1 - mask)).sum(dim=spatial_dims)
    false_negative = ((1 - probability) * mask).sum(dim=spatial_dims)
    tversky = (true_positive + 1.0) / (
        true_positive + alpha * false_positive + beta * false_negative + 1.0
    )
    focal_tversky = ((1 - tversky).clamp_min(0.0) ** gamma).mean()

    union = (probability + mask - probability * mask).sum(dim=spatial_dims)
    soft_iou = 1 - ((true_positive + 1.0) / (union + 1.0)).mean()
    return bce + focal_tversky + iou_weight * soft_iou


def irstd_v3_loss(
    pred,
    mask,
    max_pos_weight=100.0,
    alpha=0.2,
    beta=0.8,
    gamma=1.5,
    tversky_weight=1.5,
    iou_weight=0.5,
):
    """Recall-oriented segmentation loss for extremely small UAV targets."""
    spatial_dims = tuple(range(2, mask.ndim))
    positive = mask.sum(dim=spatial_dims, keepdim=True)
    pixels_per_image = mask[0, 0].numel()
    negative = pixels_per_image - positive
    positive_weight = (negative / positive.clamp_min(1.0)).clamp(1.0, max_pos_weight)
    pixel_weight = torch.where(mask > 0.5, positive_weight, torch.ones_like(mask))
    bce = F.binary_cross_entropy_with_logits(pred, mask, weight=pixel_weight, reduction="none")
    bce = (bce.sum(dim=spatial_dims) / pixel_weight.sum(dim=spatial_dims).clamp_min(1.0)).mean()

    probability = pred.sigmoid()
    true_positive = (probability * mask).sum(dim=spatial_dims)
    false_positive = (probability * (1 - mask)).sum(dim=spatial_dims)
    false_negative = ((1 - probability) * mask).sum(dim=spatial_dims)
    tversky = (true_positive + 1.0) / (
        true_positive + alpha * false_positive + beta * false_negative + 1.0
    )
    focal_tversky = ((1 - tversky).clamp_min(0.0) ** gamma).mean()

    union = (probability + mask - probability * mask).sum(dim=spatial_dims)
    soft_iou = 1 - ((true_positive + 1.0) / (union + 1.0)).mean()
    return bce + tversky_weight * focal_tversky + iou_weight * soft_iou


def irstd_targetness_loss(pred, heatmap, gamma=2.0):
    """Focal heatmap loss that suppresses easy background pixels."""
    probability = pred.sigmoid()
    cross_entropy = F.binary_cross_entropy_with_logits(pred, heatmap, reduction="none")
    probability_target = heatmap * probability + (1 - heatmap) * (1 - probability)
    focal_weight = (1 - probability_target).clamp_min(0.0) ** gamma
    target_weight = 1 + 20 * heatmap
    return (cross_entropy * focal_weight * target_weight).sum() / target_weight.sum().clamp_min(1.0)


def cal_ual(seg_logits, seg_gts):
    assert seg_logits.shape == seg_gts.shape, (seg_logits.shape, seg_gts.shape)
    sigmoid_x = seg_logits.sigmoid()
    loss_map = 1 - (2 * sigmoid_x - 1).abs().pow(2)
    return loss_map.mean()


def structure_loss_with_ual(pred, mask):
    return structure_loss(pred, mask) + 0.5 * cal_ual(pred, mask)


class Bce_iou_loss(nn.Module):

    def __init__(self):
        super(Bce_iou_loss, self).__init__()

    def forward(self, pred, mask):
        weight = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

        bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')

        pred = torch.sigmoid(pred)
        inter = pred * mask
        union = pred + mask
        iou = 1 - (inter + 1) / (union - inter + 1)

        weighted_bce = (weight * bce).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))
        weighted_iou = (weight * iou).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))

        return (weighted_bce + weighted_iou).mean()
