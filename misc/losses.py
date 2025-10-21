import random

import torch
from torch import nn
import numpy as np
import os
import datetime

from torchvision.utils import save_image
from tqdm import tqdm
import torch.nn.functional as F
import yaml
import shutil
from misc.SSIM_Torch import SSIM
from misc.sobel_op import SobelComputer
from misc.general import label2image

def l1_loss(pred, target):
    return F.l1_loss(pred, target)

def l2_loss(pred, target):
    return F.mse_loss(pred, target)

def ce_loss(pred, target):
    return F.cross_entropy(pred, target.long())

def bce_loss(pred, target):
    return F.binary_cross_entropy_with_logits(pred, (target > 0.5).float())

def ssim_loss(pred, target):
    ssim = SSIM()
    return 1 - ssim(pred, target).item()

def grad_loss(pred, target):
    sobel = SobelComputer()
    return F.l1_loss(sobel.compute_edges(pred), sobel.compute_edges(target))

def dice_loss(pred, target, smooth=1):
    pred = F.softmax(pred, dim=1)
    target_one_hot = torch.zeros_like(pred)
    target_one_hot.scatter_(1, target.long().unsqueeze(1), 1)
    batch_size = pred.size(0)
    m1 = pred.contiguous().view(batch_size, -1)
    m2 = target_one_hot.contiguous().view(batch_size, -1)
    intersection = (m1 * m2).sum(dim=1)
    union = m1.sum(dim=1) + m2.sum(dim=1)
    score = 2. * (intersection + smooth) / (union + smooth)
    score = 1 - score.sum() / batch_size
    return score

def focal_loss(pred, target, gamma=2.0, alpha=0.25):
    ce_loss = F.cross_entropy(pred, target, reduction='none')
    pt = torch.exp(-ce_loss)  # p_t = softmax(output)[target_class]
    score = alpha * (1 - pt) ** gamma * ce_loss
    return score.mean()

def align_loss(pred, target, smooth=1):
    pred = F.softmax(pred, dim=1)
    target_one_hot = torch.zeros_like(pred)
    target_one_hot.scatter_(1, target.long().unsqueeze(1), 1)
    batch_size = pred.size(0)
    m1 = pred.contiguous().view(batch_size, -1)
    m2 = target_one_hot.contiguous().view(batch_size, -1)
    intersection = (m1 * m2).sum(dim=1)
    union = m1.sum(dim=1) + m2.sum(dim=1)
    score = 2. * (intersection + smooth) / (union + smooth)
    score = 1 - score.sum() / batch_size
    return score

