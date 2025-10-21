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


VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]


class DeviceChecker:
    if torch.cuda.is_available():
        DEVICE = "cuda"
        print('CUDA is available. (' + str(torch.cuda.get_device_name(torch.cuda.current_device())) + ')')
    else:
        DEVICE = 'cpu'
        print('CUDA is not available! Calculating on CPU.')


class LoggingSaveWeights:
    def __init__(self, save_path, config, save_every_weights=True, early_stopping=-1):
        self.save_every_weights = save_every_weights
        self.start_training_time = datetime.datetime.now().strftime('%Y-%m-%d %H.%M.%S')
        os.mkdir(os.path.join(save_path, self.start_training_time))
        self.ChildDir = os.path.join(save_path, self.start_training_time)
        # 根据时间创建日志
        self.tlf = open(self.ChildDir + '\\' + self.start_training_time + '.txt', 'w')
        # self.tlf.write('Configurations:\n')
        # self.tlf.write(yaml.dump(config, default_flow_style=None, sort_keys=False))
        self.tlf.write(config)
        self.tlf.write('\n\nTraining log:\n')
        # 复制文件备份
        # shutil.copy(os.path.abspath(config_file), os.path.join(os.getcwd(), self.ChildDir, self.start_training_time + '.yaml'))
        # shutil.copy(os.path.join(os.getcwd(), 'nets', 'former_blocks.py'), os.path.join(os.getcwd(), self.ChildDir, 'former_blocks.py'))
        # shutil.copy(os.path.join(os.getcwd(), 'nets', 's2net.py'), os.path.join(os.getcwd(), self.ChildDir, 's2net.py'))
        # 显示保存位置
        print("\nTrained weights will be saved to: " + os.path.join(os.getcwd(), self.ChildDir) + "\n")
        # 早停
        self.early_stopping = early_stopping
        if self.early_stopping > 0:
            self.STOP = False
            self.patience = early_stopping
            self.delta = 1e-3
            self.counter = 0
            self.best_score = None
            self.loss_min = np.Inf

    def __call__(self, model, loss, current_epoch, log_contents):
        torch.save(model.state_dict(), os.path.join(os.getcwd(), self.ChildDir) + '/model_best.ckpt')
        if self.save_every_weights:
            # Save every model weight
            torch.save(model.state_dict(), os.path.join(os.getcwd(), self.ChildDir) + '/model' + str(current_epoch) + '.ckpt')
        # Writing logs
        self.tlf.write(log_contents)
        # For debugging
        torch.save(model.state_dict(), 'debug_model.ckpt')

        # 使用早停
        if self.early_stopping > 0:
            score = -loss
            if self.best_score is None:
                print('')
                self.best_score = score
                tqdm.write(f'Validation loss decreased ({self.loss_min:.6f} --> {loss:.6f}).  Saving models ...')
                torch.save(model.state_dict(), os.path.join(os.getcwd(), self.ChildDir) + '/best_model.ckpt')
                self.loss_min = loss
            elif score < self.best_score + self.delta:
                self.counter += 1
                print(f'\033[0;33mEarlyStopping counter: {self.counter} out of {self.patience}\033[0m')
                if self.counter >= self.patience:
                    self.STOP = True
            else:
                print('')
                self.best_score = score
                self.counter = 0
                tqdm.write(f'Validation loss decreased ({self.loss_min:.6f} --> {loss:.6f}).  Saving models ...')
                torch.save(model.state_dict(), os.path.join(os.getcwd(), self.ChildDir) + '/best_model.ckpt')
                self.loss_min = loss

    def save_model(self, model, loss, current_epoch, prefix='model'):
        torch.save(model.state_dict(), os.path.join(os.getcwd(), self.ChildDir) + '/' + str(prefix) + '_best.ckpt')
        if self.save_every_weights:
            # Save every model weight
            torch.save(model.state_dict(), os.path.join(os.getcwd(), self.ChildDir) + '/' + str(prefix) + str(current_epoch) + '.ckpt')
        # For debugging
        torch.save(model.state_dict(), 'debug_' + str(prefix) + '.ckpt')

        # 使用早停
        if self.early_stopping > 0:
            score = -loss
            if self.best_score is None:
                print('')
                self.best_score = score
                tqdm.write(f'Validation loss decreased ({self.loss_min:.6f} --> {loss:.6f}).  Saving models ...')
                torch.save(model.state_dict(), os.path.join(os.getcwd(), self.ChildDir) + '/' + str(prefix) + '_best.ckpt')
                self.loss_min = loss
            elif score < self.best_score + self.delta:
                self.counter += 1
                print(f'\033[0;33mEarlyStopping counter: {self.counter} out of {self.patience}\033[0m')
                if self.counter >= self.patience:
                    self.STOP = True
            else:
                print('')
                self.best_score = score
                self.counter = 0
                tqdm.write(f'Validation loss decreased ({self.loss_min:.6f} --> {loss:.6f}).  Saving models ...')
                torch.save(model.state_dict(), os.path.join(os.getcwd(), self.ChildDir) + '/' + str(prefix) + '_best.ckpt')
                self.loss_min = loss

    def log_content(self, log_contents):
        # Writing logs
        self.tlf.write(log_contents)

def get_one_hot(label, N):
    size = list(label.size())
    # size.insert(1, N)
    label = label.view(-1)  # reshape 为向量
    ones = torch.sparse.torch.eye(N).to(label.device)
    ones = ones.index_select(0, label)  # 用上面的办法转为换one hot
    size.append(N)  # 把类别输入添到size的尾后，准备reshape回原来的尺寸
    out = ones.view(*size)
    return out.permute(0, 3, 1, 2)

def image2label(image):
    # 将一个标记好的图像转化为类别标签图像
    # 将标签转化为每个像素值为一类数据
    colormap2label = torch.zeros(256 ** 3, dtype=torch.uint8).to(image.device)  # torch.Size([16777216])
    for i, colormap in enumerate(VOC_COLORMAP):
        # 每个通道的进制是256，这样可以保证每个 rgb 对应一个下标 i
        colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    image = image.type(torch.int32).permute(1, 2, 0)
    idx = ((image[:, :, 0] * 256 + image[:, :, 1]) * 256 + image[:, :, 2])
    return colormap2label[idx]  # colormap 映射 到colormaplabel中计算的下标

def label2image(pred):
    # 从预测的标签转化为图像的操作
    # pred: [320,480]
    colormap = torch.tensor(VOC_COLORMAP, device=pred.device, dtype=torch.uint8)
    x = pred[0].long()
    return (colormap[x,:]).data.cpu().numpy()

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def get_net_para(net):
    num_params = 0
    for p in net.parameters():
        num_params += p.numel()
    return num_params

def fusion_channel_sf(imgA, imgB, kernel_radius=5):
    """
    Perform channel sf fusion two features
    """
    device = imgA.device
    b, c, h, w = imgA.shape
    r_shift_kernel = torch.FloatTensor([[0, 0, 0],
                                        [1, 0, 0],
                                        [0, 0, 0]]).to(device).reshape((1, 1, 3, 3)).repeat(c, 1, 1, 1)
    b_shift_kernel = torch.FloatTensor([[0, 1, 0],
                                        [0, 0, 0],
                                        [0, 0, 0]]).to(device).reshape((1, 1, 3, 3)).repeat(c, 1, 1, 1)
    f1_r_shift = F.conv2d(imgA, r_shift_kernel, padding=1, groups=c)
    f1_b_shift = F.conv2d(imgA, b_shift_kernel, padding=1, groups=c)
    f2_r_shift = F.conv2d(imgB, r_shift_kernel, padding=1, groups=c)
    f2_b_shift = F.conv2d(imgB, b_shift_kernel, padding=1, groups=c)

    f1_grad = torch.pow((f1_r_shift - imgA), 2) + torch.pow((f1_b_shift - imgA), 2)
    f2_grad = torch.pow((f2_r_shift - imgB), 2) + torch.pow((f2_b_shift - imgB), 2)

    kernel_size = kernel_radius * 2 + 1
    add_kernel = torch.ones((c, 1, kernel_size, kernel_size)).float().to(device)
    kernel_padding = kernel_size // 2
    f1_sf = torch.sum(F.conv2d(f1_grad, add_kernel, padding=kernel_padding, groups=c), dim=1, keepdim=True)
    # save_image(f.conv2d(f1_grad, add_kernel, padding=kernel_padding, groups=c), '../11.png')
    f2_sf = torch.sum(F.conv2d(f2_grad, add_kernel, padding=kernel_padding, groups=c), dim=1, keepdim=True)
    # weight_zeros = torch.zeros(f1_sf.shape).to(device)
    # weight_ones = torch.ones(f1_sf.shape).to(device)

    # get decision map
    dm_tensor = torch.where(f1_sf > f2_sf, 1., 0.).to(device)
    # dm_np = dm_tensor.squeeze().cpu().numpy().astype(int)

    return dm_tensor

def tensor_erode(bin_img, ksize):  # 已测试
    eroded = 1 - tensor_dilate(1 - bin_img, ksize)
    return eroded

def tensor_dilate(bin_img, ksize): #
    # 首先为原图加入 padding，防止图像尺寸缩小
    pad = ksize // 2
    # bin_img = F.pad(bin_img, pad=[pad, pad, pad, pad], mode='reflect')
    dilate = F.max_pool2d(bin_img, kernel_size=ksize, stride=1, padding=pad)
    return dilate

def perturb_seg(s_fm_to_refine, intensity=35):
    # save_image(s_fm_to_refine, '1.png')
    s_fm_to_refine = torch.where(s_fm_to_refine > 0.5, 1., 0.)
    b, c, h, w = s_fm_to_refine.shape
    for ib in range(b):
        for _ in range(0, intensity):
            lx, ly = random.randint(0, w - 2), random.randint(0, h - 2)
            lw, lh = random.randint(lx + 1, w), random.randint(ly + 1, h)
            cx = int((lx + lw) / 2)
            cy = int((ly + lh) / 2)
            if random.random() < 1:
                s_fm_to_refine[ib, :, cy, cx] = 1 - s_fm_to_refine[ib, :, cy, cx]
            if random.random() < 0.75:
                s_fm_to_refine[ib, 0, ly:lh, lx:lw] = tensor_dilate(
                    s_fm_to_refine[ib:ib + 1, 0:, ly:lh, lx:lw], ksize=random.choice([5, 7, 9]))
            else:
                s_fm_to_refine[ib, 0, ly:lh, lx:lw] = tensor_erode(
                    s_fm_to_refine[ib:ib + 1, 0:, ly:lh, lx:lw], ksize=random.choice([5, 7, 9]))
    # save_image(s_fm_to_refine, '2.png')

    return s_fm_to_refine