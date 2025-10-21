import glob
import os

import cv2
import random
import numpy as np
from PIL import Image


import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_image, ImageReadMode

from misc.general import DeviceChecker
from misc.general import image2label

DEVICE = DeviceChecker.DEVICE

# init_size = 400
# model_input_size = 400

def rgb2yuv(rgb_tensor):
    r, g, b = torch.chunk(rgb_tensor, 3)
    # y = (0.2989 * r + 0.5870 * g + 0.1140 * b) / 255
    # u = (-0.168736 * r - 0.331264 * g + 0.500000 * b) / 255 + 0.5
    # v = (0.500000 * r - 0.418688 * g - 0.081312 * b) / 255 + 0.5
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    u = 0.5389 * (b - y) + 128
    v = 0.6350 * (r - y) + 128
    yuv_tensor = torch.cat([y, u, v])
    return yuv_tensor[0:1]

class ZeroOneNormalize(object):
    def __call__(self, img):
        return img.float().div(255)


class TrainLoader(Dataset):

    def __init__(self, file_list_A, file_list_B, file_list_DM, file_list_GT, file_list_CL, init_size, model_input_size):
        self.file_list_A = file_list_A
        self.file_list_B = file_list_B
        self.file_list_DM = file_list_DM
        self.file_list_GT = file_list_GT
        self.file_list_CL = file_list_CL
        self.img_tsf = transforms.Compose(
            [
                transforms.Resize(init_size, antialias=False),
                transforms.RandomCrop(model_input_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                # transforms.ColorJitter(brightness=(.7, 1.3), contrast=(.7, 1.3), saturation=(.8, 1.2), hue=(-0.5, 0.5)),
                # transforms.ColorJitter(brightness=.3, contrast=.3),
                ZeroOneNormalize(),
                # transforms.Normalize([0.4948052, 0.48568845, 0.44682974], [0.24580306, 0.24236229, 0.2603115]),
            ]
        )
        self.dm_tsf = transforms.Compose(
            [
                transforms.Resize(init_size, antialias=False),
                transforms.RandomCrop(model_input_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                ZeroOneNormalize(),
            ]
        )
        self.cls_tsf = transforms.Compose(
            [
                transforms.Resize(init_size, antialias=False),
                transforms.RandomCrop(model_input_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]
        )

    def __len__(self):
        return len(self.file_list_A)

    def __getitem__(self, idx):


        seed = torch.random.seed()

        imgA_path = self.file_list_A[idx]
        imgA = read_image(imgA_path, mode=ImageReadMode.RGB).to(DEVICE)
        # imgA = rgb2yuv(imgA)
        torch.random.manual_seed(seed)
        imgA = self.img_tsf(imgA)

        imgB_path = self.file_list_B[idx]
        imgB = read_image(imgB_path, mode=ImageReadMode.RGB).to(DEVICE)
        # imgB = rgb2yuv(imgB)
        torch.random.manual_seed(seed)
        imgB = self.img_tsf(imgB)

        imgDM_path = self.file_list_DM[idx]
        imgDM = read_image(imgDM_path, mode=ImageReadMode.GRAY).to(DEVICE)
        torch.random.manual_seed(seed)
        imgDM = self.dm_tsf(imgDM)

        imgGT_path = self.file_list_GT[idx]
        imgGT = read_image(imgGT_path, mode=ImageReadMode.RGB).to(DEVICE)
        # imgGT = rgb2yuv(imgGT)
        torch.random.manual_seed(seed)
        imgGT = self.img_tsf(imgGT)

        imgCL_path = self.file_list_CL[idx]
        imgCL = read_image(imgCL_path, mode=ImageReadMode.RGB).to(DEVICE)
        torch.random.manual_seed(seed)
        imgCL = self.cls_tsf(imgCL)
        imgCL = image2label(imgCL)

        return imgA, imgB, imgDM, imgGT, imgCL


class TestLoader(Dataset):
    test_trans = transforms.Compose(
        [
            # transforms.Resize((512, 512), antialias=False),
            # transforms.CenterCrop(800),
            ZeroOneNormalize(),
            # transforms.Normalize([0.4948052, 0.48568845, 0.44682974], [0.24580306, 0.24236229, 0.2603115]),
            # transforms.Normalize([0.5], [0.5]),
        ]
    )

    tt = transforms.Compose(
        [
            # transforms.Resize((384, 384), antialias=False),
            transforms.ToTensor(),
            # transforms.Normalize([0.4948052, 0.48568845, 0.44682974], [0.24580306, 0.24236229, 0.2603115]),
            # transforms.Normalize([0.5], [0.5]),
        ]
    )

    def __init__(self, file_list_A, file_list_B):
        self.file_list_A = file_list_A
        self.file_list_B = file_list_B

    def __len__(self):
        return len(self.file_list_A)

    def __getitem__(self, idx):
        seed = torch.random.seed()

        # imgA_path = self.file_list_A[idx]
        # imgA = read_image(imgA_path, mode=ImageReadMode.RGB).to(DEVICE)
        # # imgA = rgb2yuv(imgA)
        # torch.random.manual_seed(seed)
        # imgA = self.test_trans(imgA)
        #
        # imgB_path = self.file_list_B[idx]
        # imgB = read_image(imgB_path, mode=ImageReadMode.RGB).to(DEVICE)
        # # imgB = rgb2yuv(imgB)
        # torch.random.manual_seed(seed)
        # imgB = self.test_trans(imgB)

        imgA_path = self.file_list_A[idx]
        imgA = Image.open(imgA_path).convert('RGB')
        imgA = self.tt(imgA).to(DEVICE)
        # imgA = rgb2yuv(imgA)
        torch.random.manual_seed(seed)

        imgB_path = self.file_list_B[idx]
        imgB = Image.open(imgB_path).convert('RGB')
        imgB = self.tt(imgB).to(DEVICE)
        # imgB = rgb2yuv(imgB)
        torch.random.manual_seed(seed)

        return imgA, imgB


def get_trainloader(datapath, batch_size, setsize, init_size, model_input_size):
    train_list_A = []
    train_list_B = []
    train_list_DM = []
    train_list_GT = []
    train_list_CL = []

    for i in datapath:
        train_list_A = train_list_A + sorted(glob.glob(os.path.join(i, 'train/sourceA', '*.*')))[:setsize]
        train_list_B = train_list_B + sorted(glob.glob(os.path.join(i, 'train/sourceB', '*.*')))[:setsize]
        train_list_DM = train_list_DM + sorted(glob.glob(os.path.join(i, 'train/decisionmap', '*.*')))[:setsize]
        train_list_GT = train_list_GT + sorted(glob.glob(os.path.join(i, 'train/groundtruth', '*.*')))[:setsize]
        train_list_CL = train_list_CL + sorted(glob.glob(os.path.join(i, 'train/class', '*.*')))[:setsize]

    # torch.cuda.synchronize()
    train_data = TrainLoader(train_list_A, train_list_B, train_list_DM, train_list_GT, train_list_CL, init_size, model_input_size)
    train_loader = DataLoader(dataset=train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0,
                              pin_memory=False)

    # print(f"Train Data Size:{len(train_data)} , Train Loader Amount: {len(train_data)}//{batchsize} = {len(train_loader)}")
    return train_loader, len(train_data)


def get_testloader(dataset_path, batch_size, setsize):
    test_list_A = sorted(glob.glob(os.path.join(dataset_path, 'sourceA', '*.*')))[:setsize]
    test_list_B = sorted(glob.glob(os.path.join(dataset_path, 'sourceB', '*.*')))[:setsize]
    test_data = TestLoader(test_list_A, test_list_B)
    test_loader = DataLoader(dataset=test_data,
                             batch_size=batch_size,
                             shuffle=False)
    return test_loader, test_list_A, test_list_B