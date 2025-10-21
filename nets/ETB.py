import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.transforms.v2.functional as tv2f

class SeqConv3x3(nn.Module):
    # Reference to ECBSR: https://github.com/xindongzhang/ECBSR
    def __init__(self,
                 seq_type,
                 in_planes,
                 out_planes,
                 dim_multiplier=1):
        super().__init__()

        self.type = seq_type
        self.in_planes = in_planes
        self.out_planes = out_planes

        if self.type == 'conv1x1-conv3x3':
            self.mid_planes = int(out_planes * dim_multiplier)
            conv0 = torch.nn.Conv2d(self.in_planes, self.mid_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            conv1 = torch.nn.Conv2d(self.mid_planes, self.out_planes, kernel_size=3)
            self.k1 = conv1.weight
            self.b1 = conv1.bias

        elif self.type == 'conv1x1-sobelx':
            conv0 = torch.nn.Conv2d(self.in_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(scale)
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 1, 0] = 2.0
                self.mask[i, 0, 2, 0] = 1.0
                self.mask[i, 0, 0, 2] = -1.0
                self.mask[i, 0, 1, 2] = -2.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)
            self.conv1 = nn.Conv2d(self.out_planes, self.out_planes, kernel_size=3, stride=1, padding=1,
                                   groups=self.out_planes)
            self.conv1.weight = self.mask
            self.b1 = self.conv1.bias

        elif self.type == 'conv1x1-sobely':
            conv0 = torch.nn.Conv2d(self.in_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(scale)
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 0, 1] = 2.0
                self.mask[i, 0, 0, 2] = 1.0
                self.mask[i, 0, 2, 0] = -1.0
                self.mask[i, 0, 2, 1] = -2.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)
            self.conv1 = nn.Conv2d(self.out_planes, self.out_planes, kernel_size=3, stride=1, padding=1,
                                   groups=self.out_planes)
            self.conv1.weight = self.mask
            self.b1 = self.conv1.bias

        elif self.type == 'conv1x1-laplacian':
            conv0 = torch.nn.Conv2d(self.in_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(scale)
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 1] = 1.0
                self.mask[i, 0, 1, 0] = 1.0
                self.mask[i, 0, 1, 2] = 1.0
                self.mask[i, 0, 2, 1] = 1.0
                self.mask[i, 0, 1, 1] = -4.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)
            self.conv1 = nn.Conv2d(self.out_planes, self.out_planes, kernel_size=3, stride=1, padding=1,
                                   groups=self.out_planes)
            self.conv1.weight = self.mask
            self.b1 = self.conv1.bias
        else:
            raise ValueError('the type of seqconv is not supported!')

    def forward(self, x):
        if self.type == 'conv1x1-conv3x3':
            # conv-1x1
            y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)
            # explicitly padding with bias
            y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
            b0_pad = self.b0.view(1, -1, 1, 1)
            y0[:, :, 0:1, :] = b0_pad
            y0[:, :, -1:, :] = b0_pad
            y0[:, :, :, 0:1] = b0_pad
            y0[:, :, :, -1:] = b0_pad
            # conv-3x3
            y1 = F.conv2d(input=y0, weight=self.k1, bias=self.b1, stride=1)
        else:
            # conv-1x1
            y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)
            # explicitly padding with bias
            y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
            b0_pad = self.b0.view(1, -1, 1, 1)
            y0[:, :, 0:1, :] = b0_pad
            y0[:, :, -1:, :] = b0_pad
            y0[:, :, :, 0:1] = b0_pad
            y0[:, :, :, -1:] = b0_pad
            # conv-3x3
            y1 = F.conv2d(input=y0, weight=self.mask, bias=self.b1, stride=1, groups=self.out_planes)
        return y1


class ETBlock(nn.Module):
    def __init__(self,
                 in_planes,
                 out_planes,
                 dim_multiplier=1,
                 with_idt=True):
        super().__init__()

        self.dim_multiplier = dim_multiplier
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.act = nn.ReLU(inplace=True)

        if with_idt and (self.in_planes == self.out_planes):
            self.with_idt = True
        else:
            self.with_idt = False

        self.conv3x3 = torch.nn.Conv2d(self.in_planes, self.out_planes, kernel_size=3, stride=1, padding=1)
        # self.lkc1 = torch.nn.Conv2d(self.in_planes, self.out_planes, kernel_size=9, stride=1, padding=4,
        #                             groups=self.out_planes)
        # self.lkc2 = torch.nn.Conv2d(self.in_planes, self.out_planes, kernel_size=11, stride=1, padding=5,
        #                             groups=self.out_planes)
        # self.lkc3 = torch.nn.Conv2d(self.in_planes, self.out_planes, kernel_size=13, stride=1, padding=6,
        #                             groups=self.out_planes)
        self.conv1x1_3x3 = SeqConv3x3('conv1x1-conv3x3', self.in_planes, self.out_planes, self.dim_multiplier)
        self.conv1x1_sbx = SeqConv3x3('conv1x1-sobelx', self.in_planes, self.out_planes)
        self.conv1x1_sby = SeqConv3x3('conv1x1-sobely', self.in_planes, self.out_planes)
        self.conv1x1_lpl = SeqConv3x3('conv1x1-laplacian', self.in_planes, self.out_planes)
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        y = (self.conv3x3(x) + self.conv1x1_3x3(x) + self.conv1x1_sbx(x) + self.conv1x1_sby(x) + self.conv1x1_lpl(x))
        if self.with_idt:
            y += x
        # y = self.bn(y)
        y = self.act(y)
        return y