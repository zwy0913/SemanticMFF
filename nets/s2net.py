import time

import torch
from torch import nn
import torch.nn.functional as F

from nets.backbone import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5
from nets.ETB import ETBlock


class SegFormerEncoder(nn.Module):
    def __init__(self, phi='b0', pretrained=True):
        super(SegFormerEncoder, self).__init__()
        self.in_channels = {
            'b0': [32, 64, 160, 256], 'b1': [64, 128, 320, 512], 'b2': [64, 128, 320, 512],
            'b3': [64, 128, 320, 512], 'b4': [64, 128, 320, 512], 'b5': [64, 128, 320, 512],
        }[phi]
        self.backbone = {
            'b0': mit_b0, 'b1': mit_b1, 'b2': mit_b2,
            'b3': mit_b3, 'b4': mit_b4, 'b5': mit_b5,
        }[phi](pretrained)
        self.embedding_dim = {
            'b0': 128, 'b1': 256, 'b2': 768,
            'b3': 768, 'b4': 768, 'b5': 768,
        }[phi]

    def forward(self, inputs):
        # H, W = inputs.size(2), inputs.size(3)
        feats = self.backbone(inputs)
        return feats


class Upsample(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, padding=1),
            nn.SyncBatchNorm(out_dim),
            nn.ReLU(inplace=True),
            # ETBlock(in_dim, out_dim, 2)
        )

    def forward(self, x, up):
        x = F.interpolate(input=x, size=[up.shape[2], up.shape[3]], mode='bilinear', align_corners=False)
        z = self.conv(torch.cat([x, up], 1))
        return z


class OutConv(nn.Module):
    def __init__(self,
                 decoder_dim,
                 base_dim,
                 num_classes,
                 k=1,
                 s=1,
                 p=0):
        super().__init__()

        self.out_conv = nn.Sequential(
            nn.Conv2d(decoder_dim, num_classes, kernel_size=k, stride=s, padding=p),
            # nn.GELU(),
            # nn.Dropout(dropout),
            # nn.SyncBatchNorm(base_dim),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(base_dim, num_classes, kernel_size=3, stride=1, padding=1),
            # nn.Sigmoid()
            # nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        return self.out_conv(x)

# class SemanticNet(nn.Module):
#     def __init__(self, pretrained=True):
#         super(SemanticNet, self).__init__()
#         self.encoder = SegFormerEncoder(pretrained=pretrained)
#         self.up1 = Upsample(256 + 160, 160, 1)
#         self.up2 = Upsample(160 + 64, 64, 1)
#         self.up3 = Upsample(64 + 32, 32, 1)
#         self.up4 = Upsample(32 + 6, 21, 1)
#
#     def forward(self, x):
#         feats = self.encoder(x)
#         out = self.up1(feats[3], feats[2])
#         out = self.up2(out, feats[1])
#         out = self.up3(out, feats[0])
#         out = self.up4(out, x)
#         return out


class SpatialNet(nn.Module):
    def __init__(self, pretrained=True):
        super(SpatialNet, self).__init__()
        self.encoder = SegFormerEncoder(pretrained=pretrained)
        embed_dim = [32, 64, 160, 256]
        self.to_fused = nn.ModuleList([nn.Sequential(
            nn.Conv2d(dim, 32, 1),
        ) for dim in embed_dim])

        self.up = Upsample(len(embed_dim) * 32 + 6, 64)

        self.out_conv = OutConv(64, 32, 3, 3, 1, 1)

        self.act = nn.Sigmoid()

    def forward(self, input):
        feats = self.encoder(input)
        fused = []
        for x, to_fused in zip(feats, self.to_fused):
            temp = to_fused(x)
            temp = F.interpolate(temp, feats[0].size()[2:], mode='bilinear', align_corners=False)
            fused.append(temp)
        out = self.up(torch.cat(fused, dim=1), input)
        out = self.out_conv(out)
        return self.act(out), feats


class SemanticNet(nn.Module):
    def __init__(self, pretrained=True):
        super(SemanticNet, self).__init__()
        self.encoder = SegFormerEncoder(pretrained=pretrained)
        embed_dim = [32, 64, 160, 256]
        self.to_fused = nn.ModuleList([nn.Sequential(
            nn.Conv2d(dim, 32, 1),
        ) for dim in embed_dim])

        self.up = Upsample(len(embed_dim) * 32 + 6, 256)

        self.out_conv = OutConv(256, 32, 21, 1, 1, 0)

    def forward(self, input):
        feats = self.encoder(input)
        fused = []
        for x, to_fused in zip(feats, self.to_fused):
            temp = to_fused(x)
            temp = F.interpolate(temp, feats[0].size()[2:], mode='bilinear', align_corners=False)
            fused.append(temp)
        out = self.up(torch.cat(fused, dim=1), input)
        out = self.out_conv(out)
        return out, feats
