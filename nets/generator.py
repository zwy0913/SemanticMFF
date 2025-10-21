import time

import torch
from einops import rearrange
from torch import nn, einsum
import torch.nn.functional as F


class LayerNorm(nn.Module):
    # layernorm, but done in the channel dimension #1
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class ConvEmbed(nn.Module):
    """
    Patch Embedding that is implemented by a layer of conv.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """
    def __init__(self,
                 dim_in,
                 dim_embed,
                 patch_size=3,
                 stride=1):
        super().__init__()
        self.proj = nn.Conv2d(dim_in, dim_embed, kernel_size=patch_size,
                              stride=stride,
                              padding=(patch_size - 1) // 2)
        # self.norm = nn.BatchNorm2d(dim_embed)
        self.norm = LayerNorm(dim_embed)

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class ConvFFN(nn.Module):
    """
        Implementation of MLP with 1*1 convolutions.
        Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mult, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.layers(x)


class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride, bias=True):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim_in, stride=stride, bias=bias),
            LayerNorm(dim_in),
            # linear
            nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=bias)
        )

    def forward(self, x):
        x = self.layers(x)
        # s = x.clone().detach().cpu().numpy()
        return x


class S2Attention(nn.Module):
    def __init__(self, embed_dim, heads, reduction_ratio, qkv_bias=False, dropout=0.):
        super().__init__()
        self.heads = heads
        dim_per_head = embed_dim // heads
        self.scale = dim_per_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(dropout)
        self.out_drop = nn.Dropout(dropout)

        self.to_q = nn.Conv2d(embed_dim, embed_dim, 1, bias=qkv_bias)
        self.to_kv = DepthWiseConv2d(embed_dim, embed_dim, reduction_ratio, stride=reduction_ratio, bias=qkv_bias)
        # self.to_kv = nn.Conv2d(dim, dim * 2, reduction_ratio, stride=reduction_ratio, bias=qkv_bias)
        self.to_out = nn.Conv2d(embed_dim, embed_dim, 1, bias=False)

    def forward(self, sem, spa, x):
        h, w = x.shape[-2:]
        heads = self.heads

        q, k, v = self.to_q(sem), self.to_kv(spa), self.to_kv(x)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h=heads), (q, k, v))

        dots = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = self.attend(dots)
        attn = self.attn_drop(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) (x y) c -> b (h c) x y', h=heads, x=h, y=w)
        out = self.to_out(out)
        out = self.out_drop(out)
        return out


class S2Former(nn.Module):
    def __init__(self,
                 embed_dim,
                 heads,
                 reduction_ratio,
                 mlp_mult,
                 depth,
                 dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # PreNorm(embed_dim, S2Attention(dim=embed_dim, heads=heads, reduction_ratio=reduction_ratio)),
                # PreNorm(embed_dim, ConvFFN(embed_dim, mlp_mult, dropout=dropout)),
                LayerNorm(embed_dim),
                S2Attention(embed_dim=embed_dim, heads=heads, reduction_ratio=reduction_ratio),
                LayerNorm(embed_dim),
                ConvFFN(embed_dim, mlp_mult, dropout=dropout)
            ]))

    def forward(self, sem, spa, x):
        for ln1, attn, ln2, ff in self.layers:
            x = attn(ln1(sem), ln1(spa), ln1(x)) + x
            x = ff(ln2(x)) + x
        return x


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
        x = F.interpolate(input=x, size=[up.shape[2], up.shape[3]], mode='bilinear', align_corners=True)
        z = self.conv(torch.cat([x, up], 1))
        return z


class InConv(nn.Module):
    def __init__(self,
                 in_channels,
                 base_dim):
        super().__init__()

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_dim, kernel_size=3, stride=2, padding=1),
            nn.SyncBatchNorm(base_dim),
            nn.ReLU(inplace=True),
            # nn.Conv2d(base_dim // 2, base_dim, kernel_size=3, stride=1, padding=1),
            # nn.SyncBatchNorm(base_dim),
            # nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.in_conv(x)


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


class Generator(nn.Module):
    def __init__(self,
                 input_dim=3,
                 num_classes=1,
                 base_dim=32,
                 embed_dim=[32, 48, 64, 64, 64, 64, 48, 32],
                 inject_dim=[32, 64, 160, 256],
                 embed_patch=[3, 3, 3, 3, 3, 3, 3, 3],
                 embed_stride=[2, 2, 2, 2, 1, 1, 1, 1],
                 heads=[1, 2, 2, 4, 4, 2, 2, 1],
                 reduction_ratio=[8, 4, 2, 1, 1, 2, 4, 8],
                 mlp_mult=[4, 4, 4, 4, 4, 4, 4, 4],
                 depth=[2, 2, 2, 2, 2, 2, 2, 2],
                 dropout=0.
                 ):
        super(Generator, self).__init__()

        self.in_conv = InConv(in_channels=input_dim, base_dim=base_dim)

        # self.inject_1 = nn.Conv2d(inject_dim[0], base_dim, 1)
        self.embed_1 = ConvEmbed(base_dim, embed_dim[0], patch_size=embed_patch[0], stride=embed_stride[0])
        self.s2former_1 = S2Former(embed_dim[0], heads[0], reduction_ratio[0], mlp_mult[0], depth[0])

        # self.inject_2 = nn.Conv2d(inject_dim[1], embed_dim[0], 1)
        self.embed_2 = ConvEmbed(embed_dim[0], embed_dim[1], patch_size=embed_patch[1], stride=embed_stride[1])
        self.s2former_2 = S2Former(embed_dim[1], heads[1], reduction_ratio[1], mlp_mult[1], depth[1])

        # self.inject_3 = nn.Conv2d(inject_dim[2], embed_dim[1], 1)
        self.embed_3 = ConvEmbed(embed_dim[1], embed_dim[2], patch_size=embed_patch[2], stride=embed_stride[2])
        self.s2former_3 = S2Former(embed_dim[2], heads[2], reduction_ratio[2], mlp_mult[2], depth[2])

        # self.inject_4 = nn.Conv2d(inject_dim[3], embed_dim[2], 1)
        self.embed_4 = ConvEmbed(embed_dim[2], embed_dim[3], patch_size=embed_patch[3], stride=embed_stride[3])
        self.s2former_4 = S2Former(embed_dim[3], heads[3], reduction_ratio[3], mlp_mult[3], depth[3])

        self.inject_5 = ConvEmbed(inject_dim[3], embed_dim[4], patch_size=embed_patch[4], stride=embed_stride[4])
        self.embed_5 = ConvEmbed(embed_dim[3], embed_dim[4], patch_size=embed_patch[4], stride=embed_stride[4])
        self.s2former_5 = S2Former(embed_dim[4], heads[4], reduction_ratio[4], mlp_mult[4], depth[4])

        self.inject_6 = ConvEmbed(inject_dim[2], embed_dim[5], patch_size=embed_patch[5], stride=embed_stride[5])
        self.embed_6 = ConvEmbed(embed_dim[4], embed_dim[5], patch_size=embed_patch[5], stride=embed_stride[5])
        self.s2former_6 = S2Former(embed_dim[5], heads[5], reduction_ratio[5], mlp_mult[5], depth[5])

        self.inject_7 = ConvEmbed(inject_dim[1], embed_dim[6], patch_size=embed_patch[6], stride=embed_stride[6])
        self.embed_7 = ConvEmbed(embed_dim[5], embed_dim[6], patch_size=embed_patch[6], stride=embed_stride[6])
        self.s2former_7 = S2Former(embed_dim[6], heads[6], reduction_ratio[6], mlp_mult[6], depth[6])

        self.inject_8 = ConvEmbed(inject_dim[0], embed_dim[7], patch_size=embed_patch[7], stride=embed_stride[7])
        self.embed_8 = ConvEmbed(embed_dim[6], embed_dim[7], patch_size=embed_patch[7], stride=embed_stride[7])
        self.s2former_8 = S2Former(embed_dim[7], heads[7], reduction_ratio[7], mlp_mult[7], depth[7])

        self.up1 = Upsample(embed_dim[4] + embed_dim[2] * 2, embed_dim[4])
        self.up2 = Upsample(embed_dim[5] + embed_dim[1] * 2, embed_dim[5])
        self.up3 = Upsample(embed_dim[6] + embed_dim[0] * 2, embed_dim[6])
        self.up4 = Upsample(embed_dim[7] + base_dim * 2 , embed_dim[7])

        self.to_fused = nn.Sequential(
            nn.Conv2d(embed_dim[3] * 2, embed_dim[4], kernel_size=1, stride=1, padding=0),
            nn.SyncBatchNorm(embed_dim[4]),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim[4], embed_dim[4], kernel_size=3, stride=1, padding=1),
            nn.SyncBatchNorm(embed_dim[4]),
            nn.ReLU(inplace=True),
        )

        # self.to_fused = nn.Conv2d(embed_dim[3] * 2, embed_dim[4], 1)

        self.up = Upsample(embed_dim[7] + input_dim * 2, 32)

        self.out_conv = OutConv(32, 32, num_classes, 3, 1, 1)

        self.act = nn.Sigmoid()

    # xy
    def forward(self, inputX, inputY, spa_feats, sem_feats):
        x_feats = []
        y_feats = []

        x, y = self.in_conv(inputX), self.in_conv(inputY)
        x_feats.append(x)
        y_feats.append(y)

        x, y = self.embed_1(x), self.embed_1(y)
        x, y = self.s2former_1(x, y, y), self.s2former_1(y, x, x)
        x_feats.append(x)
        y_feats.append(y)

        x, y = self.embed_2(x), self.embed_2(y)
        x, y = self.s2former_2(x, y, y), self.s2former_2(y, x, x)
        x_feats.append(x)
        y_feats.append(y)

        x, y = self.embed_3(x), self.embed_3(y)
        x, y = self.s2former_3(x, y, y), self.s2former_3(y, x, x)
        x_feats.append(x)
        y_feats.append(y)

        x, y= self.embed_4(x), self.embed_4(y)
        x, y = self.s2former_4(x, y, y), self.s2former_4(y, x, x)


        f = self.to_fused(torch.cat([x, y], dim=1))
        f, spa, sem = self.embed_5(f), self.inject_5(spa_feats[3]), self.inject_5(sem_feats[3])
        f = self.s2former_5(sem, spa, f)

        f = self.up1(f, torch.cat([x_feats[3], y_feats[3]], dim=1))
        f, spa, sem = self.embed_6(f), self.inject_6(spa_feats[2]), self.inject_6(sem_feats[2])
        f = self.s2former_6(sem, spa, f)

        f = self.up2(f, torch.cat([x_feats[2], y_feats[2]], dim=1))
        f, spa, sem = self.embed_7(f), self.inject_7(spa_feats[1]), self.inject_7(sem_feats[1])
        f = self.s2former_7(sem, spa, f)

        f = self.up3(f, torch.cat([x_feats[1], y_feats[1]], dim=1))
        f, spa, sem = self.embed_8(f), self.inject_8(spa_feats[0]), self.inject_8(sem_feats[0])
        f = self.s2former_8(sem, spa, f)

        f = self.up4(f, torch.cat([x_feats[0], y_feats[0]], dim=1))

        f = self.up(f, torch.cat([inputX, inputY], dim=1))
        logits = self.out_conv(f)

        return self.act(logits), logits




