import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl

from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base.heads import SegmentationHead
from lib.pvtv2 import pvt_v2_b2
import cv2, os




def initialize_weights(*models):
    """
    Initialize Model Weights
    """
    for model in models:
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))  # 通过平均池化压缩全局空间信息: (B,C,H,W)--> (B,C,1,1) ,然后通过MLP降维升维:(B,C,1,1)
        max_out = self.mlp(self.max_pool(x))  # 通过最大池化压缩全局空间信息: (B,C,H,W)--> (B,C,1,1) ,然后通过MLP降维升维:(B,C,1,1)
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 通过平均池化压缩全局通道信息:(B,C,H,W)-->(B,1,H,W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 通过最大池化压缩全局通道信息:(B,C,H,W)-->(B,1,H,W)
        x = torch.cat([avg_out, max_out], dim=1)  # 在通道上拼接两个矩阵:(B,2,H,W)
        x = self.conv1(x)  # 通过卷积层得到注意力权重:(B,2,H,W)-->(B,1,H,W)
        return self.sigmoid(x)


class ResCBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(ResCBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)
        self.conv = Conv2dReLU(in_planes, in_planes, 3, 1, 1)

    def forward(self, x):
        out1 = self.conv(x)
        out = x * self.ca(x)  # 通过通道注意力机制得到的特征图,x:(B,C,H,W),ca(x):(B,C,1,1),out:(B,C,H,W)
        result = out * self.sa(out)  # 通过空间注意力机制得到的特征图,out:(B,C,H,W),sa(out):(B,1,H,W),result:(B,C,H,W)

        return out1+result



class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            use_batchnorm=True,
    ):

        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=not use_batchnorm,
        )
        relu = nn.ReLU(inplace=True)

        if use_batchnorm:
            bn = nn.BatchNorm2d(out_channels)

        else:
            bn = nn.Identity()

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = nn.Conv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = nn.Conv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class Agg(nn.Module):
    def __init__(self, channel=64):
        super(Agg, self).__init__()
        self.h2l_pool = nn.AvgPool2d((2, 2), stride=2)
        self.l2h_up = nn.Upsample(scale_factor=2, mode="nearest")

        # stage 1
        self.h2h_1 = nn.Sequential(
            Conv2dReLU(channel, channel, 3, 1, 1)
        )
        self.h2l_1 = nn.Sequential(
            Conv2dReLU(channel, channel, 3, 1, 1)
        )
        self.l2h_1 = nn.Sequential(
            Conv2dReLU(channel, channel, 3, 1, 1)
        )
        self.l2l_1 = nn.Sequential(
            Conv2dReLU(channel, channel, 3, 1, 1)
        )

        # stage 2
        self.h2h_2 = nn.Sequential(
            Conv2dReLU(channel, channel, 3, 1, 1)
        )
        self.l2h_2 = nn.Sequential(
            Conv2dReLU(channel, channel, 3, 1, 1)
        )

    def forward(self, h, l):
        # stage 1
        h2h = self.h2h_1(h)
        h2l = self.h2l_1(self.h2l_pool(h))
        l2l = self.l2l_1(l)
        l2h = self.l2h_1(self.l2h_up(l))
        h = h2h + l2h
        l = l2l + h2l

        # stage 2
        h2h = self.h2h_2(h)
        l2h = self.l2h_2(self.l2h_up(l))
        out = h2h + l2h
        return out

class CSAgg(nn.Module):
    def __init__(self, channel=64):
        super(CSAgg, self).__init__()
        self.h2l_pool = nn.AvgPool2d((2, 2), stride=2)
        self.l2h_up = nn.Upsample(scale_factor=2, mode="nearest")

        # stage 1
        self.h2h_1 = ResCBAM(channel)
        self.h2l_1 = ResCBAM(channel)
        self.l2h_1 = ResCBAM(channel)
        self.l2l_1 = ResCBAM(channel)

        # stage 2
        self.h2h_2 = ResCBAM(channel)
        self.l2h_2 = ResCBAM(channel)

    def forward(self, h, l):
        # stage 1
        h2h = self.h2h_1(h)
        h2l = self.h2l_1(self.h2l_pool(h))
        l2l = self.l2l_1(l)
        l2h = self.l2h_1(self.l2h_up(l))
        h = h2h + l2h
        l = l2l + h2l

        # stage 2
        h2h = self.h2h_2(h)
        l2h = self.l2h_2(self.l2h_up(l))
        out = h2h + l2h
        return out

class BDMM(nn.Module):
    def __init__(self, inplanes: list, midplanes=32, upsample=8):
        super(BDMM, self).__init__()
        assert len(inplanes) == 3

        self.rfb1 = RFB_modified(inplanes[0], midplanes)
        self.rfb2 = RFB_modified(inplanes[1], midplanes)
        self.rfb3 = RFB_modified(inplanes[2], midplanes)

        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

        self.agg1 = Agg(midplanes)
        self.agg2 = Agg(midplanes)
        self.conv_out = nn.Sequential(
            Conv2dReLU(midplanes, 1, 3, padding=1),
            nn.Upsample(scale_factor=upsample, mode='bilinear', align_corners=True),
        )

    def forward(self, x1, x2, x3):
        x1 = self.rfb1(x1)
        x2 = self.rfb2(x2)
        x3 = self.rfb3(x3)

        x2 = self.agg1(x2, x3)
        x1 = self.agg2(x1, x2)

        out = self.conv_out(x1)

        return out

class CBAMagg_BDMM(BDMM):
    def __init__(self, inplanes: list, midplanes=32, upsample=8):
        super(CBAMagg_BDMM, self).__init__(inplanes, midplanes, upsample)
        self.agg1 = CSAgg(midplanes)
        self.agg2 = CSAgg(midplanes)

class DynamicGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.conv(x)


class BDGD_A(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.l2h_up = nn.Upsample(scale_factor=2, mode="nearest")

        # stage 1
        self.l2l_0 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, 1),
        )

        # stage 2
        self.l2h_1 = nn.Sequential(
            Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.l2l_1 = nn.Sequential(
            Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1),
        )

        # stage 3
        self.l2h_2 = nn.Sequential(
            Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1),
        )
    def forward(self, x, dist):
        dist_l = F.interpolate(dist, x.size()[2:], mode='bilinear')

        # stage 1
        l = self.l2l_0(x)

        # stage 2
        l2l = self.l2l_1(l*dist_l)

        l2h = self.l2h_1(self.l2h_up(l+l2l))

        # stage 3
        out = self.l2h_2(self.l2h_up(l)+l2h)
        return out


class BDGD_B(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.h2l_pool = nn.AvgPool2d((2, 2), stride=2)
        self.l2h_up = nn.Upsample(scale_factor=2, mode="nearest")

        # stage 1
        self.h2h_0 = nn.Sequential(
            nn.Conv2d(skip_channels, skip_channels, 3, 1, 1, groups=skip_channels),
            nn.Conv2d(skip_channels, out_channels, 1),
        )

        self.l2l_0 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, 1),
        )

        # stage 2
        self.h2h_1 = nn.Sequential(
            Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.h2l_1 = nn.Sequential(
            Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.l2h_1 = nn.Sequential(
            Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.l2l_1 = nn.Sequential(
            Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1),
        )

        # stage 3
        self.h2h_2 = nn.Sequential(
            Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.l2h_2 = nn.Sequential(
            Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, skip, dist):
        dist_h = F.interpolate(dist, skip.size()[2:], mode='bilinear')
        dist_l = F.interpolate(dist, x.size()[2:], mode='bilinear')

        # stage 1
        h_in = self.h2h_0(skip)
        l_in = self.l2l_0(x)

        # stage 2
        h2h = self.h2h_1(h_in * dist_h)
        l2h = self.l2h_1(self.l2h_up(l_in))

        l2l = self.l2l_1(l_in * dist_l)
        h2l = self.h2l_1(self.h2l_pool(h_in))

        h = h2h + l2h
        l = l2l + h2l

        # stage 3
        h2h = self.h2h_2(h)
        l2h = self.l2h_2(self.l2h_up(l)) + l2h
        out = h2h + l2h
        return out


class BDMPVT_Net(pl.LightningModule):
    def __init__(self, nclass=1, max_epoch=None):
        super().__init__()
        # 初始化编码器
        self.encoder = pvt_v2_b2()  # 输出通道: [64, 128, 320, 512]
        path = '../pretrained_pth/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.encoder.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.encoder.load_state_dict(model_dict)


        self.agg = BDMM(self.encoder.out_channels[-3:], 32, upsample=8)

        # 解码器模块
        self.dec1 = BDGD_A(64, 32)
        self.dec2 = BDGD_B(128, self.encoder.out_channels[-4], 64)
        self.dec3 = BDGD_B(256, self.encoder.out_channels[-3], 128)
        self.dec4 = BDGD_B(self.encoder.out_channels[-1], self.encoder.out_channels[-2], 256)

        # 主分割头
        self.seg_head = SegmentationHead(32, nclass, upsampling=2)

        # 深度监督头（添加4个辅助输出头）
        self.ds_head1 = SegmentationHead(256, nclass,upsampling=16)  # 对应c4
        self.ds_head2 = SegmentationHead(128, nclass,upsampling=8)  # 对应c3
        self.ds_head3 = SegmentationHead(64, nclass,upsampling=4)  # 对应c2

        self.learning_rate = 1e-4
        self.max_epoch = max_epoch

        # 初始化权重
        initialize_weights(self.dec1)
        initialize_weights(self.dec2)
        initialize_weights(self.dec3)
        initialize_weights(self.dec4)
        initialize_weights(self.seg_head)
        initialize_weights(self.agg)
        initialize_weights(self.ds_head1)
        initialize_weights(self.ds_head2)
        initialize_weights(self.ds_head3)

    def forward(self, x):
        # 编码器前向传播
        x = self.encoder(x)  # 输出: [x1, x2, x3, x4]

        # 边界检测模块
        bdm = self.agg(x[-3], x[-2], x[-1])

        # 解码器前向传播
        c4 = self.dec4(x[-1], x[-2], bdm)
        c3 = self.dec3(c4, x[-3], bdm)
        c2 = self.dec2(c3, x[-4], bdm)
        c1 = self.dec1(c2, bdm)

        # 主分割输出
        seg = self.seg_head(c1)

        # print("encoder output shape:", [xx.shape for xx in x])
        # print("agg output shape:", bdm.shape)
        # print("dec4 output shape:", c4.shape)
        # print("dec3 output shape:", c3.shape)
        # print("dec2 output shape:", c2.shape)
        # print("dec1 output shape:", c1.shape)
        # print("seghead output shape:", seg.shape)

        # 深度监督输出
        ds1 = self.ds_head1(c4)
        ds2 = self.ds_head2(c3)
        ds3 = self.ds_head3(c2)


        return seg,bdm,ds1,ds2,ds3



class BDM_Net(pl.LightningModule):
    def __init__(self, nclass=1, deep_supervise=False):
        super().__init__()
        self.encoder = get_encoder('timm-efficientnet-b5', weights='noisy-student')
        self.agg = BDMM(self.encoder.out_channels[-3:], 32, upsample=8)

        self.dec1 = BDGD_A(64, 32)
        self.dec2 = BDGD_B(128, self.encoder.out_channels[-4], 64)
        self.dec3 = BDGD_B(256, self.encoder.out_channels[-3], 128)
        self.dec4 = BDGD_B(self.encoder.out_channels[-1], self.encoder.out_channels[-2], 256)

        self.seg_head = SegmentationHead(32, nclass, upsampling=2)
        self.deep_supervise=deep_supervise
        if self.deep_supervise:
            # 深度监督头（添加4个辅助输出头）
            self.ds_head1 = SegmentationHead(256, nclass, upsampling=16)  # 对应c4
            self.ds_head2 = SegmentationHead(128, nclass, upsampling=8)  # 对应c3
            self.ds_head3 = SegmentationHead(64, nclass, upsampling=4)  # 对应c2


        initialize_weights(self.dec1)
        initialize_weights(self.dec2)
        initialize_weights(self.dec3)
        initialize_weights(self.dec4)

        initialize_weights(self.seg_head)
        initialize_weights(self.agg)


    def forward(self, x):
        x = self.encoder(x)
        bdm = self.agg(x[-3], x[-2], x[-1])
        c4 = self.dec4(x[-1], x[-2], bdm)
        c3 = self.dec3(c4, x[-3], bdm)
        c2 = self.dec2(c3, x[-4], bdm)
        c1 = self.dec1(c2, bdm)
        seg = self.seg_head(c1)
        if self.deep_supervise:
            ds1 = self.ds_head1(c4)
            ds2 = self.ds_head2(c3)
            ds3 = self.ds_head3(c2)
            return seg, bdm, ds1, ds2, ds3
        else:
            return seg, bdm

class BDM_PVT_Net(BDM_Net):
    def __init__(self, nclass=1, deep_supervise=False):
        super().__init__(nclass, deep_supervise)
        self.encoder = pvt_v2_b2()  # 输出通道: [64, 128, 320, 512]
        path = '../pretrained_pth/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.encoder.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.encoder.load_state_dict(model_dict)
        # 深度监督头（添加4个辅助输出头）
        if self.deep_supervise:
            self.ds_head1 = SegmentationHead(256, nclass, upsampling=16)  # 对应c4
            self.ds_head2 = SegmentationHead(128, nclass, upsampling=8)   # 对应c3
            self.ds_head3 = SegmentationHead(64, nclass, upsampling=4)    # 对应c2
        self.agg = BDMM(self.encoder.out_channels[-3:], 32, upsample=8)

        # 解码器模块
        self.dec1 = BDGD_A(64, 32)
        self.dec2 = BDGD_B(128, self.encoder.out_channels[-4], 64)
        self.dec3 = BDGD_B(256, self.encoder.out_channels[-3], 128)
        self.dec4 = BDGD_B(self.encoder.out_channels[-1], self.encoder.out_channels[-2], 256)

class BDM_CSA_Net(BDM_Net):
    def __init__(self, nclass=1, deep_supervise=False):
        super().__init__(nclass, deep_supervise)
        self.agg = CBAMagg_BDMM(self.encoder.out_channels[-3:], 32, upsample=8)

class BDM_PVT_CSA_Net(BDM_PVT_Net):
    def __init__(self, nclass=1, deep_supervise=False):
        super().__init__(nclass, deep_supervise)
        self.agg = CBAMagg_BDMM(self.encoder.out_channels[-3:], 32, upsample=8)


if __name__ == '__main__':
    from Train import print_trainable_parameters
    model = BDM_Net(nclass=1)
    print_trainable_parameters(model.encoder,False)
    print_trainable_parameters(model,False)
    x=torch.rand(1, 3, 352, 352)
    out=model.forward(x)
    print([o.shape for o in out])
    model = BDM_PVT_Net(nclass=1)
    print_trainable_parameters(model.encoder, False)
    print_trainable_parameters(model, False)
    x = torch.rand(1, 3, 352, 352)
    out = model.forward(x)
    print([o.shape for o in out])
    model = BDM_CSA_Net(nclass=1)
    print_trainable_parameters(model.encoder,False)
    print_trainable_parameters(model, False)
    x = torch.rand(1, 3, 352, 352)
    out = model.forward(x)
    print([o.shape for o in out])
    model = BDM_PVT_CSA_Net(nclass=1)
    print_trainable_parameters(model.encoder, False)
    print_trainable_parameters(model, False)
    x = torch.rand(1, 3, 352, 352)
    out = model.forward(x)
    print([o.shape for o in out])

    model = BDM_Net(nclass=1,deep_supervise=True)
    print_trainable_parameters(model.encoder, False)
    print_trainable_parameters(model, False)
    x = torch.rand(1, 3, 352, 352)
    out = model.forward(x)
    print([o.shape for o in out])
    model = BDM_PVT_Net(nclass=1,deep_supervise=True)
    print_trainable_parameters(model.encoder, False)
    print_trainable_parameters(model, False)
    x = torch.rand(1, 3, 352, 352)
    out = model.forward(x)
    print([o.shape for o in out])
    model = BDM_CSA_Net(nclass=1,deep_supervise=True)
    print_trainable_parameters(model.encoder, False)
    print_trainable_parameters(model, False)
    x = torch.rand(1, 3, 352, 352)
    out = model.forward(x)
    print([o.shape for o in out])
    model = BDM_PVT_CSA_Net(nclass=1,deep_supervise=True)
    print_trainable_parameters(model.encoder, False)
    print_trainable_parameters(model, False)
    x = torch.rand(1, 3, 352, 352)
    out = model.forward(x)
    print([o.shape for o in out])