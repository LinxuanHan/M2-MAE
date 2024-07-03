"""
This code is referenced from https://github.com/jeya-maria-jose/KiU-Net-pytorch/blob/master/LiTS/net/models.py
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


# 该模型是一种轻量kiuNet实现，我在原始代码基础上，减去了最小分辨率的编解码过程
class KiUNet_min(nn.Module):
    def __init__(self, in_channel=1, out_channel=2, training=True):
        super(KiUNet_min, self).__init__()
        self.training = training
        self.encoder1 = nn.Conv3d(in_channel, 32, 3, stride=1, padding=1)
        self.encoder2 = nn.Conv3d(32, 64, 3, stride=1, padding=1)
        self.encoder3 = nn.Conv3d(64, 128, 3, stride=1, padding=1)
        self.encoder4 = nn.Conv3d(128, 256, 3, stride=1, padding=1)
        # self.encoder5=   nn.Conv3d(256, 512, 3, stride=1, padding=1)

        self.kencoder1 = nn.Conv3d(in_channel, 32, 3, stride=1, padding=1)
        self.kdecoder1 = nn.Conv3d(32, 2, 3, stride=1, padding=1)

        # self.decoder1 = nn.Conv3d(512, 256, 3, stride=1,padding=1)
        self.decoder2 = nn.Conv3d(256, 128, 3, stride=1, padding=1)
        self.decoder3 = nn.Conv3d(128, 64, 3, stride=1, padding=1)
        self.decoder4 = nn.Conv3d(64, 32, 3, stride=1, padding=1)
        self.decoder5 = nn.Conv3d(32, 2, 3, stride=1, padding=1)

        self.map4 = nn.Sequential(
            nn.Conv3d(2, out_channel, 1, 1),
            nn.Upsample(scale_factor=(1, 1, 1), mode='trilinear'),
            # nn.Sigmoid()
            nn.Softmax(dim=1)
        )

        # 128*128 尺度下的映射
        self.map3 = nn.Sequential(
            nn.Conv3d(64, out_channel, 1, 1),
            nn.Upsample(scale_factor=(4, 4, 4), mode='trilinear'),
            # nn.Sigmoid()
            nn.Softmax(dim=1)
        )

        # 64*64 尺度下的映射
        self.map2 = nn.Sequential(
            nn.Conv3d(128, out_channel, 1, 1),
            nn.Upsample(scale_factor=(8, 8, 8), mode='trilinear'),
            # nn.Sigmoid()
            nn.Softmax(dim=1)
        )

        # 32*32 尺度下的映射
        self.map1 = nn.Sequential(
            nn.Conv3d(256, out_channel, 1, 1),
            nn.Upsample(scale_factor=(16, 16, 16), mode='trilinear'),
            # nn.Sigmoid()
            nn.Softmax(dim=1)
        )

    def forward(self, x):

        out = F.relu(F.max_pool3d(self.encoder1(x), 2, 2))
        t1 = out
        out = F.relu(F.max_pool3d(self.encoder2(out), 2, 2))
        t2 = out
        out = F.relu(F.max_pool3d(self.encoder3(out), 2, 2))
        t3 = out
        out = F.relu(F.max_pool3d(self.encoder4(out), 2, 2))

        # t4 = out
        # out = F.relu(F.max_pool3d(self.encoder5(out),2,2))

        # t2 = out
        # out = F.relu(F.interpolate(self.decoder1(out),scale_factor=(2,2,2),mode ='trilinear'))
        # print(out.shape,t4.shape)
        # out = torch.add(F.pad(out,[0,0,0,0,0,1]),t4)
        # out = torch.add(out,t4)

        output1 = self.map1(out)
        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=(2, 2, 2), mode='trilinear'))
        out = torch.add(out, t3)
        output2 = self.map2(out)
        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=(2, 2, 2), mode='trilinear'))
        out = torch.add(out, t2)
        output3 = self.map3(out)
        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=(2, 2, 2), mode='trilinear'))
        out = torch.add(out, t1)

        out1 = F.relu(F.interpolate(self.kencoder1(x), scale_factor=(1, 2, 2), mode='trilinear'))
        out1 = F.relu(F.interpolate(self.kdecoder1(out1), scale_factor=(1, 0.5, 0.5), mode='trilinear'))

        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2, 2), mode='trilinear'))
        # print(out.shape,out1.shape)
        out = torch.add(out, out1)
        output4 = self.map4(out)

        # print(out.shape)

        # print(output1.shape,output2.shape,output3.shape,output4.shape)
        if self.training is True:
            return output1, output2, output3, output4
        else:
            return output4


if __name__ == "__main__":
    net = KiUNet_min(in_channel=4, out_channel=4, training=True)
    in1 = torch.rand((2, 4, 64, 240, 240))
    out = net(in1)
    # print(out.shape)
    for i in range(len(out)):
        print(out[i].shape)