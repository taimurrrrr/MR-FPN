import torch.nn as nn
import math
import torch.nn.functional as F
import torch

def weight_init(module):
    for n, m in module.named_children():
        if n is not 'backbone':
            # print('initialize: '+n)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Sequential):
                weight_init(m)
            elif isinstance(m, nn.ReLU):
                pass
            else:
                m.initialize()

def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, dilation=dilation, bias=bias)


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=bias)

class FFM(nn.Module):
    def __init__(self, channel):
        super(FFM, self).__init__()
        self.conv_1 = conv3x3(channel, channel)  # 3x3
        self.bn_1 = nn.BatchNorm2d(channel)
        self.conv_2 = conv3x3(channel, channel)  # 3x3
        self.bn_2 = nn.BatchNorm2d(channel)

    def forward(self, x_1, x_2):
        out = x_1 * x_2
        out = F.relu(self.bn_1(self.conv_1(out)), inplace=True)
        out = F.relu(self.bn_2(self.conv_2(out)), inplace=True)
        return out

    def initialize(self):
        weight_init(self)
# Spatial Attention Module
class SAM(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(SAM, self).__init__()
        self.conv_atten = conv3x3(2, 1)
        self.conv = conv3x3(in_chan, out_chan)
        self.bn = nn.BatchNorm2d(out_chan)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        atten = torch.cat([avg_out, max_out], dim=1)
        atten = torch.sigmoid(self.conv_atten(atten))
        out = torch.mul(x, atten)
        out = F.relu(self.bn(self.conv(out)), inplace=True)
        return out

    def initialize(self):
        weight_init(self)
# Cross Aggregation Module
class CAM(nn.Module):
    def __init__(self, channel):
        super(CAM, self).__init__()
        self.down = nn.Sequential(
            conv3x3(channel, channel, stride=2),
            nn.BatchNorm2d(channel)
        )
        self.conv_1 = conv3x3(channel, channel)
        self.bn_1 = nn.BatchNorm2d(channel)
        self.conv_2 = conv3x3(channel, channel)
        self.bn_2 = nn.BatchNorm2d(channel)
        self.mul = FFM(channel)

    def forward(self, x_high, x_low):
        left_1 = x_low
        left_2 = F.relu(self.down(x_low), inplace=True)
        right_1 = F.interpolate(x_high, size=x_low.size()[2:], mode='bilinear', align_corners=True)
        right_2 = x_high
        left = F.relu(self.bn_1(self.conv_1(left_1 * right_1)), inplace=True)
        right = F.relu(self.bn_2(self.conv_2(left_2 * right_2)), inplace=True)
        # left = F.relu(left_1 * right_1, inplace=True)
        # right = F.relu(left_2 * right_2, inplace=True)
        right = F.interpolate(right, size=x_low.size()[2:], mode='bilinear', align_corners=True)
        out = self.mul(left, right)
        return out
    def initialize(self):
        weight_init(self)
class BRM(nn.Module):
    def __init__(self, channel):
        super(BRM, self).__init__()
        self.conv_atten = conv1x1(channel, channel)
        self.conv_1 = conv3x3(channel, channel)
        self.bn_1 = nn.BatchNorm2d(channel)
        self.conv_2 = conv3x3(channel, channel)
        self.bn_2 = nn.BatchNorm2d(channel)

    def forward(self, x_1, x_edge):
        # x = torch.cat([x_1, x_edge], dim=1)
        x = x_1 + x_edge
        atten = F.avg_pool2d(x, x.size()[2:])
        atten = torch.sigmoid(self.conv_atten(atten))
        out = torch.mul(x, atten) + x
        out = F.relu(self.bn_1(self.conv_1(out)), inplace=True)
        out = F.relu(self.bn_2(self.conv_2(out)), inplace=True)
        return out

    def initialize(self):
        weight_init(self)
class Abc_merge(nn.Module):

    def __init__(self,in_channels, out_channels):
        super(Abc_merge, self).__init__()

        self.path1_1 = nn.Sequential(
            conv1x1(2048, 256),
            nn.BatchNorm2d(256)
        )
        self.path1_2 = nn.Sequential(
            conv1x1(2048, 256),
            nn.BatchNorm2d(256)
        )
        self.path1_3 = nn.Sequential(
            conv1x1(1024, 256),
            nn.BatchNorm2d(256)
        )
        self.path2 = SAM(512, 256)
        self.path3 = nn.Sequential(
            conv1x1(256, 256),
            nn.BatchNorm2d(256)
        )
        self.fuse1_1 = FFM(256)
        self.fuse1_2 = FFM(256)
        self.fuse12 = CAM(256)
        self.fuse3 = FFM(256)
        self.fuse23 = BRM(256)
        weight_init(self)


    def forward(self, f2, f3, f4, f5):
        path1_1 = F.avg_pool2d(f5, f5.size()[2:])  # 池化
        path1_1 = self.path1_1(path1_1)
        path1_1 = F.interpolate(path1_1, size=f5.size()[2:], mode='bilinear', align_corners=True)  # 1/32
        path1_2 = F.relu(self.path1_2(f5), inplace=True)  # 1/32
        path1_2 = self.fuse1_1(path1_1, path1_2)  # 1/32 stage5的特征融合
        path1_2 = F.interpolate(path1_2, size=f4.size()[2:], mode='bilinear', align_corners=True)  # 1/16 stage5的上采样到stage4

        path1_3 = F.relu(self.path1_3(f4), inplace=True)  # stage4的通道下降256
        path1 = self.fuse1_2(path1_2, path1_3)  # 1/16  # stage4与5的融合


        path2 = self.path2(f3)  # 1/8 stage3经过sam模块 通道到256
        path12 = self.fuse12(path1, path2)  # 1/8 stage3与stage4的特征送到CAM模块
        path12 = F.interpolate(path12, size=f2.size()[2:], mode='bilinear', align_corners=True)  # 1/4 上采样到stage2
        path3_1 = F.relu(self.path3(f2), inplace=True)  # 1/4
        path3_2 = F.interpolate(path1_2, size=f2.size()[2:], mode='bilinear', align_corners=True)  # 1/4
        path3 = self.fuse3(path3_1, path3_2)  # 1/4

        path_out = self.fuse23(path12, path3)


        return path_out
    def initialize(self):
        weight_init(self)


if __name__ == '__main__':

    x1 = torch.rand((4,2048, 16, 16))
    x2 = torch.rand((4,1024, 32, 32))
    x3 = torch.rand((4,512, 64, 64))
    x4 = torch.rand((4,256, 128, 128))
    model = Abc_merge(32,12)
    out = model(x4,x3,x2,x1)
    print(out.shape)
