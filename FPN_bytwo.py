import torch.nn as nn
import math
import torch
from torch.nn import functional as F

class Conv_BN_ReLU(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0):
        super(Conv_BN_ReLU, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
class ConvBNReLU(nn.Module):
    '''Module for the Conv-BN-ReLU tuple.'''

    def __init__(self, c_in, c_out, kernel_size, stride, padding, dilation,
                 use_relu=True):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            c_in, c_out, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        if use_relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
    

class CARAFE(nn.Module):
    def __init__(self, c, c_mid=64, scale=2, k_up=5, k_enc=3):
        """ The unofficial implementation of the CARAFE module.
        The details are in "https://arxiv.org/abs/1905.02188".
        Args:
            c: The channel number of the input and the output.
            c_mid: The channel number after compression.
            scale: The expected upsample scale.
            k_up: The size of the reassembly kernel.
            k_enc: The kernel size of the encoder.
        Returns:
            X: The upsampled feature map.
        """
        super(CARAFE, self).__init__()
        self.scale = scale

        self.comp = ConvBNReLU(c, c_mid, kernel_size=1, stride=1,
                               padding=0, dilation=1)
        self.enc = ConvBNReLU(c_mid, (scale * k_up) ** 2, kernel_size=k_enc,
                              stride=1, padding=k_enc // 2, dilation=1,
                              use_relu=False)
        self.pix_shf = nn.PixelShuffle(scale)

        self.upsmp = nn.Upsample(scale_factor=scale, mode='nearest')
        self.unfold = nn.Unfold(kernel_size=k_up, dilation=scale,
                                padding=k_up // 2 * scale)

    def forward(self, X):
        b, c, h, w = X.size()
        h_, w_ = h * self.scale, w * self.scale

        W = self.comp(X)  # b * m * h * w
        W = self.enc(W)  # b * 100 * h * w
        W = self.pix_shf(W)  # b * 25 * h_ * w_
        W = F.softmax(W, dim=1)  # b * 25 * h_ * w_

        X = self.upsmp(X)  # b * c * h_ * w_
        X = self.unfold(X)  # b * 25c * h_ * w_
        X = X.view(b, c, -1, h_, w_)  # b * 25 * c * h_ * w_

        X = torch.einsum('bkhw,bckhw->bchw', [W, X])  # b * c * h_ * w_
        return X


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        '''
        return :(batch, 1, h, w)
        '''
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 最大池化 (batch, 1, h, w)#
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 平均池化 (batch, 1, h, w)
        x = torch.cat([avg_out, max_out], dim=1)  # (batch, 2, h, w)
        x = self.conv1(x)  # (batch, 1, h, w)
        return self.sigmoid(x)

class MEU(nn.Module):
    # 特征融合
    def __init__(self,outchannle):
        super(MEU, self).__init__()
        self.SA = SpatialAttention()
        self.adt = nn.AdaptiveAvgPool2d(1)
        self.con = nn.Conv2d(outchannle, outchannle, kernel_size=1, padding=0, bias=False)
        self.sig1 = nn.Sigmoid()

    def forward(self, high, low):
        low_1 = self.SA(low)
        high_1 = self.sig1(self.con(self.adt(high)))
        low_2 = low * high_1
        high_2 = low_1 * self._upsample(high,low)
        out = low_2+high_2

        return out

    def _upsample(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)

class MAT_FPN(nn.Module):
    def __init__(self, in_channels=None, out_channels=256):
        super(MAT_FPN, self).__init__()

        self.mer_1 = MEU(256)
        self.mer_2 = MEU(256)
        self.mer_3 = MEU(256)

        self.adpglob = nn.AdaptiveAvgPool2d(1)
        self.convglb1x1 = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0, bias=False)

        # Top layer
        self.toplayer_ = Conv_BN_ReLU(2048, 256, kernel_size=1, stride=1, padding=0)

        # Smooth layers
        self.smooth1_ = Conv_BN_ReLU(256, 256, kernel_size=3, stride=1, padding=1)

        self.smooth2_ = Conv_BN_ReLU(256, 256, kernel_size=3, stride=1, padding=1)

        self.smooth3_ = Conv_BN_ReLU(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1_ = Conv_BN_ReLU(1024, 256, kernel_size=1, stride=1, padding=0)

        self.latlayer2_ = Conv_BN_ReLU(512, 256, kernel_size=1, stride=1, padding=0)

        self.latlayer3_ = Conv_BN_ReLU(256, 256, kernel_size=1, stride=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _upsample(self, x, y, scale=1):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H // scale, W // scale), mode='bilinear', align_corners=True)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, f2, f3, f4, f5):
        glb = self.convglb1x1(self.adpglob(f5))
        p5 = self.toplayer_(f5) + glb
        f4 = self.latlayer1_(f4)
        p4 = self.mer_1(p5, f4)
        p4 = self.smooth1_(p4)

        f3 = self.latlayer2_(f3)
        p3 = self.mer_2(p4, f3)
        p3 = self.smooth2_(p3)

        f2 = self.latlayer3_(f2)
        p2 = self.mer_2(p3, f2)
        p2 = self.smooth3_(p2)

        p3 = self._upsample(p3, p2)
        p4 = self._upsample(p4, p2)
        p5 = self._upsample(p5, p2)

        out = torch.cat((p2, p3, p4, p5), dim=1)

        return out


if __name__ == '__main__':
    model = MAT_FPN(1024,)
    input1 = torch.randn(1, 256, 512, 512)
    input2 = torch.randn(1, 512, 256, 256)
    input3 = torch.randn(1, 1024, 128, 128)
    input4 = torch.randn(1, 2048, 64, 64)
    import time
    sta = time.time()
    out = model(input1, input2, input3, input4)
    print(time.time()-sta)
    print(out.shape)
