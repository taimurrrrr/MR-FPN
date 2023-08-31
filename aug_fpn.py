import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from ..utils import Conv_BN_ReLU

class ACF(nn.Module):
    def __init__(self,inchannel,outchannekl):
        super(ACF, self).__init__()
        self.glob = nn.AdaptiveAvgPool2d(1)
        self.conv = Conv_BN_ReLU(inchannel, outchannekl, kernel_size=1)

        self.conv3x3_bn = nn.Conv2d(inchannel, outchannekl, kernel_size=3, padding=1, groups=1,padding_mode='zeros', bias=False)
        self.conv1x3_bn = nn.Conv2d(inchannel, outchannekl, kernel_size=(1, 3), padding=(0, 1), groups=1,padding_mode='zeros', bias=False)
        self.conv3x1_bn = nn.Conv2d(inchannel, outchannekl, kernel_size=(3, 1), padding=(1, 0), groups=1,padding_mode='zeros', bias=False)
        self.batch = nn.BatchNorm2d(outchannekl)
        self.relu = nn.ReLU()
    def forward(self, input):
        x = input
        x4 = self.conv(self.glob(input) * x)
        x3 = self.relu(self.batch(self.conv3x3_bn(input)))
        x2 = self.relu(self.batch(self.conv3x1_bn(input)))
        x1 = self.relu(self.batch(self.conv1x3_bn(input)))

        out = x4 +x3 +x2 + x1

        return out


class AUG_FPN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AUG_FPN, self).__init__()

        # Top layer
        self.ACF = ACF(2048, 256)
        self.toplayer_ = Conv_BN_ReLU(2048, 256,)

        # Smooth layers
        self.smooth1_ = Conv_BN_ReLU(256, 256, kernel_size=3, stride=1, padding=1)

        self.smooth2_ = Conv_BN_ReLU(256, 256, kernel_size=3, stride=1, padding=1)

        self.smooth3_ = Conv_BN_ReLU(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1_ = Conv_BN_ReLU(1024, 256,)

        self.latlayer2_ = Conv_BN_ReLU(512, 256)

        self.latlayer3_ = Conv_BN_ReLU(256, 256)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _upsample(self, x, y, scale=1):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H // scale, W // scale), mode='bilinear')

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, f2, f3, f4, f5):
        x = f5
        p5 = self.toplayer_(f5)
        c5 = self.ACF(x)
        p5 = p5 + c5
        f4 = self.latlayer1_(f4)
        p4 = self._upsample_add(p5, f4)
        p4 = self.smooth1_(p4)

        f3 = self.latlayer2_(f3)
        p3 = self._upsample_add(p4, f3)
        p3 = self.smooth2_(p3)

        f2 = self.latlayer3_(f2)
        p2 = self._upsample_add(p3, f2)
        p2 = self.smooth3_(p2)

        p3 = self._upsample(p3, p2)
        p4 = self._upsample(p4, p2)
        p5 = self._upsample(p5, p2)

        return p2, p3, p4, p5
