import torch.nn as nn
import math
import torch
from torch.nn import functional as F
from torch.nn import init
from torch.nn.parameter import Parameter

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
class ADF(nn.Module):

    def __init__(self):
        super(ADF, self).__init__()
        self.conv_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1,bias = False)
        self.bn = nn.BatchNorm2d(512)
        self.conv_2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0,bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, low_f, high_f):
        high_f = self._upsample(high_f, low_f)
        feature = torch.cat((high_f, low_f), dim=1)
        att = self.sig(self.conv_2(self.conv_1(feature)))
        out = high_f*att + low_f*(1-att)

        return out

    def _upsample(self, x, y, scale=1):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H // scale, W // scale), mode='bilinear', align_corners=True)

class MSB(nn.Module):
    def __init__(self, inchannel,out):
        super(MSB, self).__init__()
        self.conv3x3 = nn.Conv2d(inchannel, 256, kernel_size=3, padding=1)
        self.conv1x1 = nn.Conv2d(inchannel, 512, kernel_size=1, padding=0)
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(inchannel, 256, kernel_size=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=3, padding=1)
        )
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self,x):
        x1 = self.conv3x3(x)
        x2 = self.conv1_3(x)
        x3 = self.conv1x1(x)
        out = torch.cat((x1,x2,x3),dim=1)
        return self.relu(out)

class ShuffleAttention(nn.Module):

    def __init__(self, channel=512,reduction=16,G=8):
        super().__init__()
        self.G=G
        self.channel=channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))
        self.cweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))  # （1，x，1,1）
        self.cbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))  # （1，x，1,1）
        self.sweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sigmoid=nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.size()
        #group into subfeatures
        x=x.view(b*self.G,-1,h,w) #bs*G,c//G,h,w

        #channel_split
        x_0,x_1=x.chunk(2,dim=1) #bs*G,c//(2*G),h,w

        #channel attention
        x_channel=self.avg_pool(x_0) #bs*G,c//(2*G),1,1
        x_channel=self.cweight*x_channel+self.cbias #bs*G,c//(2*G),1,1
        x_channel=x_0*self.sigmoid(x_channel)

        #spatial attention
        x_spatial=self.gn(x_1) #bs*G,c//(2*G),h,w
        x_spatial=self.sweight*x_spatial+self.sbias #bs*G,c//(2*G),h,w
        x_spatial=x_1*self.sigmoid(x_spatial) #bs*G,c//(2*G),h,w

        # concatenate along channel axis
        out=torch.cat([x_channel,x_spatial],dim=1)  #bs*G,c//G,h,w
        out=out.contiguous().view(b,-1,h,w)

        # channel shuffle
        out = self.channel_shuffle(out, 2)
        return out



class ATE_FPN(nn.Module):
    def __init__(self, in_channels=None, out_channels=256):
        super(ATE_FPN, self).__init__()

        self.resu = MSB(2048,1024)
        self.SA1 = ShuffleAttention(256)
        self.SA2 = ShuffleAttention(256)
        self.SA3 = ShuffleAttention(256)

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

        x = self.resu(f5)
        p5 = self.toplayer_(f5)
        f4 = self.latlayer1_(f4)
        p4 = self._upsample_add(p5, f4)
        SA4 = self.SA1(p4)
        p4 = self.smooth1_(p4) + SA4
        p4 = self.smooth1_(p4)

        f3 = self.latlayer2_(f3)
        p3 = self._upsample_add(p4, f3)
        SA3 = self.SA2(p3)
        p3 = self.smooth2_(p3) + SA3
        p3 = self.smooth2_(p3)

        f2 = self.latlayer3_(f2)
        p2 = self._upsample_add(p3, f2)
        SA2 = self.SA3(p2)
        p2 = self.smooth3_(p2) + SA2
        p2 = self.smooth3_(p2)

        p3 = self._upsample(p3, p2)
        p4 = self._upsample(p4, p2)
        p5 = self._upsample(p5, p2)

        x = self._upsample(x, p2)
        out = torch.cat((p2, p3, p4, p5), dim=1) + x


        return out


if __name__ == '__main__':
    model = ATE_FPN(1024,)
    input1 = torch.randn(1, 256, 512, 512)
    input2 = torch.randn(1, 512, 256, 256)
    input3 = torch.randn(1, 1024, 128, 128)
    input4 = torch.randn(1, 2048, 64, 64)
    out = model(input1, input2, input3, input4)
    print(out.shape)
