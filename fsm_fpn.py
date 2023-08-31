import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from ..utils import Conv_BN_ReLU
from torch.nn import init
from torch.nn.parameter import Parameter
class FSM(nn.Module):
    '''
    降低1x1通道的损失
    '''
    def __init__(self, in_chan, out_chan,):
        super(FSM, self).__init__()
        self.conv_atten = nn.Conv2d(in_chan, in_chan, kernel_size=1, bias=False,)
        self.gn_norm1 = nn.GroupNorm(32,in_chan)
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=1, bias=False,)
        self.gn_norm2 = nn.GroupNorm(32,out_chan)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        atten = self.sigmoid(self.gn_norm1(self.conv_atten(F.avg_pool2d(x, x.size()[2:]))))  # fm 就是激活加卷积，我觉得这平均池化用的贼巧妙
        feat = torch.mul(x, atten)  # 相乘，得到重要特征
        x = x + feat # 再加上
        feat = self.gn_norm2(self.conv(x))  # 最后一层 1*1 的卷积
        return feat

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


class EAG_FPN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EAG_FPN, self).__init__()

        # Top layer
        self.sa = ShuffleAttention(256)
        self.fsm = FSM(2048, 256)

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
        return F.upsample(x, size=(H // scale, W // scale), mode='bilinear')

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, f2, f3, f4, f5):
        x = f5
        x = self.fsm(x)
        p5 = self.toplayer_(f5) + x

        f4 = self.latlayer1_(f4)
        p4 = self._upsample_add(p5, f4)
        p4 = self.smooth1_(self.sa(p4))

        f3 = self.latlayer2_(f3)
        p3 = self._upsample_add(p4, f3)
        p3 = self.smooth2_(self.sa(p3))

        f2 = self.latlayer3_(f2)
        p2 = self._upsample_add(p3, f2)
        p2 = self.smooth3_(self.sa(p2))

        p3 = self._upsample(p3, p2)
        p4 = self._upsample(p4, p2)
        p5 = self._upsample(p5, p2)

        return p2, p3, p4, p5
