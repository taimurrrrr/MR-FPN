# -*- coding: utf-8 -*-


import torch
from torch import nn
from .CoTNetBlock import CoTNetLayer

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.cot_layer = CoTNetLayer(dim=planes, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        if stride > 1:
            self.avd = nn.AvgPool2d(3, 2, padding=1)
        else:
            self.avd = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)  # 1*1 Conv
        out = self.bn1(out)
        out = self.relu(out)

        if self.avd is not None:  # new：添加AvgPooling 进行downsample
            out = self.avd(out)

        out = self.cot_layer(out)  # CoTNetLayer
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)  # 1*1 Conv
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class CoTResNet(nn.Module):

    def __init__(self, block, layers,):
        self.inplanes = 64
        super(CoTResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        C1 = self.maxpool(x)

        C2 = self.layer1(C1)
        C3 = self.layer2(C2)
        C4 = self.layer3(C3)
        C5 = self.layer4(C4)


        return C2, C3, C4, C5

def cotnet50():
    return CoTResNet(Bottleneck, [3, 4, 6, 3])

if __name__ == '__main__':
    x = torch.rand(5, 3, 512, 512)
    model = CoTResNet(Bottleneck, [3, 4, 6, 3])
    C2, C3, C4, C5 = model(x)
    print(C2.shape, C3.shape, C4.shape, C5.shape)
