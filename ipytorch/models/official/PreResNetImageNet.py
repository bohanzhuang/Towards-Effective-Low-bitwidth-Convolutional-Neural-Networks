from __future__ import division

"""
Creates a PreResNet (ResNet-v2) Model as defined in:
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. (2015). 
Identity Mappings in Deep Residual Networks. 
arXiv preprint arXiv:1603.05027.
import from https://github.com/facebook/fb.resnet.torch
Copyright (c) Yang Lu, 2017
"""
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class PreBasicBlockImageNet(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, block_type=""):
        super(PreBasicBlockImageNet, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.block_type = block_type

    def forward(self, x):
        if self.block_type == "both_preact":
            x = self.bn1(x)
            x = self.relu1(x)
            residual = x
        elif self.block_type != "no_preact":
            residual = x
            x = self.bn1(x)
            x = self.relu1(x)
        else:
            residual = x
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv2(x)

        if self.downsample:
            residual = self.downsample(residual)

        out = x + residual
        return out


class PreBottleneckImageNet(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, block_type=""):
        super(PreBottleneckImageNet, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.downsample = downsample
        self.block_type = block_type

    def forward(self, x):
        if self.block_type == "both_preact":
            x = self.bn1(x)
            x = self.relu1(x)
            residual = x
        elif self.block_type != "no_preact":
            residual = x
            x = self.bn1(x)
            x = self.relu1(x)
        else:
            residual = x
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv2(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv3(x)

        if self.downsample:
            residual = self.downsample(residual)

        out = x + residual
        return out


class PreResNetImageNet(nn.Module):
    def __init__(self, depth, num_classes=1000):
        self.inplanes = 64
        super(PreResNetImageNet, self).__init__()
        if depth < 50:
            block = PreBasicBlockImageNet
        else:
            block = PreBottleneckImageNet

        if depth == 18:
            layers = [2, 2, 2, 2]
        elif depth == 34:
            layers = [3, 4, 6, 3]
        elif depth == 50:
            layers = [3, 4, 6, 3]
        elif depth == 101:
            layers = [3, 4, 23, 3]
        elif depth == 152:
            layers = [3, 8, 36, 3]

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], block_type="no_preact")
        self.layer2 = self._make_layer(block, 128, layers[1], 2)
        self.layer3 = self._make_layer(block, 256, layers[2], 2)
        self.layer4 = self._make_layer(block, 512, layers[3], 2)
        self.bn2 = nn.BatchNorm2d(512 * block.expansion)
        self.relu2 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, block_type="both_preact"):
        """ Stack n bottleneck modules where n is inferred from the depth of the network.
        Args:
            block: block type used to construct ResNet
            planes: number of output channels (need to multiply by block.expansion)
            blocks: number of blocks to be built
            stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.
        Returns: a Module consisting of n sequential bottlenecks.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, block_type))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn2(x)
        x = self.relu2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
