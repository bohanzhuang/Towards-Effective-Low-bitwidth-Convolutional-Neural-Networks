import torch
import torch.nn as nn
from torch.nn import Parameter
import math
from ipytorch.prune import *


__all__ = ['dsPreResNet', 'dsPreBasicBlock']


def conv1x1(in_plane, out_plane, stride=1, cRate=0.7):
    """
    1x1 convolutional layer
    """
    return nn.Conv2d(in_plane, out_plane,
                     kernel_size=1, stride=stride,
                     padding=0, bias=False)


def conv3x3(in_planes, out_planes, stride=1, cRate=0.7):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes,
                     kernel_size=3, stride=stride,
                     padding=1, bias=False)


def linear(in_features, out_features, cRate=0.7):
    return nn.Linear(in_features, out_features)
# ---------------------------Small Data Sets Like CIFAR-10 or CIFAR-100----------------------------
# both-preact | half-preact


class dsPreBasicBlock(nn.Module):
    """
    base module for PreResNet on small data sets
    """

    def __init__(self, in_plane, out_plane, stride=1, downsample=None, block_type="both_preact", cRate=0.7):
        """
        init module and weights
        :param in_plane: size of input plane
        :param out_plane: size of output plane
        :param stride: stride of convolutional layers, default 1
        :param downsample: down sample type for expand dimension of input feature maps, default None
        :param block_type: type of blocks, decide position of short cut, both-preact: short cut start from beginning
        of the first segment, half-preact: short cut start from the position between the first segment and the second
        one. default: both-preact
        """
        super(dsPreBasicBlock, self).__init__()
        self.name = block_type
        self.cRate = cRate
        self.downsample = downsample

        self.bn1 = nn.BatchNorm2d(in_plane)
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = conv3x3(in_plane, out_plane, stride, cRate=self.cRate)
        self.bn2 = nn.BatchNorm2d(out_plane)
        self.conv2 = conv3x3(out_plane, out_plane, cRate=self.cRate)
        self.d = Parameter(torch.ones(1).fill_(cRate))
        # self.dt = Parameter(torch.Tensor(1).fill_(1))

    def forward(self, x):
        """
        forward procedure of residual module
        :param x: input feature maps
        :return: output feature maps
        """
        if self.name == "half_preact":
            x = self.bn1(x)
            x = self.relu(x)
            residual = x
            x = self.conv1(x)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.conv2(x)
        elif self.name == "both_preact":
            residual = x
            x = self.bn1(x)
            x = self.relu(x)
            x = self.conv1(x)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.conv2(x)

        if self.downsample:
            residual = self.downsample(residual)
        # if x.size(1) != residual.size(1):
        #     residual = torch.cat((residual, residual), 1)
        # out = self.dt*x + residual
        out = self.d*x + residual
        return out


class dsPreResNet(nn.Module):
    """
    define PreResNet on small data sets
    """

    def __init__(self, depth, wide_factor=1, num_classes=10, cRate=0.7):
        """
        init model and weights
        :param depth: depth of network
        :param wide_factor: wide factor for deciding width of network, default is 1
        :param num_classes: number of classes, related to labels. default 10
        """
        super(dsPreResNet, self).__init__()
        self.cRate = cRate
        self.in_plane = 16 * wide_factor
        self.depth = depth
        n = (depth - 2) / 6
        self.conv = conv3x3(3, 16 * wide_factor, cRate=self.cRate)
        self.layer1 = self._make_layer(dsPreBasicBlock, 16 * wide_factor, n)
        self.layer2 = self._make_layer(
            dsPreBasicBlock, 32 * wide_factor, n, stride=2)
        self.layer3 = self._make_layer(
            dsPreBasicBlock, 64 * wide_factor, n, stride=2)
        self.bn = nn.BatchNorm2d(64 * wide_factor)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(7)
        self.fc = linear(64 * wide_factor, num_classes, cRate=self.cRate)

        # init layer parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, out_plane, n_blocks, stride=1):
        """
        make residual blocks, including short cut and residual function
        :param block: type of basic block to build network
        :param out_plane: size of output plane
        :param n_blocks: number of blocks on every segment
        :param stride: stride of convolutional neural network, default 1
        :return: residual blocks
        """
        downsample = None
        if stride != 1 or self.in_plane != out_plane:
            downsample = conv1x1(self.in_plane, out_plane, stride=stride, cRate=self.cRate)

        layers = []
        layers.append(block(self.in_plane, out_plane, stride,
                            downsample, block_type="half_preact",
                            cRate=self.cRate))
        self.in_plane = out_plane
        for i in range(1, n_blocks):
            layers.append(block(self.in_plane, out_plane, cRate=self.cRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        forward procedure of model
        :param x: input feature maps
        :return: output feature maps
        """
        out = self.conv(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
