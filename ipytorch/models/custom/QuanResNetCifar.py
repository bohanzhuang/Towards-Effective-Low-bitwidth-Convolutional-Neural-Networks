import math

import torch.nn as nn

from .common import conv1x1, conv3x3


class QuanResidualBlockCifar(nn.Module):
    """
    basic block for Residual network
    """

    def __init__(self, in_plane, out_plane, stride=1, down_sample=None, is_last=False):
        """
        init model and weights
        :param in_plane: size of input plane
        :param out_plane: size of output plane
        :param stride: stride of convolutional layers, default 1
        :param down_sample: type of down sample module for expending dimension of input feature maps, default None
        """
        super(QuanResidualBlockCifar, self).__init__()
        self.down_sample = down_sample
        self.is_last = is_last

        self.conv1 = conv3x3(in_plane, out_plane, stride)
        self.bn1 = nn.BatchNorm2d(out_plane)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(out_plane, out_plane)
        self.bn2 = nn.BatchNorm2d(out_plane)
        self.relu2 = nn.ReLU(inplace=True)

        # self.segment = nn.Sequential(
        #     conv3x3(in_plane, out_plane, stride),
        #     nn.BatchNorm2d(out_plane),
        #     nn.ReLU(inplace=True),
        #     conv3x3(out_plane, out_plane),
        #     nn.BatchNorm2d(out_plane)
        # )
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        forward procedure of module
        :param x: input feature maps
        :return: output feature maps
        """
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.down_sample:
            residual = self.down_sample(x)
        out += residual
        out = self.relu2(out)
        return out


# ---------------------------Small Data Sets Like CIFAR-10 or CIFAR-100----------------------------
class QuanResNetCifar(nn.Module):
    """
    ResNet on small data sets
    """

    def __init__(self, depth, wide_factor=1, num_classes=10):
        """
        init residual model and weights
        :param depth: depth of model
        :param wide_factor: wide factor for deciding number of filters on convolutional layers
        :param num_classes: number of classes, related to labels
        """
        super(QuanResNetCifar, self).__init__()

        block = QuanResidualBlockCifar
        self.in_plane = 16 * wide_factor
        self.depth = depth
        n = (depth - 2) / 6
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.module_1 = self.make_layer(block, 16 * wide_factor, n)
        self.module_2 = self.make_layer(block, 32 * wide_factor, n, 2)
        self.module_3 = self.make_layer(block, 64 * wide_factor, n, 2, is_last=True)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * wide_factor, num_classes)
        # self.scalar = torch.nn.Parameter(torch.Tensor([0.01]))

        self.init_weight()

    def init_weight(self):
        # init layer parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def make_layer(self, block, out_plane, n_blocks, stride=1, is_last=False):
        """
        make basic residual block including short cut and residual function
        :param block: basic block to build model
        :param out_plane: size of output plane
        :param n_blocks: number of blocks on every segment
        :param stride: stride of convolutional layers, default 1
        :return: residual blocks
        """
        down_sample = None
        if (stride != 1) or (self.in_plane != out_plane):
            down_sample = nn.Sequential(
                conv1x1(self.in_plane, out_plane, stride=stride),
                nn.BatchNorm2d(out_plane))

        layers = []
        layers.append(block(self.in_plane, out_plane, stride, down_sample))
        self.in_plane = out_plane
        for i in range(1, int(n_blocks)):
            if i == n_blocks - 1:
                layers.append(block(out_plane, out_plane, is_last=is_last))
            else:
                layers.append(block(out_plane, out_plane))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        forward procedure of module
        :param x: input feature maps
        :return: output feature maps
        """
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.module_1(out)
        out = self.module_2(out)
        out = self.module_3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        # out = out * self.scalar
        return out


# ---------------------------ImageNet----------------------------------------------------------
class QuanBottleNeckCifar(nn.Module):
    """
    basic block for ResNet on ImageNet
    """

    def __init__(self, in_plane, out_plane, stride=1, downsample=None):
        """
        init module and weights
        :param in_plane: size of input plane
        :param out_plane: size of output plane
        :param stride: stride of convolutional layers, default 1
        :param downsample: type of down sample blocks for expending dimension of input feature maps
        """
        super(QuanBottleNeckCifar, self).__init__()
        self.downsample = downsample
        self.stride = stride
        inner_output = out_plane // 4
        self.segment = nn.Sequential(
            conv1x1(in_plane, inner_output),
            nn.BatchNorm2d(inner_output),
            nn.ReLU(inplace=True),
            conv3x3(inner_output, inner_output, stride),
            nn.BatchNorm2d(inner_output),
            nn.ReLU(inplace=True),
            conv1x1(inner_output, out_plane),
            nn.BatchNorm2d(out_plane),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        forward procedure of module
        :param x: input feature maps
        :return: output feature maps
        """
        residual = x
        out = self.segment(x)
        if self.downsample is not None:
            residual = self.downsample(residual)
        out += residual
        out = self.relu(out)
        return out
