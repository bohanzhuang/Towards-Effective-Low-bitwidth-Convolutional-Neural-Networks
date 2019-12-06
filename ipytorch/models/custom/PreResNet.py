import torch
import torch.nn as nn
from .common import conv1x1, conv3x3
import math

# ---------------------------Small Data Sets Like CIFAR-10 or CIFAR-100----------------------------
# both-preact | half-preact | no-preact
class PreResidualBlock(nn.Module):
    """
    base module for PreResNet on small data sets
    """
    def __init__(self, in_plane, out_plane, stride=1, down_sample=None, block_type="both-preact"):
        """
        init module and weights
        :param in_plane: size of input plane
        :param out_plane: size of output plane
        :param stride: stride of convolutional layers, default 1
        :param down_sample: down sample type for expand dimension of input feature maps, default None
        :param block_type: type of blocks, decide position of short cut, both-preact: short cut start from beginning
        of the first segment, half-preact: short cut start from the position between the first segment and the second
        one. no-preact: short cut start from beginning of second segment and remove first segment. default: both-preact
        """
        super(PreResidualBlock, self).__init__()
        self.block_type = block_type
        self.down_sample = down_sample

        self.first_segment = nn.Sequential(
            nn.BatchNorm2d(in_plane),
            nn.ReLU(inplace=True)
        )
        self.second_segment = nn.Sequential(
            conv3x3(in_plane, out_plane, stride),
            nn.BatchNorm2d(out_plane),
            nn.ReLU(inplace=True),
            conv3x3(out_plane, out_plane)
        )

    def forward(self, x):
        """
        forward procedure of residual module
        :param x: input feature maps
        :return: output feature maps
        """
        if self.block_type == "half-preact":
            out = self.first_segment(x)
            residual = out
            out = self.second_segment(out)
        elif self.block_type == "no-preact":
            residual = x
            out = self.second_segment(x)
        else:
            residual = x
            out = self.first_segment(x)
            out = self.second_segment(out)

        if self.down_sample:
            residual = self.down_sample(residual)
        out = out + residual
        return out


class PreResNet(nn.Module):
    """
    define PreResNet on small data sets
    """
    def __init__(self, depth, wide_factor=1, num_classes=10):
        """
        init model and weights
        :param depth: depth of network
        :param wide_factor: wide factor for deciding width of network, default is 1
        :param num_classes: number of classes, related to labels. default 10
        """
        super(PreResNet, self).__init__()

        block = PreResidualBlock
        self.in_plane = 16 * wide_factor
        self.depth = depth
        n = (depth - 2) / 6
        self.conv = conv3x3(3, 16 * wide_factor)
        self.module_1 = self.make_layer(block, 16 * wide_factor, n)
        self.module_2 = self.make_layer(block, 32 * wide_factor, n, 2)
        self.module_3 = self.make_layer(block, 64 * wide_factor, n, 2)
        self.bn = nn.BatchNorm2d(64 * wide_factor)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(7)
        self.fc = nn.Linear(64 * wide_factor, num_classes)
        
        self._init_weight()

    def _init_weight(self):
        # init layer parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def make_layer(self, block, out_plane, n_blocks, stride=1):
        """
        make residual blocks, including short cut and residual function
        :param block: type of basic block to build network
        :param out_plane: size of output plane
        :param n_blocks: number of blocks on every segment
        :param stride: stride of convolutional neural network, default 1
        :return: residual blocks
        """
        down_sample = None
        if (stride != 1) or (self.in_plane != out_plane):
            down_sample = nn.Sequential(
                conv1x1(self.in_plane, out_plane, stride=stride))

        layers = []
        layers.append(block(self.in_plane, out_plane, stride, down_sample, block_type="half-preact"))
        self.in_plane = out_plane
        for i in range(1, n_blocks):
            layers.append(block(out_plane, out_plane))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        forward procedure of model
        :param x: input feature maps
        :return: output feature maps
        """
        out = self.conv(x)
        out = self.module_1(out)
        out = self.module_2(out)
        out = self.module_3(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# ---------------------------ImageNet----------------------------------------------------------
# both-preact | half-preact | no-preact
class PreBottleNeck(nn.Module):
    """
    basic module for ImageNet PreResNet
    """
    def __init__(self, in_plane, out_plane, stride=1, down_sample=None, block_type="both-preact"):
        """
        init module and weights
        :param in_plane: size of input plane
        :param out_plane: size of output plane
        :param stride: stride of convolutional layers, default 1
        :param down_sample: down sample type for expand dimension of input feature maps, default None
        :param block_type: type of blocks, decide position of short cut, both-preact: short cut start from beginning
        of the first segment, half-preact: short cut start from the position between the first segment and the second
        one. no-preact: short cut start from beginning of second segment and remove first segment. default: both-preact
        """
        super(PreBottleNeck, self).__init__()
        self.block_type = block_type
        self.down_sample = down_sample

        inner_outplane = out_plane//4
        if block_type != "no-preact":
            self.first_segment = nn.Sequential(
                nn.BatchNorm2d(in_plane),
                nn.ReLU(inplace=True)
            )
        self.second_segment = nn.Sequential(
            conv1x1(in_plane, inner_outplane, stride),
            nn.BatchNorm2d(inner_outplane),
            nn.ReLU(inplace=True),
            conv3x3(inner_outplane, inner_outplane),
            nn.BatchNorm2d(inner_outplane),
            nn.ReLU(inplace=True),
            conv1x1(inner_outplane, out_plane)
        )

    def forward(self, x):
        """
        forward procedure of module
        :param x: input feature maps
        :return: output feature maps
        """
        if self.block_type == "half-preact":
            out = self.first_segment(x)
            residual = out
            out = self.second_segment(out)
        elif self.block_type == "no-preact":
            residual = x
            out = self.second_segment(x)
        else:
            residual = x
            out = self.first_segment(x)
            out = self.second_segment(out)
        if self.down_sample:
            residual = self.down_sample(residual)
        out = out + residual
        return out

