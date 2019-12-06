# this code is writen by liujing
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from .margin_linear import MarginLinear
# from margin_linear_old import MarginLinear

__all__ = ["SphereMobleNet_v2"]


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes):
    "1x1 convolution"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)


def dwconv3x3(in_planes, out_planes, stride=1):
    "3x3 depth wise convolution"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=in_planes, bias=False)


class MobileBottleneck(nn.Module):

    def __init__(self, in_planes, out_planes, expand=1, stride=1):
        super(MobileBottleneck, self).__init__()
        self.name = "mobile-bottleneck"

        intermedia_planes = in_planes * expand
        self.conv1 = conv1x1(in_planes, intermedia_planes)
        self.bn1 = nn.BatchNorm2d(intermedia_planes)
        # self.relu1 = nn.ReLU6(inplace=True)
        self.relu1 = nn.PReLU(in_planes * expand)

        self.conv2 = dwconv3x3(
            intermedia_planes, intermedia_planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(intermedia_planes)
        # self.relu2 = nn.ReLU6(inplace=True)
        self.relu2 = nn.PReLU(in_planes * expand)

        self.conv3 = conv1x1(intermedia_planes, out_planes)
        self.bn3 = nn.BatchNorm2d(out_planes)
        # self.relu3 = nn.PReLU(out_planes)

        self.shortcut = (stride == 1)
        self.block_index = 0

        if in_planes != out_planes and self.shortcut:
            self.downsample = nn.Sequential(
                conv1x1(in_planes, out_planes),
                nn.BatchNorm2d(out_planes)
            )
        else:
            self.downsample = None

    def forward(self, x):
        if self.shortcut:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.shortcut:
            if self.downsample is not None:
                out += self.downsample(residual)
            else:
                out += residual
        # out = self.relu3(out)

        return out


class SphereMobleNet_v2(nn.Module):
    """SphereNet class
    Note: Input must be 112x96
    """

    def __init__(self, num_output=10572, num_features=512,
                 wide_scale=1.0,
                 margin_inner_product_type='quadruple'):
        super(SphereMobleNet_v2, self).__init__()
        """
        if depth == 4:
            layers = [0, 0, 0, 0]
        elif depth == 10:
            layers = [0, 1, 2, 0]
        elif depth == 20:
            layers = [1, 2, 4, 1]
        elif depth == 38:
            layers = [2, 4, 8, 2]
        elif depth == 64:
            layers = [3, 8, 16, 3]
        else:
            assert False, "invalid depth: %d, only support: 4, 10, 20, 38, 64" % depth
        """
        self.base = 1000.
        # self.gamma = 0.000003
        # self.power = 45
        self.gamma = 0.12
        self.power = 1.

        # self.depth = depth
        block = MobileBottleneck
        # define network structure
        self.layer_width = np.array([32, 16, 24, 32, 64, 96, 160, 320])
        self.layer_width = np.around(self.layer_width * wide_scale)
        self.layer_width = self.layer_width.astype(int)

        self.in_planes = self.layer_width[0]
        self.conv1 = conv3x3(3, self.in_planes, stride=2)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        # self.relu1 = nn.ReLU6(inplace=True)
        self.relu1 = nn.PReLU(self.in_planes)
        self.layer1 = self._make_layer(block, self.layer_width[1], blocks=1)
        self.layer2 = self._make_layer(
            block, self.layer_width[2], blocks=2, expand=6, stride=2)
        self.layer3 = self._make_layer(
            block, self.layer_width[3], blocks=3, expand=6, stride=2)
        self.layer4 = self._make_layer(
            block, self.layer_width[4], blocks=4, expand=6, stride=2)
        self.layer5 = self._make_layer(
            block, self.layer_width[5], blocks=3, expand=6, stride=1)
        self.layer6 = self._make_layer(
            block, self.layer_width[6], blocks=3, expand=6, stride=2)
        self.layer7 = self._make_layer(
            block, self.layer_width[7], blocks=1, expand=6)

        self.conv2 = conv1x1(
            in_planes=self.layer_width[7], out_planes=1280)
        self.bn2 = nn.BatchNorm2d(1280)
        # self.relu2 = nn.ReLU6(inplace=True)
        self.relu2 = nn.PReLU(1280)
        # self.avgpool = nn.AvgPool2d((4, 3))
        # self.dropout = nn.Dropout()
        # self.conv3 = conv1x1(in_planes=1280, out_planes=num_features)
        self.fc = nn.Linear(1280 * 4 * 3, num_features)

        self.margin_inner_product_type = margin_inner_product_type
        if margin_inner_product_type == 'single':
            margin_inner_product_type = 1
        elif margin_inner_product_type == 'double':
            margin_inner_product_type = 2
        elif margin_inner_product_type == 'triple':
            margin_inner_product_type = 3
        elif margin_inner_product_type == 'quadruple':
            margin_inner_product_type = 4
        else:
            print("Unknown margin type.")
        self.margin_linear = MarginLinear(
            num_output=num_output,
            num_features=num_features,
            margin_inner_product_type=margin_inner_product_type)

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                init.xavier_normal(m.weight)
                if m.bias is not None:
                    init.constant(m.bias, 0)

    def _make_layer(self, block, out_planes, blocks, expand=1, stride=1):
        layers = []
        layers.append(block(self.in_planes, out_planes,
                            expand=expand, stride=stride))
        self.in_planes = out_planes
        for i in range(1, blocks):
            layers.append(block(self.in_planes, out_planes, expand=expand))
            self.in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x, target=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        # x = self.avgpool(x)
        # x = self.dropout(x)
        # x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        if target is not None:
            # print "enter margin linear"
            x = self.margin_linear(x, target)
        return x
