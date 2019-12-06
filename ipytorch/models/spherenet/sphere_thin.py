# this code is writen by liujing
import math
import torch
import torch.nn as nn
import torch.nn.init as init
from .margin_linear import MarginLinear

__all__ = ["SphereNet"]


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1)


class SphereBlock(nn.Module):

    def __init__(self, planes):
        super(SphereBlock, self).__init__()
        self.conv1 = conv3x3(planes, planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)

        self._init_weight()

    def _init_weight(self):
        # init conv1
        init.normal(self.conv1.weight, std=0.01)
        init.constant(self.conv1.bias, 0)
        # init conv2
        init.normal(self.conv2.weight, std=0.01)
        init.constant(self.conv2.bias, 0)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        out += residual
        return out


class SphereNet(nn.Module):
    """SphereNet class

    Note: Input must be 112x96
    """

    def __init__(self, depth, num_output=10572, num_features=512, margin_inner_product_type='quadruple'):
        super(SphereNet, self).__init__()
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

        self.base = 1000
        # self.gamma = 0.000003
        # self.power = 45
        self.gamma = 0.12
        self.power = 1

        self.depth = depth
        block = SphereBlock
        # define network structure
        self.conv1 = nn.Conv2d(3, 43, kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.layer1 = self._make_layer(block, 64, layers[0])

        self.conv2 = nn.Conv2d(43, 119, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        
        self.conv3 = nn.Conv2d(119, 253, kernel_size=3, stride=2, padding=1)
        self.relu3 = nn.ReLU()
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        
        self.conv4 = nn.Conv2d(253, 512, kernel_size=3, stride=2, padding=1)
        self.relu4 = nn.ReLU()
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.pooling = nn.MaxPool2d(kernel_size=(7, 6))
        self.fc = nn.Linear(512 * 7 * 6, num_features)
        # self.fc = nn.Linear(512, num_features)

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
            print('Unknown margin type.')
        self.margin_linear = MarginLinear(base=self.base,
                                          gamma=self.gamma,
                                          power=self.power,
                                          num_output=num_output,
                                          num_features=num_features,
                                          margin_inner_product_type=margin_inner_product_type)
        self._init_weight()

    def _init_weight(self):
        # init conv1
        init.xavier_normal(self.conv1.weight)
        # init.kaiming_normal(self.conv1.weight)
        init.constant(self.conv1.bias, 0)
        # init conv2
        init.xavier_normal(self.conv2.weight)
        # init.kaiming_normal(self.conv2.weight)
        init.constant(self.conv2.bias, 0)
        # init conv3
        init.xavier_normal(self.conv3.weight)
        # init.kaiming_normal(self.conv3.weight)
        init.constant(self.conv3.bias, 0)
        # init conv4
        init.xavier_normal(self.conv4.weight)
        # init.kaiming_normal(self.conv4.weight)
        init.constant(self.conv4.bias, 0)
        # init fc
        init.xavier_normal(self.fc.weight)
        # init.kaiming_normal(self.fc.weight)
        init.constant(self.fc.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        for i in range(blocks):
            layers.append(block(planes))
        return nn.Sequential(*layers)

    def forward(self, x, target=None):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.layer1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.layer2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.layer3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.layer4(x)
        # x = self.pooling(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if target is not None:
            x = self.margin_linear(x, target)
        return x
