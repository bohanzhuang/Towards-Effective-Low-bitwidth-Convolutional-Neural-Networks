import math
import torch
import torch.nn as nn
import torch.nn.init as init
from .margin_linear import MarginLinear

__all__ = ["SphereNIN"]


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class SphereBlock(nn.Module):

    def __init__(self, in_plane, out_plane, stride=1, kernel_size=3, pad=1):
        super(SphereBlock, self).__init__()
        self.conv_1 = nn.Conv2d(
            in_plane, out_plane, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)
        self.relu_1 = nn.ReLU(inplace=True)
        self.conv_2 = nn.Conv2d(out_plane, out_plane,
                                kernel_size=1, stride=1, padding=0, bias=False)
        self.relu_2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv_1(x)
        out = self.relu_1(out)
        out = self.conv_2(out)
        out = self.relu_2(out)
        return out


class SphereNIN(nn.Module):
    """SphereNIN class

    Note: Input must be 112x96
    """

    def __init__(self, num_output=10572, num_features=512, margin_inner_product_type='quadruple'):
        super(SphereNIN, self).__init__()

        block = SphereBlock
        self.layer1 = block(in_plane=3, out_plane=32,
                               kernel_size=5, pad=2, stride=2)
        self.max_pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer2 = block(in_plane=32, out_plane=96,
                               kernel_size=3, pad=1, stride=2)
        self.layer3 = block(in_plane=96, out_plane=128,
                               kernel_size=3, pad=1, stride=2)
        self.layer4 = block(in_plane=128, out_plane=128,
                               kernel_size=3, pad=1, stride=1)
        self.layer5 = block(in_plane=128, out_plane=128,
                               kernel_size=3, pad=1, stride=1)

        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3,
                      padding=1, stride=2, bias=False),
            nn.Conv2d(128, 256, kernel_size=(4,3),
                      padding=0, stride=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=1,
                      padding=0, stride=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_features, kernel_size=1,
                      padding=0, stride=1, bias=False),
        )

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
        self.margin_linear = MarginLinear(num_output=num_output,
                                          num_features=num_features,
                                          margin_inner_product_type=margin_inner_product_type)

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, target=None):
        x = self.layer1(x)
        x = self.max_pooling(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = x.view(x.size(0), -1)

        if target is not None:
            # print "enter margin linear"
            x = self.margin_linear(x, target)
        return x
