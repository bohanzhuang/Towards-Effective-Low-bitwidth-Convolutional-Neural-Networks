'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import random
import math

__all__ = ['dsVGG_CIFAR', 'dsVGGBlock']

cfg = {
    'dsVGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'dsVGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'dsVGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'dsVGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class dsVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=None, cRate=1.0):
        super(dsVGGBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.d = Parameter(torch.Tensor(1).fill_(cRate))
        self.d_p = Parameter(torch.Tensor(1).fill_(cRate))
        # self.d = 1.0


    def forward(self, x):    
        # if d is not 0, then keep the original block, otherwise, replace it by a short-cut
        residual = x
        out = self.conv(x)
        out = self.bn(out)
        out = out*self.d
        if self.d.data[0] == 0:
            if self.downsample is not None:
                residual = self.downsample(residual)
            out = out + residual
        out = self.relu(out)
        return out

class dsVGG_CIFAR(nn.Module):
    def __init__(self, depth, n_class=10, cRate=1.0):
        super(dsVGG_CIFAR, self).__init__()
        vgg_name = 'dsVGG'+str(depth)
        self.cRate = cRate
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, n_class)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if in_channels != x:
                    downsample = nn.Conv2d(in_channels, x, kernel_size=1, bias=False)
                else:
                    downsample = None
                layers += [dsVGGBlock(in_channels, x, downsample, cRate=self.cRate)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
