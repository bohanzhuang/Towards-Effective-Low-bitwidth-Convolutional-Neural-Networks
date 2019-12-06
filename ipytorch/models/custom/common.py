import torch.nn as nn
import math
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import functional as F
import random


def model2list(model):
    """ 
    convert model to list type
    :param model: should be type of list or nn.DataParallel or nn.Sequential
    :return: no return params
    """
    if isinstance(model, nn.DataParallel):
        model = list(model.module)
    elif isinstance(model, nn.Sequential):
        model = list(model)
    return model


def initweights(layer):
    """ 
    init weights on neural networks, usage: model.applay(initweights)
    :param m: <nn.Module> target model
    :return: no return parameters
    """
    orthogonal_flag = False
    # for layer in m.modules():
    if isinstance(layer, nn.Conv2d) or isinstance(layer, qConv2d):
        n = layer.kernel_size[0] * layer.kernel_size[1] * layer.out_channels
        layer.weight.data.normal_(0, math.sqrt(2. / n))

        # orthogonal initialize
        """Reference:
        [1] Saxe, Andrew M., James L. McClelland, and Surya Ganguli.
           "Exact solutions to the nonlinear dynamics of learning in deep
           linear neural networks." arXiv preprint arXiv:1312.6120 (2013)."""
        if orthogonal_flag:
            weight_shape = layer.weight.data.cpu().numpy().shape
            u, _, v = np.linalg.svd(
                layer.weight.data.cpu().numpy(), full_matrices=False)
            flat_shape = (weight_shape[0], np.prod(weight_shape[1:]))
            q = u if u.shape == flat_shape else v
            q = q.reshape(weight_shape)
            layer.weight.data.copy_(torch.Tensor(q))

    elif isinstance(layer, nn.BatchNorm2d):
        layer.weight.data.fill_(1)
        layer.bias.data.zero_()
    elif isinstance(layer, nn.Linear) or isinstance(layer, qLinear):
        layer.bias.data.zero_()


def conv3x3(in_plane, out_plane, stride=1, padding=1):
    """
    3x3 convolutional layer
    :param in_plane: size of input plane
    :param out_plane: size of output plane
    :param stride: default 1
    :param padding: default 1
    :return: <nn.Module> convolutional layer with kernel size of 3
    """
    "3x3 convolutional layer with padding"
    return nn.Conv2d(in_plane, out_plane, kernel_size=3, stride=stride, padding=padding, bias=False)
    # return qConv2d(in_plane, out_plane, kernel_size=3, stride=stride, padding=padding, bias=False)


def conv1x1(in_plane, out_plane, stride=1):
    """
    1x1 convolutional layer
    :param in_plane: size of input plane
    :param out_plane: size of output plane
    :param stride: default 1
    :return: <nn.Module> convolutional layer with kernel size of 1
    """
    "3x3 convolutional layer with padding"
    return nn.Conv2d(in_plane, out_plane, kernel_size=1, stride=stride, padding=0, bias=False)
    # return qConv2d(in_plane, out_plane, kernel_size=1, stride=stride, padding=0, bias=False)


class ShortCut(nn.Module):
    """
    define short cut
    """

    def __init__(self, in_plane, out_plane):
        """
        init module, if in_plane != out_plane, then a conv1x1 mapping will be added
        :param in_plane: size of input plane 
        :param out_plane: size of output plane
        """
        super(ShortCut, self).__init__()
        self.in_plane = in_plane
        self.out_plane = out_plane
        if in_plane != out_plane:
            self.short_cut = conv1x1(in_plane, out_plane, out_plane / in_plane)

        # init layer parameters
        self.apply(initweights)

    def forward(self, x):
        """
        forward procedure of the module
        :param x: input feature maps
        :return: output feature maps
        """
        if self.in_plane == self.out_plane:
            return x
        else:
            out = self.short_cut(x)
            return out


class GreedyFC(nn.Module):
    """
    common classifier for all GreedyNet
    """

    def __init__(self, in_plane, avg_size=8, fc_type="BN", num_classes=10):
        """
        init module
        :param in_plane: size of input plane
        :param avg_size: dimension of input feature maps
        :param fc_type: type of classifier, BN means the classifier has BN+ReLU layer, 
        otherwise only use average pooling
        :param num_classes: number of output classes, related to number of labels 
        """
        super(GreedyFC, self).__init__()
        self.fc_type = fc_type
        if self.fc_type == "BN":
            self.bn = nn.BatchNorm2d(in_plane)
            self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(avg_size)
        # self.fc = qLinear(in_plane, num_classes)
        self.fc = nn.Linear(in_plane, num_classes)

        # init layer parameters
        self.apply(initweights)

    def forward(self, x):
        """
        forward procedure of the module
        :param x: input feature maps
        :return: output feature maps
        """
        if self.fc_type == "BN":
            out = self.bn(x)
            out = self.relu(out)
            out = self.avg_pool(out)
        else:
            out = self.avg_pool(x)
        out = out.view(out.size(0), -1)
        # feature = out
        out = self.fc(out)
        return out


def getplanesize(i, depth):
    """
    function for decide input plane and output plane of residual network
    :param i: position of target block
    :param depth: depth of network
    :return: size of input plane and output plane
    """
    key_pivot = np.array([1, 2]) * ((depth - 2) / 6)
    if i <= key_pivot[0]:
        in_plane = 16
    elif i <= key_pivot[1]:
        in_plane = 32
    else:
        in_plane = 64
        # in_plane = 32  # 64

    if i < key_pivot[0]:
        out_plane = 16
    elif i < key_pivot[1]:
        out_plane = 32
    else:
        out_plane = 64
        # out_plane = 32  # 64
    return in_plane, out_plane
