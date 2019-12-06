import torch
import torch.nn as nn
import math
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import Variable
__all__ = ['QuanPreResNet', 'QuanPreBasicBlock']

# ---------------------------Small Data Sets Like CIFAR-10 or CIFAR-100----------------------------

def conv1x1(in_plane, out_plane, stride=1):
    """
    1x1 convolutional layer
    """
    return nn.Conv2d(in_plane, out_plane,
                     kernel_size=1, stride=stride, padding=0, bias=False)


def forward_conv1x1(input, weight, stride=1):
    return F.conv2d(input, weight, bias=None, stride=stride, padding=0)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def forward_conv3x3(input, weight, stride=1):
    return F.conv2d(input, weight, bias=None, stride=stride, padding=1)


def linear(in_features, out_features):
    return nn.Linear(in_features, out_features)


def forward_linear(input, weight, bias):
    return F.linear(input, weight, bias)


class QuantizationFunction(Function):

    @staticmethod
    def forward(ctx, input, k):
        if input.is_cuda:
            k = k.cuda()
        pow2_k_1 = 2 ** k - 1
        output = torch.round(pow2_k_1 * input) / pow2_k_1
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        # straight-through estimator (STE)
        return grad_output, None


def normalization_on_weights(input):
    # output = (input + 1) / 2
    tanh_data = nn.functional.tanh(input)
    output = tanh_data / torch.max(torch.abs(tanh_data)) * 0.5 + 0.5
    return output


def normalization_on_activations(input):
    return torch.clamp(input, 0, 1)


def quantization_on_activations(input, k):
    # quantize input to k-bit
    output = normalization_on_activations(input)
    output = QuantizationFunction.apply(output, k)
    return output


def quantization_on_weights(input, k):
    # quantize data to k-bit
    output = normalization_on_weights(input)
    # after quantization, normalize weight to -1 to 1
    output = 2 * QuantizationFunction.apply(output, k) - 1
    return output


class QuanPreBasicBlock(nn.Module):
    """
    base module for PreResNet on small data sets
    """

    def __init__(self, in_plane, out_plane, k, stride=1, downsample=None, block_type="both_preact"):
        """
        init module and weights
        :param in_plane: size of input plane
        :param out_plane: size of output plane
        :param k: k-bit quantization
        :param stride: stride of convolutional layers, default 1
        :param downsample: down sample type for expand dimension of input feature maps, default None
        :param block_type: type of blocks, decide position of short cut, both-preact: short cut start from beginning
        of the first segment, half-preact: short cut start from the position between the first segment and the second
        one. default: both-preact
        """
        super(QuanPreBasicBlock, self).__init__()
        self.name = block_type
        self.downsample = downsample
        self.k = k

        self.bn1 = nn.BatchNorm2d(in_plane)
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = conv3x3(in_plane, out_plane, stride)
        self.bn2 = nn.BatchNorm2d(out_plane)
        self.conv2 = conv3x3(out_plane, out_plane)
        self.block_index = 0

    def forward(self, x):
        """
        forward procedure of residual module
        :param x: input feature maps
        :return: output feature maps
        """
        if self.name == "half_preact":
            x = self.bn1(x)
            x = self.relu(x)
            x = quantization_on_activations(x, self.k)
            residual = x
            conv1_quan_weight = quantization_on_weights(self.conv1.weight, self.k)
            x = forward_conv3x3(x, conv1_quan_weight, self.conv1.stride)
            x = self.bn2(x)
            x = self.relu(x)
            x = quantization_on_activations(x, self.k)
            conv2_quan_weight = quantization_on_weights(self.conv2.weight, self.k)
            x = forward_conv3x3(x, conv2_quan_weight, self.conv2.stride)
        elif self.name == "both_preact":
            residual = x
            x = self.bn1(x)
            x = self.relu(x)
            x = quantization_on_activations(x, self.k)
            conv1_quan_weight = quantization_on_weights(self.conv1.weight, self.k)
            x = forward_conv3x3(x, conv1_quan_weight, self.conv1.stride)
            x = self.bn2(x)
            x = self.relu(x)
            x = quantization_on_activations(x, self.k)
            conv2_quan_weight = quantization_on_weights(self.conv2.weight, self.k)
            x = forward_conv3x3(x, conv2_quan_weight, self.conv2.stride)

        if self.downsample:
            down_sample_weight = quantization_on_weights(self.downsample.weight, self.k)
            residual = forward_conv1x1(residual, down_sample_weight, self.downsample.stride)

        out = x + residual
        return out


class QuanPreResNet(nn.Module):
    """
    define QuanPreResNet on small data sets
    """

    def __init__(self, depth, k, wide_factor=1, num_classes=10):
        """
        init model and weights
        :param depth: depth of network
        :param wide_factor: wide factor for deciding width of network, default is 1
        :param num_classes: number of classes, related to labels. default 10
        :ori_model: original model
        :k: k-bit quantization
        """
        super(QuanPreResNet, self).__init__()
        self.k = Variable(torch.ones(1), requires_grad=False)
        self.k.data.fill_(k)

        self.in_plane = 16 * wide_factor
        self.depth = depth
        n = (depth - 2) / 6
        self.conv = conv3x3(3, 16 * wide_factor)
        self.layer1 = self._make_layer(QuanPreBasicBlock, 16 * wide_factor, n)
        self.layer2 = self._make_layer(
            QuanPreBasicBlock, 32 * wide_factor, n, stride=2)
        self.layer3 = self._make_layer(
            QuanPreBasicBlock, 64 * wide_factor, n, stride=2)
        self.bn = nn.BatchNorm2d(64 * wide_factor)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(7)
        self.fc = linear(64 * wide_factor, num_classes)
        self.scalar = Parameter(torch.Tensor([0.01]))

    def init_weight(self, ori_model):
        ori_model_dict = ori_model.state_dict()
        quan_model_dict = self.state_dict()
        for k, _ in list(quan_model_dict.items()):
            if k in ori_model_dict:
                quan_model_dict[k] = ori_model_dict[k]
        self.load_state_dict(quan_model_dict)

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
            downsample = conv1x1(self.in_plane, out_plane, stride=stride)

        layers = []
        layers.append(block(self.in_plane, out_plane, self.k, stride,
                            downsample, block_type="half_preact"))
        self.in_plane = out_plane
        for i in range(1, n_blocks):
            layers.append(block(self.in_plane, out_plane, self.k))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        forward procedure of model
        :param x: input feature maps
        :return: output feature maps
        """
        conv_quan_weight = quantization_on_weights(self.conv.weight, self.k)
        out = forward_conv3x3(x, conv_quan_weight, self.conv.stride)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.bn(out)
        out = self.relu(out)
        out = quantization_on_activations(out, self.k)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        fc_quan_weight = quantization_on_weights(self.fc.weight, self.k)
        fc_quan_bias = quantization_on_weights(self.fc.bias, self.k)
        out = forward_linear(out, fc_quan_weight, fc_quan_bias)
        out = out * self.scalar
        return out
