import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import numpy as np
import random
import torch.nn.functional as F
from .wc_v5 import wcConv2d, wcLinear, thinLinear, thinConv2d
from ipytorch.models import wcBasicBlock, wcTransitionBlock
__all__ = ['SeqModelPrune', 'ResModelPrune',
           'set_d_weight', 'SpherePrune', 'DenseNetCifarPrune']

# set_weight


def set_d_weight(model):
    for layer in model.modules():
        if isinstance(layer, wcConv2d):

            new_weight = layer.binary_weight.unsqueeze(0).unsqueeze(
                2).unsqueeze(3).expand_as(layer.weight) * layer.weight
            layer.weight.data.copy_(new_weight.data)
        elif isinstance(layer, wcLinear):
            new_weight = layer.binary_weight.unsqueeze(
                0).expand_as(layer.weight) * layer.weight
            layer.weight.data.copy_(new_weight.data)


# code for adaptive filter pruning
def get_select_channels(d, h=1, w=1):
    d_np = d.data.cpu().numpy()
    d_np = np.reshape(d_np, (h * w, -1))
    d_sum = d_np.sum(0)
    select_channels = np.where(d_sum > 0)
    select_channels = torch.LongTensor(select_channels).cuda()
    select_channels = select_channels.squeeze()
    # print select_channels.size()
    return select_channels


# get thin parameters form old layer
def get_thin_params(layer, select_channels, dim=0, h=1, w=1):
    if isinstance(layer, nn.Conv2d) or isinstance(layer, wcConv2d):
        # print "get thin params"
        thin_weight = layer.weight.data.index_select(
            dim, select_channels)
        if layer.bias is not None:
            if dim == 0:
                thin_bias = layer.bias.data.index_select(
                    dim, select_channels)
            else:
                thin_bias = layer.bias.data
        else:
            thin_bias = None

    elif isinstance(layer, nn.Linear) or isinstance(layer, wcLinear):
        if dim == 0:
            thin_weight = layer.weight.data.index_select(
                dim, select_channels)
            if layer.bias is not None:
                thin_bias = layer.bias.data.index_select(
                    dim, select_channels)
            else:
                thin_bias = None
        else:
            fat_weight = layer.weight.data.view(layer.weight.size(
                0), layer.weight.size(1) / (h * w), h, w)
            thin_weight = fat_weight.index_select(
                dim, select_channels)
            thin_weight = thin_weight.view(layer.weight.size(0), -1)
            if layer.bias is not None:
                thin_bias = layer.bias.data
            else:
                thin_bias = None

    elif isinstance(layer, nn.BatchNorm2d):
        assert dim == 0, "invalid dimension for bn_layer"

        thin_weight = layer.weight.data.index_select(
            dim, select_channels)
        thin_mean = layer.running_mean.index_select(
            dim, select_channels
        )
        thin_var = layer.running_var.index_select(
            dim, select_channels
        )
        if layer.bias is not None:
            thin_bias = layer.bias.data.index_select(
                dim, select_channels)
        else:
            thin_bias = None
        return (thin_weight, thin_mean), (thin_bias, thin_var)
    elif isinstance(layer, nn.PReLU):
        thin_weight = layer.weight.data.index_select(dim, select_channels)
        thin_bias = None

    return thin_weight, thin_bias


# --------------------------------------------------------------------------------------------------
def replace_segment(model, index, init_weight, init_bias=None):
    """
    replace specific layer of model
    :params model: original model
    :params index: index of the layer to be replaced
    :params init_weight: thin_weight get from FilterPrune.get_thin_params
    :params init_bias: thin_bias get from FilterPrune.get_thin_params
    :returns <nn.Sequential> new model
    """

    old_layer = model[index]
    new_model = []
    for i, _ in enumerate(model):
        if i == index:
            new_layer = replace_layer(old_layer, init_weight, init_bias)
            new_model.append(new_layer)
        else:
            new_model.append(model[i])
    return nn.Sequential(*new_model)


def replace_thin_layer(old_layer, init_weight, init_bias=None):
    """
    replace specific layer of model
    :params layer: original layer
    :params init_weight: thin_weight get from FilterPrune.get_thin_params
    :params init_bias: thin_bias get from FilterPrune.get_thin_params
    :returns new_layer
    """
    if old_layer.bias is not None:
        bias_flag = True
    else:
        bias_flag = False
    if isinstance(old_layer, wcConv2d):
        new_layer = thinConv2d(init_weight.size(1),
                               init_weight.size(0),
                               kernel_size=old_layer.kernel_size,
                               stride=old_layer.stride,
                               padding=old_layer.padding,
                               bias=bias_flag,
                               bw_size=old_layer.binary_weight.size(0))
        new_layer.binary_weight.data.copy_(old_layer.binary_weight.data)
        new_layer.weight.data.copy_(init_weight)
        if init_bias is not None:
            new_layer.bias.data.copy_(init_bias)

    elif isinstance(old_layer, wcLinear):
        new_layer = thinLinear(init_weight.size(1),
                               init_weight.size(0),
                               bias=bias_flag,
                               bw_size=old_layer.binary_weight.size(0))
        new_layer.binary_weight.data.copy_(old_layer.binary_weight.data)
        new_layer.weight.data.copy_(init_weight)
        if init_bias is not None:
            new_layer.bias.data.copy_(init_bias)

    else:

        assert False, "unsupport layer type:" + \
            str(type(old_layer))
    return new_layer


def replace_layer(old_layer, init_weight, init_bias=None, keeping=False):
    """
    replace specific layer of model
    :params layer: original layer
    :params init_weight: thin_weight get from FilterPrune.get_thin_params
    :params init_bias: thin_bias get from FilterPrune.get_thin_params
    :returns new_layer
    """
    if hasattr(old_layer, "bias") and old_layer.bias is not None:
        bias_flag = True
    else:
        bias_flag = False
    if isinstance(old_layer, wcConv2d) and keeping:
        new_layer = wcConv2d(init_weight.size(1),
                             init_weight.size(0),
                             kernel_size=old_layer.kernel_size,
                             stride=old_layer.stride,
                             padding=old_layer.padding,
                             bias=bias_flag,
                             rate=old_layer.rate[0])
        new_layer.weight.data.copy_(init_weight)
        if init_bias is not None:
            new_layer.bias.data.copy_(init_bias)
        new_layer.binary_weight.data.copy_(old_layer.binary_weight.data)
        new_layer.float_weight.data.copy_(old_layer.binary_weight.data)


    elif isinstance(old_layer, (nn.Conv2d, wcConv2d)):
        new_layer = nn.Conv2d(init_weight.size(1),
                              init_weight.size(0),
                              kernel_size=old_layer.kernel_size,
                              stride=old_layer.stride,
                              padding=old_layer.padding,
                              bias=bias_flag)
        new_layer.weight.data.copy_(init_weight)
        if init_bias is not None:
            new_layer.bias.data.copy_(init_bias)

    elif isinstance(old_layer, (nn.Linear, wcLinear)):

        new_layer = nn.Linear(init_weight.size(1),
                              init_weight.size(0),
                              bias=bias_flag)
        new_layer.weight.data.copy_(init_weight)
        if init_bias is not None:
            new_layer.bias.data.copy_(init_bias)

    elif isinstance(old_layer, nn.BatchNorm2d):
        weight = init_weight[0]
        mean_ = init_weight[1]
        bias = init_bias[0]
        var_ = init_bias[1]
        new_layer = nn.BatchNorm2d(weight.size(0))
        new_layer.weight.data.copy_(weight)
        assert init_bias is not None, "batch normalization needs bias"
        new_layer.bias.data.copy_(bias)
        new_layer.running_mean.copy_(mean_)
        new_layer.running_var.copy_(var_)
    elif isinstance(old_layer, nn.PReLU):
        new_layer = nn.PReLU(init_weight.size(0))
        new_layer.weight.data.copy_(init_weight)

    else:
        assert False, "unsupport layer type:" + \
            str(type(old_layer))
    return new_layer


# -----------------------------------------------------------------------------------------------------------------
class SeqPrune:
    def __init__(self, segment, h=1, w=1):
        """
        :params segment: target segment or sequential
        """
        self.segment = segment
        self.segment_length = None
        self.first_fc = True
        self.h = h
        self.w = w
        self.select_channels = None
        print("|===>init SegmentPrune")

        # get length of segment
        if isinstance(self.segment, nn.DataParallel):
            self.segment = list(self.segment.module)
        elif isinstance(self.segment, nn.Sequential):
            self.segment = list(self.segment)
        else:
            self.segment = [self.segment]
        self.segment_length = len(self.segment)
        self.segment = nn.Sequential(*self.segment)
        self.segment.cuda()

    def pruning(self):
        print("|===>pruning layers")
        for i in range(self.segment_length):
            if isinstance(self.segment[i], (wcConv2d, wcLinear)):
                if self.first_fc and isinstance(self.segment[i], wcLinear):
                    h = self.h
                    w = self.w
                    self.first_fc = False
                else:
                    h = 1
                    w = 1
                binary_weight = self.segment[i].binary_weight

                # compute selected channels
                select_channels = get_select_channels(binary_weight, h, w)
                if self.select_channels is not None:
                    self.select_channels = select_channels

                # replace current layer
                thin_weight, thin_bias = get_thin_params(
                    self.segment[i], select_channels, 1, h, w)
                self.segment = replace_segment(
                    self.segment, i, thin_weight, thin_bias)

                self.segment.cuda()

                for j in range(i - 1, -1, -1):
                    if isinstance(self.segment[j], (nn.Linear, nn.Conv2d, nn.BatchNorm2d, wcConv2d, wcLinear)):
                        thin_weight, thin_bias = get_thin_params(
                            self.segment[j], select_channels, 0)
                        self.segment = replace_segment(
                            self.segment, j, thin_weight, thin_bias)
                        self.segment.cuda()
                        if not isinstance(self.segment[j], nn.BatchNorm2d):
                            break

        print("|===>binary_weight segment channels select...")
        return select_channels

    def extra_prune(self, select_channels):
        print("|===>enter extra_prune")
        assert select_channels is not None, "select_channels is NoneType"

        for i in range(self.segment_length - 1, -1, -1):
            if isinstance(self.segment[i], (nn.Conv2d, wcConv2d, nn.BatchNorm2d)):
                thin_weight, thin_bias = get_thin_params(
                    self.segment[i], select_channels, 0)
                self.segment = replace_segment(
                    self.segment, i, thin_weight, thin_bias)
                if not isinstance(self.segment[i], nn.BatchNorm2d):
                    break


class LeNet500300Prune:
    def __init__(self, model):
        self.model = model.cuda()
        print("|===>init pruning")

    def run(self):
        pass


class DenseNetCifarPrune:
    def __init__(self, model):
        self.model = model.cuda()
        print("|===>init pruning")

    def run(self):
        for block in self.model.modules():
            if isinstance(block, (wcBasicBlock, wcTransitionBlock)):
                binary_weight = block.conv1.binary_weight
                select_channels = get_select_channels(binary_weight)
                # prune and replace conv1
                thin_weight, thin_bias = get_thin_params(
                    block.conv1, select_channels, 1)
                block.conv1 = replace_thin_layer(
                    block.conv1, thin_weight, thin_bias)
        self.model.cuda()
        print((self.model))


# prune spherenet depth 4
class SpherePrune:
    def __init__(self, model):
        self.model = model.cuda()
        print("|===>init ModelPrune")

    def run(self):
        # compute selected channels of conv2
        binary_weight = self.model.conv2.binary_weight
        select_channels = get_select_channels(binary_weight)
        self.select_channels = select_channels

        # prune and replace conv2
        thin_weight, thin_bias = get_thin_params(
            self.model.conv2, select_channels, 1)
        self.model.conv2 = replace_layer(
            self.model.conv2, thin_weight, thin_bias)

        # prune and replace conv1
        thin_weight, thin_bias = get_thin_params(
            self.model.conv1, select_channels, 0)
        self.model.conv1 = replace_layer(
            self.model.conv1, thin_weight, thin_bias)
        self.model.cuda()

        # prune and repalce prelu1

        thin_weight, thin_bias = get_thin_params(
            self.model.relu1, select_channels, 0)
        self.model.relu1 = replace_layer(
            self.model.relu1, thin_weight, thin_bias)
        self.model.cuda()

        # compute selected channels of conv3
        binary_weight = self.model.conv3.binary_weight
        select_channels = get_select_channels(binary_weight)
        self.select_channels = select_channels

        # prune and replace conv3
        thin_weight, thin_bias = get_thin_params(
            self.model.conv3, select_channels, 1)
        self.model.conv3 = replace_layer(
            self.model.conv3, thin_weight, thin_bias)

        # prune and replace conv2
        thin_weight, thin_bias = get_thin_params(
            self.model.conv2, select_channels, 0)
        self.model.conv2 = replace_layer(
            self.model.conv2, thin_weight, thin_bias)
        self.model.cuda()

        # prune and repalce prelu2

        thin_weight, thin_bias = get_thin_params(
            self.model.relu2, select_channels, 0)
        self.model.relu2 = replace_layer(
            self.model.relu2, thin_weight, thin_bias)
        self.model.cuda()

        # compute selected channels of conv4
        binary_weight = self.model.conv4.binary_weight
        select_channels = get_select_channels(binary_weight)
        self.select_channels = select_channels

        # prune and replace conv4
        thin_weight, thin_bias = get_thin_params(
            self.model.conv4, select_channels, 1)
        self.model.conv4 = replace_layer(
            self.model.conv4, thin_weight, thin_bias)

        # prune and replace conv3
        thin_weight, thin_bias = get_thin_params(
            self.model.conv3, select_channels, 0)
        self.model.conv3 = replace_layer(
            self.model.conv3, thin_weight, thin_bias)
        self.model.cuda()

        # prune and repalce prelu2

        thin_weight, thin_bias = get_thin_params(
            self.model.relu3, select_channels, 0)
        self.model.relu3 = replace_layer(
            self.model.relu3, thin_weight, thin_bias)
        self.model.cuda()

        # compute selected channels of fc
        # binary_weight = self.model.fc.binary_weight
        # select_channels = get_select_channels(binary_weight, h=1, w=1)
        # self.select_channels = select_channels

        # prune and replace fc
        """thin_weight, thin_bias = get_thin_params(
            self.model.fc, select_channels, 1, h=1, w=1)
        self.model.fc = replace_thin_layer(
            self.model.fc, thin_weight, thin_bias)"""

        # compute selected channels of fc
        """binary_weight = self.model.fc.binary_weight
        select_channels = get_select_channels(binary_weight, h=7, w=6)
        self.select_channels = select_channels

        thin_weight, thin_bias = get_thin_params(
            self.model.fc, select_channels, 1, h=7, w=6)
        self.model.fc = replace_layer(
            self.model.fc, thin_weight, thin_bias)

        # prune and replace conv4
        thin_weight, thin_bias = get_thin_params(
            self.model.conv4, select_channels, 0)
        self.model.conv4 = replace_layer(
            self.model.conv4, thin_weight, thin_bias)"""
        self.model.cuda()
        print(("|===>new model is:", self.model))


# prune sequential model like vgg or lenet5
class SeqModelPrune:
    def __init__(self, model, net_type):
        self.model = model
        self.net_type = net_type
        print("|===>init ModelPrune")

    def run(self):
        # divide model into several segment
        if self.net_type in ["LeNet5", "VGG", "AlexNet", "VGG_GAP"]:
            if self.net_type == "LeNet5":
                h = w = 4
            elif self.net_type == "VGG":
                h = w = 7
            elif self.net_type == "AlexNet":
                h = w = 6
            elif self.net_type == "VGG_GAP":
                h = w = 1
            feature_prune = SeqPrune(self.model.features)
            # segment level prune
            # prune features
            feature_prune.pruning()
            self.model.features = feature_prune.segment
            # self.model.features = feature_prune.segment
        
            if self.net_type != "VGG_GAP":
                # prune classifier
                classifier_prune = SeqPrune(self.model.classifier, h, w)
                select_channels = classifier_prune.pruning()
                feature_prune.extra_prune(select_channels)
                self.model.features = feature_prune.segment
                self.model.classifier = classifier_prune.segment
            
            self.model.cuda()
            print((self.model))
        
        else:
            assert False, "invalid net_type: " + self.net_type


# resnet only ---------------------------------------------------------------------------
class ResBlockPrune:
    def __init__(self, block, block_type):
        self.block = block
        self.block_type = block_type
        self.select_channels = None

    def pruning(self):
        # prune pre-resnet on cifar
        if self.block_type == "PreResNet":
            if self.block.conv2.binary_weight.data.sum() == 0 or self.block.conv1.binary_weight.data.sum() == 0:
                # self.block.bn1.weight.data.fill_(0)
                # self.block.bn2.weight.data.fill_(0)
                self.block = self.block.downsample
                
                if self.block is not None:
                    # replace wcconv in downsample block
                    binary_weight = self.block.binary_weight
                    select_channels = get_select_channels(binary_weight)
                    self.select_channels = select_channels
                    thin_weight, thin_bias = get_thin_params(
                        self.block, select_channels, 1
                    )
                    self.block = replace_thin_layer(
                        self.block, thin_weight, thin_bias
                    )
                
                print("remove whole block")
                return None
            # compute selected channels
            binary_weight = self.block.conv2.binary_weight
            select_channels = get_select_channels(binary_weight)
            self.select_channels = select_channels

            # prune and replace conv2
            thin_weight, thin_bias = get_thin_params(
                self.block.conv2, select_channels, 1)
            self.block.conv2 = replace_layer(
                self.block.conv2, thin_weight, thin_bias)

            # prune and replace bn2
            thin_weight, thin_bias = get_thin_params(
                self.block.bn2, select_channels, 0)
            self.block.bn2 = replace_layer(
                self.block.bn2, thin_weight, thin_bias)

            # prune and replace conv1
            thin_weight, thin_bias = get_thin_params(
                self.block.conv1, select_channels, 0)
            self.block.conv1 = replace_layer(
                self.block.conv1, thin_weight, thin_bias , keeping=True)
            self.block.cuda()

            
            # compute conv1
            binary_weight = self.block.conv1.binary_weight
            select_channels = get_select_channels(binary_weight)

            # prune and replace conv1
            self.select_channels = select_channels
            thin_weight, thin_bias = get_thin_params(
                self.block.conv1, select_channels, 1
            )
            self.block.conv1 = replace_thin_layer(
                self.block.conv1, thin_weight, thin_bias
            )
            """
            """
            # replace wcconv in downsample block
            if self.block.downsample is not None and self.block.downsample.binary_weight.data.sum() != 0:                
                binary_weight = self.block.downsample.binary_weight
                select_channels = get_select_channels(binary_weight)
                self.select_channels = select_channels
                thin_weight, thin_bias = get_thin_params(
                    self.block.downsample, select_channels, 1
                )
                self.block.downsample = replace_thin_layer(
                    self.block.downsample, thin_weight, thin_bias
                )
            

        # prune shallow resnet on imagenet
        elif self.block_type == "resnet_basic":
            if self.block.conv1.binary_weight.data.sum() == 0 or self.block.conv2.binary_weight.data.sum() == 0:
                self.block = self.block.downsample
                print("remove whole block")
                return None

            # compute selected channels
            binary_weight = self.block.conv2.binary_weight
            select_channels = get_select_channels(binary_weight)
            self.select_channels = select_channels

            # prune and replace conv2
            thin_weight, thin_bias = get_thin_params(
                self.block.conv2, select_channels, 1)
            self.block.conv2 = replace_layer(
                self.block.conv2, thin_weight, thin_bias)

            # prune and replace bn1
            thin_weight, thin_bias = get_thin_params(
                self.block.bn1, select_channels, 0)
            self.block.bn1 = replace_layer(
                self.block.bn1, thin_weight, thin_bias)

            # prune and replace conv1
            thin_weight, thin_bias = get_thin_params(
                self.block.conv1, select_channels, 0)
            self.block.conv1 = replace_layer(
                self.block.conv1, thin_weight, thin_bias)

        # prune deep resnet on imagenet
        elif self.block_type == "resnet_bottleneck":
            if self.block.conv1.binary_weight.data.sum() == 0 or self.block.conv2.binary_weight.data.sum() == 0 or self.block.conv3.binary_weight.data.sum() == 0:
                self.block = self.block.downsample
                """
                if self.block is not None:
                    # replace wcconv in downsample block
                    binary_weight = self.block[0].binary_weight
                    select_channels = get_select_channels(binary_weight)
                    self.select_channels = select_channels
                    thin_weight, thin_bias = get_thin_params(
                        self.block, select_channels, 1
                    )
                    self.block[0] = replace_thin_layer(
                        self.block[0], thin_weight, thin_bias
                    )
                """
                print("remove whole block")
                return None

            # compute selected channels of conv2
            binary_weight = self.block.conv2.binary_weight
            select_channels = get_select_channels(binary_weight)
            self.select_channels = select_channels

            # prune and replace conv2
            thin_weight, thin_bias = get_thin_params(
                self.block.conv2, select_channels, 1)
            self.block.conv2 = replace_layer(
                self.block.conv2, thin_weight, thin_bias)

            # prune and replace bn1
            thin_weight, thin_bias = get_thin_params(
                self.block.bn1, select_channels, 0)
            self.block.bn1 = replace_layer(
                self.block.bn1, thin_weight, thin_bias)

            # prune and replace conv1
            thin_weight, thin_bias = get_thin_params(
                self.block.conv1, select_channels, 0)
            self.block.conv1 = replace_layer(
                self.block.conv1, thin_weight, thin_bias)# , keeping=True)

            self.block.cuda()
            # compute selected channels of conv3
            binary_weight = self.block.conv3.binary_weight
            select_channels = get_select_channels(binary_weight)

            # prune and replace conv3
            thin_weight, thin_bias = get_thin_params(
                self.block.conv3, select_channels, 1)
            self.block.conv3 = replace_layer(
                self.block.conv3, thin_weight, thin_bias)

            # prune and replace bn2
            thin_weight, thin_bias = get_thin_params(
                self.block.bn2, select_channels, 0)
            self.block.bn2 = replace_layer(
                self.block.bn2, thin_weight, thin_bias)

            # prune and replace conv2
            thin_weight, thin_bias = get_thin_params(
                self.block.conv2, select_channels, 0)
            self.block.conv2 = replace_layer(
                self.block.conv2, thin_weight, thin_bias)
            self.block.cuda()

            """
            # compute select channels of conv1
            binary_weight = self.block.conv1.binary_weight
            select_channels = get_select_channels(binary_weight)
            
            # prune and replace conv1
            self.select_channels = select_channels
            thin_weight, thin_bias = get_thin_params(
                self.block.conv1, select_channels, 1
            )
            self.block.conv1 = replace_thin_layer(
                self.block.conv1, thin_weight, thin_bias
            )
            self.block.cuda()
            """

        else:
            assert False, "invalid block type: " + self.block_type


class ResSeqPrune:
    def __init__(self, sequential, seq_type):
        self.sequential = sequential
        self.sequential_length = len(list(self.sequential))
        self.res_block_prune = []
        self.select_channels = None

        for i in range(self.sequential_length):
            self.res_block_prune.append(
                ResBlockPrune(self.sequential[i], block_type=seq_type)
            )

    def pruning(self):
        for i in range(self.sequential_length):
            self.res_block_prune[i].pruning()

        temp_seq = []
        for i in range(self.sequential_length):
            if self.res_block_prune[i].block is not None:
                temp_seq.append(self.res_block_prune[i].block)
        self.sequential = nn.Sequential(*temp_seq)
        self.select_channels = self.res_block_prune[0].select_channels


class ResModelPrune:
    def __init__(self, model, net_type, depth):
        self.model = model
        if net_type =="ResNet":
            if depth >= 50:
                self.net_type = "resnet_bottleneck"
            else:
                self.net_type = "resnet_basic"
        else:      
            self.net_type = net_type
        print("|===>init ModelPrune")

    def run(self):
        if self.net_type in ["resnet_basic", "resnet_bottleneck"]:
            res_seq_prune = [
                ResSeqPrune(self.model.layer1, seq_type=self.net_type),
                ResSeqPrune(self.model.layer2, seq_type=self.net_type),
                ResSeqPrune(self.model.layer3, seq_type=self.net_type),
                ResSeqPrune(self.model.layer4, seq_type=self.net_type)
            ]

            for i in range(4):
                res_seq_prune[i].pruning()
            """
            select_channels = res_seq_prune[0].select_channels

            # prune and replace conv1
            thin_weight, thin_bias = get_thin_params(
                self.model.conv1, select_channels, 0)
            self.model.conv1 = replace_layer(
                self.model.conv1, thin_weight, thin_bias)

            # prune and replace bn1
            thin_weight, thin_bias = get_thin_params(
                self.model.bn1, select_channels, 0)
            self.model.bn1 = replace_layer(
                self.model.bn1, thin_weight, thin_bias)
            """
            self.model.layer1 = res_seq_prune[0].sequential
            self.model.layer2 = res_seq_prune[1].sequential
            self.model.layer3 = res_seq_prune[2].sequential
            self.model.layer4 = res_seq_prune[3].sequential
            self.model.cuda()
            print((self.model))

        elif self.net_type == "PreResNet":
            # replace first conv layer
            conv_weight = self.model.conv.weight
            conv = nn.Conv2d(conv_weight.size(1),
                             conv_weight.size(0),
                             kernel_size=self.model.conv.kernel_size,
                             stride=self.model.conv.stride,
                             padding=self.model.conv.padding,
                             bias=False)
            conv.weight.data.copy_(conv_weight.data)
            self.model.conv = conv
            self.model.cuda()

            res_seq_prune = [
                ResSeqPrune(self.model.layer1, seq_type=self.net_type),
                ResSeqPrune(self.model.layer2, seq_type=self.net_type),
                ResSeqPrune(self.model.layer3, seq_type=self.net_type)
            ]
            for i in range(3):
                res_seq_prune[i].pruning()

            select_channels = res_seq_prune[0].select_channels

            # prune and replace conv
            # thin_weight, thin_bias = get_thin_params(
            #     self.model.conv, select_channels, 0)
            # self.model.conv = replace_layer(
            #     self.model.conv, thin_weight, thin_bias)

            self.model.layer1 = res_seq_prune[0].sequential
            self.model.layer2 = res_seq_prune[1].sequential
            self.model.layer3 = res_seq_prune[2].sequential
            print((self.model))
            self.model.cuda()
