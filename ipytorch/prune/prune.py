import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import numpy as np
import random
import torch.nn.functional as F
from ipytorch.quantization import twnConv2d, twnLinear


__all__ = ['FilterPrune', 'replace_segment', 'replace_layer',
           'SegmentPrune', 'ModelPrune', 'ComplexSeqPrune']

# re-implementation of ThiNet: a filter level pruning method for convolutional network compression
class FilterPrune(object):
    def __init__(self, h=None, w=None, locations=10, prune_ratio=0.5):
        """
        filter level pruning
        :params h: <int> height of feature maps. We always reshape the inputs of linear layer into 2D matrix which
        leads to changing of channels of inputs, so we need to define the height and weight of feature maps for linear layer
        :params w: <int> width of feature maps.
        :params locations: <int> number of locations sampling from output features of every input image.
        :params prune_ratio: <float> [0, 1), percentage of preserved channels
        """
        self.xhat_cache = None
        self.locations = locations
        self.prune_ratio = prune_ratio
        self.remove_channels = None
        self.select_channels = None
        self.process_feature = False
        self.h = h
        self.w = w
        self.y_cache = None

    def feature_extract(self, x, y, layer):
        """
        :params x: input feature
        :params y: output feature
        :params layer: pruned layer
        """
        if isinstance(layer, nn.Conv2d):
            # padding
            padding_size = layer.padding
            padding_layer = nn.ZeroPad2d(
                (padding_size[1], padding_size[1], padding_size[0], padding_size[0]))
            x = padding_layer(x).data

            # generate random location of y
            y_d = torch.LongTensor(np.random.randint(
                y.size(1), size=self.locations * y.size(0))).cuda()
            y_h = torch.LongTensor(np.random.randint(
                y.size(2), size=self.locations * y.size(0))).cuda()
            y_w = torch.LongTensor(np.random.randint(
                y.size(3), size=self.locations * y.size(0))).cuda()

            # select weight according to y
            w_select = layer.weight.data[y_d]

            # compute locations of x according to y
            x_h = y_h * layer.stride[0]
            x_w = y_w * layer.stride[1]

            # compute x of every channel
            temp_xhat_cache = tuple()
            temp_y_cache = []

            x_n = torch.LongTensor(
                np.arange(y_h.size(0)) / self.locations).cuda()

            for i in range(y_h.size(0)):
                x_select = x[i / self.locations, :, x_h[i]:x_h[i] + layer.kernel_size[0],
                             x_w[i]:x_w[i] + layer.kernel_size[1]].unsqueeze(0)

                temp_xhat_cache = temp_xhat_cache + (x_select, )

            temp_xhat_cache = torch.cat(temp_xhat_cache, 0)
            temp_xhat_cache = (temp_xhat_cache * w_select).sum(2).sum(2)

            temp_y_cache = y.data[x_n, y_d, y_h, y_w]

            # add y to cache
            if self.y_cache is None:
                self.y_cache = temp_y_cache
            else:
                self.y_cache = torch.cat(
                    (self.y_cache, temp_y_cache), 0)

            # add results to a larger cache
            # temp_xhat_cache = torch.cat(temp_xhat_cache, 0)
            if self.xhat_cache is None:
                self.xhat_cache = temp_xhat_cache
            else:
                self.xhat_cache = torch.cat(
                    (self.xhat_cache, temp_xhat_cache), 0)

            # set flag true
            self.process_feature = True
            # print self.xhat_cache.size()

        elif isinstance(layer, nn.Linear):
            # linear layer

            if self.h is None or self.w is None:
                assert False, "Linear layer needs parameters <h, w>"

            # change data type of x
            x = x.data

            # generate locations of y
            y_d = torch.LongTensor(np.random.randint(
                y.size(1), size=self.locations * y.size(0))).cuda()

            # select w from W
            w_select = layer.weight.data[y_d]
            temp_xhat_cache = tuple()

            # compute x of every channel
            x_n = torch.LongTensor(
                list(range(x.size(0)))).cuda().repeat(self.locations)

            temp_xhat_cache = x[x_n] * w_select

            # reshape result to <N,C,H,W>
            temp_xhat_cache = temp_xhat_cache.view(
                y_d.size(0), x.size(1) / (self.h * self.w), self.h, self.w).sum(2).sum(2)
            temp_y_cache = y.data[x_n, y_d]

            if self.y_cache is None:
                self.y_cache = temp_y_cache
            else:
                self.y_cache = torch.cat((self.y_cache, temp_y_cache), 0)

            if self.xhat_cache is None:
                self.xhat_cache = temp_xhat_cache
            else:
                self.xhat_cache = torch.cat((self.xhat_cache, temp_xhat_cache))
            self.process_feature = True

    def channel_select(self, layer):
        """
        select channels according to value of x_hat
        :params layer: pruned layer
        :return w_hat: scalar of weights
        """
        assert self.process_feature, "Please run feature_extract() first!"

        if isinstance(layer, nn.Conv2d):
            channels = layer.in_channels
        elif isinstance(layer, nn.Linear):
            if self.h is None or self.w is None:
                assert False, "Linear layer needs parameters <h, w>"
            channels = layer.in_features / (self.h * self.w)
        else:
            assert False, "unsupported layer type: " + str(type(layer))

        # init I and T: I is the set of all channels, T is the set of removed channels
        I = list(range(channels))
        T = []
        print(("xhat_size: ", self.xhat_cache.size()))

        sum_cache = None
        for c in range(int(len(I) * (1 - self.prune_ratio))):
            min_value = None
            # print len(I)
            select_value = None
            for i in I:
                tempT = T[:]
                tempT.append(i)
                tempT = torch.LongTensor(tempT).cuda()
                # print i, self.xhat_cache.size()  # , self.xhat_cache.index_select(1, tempT).size()

                temp_value = self.xhat_cache.index_select(
                    1, torch.LongTensor([i]).cuda())
                if sum_cache is None:
                    value = temp_value.abs().sum()
                else:
                    value = (sum_cache + temp_value).abs().sum()

                if min_value is None or min_value > value:
                    select_value = temp_value
                    min_value = value
                    min_i = i
            if sum_cache is None:
                sum_cache = select_value
            else:
                sum_cache += select_value

            I.remove(min_i)
            T.append(min_i)
            # print "min_value: ", min_value

        # T = np.random.randint(len(I), size=int(len(I)*(1-self.prune_ratio)))
        self.remove_channels = torch.LongTensor(sorted(T)).cuda()
        S = list(range(channels))
        for c in T:
            if c in S:
                S.remove(c)
        self.select_channels = torch.LongTensor(S).cuda()

        # compute scale
        x = self.xhat_cache.index_select(1, self.select_channels)
        x = x.view(x.size(0), -1)
        y = self.y_cache.unsqueeze(1)

        print(("size of x: ", x.size()))
        print(("size of y: ", y.size()))
        print(("size of xhat_cache: ", self.xhat_cache.size()))
        print(("size of x.transpose: ", x.transpose(0, 1).size()))

        """
        print "operation checking:-----------"
        xt_x = torch.mm(x.transpose(0, 1), x)
        print xt_x.size()
        xt_x_inverse = xt_x.inverse()
        print xt_x_inverse.size()
        xt_x_inverse_xt = torch.mm(xt_x_inverse, x.transpose(0, 1))
        print xt_x_inverse_xt.size()
        w_hat = torch.mm(xt_x_inverse_xt, y)
        print w_hat.size()
        """
        if isinstance(layer, twnConv2d) or isinstance(layer,twnLinear):
            return None

        w_hat = torch.mm(
            torch.mm(torch.mm(x.transpose(0, 1), x).inverse(), x.transpose(0, 1)), y)
        print((w_hat.size()))
        print((layer.weight.data.dim()))
        if layer.weight.data.dim() > 2:
            w_hat = w_hat.unsqueeze(0).unsqueeze(2)
            w_hat = w_hat.expand(layer.weight.size(0), w_hat.size(
                1), layer.weight.size(2), layer.weight.size(3))
            print((w_hat.size()))
        else:
            w_hat = w_hat.repeat(1, self.h * self.w).view(-1)
            print((layer.weight.data.size()))
            w_hat = w_hat.unsqueeze(0).expand(
                layer.weight.size(0), w_hat.size(0))
        # assert False
        print(("size of w_hat is:", w_hat.size()))
        w_hat = None
        # assert False
        return w_hat

    def get_thin_params(self, layer, dim=0):
        """
        prune weights accordingt to the select channels
        :params layer: pruned layer
        :params dim: pruned dimension
        :return thin_weight, thin_bias
        """
        if isinstance(layer, nn.Conv2d) or isinstance(layer, twnConv2d):
            thin_weight = layer.weight.data.index_select(
                dim, self.select_channels)
            if layer.bias is not None:
                if dim == 0:
                    thin_bias = layer.bias.data.index_select(
                        dim, self.select_channels)
                else:
                    thin_bias = layer.bias.data
            else:
                thin_bias = None

        elif isinstance(layer, nn.Linear) or isinstance(layer, twnLinear):
            if dim == 0:
                thin_weight = layer.weight.data.index_select(
                    dim, self.select_channels)
                if layer.bias is not None:
                    thin_bias = layer.bias.data.index_select(
                        dim, self.select_channels)
                else:
                    thin_bias = None
            else:

                fat_weight = layer.weight.data
                fat_weight = fat_weight.view(layer.weight.size(
                    0), layer.weight.size(1) / (self.h * self.w), self.h, self.w)
                thin_weight = fat_weight.index_select(
                    dim, self.select_channels)
                thin_weight = thin_weight.view(layer.weight.size(0), -1)
                if layer.bias is not None:
                    thin_bias = layer.bias.data
                else:
                    thin_bias = None

        elif isinstance(layer, nn.BatchNorm2d):
            assert dim == 0, "invalid dimension for bn_layer"

            thin_weight = layer.weight.data.index_select(
                dim, self.select_channels)
            if layer.bias is not None:
                thin_bias = layer.bias.data.index_select(
                    dim, self.select_channels)
            else:
                thin_bias = None

        return thin_weight, thin_bias


# --------------------------------------------------------------------------------------------------
def replace_segment(model, index, init_weight, init_bias=None, scale=None):
    """
    replace specific layer of model
    :params model: original model
    :params index: index of the layer to be replaced
    :params init_weight: thin_weight get from FilterPrune.get_thin_params
    :params init_bias: thin_bias get from FilterPrune.get_thin_params
    :params scale: w_hat get from FilterPrune.select_channels
    :returns <nn.Sequential> new model
    """

    old_layer = model[index]
    new_model = []
    for i, _ in enumerate(model):
        if i == index:
            if scale is not None:
                init_weight = init_weight * scale

            if isinstance(old_layer, twnConv2d):
                new_layer = twnConv2d(init_weight.size(1),
                                      init_weight.size(0),
                                      kernel_size=old_layer.kernel_size,
                                      stride=old_layer.stride,
                                      padding=old_layer.padding,
                                      cRate=old_layer.cRate)
                new_layer.weight.data.copy_(init_weight)
                if init_bias is not None:
                    new_layer.bias.data.copy_(init_bias)

            elif isinstance(old_layer, nn.Conv2d):
                new_layer = nn.Conv2d(init_weight.size(1),
                                      init_weight.size(0),
                                      kernel_size=old_layer.kernel_size,
                                      stride=old_layer.stride,
                                      padding=old_layer.padding)
                new_layer.weight.data.copy_(init_weight)
                if init_bias is not None:
                    new_layer.bias.data.copy_(init_bias)

            elif isinstance(old_layer, twnLinear):
                new_layer = twnLinear(init_weight.size(1),
                                      init_weight.size(0),
                                      cRate=old_layer.cRate)

                new_layer.weight.data.copy_(init_weight)
                if init_bias is not None:
                    new_layer.bias.data.copy_(init_bias)

            elif isinstance(old_layer, nn.Linear):
                new_layer = nn.Linear(init_weight.size(1),
                                      init_weight.size(0))
                new_layer.weight.data.copy_(init_weight)
                if init_bias is not None:
                    new_layer.bias.data.copy_(init_bias)

            elif isinstance(old_layer, nn.BatchNorm2d):
                new_layer = nn.BatchNorm2d(init_weight.size(0))
                new_layer.weight.data.copy_(init_weight)
                assert init_bias is not None, "batch normalization needs bias"
                new_layer.bias.data.copy_(init_bias)

            else:
                assert False, "unsupport layer type:" + \
                    str(type(old_layer))
            new_model.append(new_layer)
        else:
            new_model.append(model[i])
    return nn.Sequential(*new_model)


def replace_layer(old_layer, init_weight, init_bias=None, scale=None):
    """
    replace specific layer of model
    :params layer: original layer
    :params init_weight: thin_weight get from FilterPrune.get_thin_params
    :params init_bias: thin_bias get from FilterPrune.get_thin_params
    :params scale: w_hat get from FilterPrune.select_channels
    :returns new_layer
    """
    if scale is not None:
        init_weight = init_weight * scale

    if isinstance(old_layer, twnConv2d):
        new_layer = twnConv2d(init_weight.size(1),
                              init_weight.size(0),
                              kernel_size=old_layer.kernel_size,
                              stride=old_layer.stride,
                              padding=old_layer.padding,
                              cRate=old_layer.cRate)
        new_layer.weight.data.copy_(init_weight)
        if init_bias is not None:
            new_layer.bias.data.copy_(init_bias)

    elif isinstance(old_layer, nn.Conv2d):
        new_layer = nn.Conv2d(init_weight.size(1),
                              init_weight.size(0),
                              kernel_size=old_layer.kernel_size,
                              stride=old_layer.stride,
                              padding=old_layer.padding)
        new_layer.weight.data.copy_(init_weight)
        if init_bias is not None:
            new_layer.bias.data.copy_(init_bias)

    elif isinstance(old_layer, twnLinear):
        new_layer = twnLinear(init_weight.size(1),
                              init_weight.size(0),
                              cRate=old_layer.cRate)

        new_layer.weight.data.copy_(init_weight)
        if init_bias is not None:
            new_layer.bias.data.copy_(init_bias)

    elif isinstance(old_layer, nn.Linear):
        new_layer = nn.Linear(init_weight.size(1),
                              init_weight.size(0))
        new_layer.weight.data.copy_(init_weight)
        if init_bias is not None:
            new_layer.bias.data.copy_(init_bias)

    elif isinstance(old_layer, nn.BatchNorm2d):
        new_layer = nn.BatchNorm2d(init_weight.size(0))
        new_layer.weight.data.copy_(init_weight)
        assert init_bias is not None, "batch normalization needs bias"
        new_layer.bias.data.copy_(init_bias)

    else:
        assert False, "unsupport layer type:" + \
            str(type(old_layer))
    return new_layer


# -----------------------------------------------------------------------------------------------------------------
class SegmentPrune:
    def __init__(self, segment, segment_type="sequential", skip_layer=False, prune_ratio=0.5):
        """
        :params segment: target segment or sequential
        :params segment_type: option: ["sequential", "resnet_basic", "resnet_bottleneck", "pre_resnet"]
        "sequantial" is for pruning nn.Sequential; 
        "resnet_basic and "resnet_bottleneck" are working for residual network
        "half_preact" and "both_preact" are working for pre_resnet
        more specific type will added for pruning complicated network like squeezeNet, denseNet, dual_Path Network, etc
        :params skip_layer: skip first pruned layer, default: False, for some case, we do not need to prune first conv_layer
        as it only have 3 channels(RGB)
        """
        self.segment = segment
        self.segment_type = segment_type
        self.finish_flag = False
        self.filter_prune = None
        self.segment_length = None
        self.prune_count = 0
        self.prune_ratio = prune_ratio
        print(("|===>init SegmentPrune" + segment_type))

        assert self.segment_type in ["sequential", "resnet_basic", "resnet_bottleneck",
                                     "pre_resnet"], "invalid segment_type: " + self.segment_type

        if segment_type == "sequential":
            # get length of segment
            if isinstance(self.segment, nn.DataParallel):
                self.segment = list(self.segment.module)
            elif isinstance(self.segment, nn.Sequential):
                self.segment = list(self.segment)
            self.segment_length = len(self.segment)
            self.segment = nn.Sequential(*self.segment)
            self.segment.cuda()
            self.finish_flag = True
            # get firt prune layer
            for i in range(self.segment_length):
                if isinstance(self.segment[i], nn.Conv2d) or isinstance(self.segment[i], nn.Linear):
                    # for some case, we will skip the first layer
                    if skip_layer:
                        skip_layer = False
                        continue
                    self.prune_count = i
                    self.finish_flag = False
                    break
            if self.finish_flag:
                print("|===>Warning: this segment will not be pruned!")

    def feature_extract(self, x):
        if self.segment_type == "sequential":
            assert self.prune_count < self.segment_length, "prune times should smaller than lenght of segment, prune_count:%d, segment_len" % (
                self.prune_count, self.segment_length)

            if self.filter_prune is None:
                self.filter_prune = FilterPrune(prune_ratio=self.prune_ratio)

            for i in range(self.prune_count):
                if isinstance(self.segment[i], nn.Linear) and x.dim() != 2:
                    x = x.view(x.size(0), -1)
                x = self.segment[i](x)

            # set 3rd and 4th dimension of linear layer
            if isinstance(self.segment[self.prune_count], nn.Linear):
                if x.dim() != 2:
                    self.filter_prune.h = x.size(2)
                    self.filter_prune.w = x.size(3)
                    x = x.view(x.size(0), -1)
                else:
                    self.filter_prune.h = 1
                    self.filter_prune.w = 1
            y = self.segment[self.prune_count](x)
            self.filter_prune.feature_extract(
                x, y, self.segment[self.prune_count])

        elif self.segment_type == "resnet_basic":
            if self.filter_prune is None:
                self.filter_prune = FilterPrune(prune_ratio=self.prune_ratio)

            if self.prune_count == 0:
                y = self.segment.conv1(x)
                print(("feature extract: x_size, y_size", x.size(), y.size()))
                self.filter_prune.feature_extract(x, y, self.segment.conv1)

            elif self.prune_count == 1:
                x = self.segment.conv1(x)
                x = self.segment.bn1(x)
                x = self.segment.relu(x)
                y = self.segment.conv2(x)
                print(("feature extract: x_size, y_size", x.size(), y.size()))
                self.filter_prune.feature_extract(x, y, self.segment.conv2)
            else:
                assert False, "no more layer to prune"

        elif self.segment_type == "resnet_bottleneck":
            if self.filter_prune is None:
                self.filter_prune = FilterPrune(prune_ratio=self.prune_ratio)

            if self.prune_count == 0:
                y = self.segment.conv1(x)
                self.filter_prune.feature_extract(x, y, self.segment.conv1)
            elif self.prune_count == 1:
                x = self.segment.conv1(x)
                x = self.segment.bn1(x)
                x = self.segment.relu(x)
                y = self.segment.conv2(x)
                self.filter_prune.feature_extract(x, y, self.segment.conv2)
            elif self.prune_count == 2:
                x = self.segment.conv1(x)
                x = self.segment.bn1(x)
                x = self.segment.relu(x)
                x = self.segment.conv2(x)
                x = self.segment.bn2(x)
                x = self.segment.relu(x)
                y = self.segment.conv3(x)
                self.filter_prune.feature_extract(x, y, self.segment.conv3)
            else:
                assert False, "no more layer to prune"

        elif self.segment_type == "pre_resnet":
            if self.filter_prune is None:
                self.filter_prune = FilterPrune(prune_ratio=self.prune_ratio)
            if self.prune_count == 2:
                x = self.segment.bn1(x)
                x = self.segment.relu(x)
                y = self.segment.conv1(x)
                self.filter_prune.feature_extract(x, y, self.segment.conv1)
                # print "feature extract: x_size, y_size", x.size(), y.size()
            elif self.prune_count == 0:
                x = self.segment.bn1(x)
                x = self.segment.relu(x)
                x = self.segment.conv1(x)
                x = self.segment.bn2(x)
                x = self.segment.relu(x)
                y = self.segment.conv2(x)
                # print "feature extract: x_size, y_size", x.size(), y.size()
                self.filter_prune.feature_extract(x, y, self.segment.conv2)
            else:
                assert False, "no more layer to prune"
        else:
            assert False, "invalid segment type: " + self.segment_type

    def channel_select(self):
        print("|===>segment prune, channels select")
        if self.segment_type == "sequential":
            w_hat = self.filter_prune.channel_select(
                self.segment[self.prune_count])
            # replace current layer
            thin_weight, thin_bias = self.filter_prune.get_thin_params(
                self.segment[self.prune_count], 1)
            self.segment = replace_segment(
                self.segment, self.prune_count, thin_weight, thin_bias, w_hat)

            # replace previous layer
            previous_layer_flag = False
            for i in range(self.prune_count - 1, -1, -1):
                if (isinstance(self.segment[i], nn.Linear)
                        or isinstance(self.segment[i], nn.Conv2d)
                        or isinstance(self.segment[i], nn.BatchNorm2d)):

                    thin_weight, thin_bias = self.filter_prune.get_thin_params(
                        self.segment[i], 0)
                    self.segment = replace_segment(
                        self.segment, i, thin_weight, thin_bias)
                    if not isinstance(self.segment[i], nn.BatchNorm2d):
                        previous_layer_flag = True
                        break

            # get next layer to prune
            self.finish_flag = True
            for i in range(self.prune_count + 1, self.segment_length):
                if isinstance(self.segment[i], nn.Conv2d) or isinstance(self.segment[i], nn.Linear):
                    self.prune_count = i
                    self.finish_flag = False
                    break
            # for vgg, only prune 10 conv
            # TODO: remove these code then
            if self.prune_count >= 24:
                self.finish_flag = True

            if not previous_layer_flag:
                select_channels = self.filter_prune.select_channels
            else:
                select_channels = None
            if self.finish_flag:
                # convert segment type to nn.Sequential
                self.segment = nn.Sequential(*self.segment)

            self.filter_prune = None
            print(("|===>%d segment channels select..." % self.prune_count))
            return select_channels

        elif self.segment_type == "resnet_basic":
            if self.prune_count == 0:
                w_hat = self.filter_prune.channel_select(self.segment.conv1)

                thin_weight, thin_bias = self.filter_prune.get_thin_params(
                    self.segment.conv1, 1)
                self.segment.conv1 = replace_layer(
                    self.segment.conv1, thin_weight, thin_bias, w_hat)

                select_channels = self.filter_prune.select_channels
                if self.segment.downsample is not None:
                    thin_weight, thin_bias = self.filter_prune.get_thin_params(
                        self.segment.downsample[0], 1)
                    self.segment.downsample = replace_segment(
                        self.segment.downsample, 0, thin_weight, thin_bias)
                self.prune_count += 1
                self.filter_prune = None

            elif self.prune_count == 1:
                # get scalar of conv2
                w_hat = self.filter_prune.channel_select(self.segment.conv2)

                # compute thin_weight and thin_bias of conv2
                thin_weight, thin_bias = self.filter_prune.get_thin_params(
                    self.segment.conv2, 1)
                # replace conv2
                self.segment.conv2 = replace_layer(
                    self.segment.conv2, thin_weight, thin_bias)

                # compute thin_weight and thin_bias of conv1
                thin_weight, thin_bias = self.filter_prune.get_thin_params(
                    self.segment.conv1, 0)
                # replace conv1
                self.segment.conv1 = replace_layer(
                    self.segment.conv1, thin_weight, thin_bias)

                # compute thin_weight and thin_bias of bn1
                thin_weight, thin_bias = self.filter_prune.get_thin_params(
                    self.segment.bn1, 0)
                # replace bn1
                self.segment.bn1 = replace_layer(
                    self.segment.bn1, thin_weight, thin_bias)
                self.prune_count = 2
                self.filter_prune = None
                select_channels = None
                self.finish_flag = True
            return select_channels

        elif self.segment_type == "resnet_bottleneck":
            if self.prune_count == 0:
                w_hat = self.filter_prune.channel_select(self.segment.conv1)

                thin_weight, thin_bias = self.filter_prune.get_thin_params(
                    self.segment.conv1, 1)
                self.segment.conv1 = replace_layer(
                    self.segment.conv1, thin_weight, thin_bias, w_hat)

                select_channels = self.filter_prune.select_channels

                if self.segment.downsample is not None:
                    thin_weight, thin_bias = self.filter_prune.get_thin_params(
                        self.segment.downsample[0], 1)
                    self.segment.downsample = replace_segment(
                        self.segment.downsample, 0, thin_weight, thin_bias)
                self.prune_count += 1
                self.filter_prune = None

            elif self.prune_count == 1:
                # get scalar of conv2
                w_hat = self.filter_prune.channel_select(self.segment.conv2)

                # compute thin_weight and thin_bias of conv2
                thin_weight, thin_bias = self.filter_prune.get_thin_params(
                    self.segment.conv2, 1)
                # replace conv2
                self.segment.conv2 = replace_layer(
                    self.segment.conv2, thin_weight, thin_bias)

                # compute thin_weight and thin_bias of conv1
                thin_weight, thin_bias = self.filter_prune.get_thin_params(
                    self.segment.conv1, 0)
                # replace conv1
                self.segment.conv1 = replace_layer(
                    self.segment.conv1, thin_weight, thin_bias)

                # compute thin_weight and thin_bias of bn1
                thin_weight, thin_bias = self.filter_prune.get_thin_params(
                    self.segment.bn1, 0)
                # replace bn1
                self.segment.bn1 = replace_layer(
                    self.segment.bn1, thin_weight, thin_bias)
                self.prune_count += 1
                self.filter_prune = None
                select_channels = None

            elif self.prune_count == 2:
                # get scalar of conv3
                w_hat = self.filter_prune.channel_select(self.segment.conv3)

                # compute thin_weight and thin_bias of conv2
                thin_weight, thin_bias = self.filter_prune.get_thin_params(
                    self.segment.conv3, 1)
                # replace conv2
                self.segment.conv3 = replace_layer(
                    self.segment.conv3, thin_weight, thin_bias)

                # compute thin_weight and thin_bias of conv2
                thin_weight, thin_bias = self.filter_prune.get_thin_params(
                    self.segment.conv2, 0)
                # replace conv1
                self.segment.conv2 = replace_layer(
                    self.segment.conv2, thin_weight, thin_bias)

                # compute thin_weight and thin_bias of bn1
                thin_weight, thin_bias = self.filter_prune.get_thin_params(
                    self.segment.bn2, 0)
                # replace bn1
                self.segment.bn2 = replace_layer(
                    self.segment.bn2, thin_weight, thin_bias)

                self.prune_count += 1
                self.filter_prune = None
                select_channels = None
                self.finish_flag = True
            self.segment.cuda()
            return select_channels

        elif self.segment_type == "pre_resnet":
            if self.prune_count == 2:
                w_hat = self.filter_prune.channel_select(self.segment.conv1)
                thin_weight, thin_bias = self.filter_prune.get_thin_params(
                    self.segment.conv1, 1)
                self.segment.conv1 = replace_layer(
                    self.segment.conv1, thin_weight, thin_bias, w_hat)

                thin_weight, thin_bias = self.filter_prune.get_thin_params(
                    self.segment.bn1, 0)
                self.segment.bn1 = replace_layer(
                    self.segment.bn1, thin_weight, thin_bias)

                select_channels = self.filter_prune.select_channels
                if self.segment.downsample is not None:
                    thin_weight, thin_bias = self.filter_prune.get_thin_params(
                        self.segment.downsample, 1)
                    self.segment.downsample = replace_layer(
                        self.segment.downsample, thin_weight, thin_bias)
                self.prune_count += 1
                self.filter_prune = None
                print("channels select done!")

            elif self.prune_count == 0:
                w_hat = self.filter_prune.channel_select(self.segment.conv2)
                thin_weight, thin_bias = self.filter_prune.get_thin_params(
                    self.segment.conv2, 1)
                self.segment.conv2 = replace_layer(
                    self.segment.conv2, thin_weight, thin_bias, w_hat)

                thin_weight, thin_bias = self.filter_prune.get_thin_params(
                    self.segment.conv1, 0)
                self.segment.conv1 = replace_layer(
                    self.segment.conv1, thin_weight, thin_bias)

                thin_weight, thin_bias = self.filter_prune.get_thin_params(
                    self.segment.bn2, 0)
                self.segment.bn2 = replace_layer(
                    self.segment.bn2, thin_weight, thin_bias)

                self.filter_prune = None
                self.prune_count += 1
                select_channels = None
                self.finish_flag = True
                print("prune second conv")
            else:
                assert False, "invalid prune count: %d" % self.prune_count
            self.segment.cuda()
            return select_channels
        else:
            assert False, "invalid segment type: " + self.segment_type

    def extra_prune(self, select_channels):
        print("|===>enter extra_prune")
        assert select_channels is not None, "select_channels is NoneType"

        extra_filter_prune = FilterPrune(prune_ratio=self.prune_ratio)
        extra_filter_prune.select_channels = select_channels
        if self.segment_type == "sequential":
            for i in range(self.segment_length - 1, -1, -1):
                if (isinstance(self.segment[i], nn.Conv2d)
                        or isinstance(self.segment[i], nn.BatchNorm2d)):

                    thin_weight, thin_bias = extra_filter_prune.get_thin_params(
                        self.segment[i], 0)
                    self.segment = replace_segment(
                        self.segment, i, thin_weight, thin_bias)
                    if not isinstance(self.segment[i], nn.BatchNorm2d):
                        break
        elif self.segment_type == "resnet_basic":
            thin_weight, thin_bias = extra_filter_prune.get_thin_params(
                self.segment.conv2, 0)
            self.segment.conv2 = replace_layer(
                self.segment.conv2, thin_weight, thin_bias)

            thin_weight, thin_bias = extra_filter_prune.get_thin_params(
                self.segment.bn2, 0)
            self.segment.bn2 = replace_layer(
                self.segment.bn2, thin_weight, thin_bias)

            if self.segment.downsample is not None:

                thin_weight, thin_bias = extra_filter_prune.get_thin_params(
                    self.segment.downsample[0], 0)
                self.segment.downsample = replace_segment(
                    self.segment.downsample, 0, thin_weight, thin_bias)

                thin_weight, thin_bias = extra_filter_prune.get_thin_params(
                    self.segment.downsample[1], 0)
                self.segment.downsample = replace_segment(
                    self.segment.downsample, 1, thin_weight, thin_bias)

        elif self.segment_type == "resnet_bottleneck":
            thin_weight, thin_bias = extra_filter_prune.get_thin_params(
                self.segment.conv3, 0)
            self.segment.conv3 = replace_layer(
                self.segment.conv3, thin_weight, thin_bias)

            thin_weight, thin_bias = extra_filter_prune.get_thin_params(
                self.segment.bn3, 0)
            self.segment.bn3 = replace_layer(
                self.segment.bn3, thin_weight, thin_bias)

            if self.segment.downsample is not None:
                thin_weight, thin_bias = extra_filter_prune.get_thin_params(
                    self.segment.downsample[0], 0)
                self.segment.downsample = replace_segment(
                    self.segment.downsample, 0, thin_weight, thin_bias)

                thin_weight, thin_bias = extra_filter_prune.get_thin_params(
                    self.segment.downsample[1], 0)
                self.segment.downsample = replace_segment(
                    self.segment.downsample, 1, thin_weight, thin_bias)
        elif self.segment_type == "pre_resnet":
            thin_weight, thin_bias = extra_filter_prune.get_thin_params(
                self.segment.conv2, 0)
            self.segment.conv2 = replace_layer(
                self.segment.conv2, thin_weight, thin_bias)

            if self.segment.downsample is not None:
                thin_weight, thin_bias = extra_filter_prune.get_thin_params(
                    self.segment.downsample, 0)
                self.segment.downsample = replace_layer(
                    self.segment.downsample, thin_weight, thin_bias)

        else:
            assert False, "invalid segment type: " + self.segment_type

    def __call__(self, x):
        out = self.segment(x)
        return out

# imagenet only ---------------------------------------------------------------------------


class ComplexSeqPrune:
    def __init__(self, sequential, seq_type="resnet_basic", prune_ratio=0.5):
        self.sequential = sequential
        self.sequential_length = len(list(self.sequential))
        self.segment_prune = []
        self.finish_flag = False
        self.prune_ratio = prune_ratio
        for i in range(self.sequential_length):
            self.segment_prune.append(SegmentPrune(
                self.sequential[i], segment_type=seq_type, prune_ratio=self.prune_ratio))
        self.segment_count = 0

        print("|===>init ComplexSeqPrune")

    def feature_extract(self, x):
        for i in range(self.segment_count):
            x = self.segment_prune[i](x)
        self.segment_prune[self.segment_count].feature_extract(x)

    def channel_select(self):
        # segment level prune
        select_channels = self.segment_prune[self.segment_count].channel_select(
        )

        # prune previous segment
        if select_channels is not None and self.segment_count > 0:
            self.segment_prune[self.segment_count -
                               1].extra_prune(select_channels)
            select_channels = None

        # replace original segment
        temp_seq = []
        for i in range(self.sequential_length):
            if i == self.segment_count:
                temp_seq.append(self.segment_prune[self.segment_count].segment)
            else:
                temp_seq.append(self.sequential[i])
        self.sequential = nn.Sequential(*temp_seq)

        if self.segment_prune[self.segment_count].finish_flag:
            self.segment_count += 1
        if self.segment_count == self.sequential_length:
            self.finish_flag = True
        return select_channels

    def extra_prune(self, select_channels):

        self.segment_prune[-1].extra_prune(select_channels)
        # replace original segment
        temp_seq = []
        for i in range(self.sequential_length - 1):
            temp_seq.append(self.sequential[i])
        temp_seq.append(self.segment_prune[-1].segment)
        self.sequential = nn.Sequential(*temp_seq)

    def __call__(self, x):
        out = self.sequential(x)
        return out


# -------------------------------------------------------------------------------------------------------------
class ModelPrune:
    def __init__(self, model, trainer, net_type, thi_loader=None, prune_ratio=0.5, fine_tuning=2):
        self.model = model
        self.trainer = trainer
        self.train_loader = thi_loader or self.trainer.train_loader
        self.net_type = net_type
        self.segment_prune = None
        self.prune_ratio = prune_ratio
        self.fine_tuning = fine_tuning
        print("|===>init ModelPrune")

    def run(self):
        # divide model into several segment
        if self.net_type in ["LeNet5", "VGG", "AlexNet"]:
            self.segment_prune = [SegmentPrune(self.model.features, segment_type="sequential", skip_layer=True, prune_ratio=self.prune_ratio),
                                  SegmentPrune(self.model.classifier, segment_type="sequential", prune_ratio=self.prune_ratio)]

            segment_count = 0
            while segment_count < len(self.segment_prune)-1:
                # feature extraction
                for i, (images, labels) in enumerate(self.train_loader):
                    images = images.cuda()
                    x = Variable(images, volatile=True)
                    for j in range(segment_count):
                        x = self.segment_prune[j](x)
                    self.segment_prune[segment_count].feature_extract(x)
                    print(("# feature_extract ", i))
                    # break
                # segment level prune
                select_channels = self.segment_prune[segment_count].channel_select(
                )

                # prune previous segment
                if select_channels is not None and segment_count > 0:
                    self.segment_prune[segment_count -
                                       1].extra_prune(select_channels)
                    self.model.features = self.segment_prune[0].segment

                # replace old segment
                if segment_count == 0:
                    self.model.features = self.segment_prune[0].segment
                    self.model.cuda()
                elif segment_count == 1:
                    self.model.classifier = self.segment_prune[1].segment
                    self.model.cuda()
                else:
                    assert False, "segment count out of range: %d" % segment_count

                if self.segment_prune[segment_count].finish_flag:
                    segment_count += 1

                self.trainer.reset_model(self.model)
                # fine-tuning
                for iters in range(self.fine_tuning):
                    self.trainer.train(epoch=iters)
                    self.trainer.test(epoch=iters)
                print(("|===>%d new model:" % segment_count))
                print((self.model))
                self.model = self.trainer.model.module
                self.model.cuda()

        elif self.net_type in ["resnet_basic", "resnet_bottleneck"]:
            """
            architecture of ImageNet-ResNet-<18-34>
            ResNet(
                conv1, bn1, relu, maxpool
                layer1: sequential(
                    basicblock(
                        conv1, bn1, relu, conv2, bn2, relu
                          |----downsample---------|
                        downsample(if exist)(
                            conv, bn
                        )
                    )
                )
                layer2: ...
                layer3: ... 
                layer4: ... 
                avgpool
                fc
            )

            architecture of ImageNet-ResNet-<50-152-?>
            ResNet(
                conv1, bn1, relu, maxpool
                layer1: sequential(
                    basicblock(
                        conv1, bn1, relu, conv2, bn2, relu, conv3, bn3, relu
                          |----downsample---------|
                        downsample(if exist)(
                            conv, bn
                        )
                    )
                )
                layer2: ...
                layer3: ... 
                layer4: ... 
                avgpool
                fc
            )
            """
            complex_seq_prune = [
                ComplexSeqPrune(
                    self.model.layer1, seq_type=self.net_type, prune_ratio=self.prune_ratio),
                ComplexSeqPrune(
                    self.model.layer2, seq_type=self.net_type, prune_ratio=self.prune_ratio),
                ComplexSeqPrune(
                    self.model.layer3, seq_type=self.net_type, prune_ratio=self.prune_ratio),
                ComplexSeqPrune(
                    self.model.layer4, seq_type=self.net_type, prune_ratio=self.prune_ratio),
            ]
            layer_prune = FilterPrune(prune_ratio=self.prune_ratio)

            segment_count = 0
            while segment_count < 5:
                for i, (images, labels) in enumerate(self.train_loader):
                    images = images.cuda()
                    x = Variable(images, volatile=True)
                    x = self.model.conv1(x)
                    x = self.model.bn1(x)
                    x = self.model.relu(x)
                    x = self.model.maxpool(x)
                    for j in range(segment_count):
                        x = complex_seq_prune[j](x)
                    if segment_count < 4:
                        # print "|===>%d: x size:" % segment_count, x.size()
                        complex_seq_prune[segment_count].feature_extract(x)
                    else:
                        x = self.model.avgpool(x)
                        layer_prune.h = x.size(2)
                        layer_prune.w = x.size(3)
                        x = x.view(x.size(0), -1)
                        y = self.model.fc(x)
                        layer_prune.feature_extract(x, y, self.model.fc)
                    # break

                if segment_count < 4:
                    select_channels = complex_seq_prune[segment_count].channel_select(
                    )
                    if select_channels is not None:
                        if segment_count > 0:
                            complex_seq_prune[segment_count -
                                              1].extra_prune(select_channels)
                            if segment_count == 1:
                                self.model.layer1 = complex_seq_prune[0].sequential
                            elif segment_count == 2:
                                self.model.layer2 = complex_seq_prune[1].sequential
                            elif segment_count == 3:
                                self.model.layer3 = complex_seq_prune[2].sequential
                        else:
                            extra_filter_prune = FilterPrune(
                                prune_ratio=self.prune_ratio)
                            extra_filter_prune.select_channels = select_channels
                            thin_weight, thin_bias = extra_filter_prune.get_thin_params(
                                self.model.conv1, 0)
                            self.model.conv1 = replace_layer(
                                self.model.conv1, thin_weight, thin_bias)
                            thin_weight, thin_bias = extra_filter_prune.get_thin_params(
                                self.model.bn1, 0)
                            self.model.bn1 = replace_layer(
                                self.model.bn1, thin_weight, thin_bias)

                    if segment_count == 0:
                        self.model.layer1 = complex_seq_prune[0].sequential
                    elif segment_count == 1:
                        self.model.layer2 = complex_seq_prune[1].sequential
                    elif segment_count == 2:
                        self.model.layer3 = complex_seq_prune[2].sequential
                    elif segment_count == 3:
                        self.model.layer4 = complex_seq_prune[3].sequential
                    if complex_seq_prune[segment_count].finish_flag:
                        segment_count += 1
                else:
                    w_hat = layer_prune.channel_select(self.model.fc)
                    thin_weight, thin_bias = layer_prune.get_thin_params(
                        self.model.fc, 1)
                    self.model.fc = replace_layer(
                        self.model.fc, thin_weight, thin_bias, w_hat)
                    complex_seq_prune[3].extra_prune(
                        layer_prune.select_channels)
                    self.model.layer4 = complex_seq_prune[3].sequential
                    segment_count += 1
                self.model.cuda()
                print(("new model:", self.model))
                self.trainer.reset_model(self.model)

                # fine-tuning
                for iters in range(self.fine_tuning):
                    self.trainer.train(epoch=iters)
                    self.trainer.test(epoch=iters)
        elif self.net_type == "pre_resnet":
            """
            architecture of CIFAR-10/100-PreResNet
            ResNet(
                conv
                layer1: sequential(
                    basicblock(
                        bn1, relu, conv1, bn2, relu, conv1
                          |----downsample-------------|
                        downsample(if exist)=conv
                    )
                )
                layer2: ...
                layer3: ... 
                bn
                relu
                avg_pool
                fc
            )
            """
            complex_seq_prune = [
                ComplexSeqPrune(
                    self.model.layer1, seq_type=self.net_type, prune_ratio=self.prune_ratio),
                ComplexSeqPrune(
                    self.model.layer2, seq_type=self.net_type, prune_ratio=self.prune_ratio),
                ComplexSeqPrune(
                    self.model.layer3, seq_type=self.net_type, prune_ratio=self.prune_ratio),
            ]
            layer_prune = FilterPrune(prune_ratio=self.prune_ratio)

            segment_count = 0
            while segment_count < 3:
                for i, (images, labels) in enumerate(self.train_loader):
                    images = images.cuda()
                    x = Variable(images, volatile=True)
                    x = self.model.conv(x)
                    for j in range(segment_count):
                        x = complex_seq_prune[j](x)
                    if segment_count < 3:
                        # print "|===>%d: x size:" % segment_count, x.size()
                        complex_seq_prune[segment_count].feature_extract(x)
                    else:
                        x = self.model.bn(x)
                        x = self.model.relu(x)
                        x = self.model.avg_pool(x)
                        layer_prune.h = x.size(2)
                        layer_prune.w = x.size(3)
                        x = x.view(x.size(0), -1)
                        y = self.model.fc(x)
                        layer_prune.feature_extract(x, y, self.model.fc)
                    # break

                if segment_count < 3:
                    select_channels = complex_seq_prune[segment_count].channel_select(
                    )
                    if select_channels is not None:
                        if segment_count > 0:
                            complex_seq_prune[segment_count -
                                              1].extra_prune(select_channels)
                            if segment_count == 1:
                                self.model.layer1 = complex_seq_prune[0].sequential
                            elif segment_count == 2:
                                self.model.layer2 = complex_seq_prune[1].sequential
                        else:
                            extra_filter_prune = FilterPrune(
                                prune_ratio=self.prune_ratio)
                            extra_filter_prune.select_channels = select_channels
                            thin_weight, thin_bias = extra_filter_prune.get_thin_params(
                                self.model.conv, 0)
                            self.model.conv = replace_layer(
                                self.model.conv, thin_weight, thin_bias)

                    if segment_count == 0:
                        self.model.layer1 = complex_seq_prune[0].sequential
                    elif segment_count == 1:
                        self.model.layer2 = complex_seq_prune[1].sequential
                    elif segment_count == 2:
                        self.model.layer3 = complex_seq_prune[2].sequential

                    if complex_seq_prune[segment_count].finish_flag:
                        segment_count += 1
                else:
                    w_hat = layer_prune.channel_select(self.model.fc)

                    thin_weight, thin_bias = layer_prune.get_thin_params(
                        self.model.fc, 1)
                    self.model.fc = replace_layer(
                        self.model.fc, thin_weight, thin_bias, w_hat)

                    thin_weight, thin_bias = layer_prune.get_thin_params(
                        self.model.bn, 0)
                    self.model.bn = replace_layer(
                        self.model.bn, thin_weight, thin_bias)

                    complex_seq_prune[2].extra_prune(
                        layer_prune.select_channels)
                    self.model.layer3 = complex_seq_prune[2].sequential
                    segment_count += 1

                print(("new_model is ", self.model))
                self.model.cuda()
                self.trainer.reset_model(self.model)
                # fine-tuning
                for iters in range(self.fine_tuning):
                    self.trainer.train(epoch=iters)
                    self.trainer.test(epoch=iters)
                # self.model = self.trainer.model

        else:
            assert False, "invalid net_type: " + self.net_type
