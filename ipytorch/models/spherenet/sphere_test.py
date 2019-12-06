# this code is writen by liujing
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

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
            assert False, "invalid depth: %d, only support: 4, 10, 20, 38, 64"%depth

        self.depth = depth

        # Network parameters
        block = SphereBlock
        self.base = 1000
        # self.gamma = 0.000003
        # self.power = 45

        self.gamma = 0.12
        self.power = 1
        self.lambda_min = 5
        self.current_lambda = 1000
        self.iteration = 0
        self.margin_inner_product_type = margin_inner_product_type
        self.num_features = num_features
        self.num_output = num_output

        # define network structure
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.layer1 = self._make_layer(block, 64, layers[0])

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.relu3 = nn.ReLU()
        
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.relu4 = nn.ReLU()
        
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.fc = nn.Linear(512 * 7 * 6, num_features)

        self.margin_inner_product_weight = nn.Parameter(torch.zeros(num_features, num_output))
        self._init_weight()

    def _init_weight(self):
        # init conv1
        init.xavier_normal(self.conv1.weight)
        init.constant(self.conv1.bias, 0)
        # init conv2
        init.xavier_normal(self.conv2.weight)
        init.constant(self.conv2.bias, 0)
        # init conv3
        init.xavier_normal(self.conv3.weight)
        init.constant(self.conv3.bias, 0)
        # init conv4
        init.xavier_normal(self.conv4.weight)
        init.constant(self.conv4.bias, 0)
        # init fc
        init.xavier_normal(self.fc.weight)
        init.constant(self.fc.bias, 0)
        # init margin inner product weight
        init.xavier_normal(self.margin_inner_product_weight)

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

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # x = self.margin_linear(x, target)
        # return x

        if target is not None:
            self.iteration += 1
            self.current_lambda = self.base * math.pow(1.0 + self.gamma * self.iteration, -self.power)
            self.current_lambda = max(self.current_lambda, self.lambda_min)

            # normalize weight
            # with type of variable, pytorch can compute gradient of norm_weight automatically
            norm_weight = F.normalize(self.margin_inner_product_weight, p=2, dim=0)

            # common variables
            # x_norm_ = |x|
            x_norm = torch.norm(x, 2, 1)
            # cos_theta = xw'/|x|
            xw = x.matmul(norm_weight)
            cos_theta = xw / torch.unsqueeze(x_norm, 1).expand_as(xw)
            # sign_0 = sign(cos_theta)
            sign_0 = torch.sign(cos_theta)

            # optional variables
            if self.margin_inner_product_type == 'single':
                pass
            elif self.margin_inner_product_type == 'double':
                cos_theta_quadratic = torch.pow(cos_theta, 2)
            elif self.margin_inner_product_type == 'triple':
                cos_theta_quadratic = torch.pow(cos_theta, 2)
                cos_theta_cubic = torch.pow(cos_theta, 3)
                # sign_1 = sign(abs(cos_theta) - 0.5)
                sign_1 = torch.sign(torch.abs(cos_theta) - 0.5)
                # sign_2 = sign_0 * (1 + sign_1) - 2
                sign_2 = sign_0 * (1. + sign_1) - 2.
            elif self.margin_inner_product_type == 'quadruple':
                cos_theta_quadratic = torch.pow(cos_theta, 2)
                cos_theta_cubic = torch.pow(cos_theta, 3)
                cos_theta_quartic = torch.pow(cos_theta, 4)
                # sign_3 = sign_0 * sign(2 * cos_theta_quadratic_ - 1)
                sign_3 = sign_0 * torch.sign(2. * cos_theta_quadratic - 1.)
                # sign_4 = 2 * sign_0 + sign_3 - 3
                sign_4 = 2. * sign_0 + sign_3 - 3.
            else:
                print('Unknown margin type.')

            feature = x.clone()
            x = x.matmul(norm_weight)

            target_ = torch.unsqueeze(target, 1)
            y_range = torch.arange(0, self.num_output).long()
            if target_.is_cuda:
                y_range = y_range.cuda()
            y_range = torch.unsqueeze(y_range, 0).expand(x.size(0), self.num_output)

            if self.margin_inner_product_type == 'single':
                pass
            elif self.margin_inner_product_type == 'double':
                # |x| * (2 * sign_0 * cos_theta_quadratic - 1)
                x[target_.data == y_range] = x_norm * (2. * sign_0[target_.data == y_range] * cos_theta_quadratic[target_.data == y_range] - 1)
                # + lambda * x'w
                xw = feature.matmul(norm_weight)
                x = x + self.current_lambda * xw
                # / (1 + lambda )
                x = x / (1. + self.current_lambda)
            elif self.margin_inner_product_type == 'triple':
                # |x| * (sign_1 * (4 * cos_theta_cubic - 3 * cos_theta) + sign_2)
                x[target_.data == y_range] = x_norm * (sign_1[target_.data == y_range] * (4. * cos_theta_cubic[target_.data == y_range]
                                                                                          - 3. * cos_theta[target_.data == y_range]) + sign_2[target_.data == y_range])
                # + lambda * x'w
                xw = feature.matmul(norm_weight)
                x = x + self.current_lambda * xw
                # / (1 + lambda)
                x = x / (1. + self.current_lambda)
            elif self.margin_inner_product_type == 'quadruple':
                # |x| * (sign_3 * (8 * cos_theta_quartic - 8 * cos_theta_quadratic + 1) + sign_4)
                x[target_.data == y_range] = x_norm * (sign_3[target_.data == y_range] * (8. * cos_theta_quartic[target_.data == y_range]
                                              - 8. * cos_theta_quadratic[target_.data == y_range] + 1.) + sign_4[target_.data == y_range])
                # + lambda * x'w
                xw = feature.matmul(norm_weight)
                x = x + self.current_lambda * xw
                # / (1 + lambda)
                x = x / (1. + self.current_lambda)
            else:
                print('Unknown margin type.')
                exit(-1)
            return x
        else:
            return x

