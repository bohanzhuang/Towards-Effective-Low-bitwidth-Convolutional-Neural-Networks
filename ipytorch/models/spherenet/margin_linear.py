import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn.init as init
from torch.autograd import Function
from torch.autograd import Variable

__all__ = ["MarginLinear"]


class MarginLinear(nn.Module):
    def __init__(self, num_output=10572, num_features=512,
                 margin_inner_product_type=4):
        super(MarginLinear, self).__init__()

        self.base = 1000
        self.gamma = 0.12
        self.power = 1
        self.lambda_min = 5.0
        
        self.register_buffer("iteration", torch.zeros(1))
        self.current_lambda = Variable(torch.ones(1))
        self.margin_type = Variable(torch.Tensor(
            1).fill_(margin_inner_product_type))

        self.weight = nn.Parameter(torch.zeros(num_output, num_features))
        self.bias = None
        # init margin inner product weight
        init.xavier_normal(self.weight)

    def forward(self, x, target):
        
        self.iteration += 1.
        iteration = self.iteration[0]
        self.current_lambda.data.fill_(max(
            self.base * math.pow(1. + self.gamma * iteration, -self.power), self.lambda_min))
        
        # self.current_lambda.data.fill_(self.lambda_min)

        # normalize weight
        self.weight.data.copy_(F.normalize(self.weight.data, p=2, dim=1))
        out = F_MarginLinear()(x, target, self.weight, self.current_lambda, self.margin_type)
        # print self.iteration, self.current_lambda
        return out


class F_MarginLinear(Function):
    """MarginLinear class

    Note: This class define custom forward and backward propagation function
    """

    def forward(self, x, target, weight, current_lambda, margin_type):
        # variables need to be saved
        """x_norm = None
        cos_theta = None"""
        cos_theta_quadratic = None
        cos_theta_cubic = None
        cos_theta_quartic = None
        sign_0 = None
        sign_1 = None
        sign_2 = None
        sign_3 = None
        sign_4 = None
        # common variables

        # cos_theta = x'w/|x|
        xw = x.mm(weight.t())

        # x_norm_ = |x|
        x_norm = torch.norm(x, 2, 1).unsqueeze(1).expand_as(xw)

        cos_theta = xw / x_norm
        # sign_0 = sign(cos_theta)
        sign_0 = torch.sign(cos_theta)

        # optional variables
        # single
        if margin_type[0] == 1:
            pass
        # double
        elif margin_type[0] == 2:
            cos_theta_quadratic = torch.pow(cos_theta, 2)
        # triple
        elif margin_type[0] == 3:
            cos_theta_quadratic = torch.pow(cos_theta, 2)
            cos_theta_cubic = torch.pow(cos_theta, 3)
            # sign_1 = sign(abs(cos_theta) - 0.5)
            sign_1 = torch.sign(torch.abs(cos_theta) - 0.5)
            # sign_2 = sign_0 * (1 + sign_1) - 2
            sign_2 = sign_0 * (1. + sign_1) - 2.
        # quadruple
        elif margin_type[0] == 4:
            cos_theta_quadratic = torch.pow(cos_theta, 2)
            cos_theta_cubic = torch.pow(cos_theta, 3)
            cos_theta_quartic = torch.pow(cos_theta, 4)
            # sign_3 = sign_0 * sign(2 * cos_theta_quadratic_ - 1)
            sign_3 = sign_0 * torch.sign(2. * cos_theta_quadratic - 1.)
            # sign_4 = 2 * sign_0 + sign_3 - 3
            sign_4 = 2. * sign_0 + sign_3 - 3.
        else:
            print('Unknown margin type.')

        # get ont-hot vector
        target_ = torch.unsqueeze(target, 1)
        target_ = target_.expand(x.size(0), weight.size(0))
        y_range = torch.arange(0, weight.size(0)).unsqueeze(
            0).expand(x.size(0), weight.size(0)).type_as(target_).long()

        # save tensor for backward
        self.x = x
        self.target = target
        self.weight = weight
        self.current_lambda = current_lambda[0]
        self.margin_type = margin_type[0]
        self.x_norm = x_norm
        self.cos_theta = cos_theta
        self.cos_theta_quadratic = cos_theta_quadratic
        self.cos_theta_cubic = cos_theta_cubic
        self.cos_theta_quartic = cos_theta_quartic
        self.sign_0 = sign_0
        self.sign_1 = sign_1
        self.sign_2 = sign_2
        self.sign_3 = sign_3
        self.sign_4 = sign_4
        self.target_ = target_
        self.y_range = y_range

        out = xw.clone()
        # single
        if self.margin_type == 1:
            pass
        # double
        elif self.margin_type == 2:
            # |x| * (2 * sign_0 * cos_theta_quadratic - 1)
            out[target_ == y_range] = x_norm[target_ == y_range] * (
                2. * sign_0[target_ == y_range] * cos_theta_quartic[target_ == y_range] - 1.)
            # + lambda * x'w
            out = out + self.current_lambda * xw
            # / (1 + lambda )
            out = out / (1. + self.current_lambda)
        # triple
        elif self.margin_type == 3:
            # |x| * (sign_1 * (4 * cos_theta_cubic - 3 * cos_theta) + sign_2)
            out[target_ == y_range] = x_norm[target_ == y_range] * (sign_1[target_ == y_range] * (
                4. * cos_theta_cubic[target_ == y_range] - 3. * cos_theta[target_ == y_range]) + sign_2[target_ == y_range])
            # + lambda * x'w
            out = out + self.current_lambda * xw
            # / (1 + lambda)
            out = out / (1. + self.current_lambda)
        # quadruple
        elif self.margin_type == 4:
            # |x| * (sign_3 * (8 * cos_theta_quartic - 8 * cos_theta_quadratic + 1) + sign_4)
            out[target_ == y_range] = x_norm[target_ == y_range] * (sign_3[target_ == y_range] * (
                8. * cos_theta_quartic[target_ == y_range] - 8. * cos_theta_quadratic[target_ == y_range] + 1.) + sign_4[target_ == y_range])

            # + lambda * x'w
            out = out + self.current_lambda * xw
            # / (1 + lambda)
            out = out / (1. + self.current_lambda)
        else:
            print('Unknown margin type.')
            exit(-1)
        return out

    def backward(self, grad_output):
        x = self.x
        target = self.target
        weight = self.weight
        current_lambda = self.current_lambda
        margin_type = self.margin_type
        x_norm = self.x_norm
        cos_theta = self.cos_theta
        cos_theta_quadratic = self.cos_theta_quadratic
        cos_theta_cubic = self.cos_theta_cubic
        cos_theta_quartic = self.cos_theta_quartic
        sign_0 = self.sign_0
        sign_1 = self.sign_1
        sign_2 = self.sign_2
        sign_3 = self.sign_3
        sign_4 = self.sign_4
        target_ = self.target_
        y_range = self.y_range

        grad_output_copy_1 = grad_output.clone()
        grad_output_copy_2 = grad_output.clone()
        grad_output_copy_1[target_ == y_range] = 0
        grad_output_copy_2[target_ != y_range] = 0

        grad_input = torch.zeros(x.size()).type_as(grad_output)

        # gradient with respect to weight
        grad_weight = grad_output.t().mm(x)

        # gradient with respect to input
        # single
        grad_input.fill_(0)
        if margin_type == 1:
            grad_input = grad_output.mm(weight)
        # double
        elif margin_type == 2:
            # 1 / (1 + lambda) * w
            grad_input = (1. / (1. + current_lambda) *
                          grad_output_copy_1).mm(weight)

            # 4 * sign_0 * cos_theta * w
            coeff_w = 4. * sign_0 * cos_theta

            # 1 / (-|x|) * (2 * sign_0 * cos_theta_quadratic + 1) * x
            coeff_x = 1. / (-x_norm) * (2. * sign_0 * cos_theta_quadratic + 1.)
            coeff_norm = torch.sqrt(coeff_w * coeff_w + coeff_x * coeff_x)
            coeff_w = coeff_w / coeff_norm
            coeff_x = coeff_x / coeff_norm

            grad_input = grad_input + (
                1. / (1. + current_lambda) *
                grad_output_copy_2 * coeff_w).mm(weight)
            intermediate_value = 1. / (
                1. + current_lambda) * grad_output_copy_2 * coeff_x

            grad_input = grad_input + intermediate_value.sum(
                1).unsqueeze(1).expand_as(x) * x

            # + lambda/(1 + lambda) * w
            grad_input = grad_input + (
                current_lambda / (1. + current_lambda) * grad_output).mm(weight)

        # triple
        elif margin_type == 3:
            # 1 / (1 + lambda) * w
            grad_input = (1. / (
                1 + current_lambda) * grad_output_copy_1).mm(weight)

            # sign_1 * (12 * cos_theta_quadratic - 3) * w
            coeff_w = sign_1 * (12. * cos_theta_quadratic - 3.)
            # 1 / (-|x|) * (8 * sign_1 * cos_theta_cubic - sign_2) * x
            coeff_x = 1. / (-x_norm) * (8. * sign_1 * cos_theta_cubic - sign_2)

            coeff_norm = torch.sqrt(coeff_w * coeff_w + coeff_x * coeff_x)

            coeff_w = coeff_w / coeff_norm
            coeff_x = coeff_x / coeff_norm

            grad_input = grad_input + (
                1. / (1. + current_lambda) * grad_output_copy_2 *
                coeff_w).mm(weight)

            intermediate_value = 1. / (
                1. + current_lambda) * grad_output_copy_2 * coeff_x

            grad_input = grad_input + intermediate_value.sum(1).unsqueeze(
                1).expand_as(x) * x

            # + lambda/(1 + lambda) * w
            grad_input = grad_input + (
                current_lambda / (1. + current_lambda) * grad_output).mm(weight)

        # quadruple
        elif margin_type == 4:
            grad_input = grad_input + (1. / (1. + current_lambda) *
                                       grad_output_copy_1).mm(weight)

            # sign_3 * (32 * cos_theta_cubic - 16 * cos_theta) * w
            coeff_w = sign_3 * (32. * cos_theta_cubic - 16. * cos_theta)
            # 1 / (-|x|) * (sign_3 * (24 * cos_theta_quartic - 8 * cos_theta_quadratic - 1) - sign_4) * x
            coeff_x = -1. / x_norm * (
                sign_3 * (24. * cos_theta_quartic - 8. * cos_theta_quadratic - 1.) - sign_4)

            coeff_norm = torch.sqrt(coeff_w * coeff_w + coeff_x * coeff_x)
            coeff_w = coeff_w / coeff_norm
            coeff_x = coeff_x / coeff_norm

            # print coeff_w.size(), coeff_x.size()
            grad_input = grad_input + (
                1. / (1. + current_lambda) * grad_output_copy_2 *
                coeff_w).mm(weight)

            intermediate_value = 1. / (
                1. + current_lambda) * grad_output_copy_2 * coeff_x
            # print intermediate_value.size()
            grad_input = grad_input + intermediate_value.sum(1).unsqueeze(
                1).expand_as(x) * x

            # + lambda/(1 + lambda) * w
            grad_input = grad_input + (
                current_lambda / (1. + current_lambda) * grad_output).mm(weight)

        # assert False
        return grad_input, None, grad_weight, None, None
