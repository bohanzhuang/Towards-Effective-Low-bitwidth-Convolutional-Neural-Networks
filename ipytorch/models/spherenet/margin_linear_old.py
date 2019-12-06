import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn.init as init
from torch.autograd import Function
from torch.autograd import Variable

__all__ = ["MarginLinear"]


class MarginLinear(nn.Module):
    def __init__(self, base=1000, gamma=0.12, power=1,
                 num_output=10572, num_features=512,
                 margin_inner_product_type='quadruple'):
        super(MarginLinear, self).__init__()

        self.base = base
        self.gamma = gamma
        self.power = power
        self.lambda_min = 5.0
        self.current_lambda = 1000
        self.iteration = 0.
        self.current_lambda_var = Variable(torch.ones(1))
        self.margin_type_var = Variable(torch.Tensor(1).fill_(margin_inner_product_type))

        self.num_features = num_features
        self.num_output = num_output

        self.margin_inner_product_weight = nn.Parameter(
            torch.zeros(num_features, num_output))
        # self.f_linear = F_MarginLinear.apply
        # init margin inner product weight
        init.xavier_normal(self.margin_inner_product_weight)

    def forward(self, x, target):
        self.iteration += 1.
        self.current_lambda = self.base * \
            math.pow(1.0 + self.gamma * self.iteration, -self.power)
        self.current_lambda = max(self.current_lambda, self.lambda_min)
        self.current_lambda_var.data.fill_(self.current_lambda)
        # normalize weight
        self.margin_inner_product_weight.data.copy_(F.normalize(self.margin_inner_product_weight.data, p=2, dim=0))
        x = F_MarginLinear()(x, target, self.margin_inner_product_weight,
                             self.current_lambda_var, self.margin_type_var)
        return x


class F_MarginLinear(Function):
    """MarginLinear class

    Note: This class define custom forward and backward propagation function
    """

    def forward(self, x, target, weight, current_lambda, margin_inner_product_type):
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
        # weight = F.normalize(weight, p=2, dim=0, eps=0)
        # common variables
        # x_norm_ = |x|
        x_norm = torch.norm(x, 2, 1)
        # cos_theta = xw'/|x|
        xw = x.mm(weight)
        cos_theta = xw / (torch.unsqueeze(x_norm, 1).expand_as(xw))
        # sign_0 = sign(cos_theta)
        sign_0 = torch.sign(cos_theta)

        # optional variables
        # single
        if margin_inner_product_type[0] == 1:
            pass
        # double
        elif margin_inner_product_type[0] == 2:
            cos_theta_quadratic = torch.pow(cos_theta, 2)
        # triple
        elif margin_inner_product_type[0] == 3:
            cos_theta_quadratic = torch.pow(cos_theta, 2)
            cos_theta_cubic = torch.pow(cos_theta, 3)
            # sign_1 = sign(abs(cos_theta) - 0.5)
            sign_1 = torch.sign(torch.abs(cos_theta) - 0.5)
            # sign_2 = sign_0 * (1 + sign_1) - 2
            sign_2 = sign_0 * (1. + sign_1) - 2.
        # quadruple
        elif margin_inner_product_type[0] == 4:
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
        target_ = target_.expand(x.size(0), weight.size(1))
        y_range = torch.arange(0, weight.size(1)).long()
        if target_.is_cuda:
            y_range = y_range.cuda()
        y_range = torch.unsqueeze(y_range, 0).expand(x.size(0), weight.size(1))

        # save tensor for backward
        self.x = x
        self.target = target
        self.weight = weight
        self.current_lambda = current_lambda[0]
        self.margin_inner_product_type = margin_inner_product_type[0]
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

        feature = x.clone()
        x = x.mm(weight)

        # single
        if self.margin_inner_product_type == 1:
            pass
        # double
        elif self.margin_inner_product_type == 2:
            # |x| * (2 * sign_0 * cos_theta_quadratic - 1)
            x[target_ == y_range] = x_norm * (
                2. * sign_0[target_ == y_range] * cos_theta_quadratic[target_ == y_range] - 1.)
            # + lambda * x'w
            xw = feature.mm(weight)
            x = x + self.current_lambda * xw
            # / (1 + lambda )
            x = x / (1. + self.current_lambda)
        # triple
        elif self.margin_inner_product_type == 3:
            # |x| * (sign_1 * (4 * cos_theta_cubic - 3 * cos_theta) + sign_2)
            x[target_ == y_range] = x_norm * (
                sign_1[target_ == y_range] * (4. * cos_theta_cubic[target_ == y_range]
                                                   - 3. * cos_theta[target_ == y_range]) + sign_2[target_ == y_range])
            # + lambda * x'w
            xw = feature.mm(weight)
            x = x + self.current_lambda * xw
            # / (1 + lambda)
            x = x / (1. + self.current_lambda)
        # quadruple
        elif self.margin_inner_product_type == 4:
            # |x| * (sign_3 * (8 * cos_theta_quartic - 8 * cos_theta_quadratic + 1) + sign_4)
            x[target_ == y_range] = x_norm * (
                sign_3[target_ == y_range] * (8. * cos_theta_quartic[target_ == y_range]
                                                   - 8. * cos_theta_quadratic[target_ == y_range] + 1.) + sign_4[target_ == y_range])
            # + lambda * x'w
            xw = feature.mm(weight)
            x = x + self.current_lambda * xw
            # / (1 + lambda)
            x = x / (1. + self.current_lambda)
        else:
            print('Unknown margin type.')
            exit(-1)
        return x

    def backward(self, grad_output):
        x = self.x
        target = self.target
        weight = self.weight
        current_lambda = self.current_lambda
        margin_inner_product_type = self.margin_inner_product_type
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

        grad_input = grad_weight = None

        grad_output_copy_1 = grad_output.clone()
        grad_output_copy_2 = grad_output.clone()
        grad_output_copy_1[target_ == y_range] = 0
        grad_output_copy_2[target_ != y_range] = 0

        grad_input = torch.zeros(grad_output.shape[0], weight.shape[0])
        if grad_output.is_cuda:
            grad_input = grad_input.cuda()

        # gradient with respect to input
        # single
        if margin_inner_product_type == 1:
            grad_input = grad_output.mm(weight.t())
        # double
        elif margin_inner_product_type == 2:
            # 1 / (1 + lambda) * w
            grad_input = (1. / 1. + current_lambda) * \
                grad_output_copy_1.mm(weight.t())

            # 4 * sign_0 * cos_theta * w
            coeff_w = 4. * sign_0 * cos_theta
            # 1 / (-|x|) * (2 * sign_0 * cos_theta_quadratic + 1) * x
            coeff_x = 1. / (-x_norm) * (2 * sign_0 * cos_theta_quadratic + 1)
            coeff_norm = torch.sqrt(coeff_w * coeff_w + coeff_x * coeff_x)
            coeff_w = coeff_w / coeff_norm
            coeff_x = coeff_x / coeff_norm
            grad_input = grad_input + \
                (1. / 1. + current_lambda) * \
                grad_output_copy_2 * coeff_w.mm(weight.t())
            grad_input = grad_input + \
                (1. / 1. + current_lambda) * \
                grad_output_copy_2 * (coeff_x.t().mm(x))
            # + lambda/(1 + lambda) * w
            grad_input = grad_input + \
                (1. / 1. + current_lambda) * grad_output.mm(weight.t())
        # triple
        elif margin_inner_product_type == 3:
            # 1 / (1 + lambda) * w
            grad_input = (1. / 1. + current_lambda) * \
                grad_output_copy_1.mm(weight.t())

            # sign_1 * (12 * cos_theta_quadratic - 3) * w
            coeff_w = sign_1 * (12 * cos_theta_quadratic - 3)
            # 1 / (-|x|) * (8 * sign_1 * cos_theta_cubic - sign_2) * x
            coeff_x = 1. / (-x_norm) * (8 * sign_1 * cos_theta_cubic - sign_2)
            coeff_norm = torch.sqrt(coeff_w * coeff_w + coeff_x * coeff_x)
            coeff_w = coeff_w / coeff_norm
            coeff_x = coeff_x / coeff_norm
            grad_input = grad_input + \
                (1. / 1. + current_lambda) * \
                grad_output_copy_2 * coeff_w.mm(weight.t())
            grad_input = grad_input + \
                (1. / 1. + current_lambda) * \
                grad_output_copy_2 * (coeff_x.t().mm(x))
            # + lambda/(1 + lambda) * w
            grad_input = grad_input + \
                (1. / 1. + current_lambda) * grad_output.mm(weight.t())
        # quadruple
        elif margin_inner_product_type == 4:
            grad_input = grad_input + \
                (1. / (1. + current_lambda) * grad_output_copy_1).mm(weight.t())

            # sign_3 * (32 * cos_theta_cubic - 16 * cos_theta) * w
            coeff_w = sign_3 * (32. * cos_theta_cubic - 16. * cos_theta)
            # 1 / (-|x|) * (sign_3 * (24 * cos_theta_quartic - 8 * cos_theta_quadratic - 1) - sign_4) * x
            coeff_x = 1. / (-torch.unsqueeze(x_norm, 1).expand_as(sign_3)) * (
                sign_3 * (24. * cos_theta_quartic - 8. * cos_theta_quadratic - 1.) - sign_4)
  
            coeff_norm = torch.sqrt(coeff_w * coeff_w + coeff_x * coeff_x)
            coeff_w = coeff_w / coeff_norm
            coeff_x = coeff_x / coeff_norm

            # print coeff_w.size(), coeff_x.size()
            grad_input += (1. / (1. + current_lambda) * grad_output_copy_2 * coeff_w).mm(weight.t())
            
            intermediate_value = 1./(1. + current_lambda) * grad_output_copy_2 * coeff_x
            # print intermediate_value.size()
            grad_input += x*intermediate_value.sum(1).unsqueeze(1).expand_as(x)

            # + lambda/(1 + lambda) * w
            grad_input += (current_lambda / (1. + current_lambda) * grad_output).mm(weight.t())

        # gradient with respect to weight
        grad_weight = x.t().mm(grad_output)
        # assert False
        return grad_input, None, grad_weight, None, None
