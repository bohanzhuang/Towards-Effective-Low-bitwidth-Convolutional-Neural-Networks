import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn import functional as F

MOVING_AVERAGES_FACTOR = 0.9
EPS = 0.0001
NORM_PPF_0_75 = 0.6745


def quantize_w(weight, basis, level_codes, thrs_multiplier, num_levels, num_filters, k):
    # calculate levels and sort
    levels = torch.matmul(level_codes, basis)
    levels, sort_id = torch.topk(levels, num_levels, 0, largest=False)

    # calculate threshold
    thrs = torch.matmul(thrs_multiplier, levels)

    # calculate level codes per channel
    # n c h w -> h c n w -> h w n c -> h w c n
    transpose_w = weight.transpose(0, 2).transpose(1, 3).transpose(2, 3)
    # h w c n -> hwc n
    reshape_w = torch.reshape(transpose_w, (-1, num_filters))
    level_codes_channelwise = weight.new_zeros(num_levels * num_filters, k)
    for i in range(num_levels):
        eq = torch.eq(sort_id, i)
        level_codes_channelwise = torch.where(torch.reshape(eq, (-1, 1)), level_codes_channelwise + level_codes[i],
                                              level_codes_channelwise)
    level_codes_channelwise = torch.reshape(level_codes_channelwise, (num_levels, num_filters, k))

    # calculate output y and its binary code
    y = weight.new_zeros(transpose_w.shape) + levels[0]  # output
    zero_dims = (reshape_w.shape[0] * num_filters, k)
    bits_y = weight.new_full(zero_dims, -1.)
    zero_y = weight.new_zeros(transpose_w.shape)
    zero_bits_y = weight.new_full(zero_dims, 0)
    zero_bits_y = torch.reshape(zero_bits_y, (-1, num_filters, k))
    for i in range(num_levels - 1):
        g = transpose_w > thrs[i]
        y = torch.where(g, zero_y + levels[i + 1], y)
        bits_y = torch.where(torch.reshape(g, (-1, 1)),
                             torch.reshape(zero_bits_y + level_codes_channelwise[i + 1], (-1, k)), bits_y)
    bits_y = torch.reshape(bits_y, (-1, num_filters, k))
    # h w c n -> h w n c -> h c n w -> n c h w
    reshape_y = y.transpose(2, 3).transpose(1, 3).transpose(0, 2)
    return reshape_y, bits_y, reshape_w


def quantize_ac(ac, basis, level_codes, thrs_multiplier, num_levels, k):
    # calculate levels and sort
    levels = torch.matmul(level_codes, basis)
    levels, sort_id = torch.topk(levels, num_levels, 0, largest=False)

    # calculate threshold
    thrs = torch.matmul(thrs_multiplier, levels)

    # calculate output y and its binary code
    y = ac.new_zeros(ac.shape)
    reshape_ac = torch.reshape(ac, (-1,))
    zero_dims = (reshape_ac.shape[0], k)
    bits_y = ac.new_full(zero_dims, 0.)
    zero_y = ac.new_zeros(ac.shape)
    zero_bits_y = ac.new_full(zero_dims, 0.)
    for i in range(num_levels - 1):
        g = ac > thrs[i]
        y = torch.where(g, zero_y + levels[i + 1], y)
        bits_y = torch.where(torch.reshape(g, (-1, 1)), zero_bits_y + level_codes[sort_id[i + 1].item()], bits_y)
    return y, bits_y, reshape_ac, levels


# https://stackoverflow.com/questions/46595157/how-to-apply-the-torch-inverse-function-of-pytorch-to-every-sample-in-the-batc?rq=1
def b_inv(b_mat):
    eye = b_mat.new_ones(b_mat.size(-1)).diag().expand_as(b_mat)
    b_inv, _ = torch.gesv(eye, b_mat)
    return b_inv


class QuantizeW(Function):
    @staticmethod
    def forward(ctx, weight, basis, level_codes, thrs_multiplier, num_levels, num_filters, k):
        reshape_y, bits_y, reshape_w = quantize_w(weight, basis, level_codes, thrs_multiplier,
                                                                  num_levels, num_filters, k)
        return reshape_y, bits_y, reshape_w

    @staticmethod
    def backward(ctx, grad_output, grad_bits_y, grad_reshape_w):
        return grad_output, None, None, None, None, None, None


class QuantizeAC(Function):
    @staticmethod
    def forward(ctx, ac, basis, level_codes, thrs_multiplier, num_levels, k):
        y, bits_y, reshape_ac, levels = quantize_ac(ac, basis, level_codes, thrs_multiplier, num_levels, k)
        return y, bits_y, reshape_ac, levels

    @staticmethod
    def backward(ctx, grad_outputs, grad_bits_y, grad_reshape_ac, grad_levels):
        return grad_outputs, None, None, None, None, None


class QConv2d(nn.Conv2d):
    """
    custom convolutional layers for quantization
    """

    def __init__(self, in_channels, out_channels, kernel_size, k, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(QConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, groups, bias)

        self.num_filters = out_channels
        self.k = k
        self.num_levels = 2 ** k
        self.delta = EPS
        self.n = kernel_size * kernel_size * out_channels
        self.init_basis()

    def init_basis(self):
        init_basis = []
        base = NORM_PPF_0_75 * ((2. / self.n) ** 0.5) / (2 ** (self.k - 1))
        for j in range(self.k):
            init_basis.append([(2 ** j) * base for i in range(self.num_filters)])

        # initialize level multiplier
        init_level_multiplier = []
        for i in range(self.num_levels):
            level_multiplier_i = [0. for j in range(self.k)]
            level_number = i
            for j in range(self.k):
                binary_code = level_number % 2
                if binary_code == 0:
                    binary_code = -1
                level_multiplier_i[j] = float(binary_code)
                level_number = level_number // 2
            init_level_multiplier.append(level_multiplier_i)

        # initialize threshold multiplier
        init_thrs_multiplier = []
        for i in range(1, self.num_levels):
            thrs_multiplier_i = [0. for j in range(self.num_levels)]
            thrs_multiplier_i[i - 1] = 0.5
            thrs_multiplier_i[i] = 0.5
            init_thrs_multiplier.append(thrs_multiplier_i)

        self.register_buffer('basis', torch.FloatTensor(init_basis))
        self.register_buffer('level_codes', torch.FloatTensor(init_level_multiplier))
        self.register_buffer('thrs_multiplier', torch.FloatTensor(init_thrs_multiplier))
        self.register_buffer('sum_multiplier', torch.ones(1, self.kernel_size[0] * self.kernel_size[1] * self.in_channels))
        self.register_buffer('sum_multiplier_basis', torch.ones(1, self.k))

    def train_basis(self, bits_y, reshape_w):
        # bits_y [-1, num_filters, k] -> [k, num_filters, -1] -> [k, -1, num_filters]
        BT = bits_y.transpose(0, 2).transpose(1, 2)
        # calculate BTxB
        BTxB = []
        for i in range(self.k):
            for j in range(self.k):
                BTxBij = BT[i] * BT[j]
                BTxBij = torch.matmul(self.sum_multiplier, BTxBij)
                if i == j:
                    mat_one = self.weight.new_ones(1, self.num_filters)
                    BTxBij = BTxBij + (self.delta * mat_one)  # + E
                BTxB.append(BTxBij)
        BTxB = torch.reshape(torch.stack(BTxB), (self.k, self.k, self.num_filters))
        # calculate inverse of BTxB
        if self.k > 2:
            # kw kh num_filters -> num_filters kh kw -> num_filters kw kh
            BTxB_transpose = BTxB.transpose(0, 2).transpose(1, 2)
            BTxB_inv = b_inv(BTxB_transpose)
            # num_filters kw kh -> num_filters kh kw -> kw kh num_filters
            BTxB_inv = BTxB_inv.transpose(1, 2).transpose(0, 2)
        elif self.k == 2:
            det = BTxB[0][0] * BTxB[1][1] - BTxB[0][1] * BTxB[1][0]
            inv = []
            inv.append(BTxB[1][1] / det)
            inv.append(-BTxB[0][1] / det)
            inv.append(-BTxB[1][0] / det)
            inv.append(BTxB[0][0] / det)
            BTxB_inv = torch.reshape(torch.stack(inv), (self.k, self.k, self.num_filters))
        elif self.k == 1:
            BTxB_inv = torch.reciprocal(BTxB)
        # calculate BTxX
        BTxX = []
        for i in range(self.k):
            BTxXi0 = BT[i] * reshape_w
            BTxXi0 = torch.matmul(self.sum_multiplier, BTxXi0)
            BTxX.append(BTxXi0)
        BTxX = torch.reshape(torch.stack(BTxX), (self.k, self.num_filters))
        BTxX = BTxX + (self.delta * self.basis)  # + basis

        # calculate new basis
        new_basis = []
        for i in range(self.k):
            new_basis_i = BTxB_inv[i] * BTxX
            new_basis_i = torch.matmul(self.sum_multiplier_basis, new_basis_i)
            new_basis.append(new_basis_i)
        new_basis = torch.reshape(torch.stack(new_basis), (self.k, self.num_filters))
        # moving average
        self.basis = MOVING_AVERAGES_FACTOR * self.basis + (1 - MOVING_AVERAGES_FACTOR) * new_basis

    def forward(self, input):
        quantized_weight, bits_y, reshape_w = QuantizeW.apply(self.weight, self.basis,
                                                                              self.level_codes,
                                                                              self.thrs_multiplier,
                                                                              self.num_levels,
                                                                              self.num_filters,
                                                                              self.k)
        self.train_basis(bits_y.detach(), reshape_w.detach())
        return F.conv2d(input, quantized_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class QReLU(nn.ReLU):
    """
    custom ReLU for quantization
    """

    def __init__(self, k, inplace=False):
        super(QReLU, self).__init__(inplace=inplace)

        self.k = k
        self.num_levels = 2 ** k
        self.delta = EPS
        self.init_basis()

    def init_basis(self):
        init_basis = [(NORM_PPF_0_75 * 2 / (2 ** self.k - 1)) * (2. ** i) for i in range(self.k)]

        # initialize level multiplier
        init_level_multiplier = []
        for i in range(0, self.num_levels):
            level_multiplier_i = [0. for j in range(self.k)]
            level_number = i
            for j in range(self.k):
                level_multiplier_i[j] = float(level_number % 2)
                level_number = level_number // 2
            init_level_multiplier.append(level_multiplier_i)

        # initialize threshold multiplier
        init_thrs_multiplier = []
        for i in range(1, self.num_levels):
            thrs_multiplier_i = [0. for j in range(self.num_levels)]
            thrs_multiplier_i[i - 1] = 0.5
            thrs_multiplier_i[i] = 0.5
            init_thrs_multiplier.append(thrs_multiplier_i)

        self.register_buffer('basis', torch.FloatTensor(init_basis))
        self.register_buffer('level_codes', torch.FloatTensor(init_level_multiplier))
        self.register_buffer('thrs_multiplier', torch.FloatTensor(init_thrs_multiplier))
        self.basis = self.basis.reshape(-1, 1)

    def train_basis(self, bits_y, reshape_ac, levels):
        BT = bits_y.t()
        # calculate BTxB
        BTxB = []
        for i in range(self.k):
            for j in range(self.k):
                BTxBij = BT[i] * BT[j]
                BTxBij = torch.sum(BTxBij)
                BTxB.append(BTxBij)
        BTxB = torch.reshape(torch.stack(BTxB), (self.k, self.k))
        # BTxB_inv = torch.inverse(BTxB)
        if self.k > 2:
            BTxB_inv = torch.inverse(BTxB)
        elif self.k == 2:
            det = BTxB[0][0] * BTxB[1][1] - BTxB[0][1] * BTxB[1][0]
            inv = []
            inv.append(BTxB[1][1] / det)
            inv.append(-BTxB[0][1] / det)
            inv.append(-BTxB[1][0] / det)
            inv.append(BTxB[0][0] / det)
            BTxB_inv = torch.reshape(torch.stack(inv), (self.k, self.k))
        elif self.k == 1:
            BTxB_inv = torch.reciprocal(BTxB)
        # calculate BTxX
        BTxX = []
        for i in range(self.k):
            BTxXi0 = BT[i] * reshape_ac
            BTxXi0 = torch.sum(BTxXi0)
            BTxX.append(BTxXi0)
        BTxX = torch.reshape(torch.stack(BTxX), (self.k, 1))

        new_basis = torch.matmul(BTxB_inv, BTxX)  # calculate new basis
        self.basis = MOVING_AVERAGES_FACTOR * self.basis + (1 - MOVING_AVERAGES_FACTOR) * new_basis
        return levels

    def forward(self, input):
        out = F.relu(input, self.inplace)
        ac, bits_y, reshape_ac, levels = QuantizeAC.apply(out, self.basis, self.level_codes,
                                                          self.thrs_multiplier,
                                                          self.num_levels,
                                                          self.k)
        levels = self.train_basis(bits_y.detach(), reshape_ac.detach(), levels.detach())
        out = ac
        out = torch.clamp(out, max=levels[self.num_levels - 1].item())
        return out
