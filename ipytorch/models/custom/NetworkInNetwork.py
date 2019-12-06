import torch
import torch.nn as nn


class NINBlock(nn.Module):
    def __init__(self, in_plane, out_plane, stride=1, kernel_size=3, padding=1, pooling_kernel_size=3,
                 pooling_stride=1, pooling_type="avg", view_flag=False, qtype="middle"):
        """
        init network in network
        :param in_plane: size of input plane
        :param out_plane: size of output plane
        :param stride: stide of convolutional layers, default 1
        :param kernel_size: kernel size of convolutional layers, default 3
        :param padding: default 1
        :param pooling_kernel_size: kernel size of pooling layers, default 3 
        :param pooling_stride: stride of pooling layers, default 1
        :param pooling_type: pooling type, avg: average pooling, max: max pooling, default avg
        :param view_flag: flag for represent whether to view input data into one dimension data
        """
        super(NINBlock, self).__init__()
        self.view_flag = view_flag
        if qtype == "first":
            self.segment = nn.Sequential(
                nn.Conv2d(in_plane, out_plane[0], stride=stride,
                        kernel_size=kernel_size, padding=padding),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_plane[0], out_plane[1], kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_plane[1], out_plane[2], kernel_size=1),
                nn.ReLU(inplace=True)
            )
        elif qtype == "final":
            self.segment = nn.Sequential(
                nn.Conv2d(in_plane, out_plane[0], stride=stride,
                        kernel_size=kernel_size, padding=padding),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_plane[0], out_plane[1], kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_plane[1], out_plane[2], kernel_size=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.segment = nn.Sequential(
                nn.Conv2d(in_plane, out_plane[0], stride=stride,
                        kernel_size=kernel_size, padding=padding),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_plane[0], out_plane[1], kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_plane[1], out_plane[2], kernel_size=1),
                nn.ReLU(inplace=True)
            )
        if pooling_type == "avg":
            self.pooling = nn.AvgPool2d(
                kernel_size=pooling_kernel_size, stride=pooling_stride)
        else:
            self.pooling = nn.MaxPool2d(
                kernel_size=pooling_kernel_size, stride=pooling_stride)

    def forward(self, x):
        """
        forward procedure of model
        :param x: input feature maps
        :return: output feature maps
        """
        out = self.segment(x)
        out = self.pooling(out)
        if self.view_flag:
            out = out.view(out.size(0), -1)
        return out


# this code have not been tested yet.
class NetworkInNetwork(nn.Module):
    """
    Network in Network
    """

    def __init__(self):
        super(NetworkInNetwork, self).__init__()

        self.segment = nn.Sequential(
            NINBlock(in_plane=3, out_plane=[192, 160, 96], kernel_size=5, padding=2, stride=1,
                     pooling_kernel_size=3, pooling_stride=2, pooling_type="max", qtype="first"),
            nn.Dropout(),
            NINBlock(in_plane=96, out_plane=[192, 192, 192], kernel_size=5, padding=2, stride=1,
                     pooling_kernel_size=3, pooling_stride=2, pooling_type="avg"),
            nn.Dropout(),
            NINBlock(in_plane=192, out_plane=[192, 192, 10], kernel_size=3, padding=1, stride=1,
                     pooling_kernel_size=7, pooling_stride=1, pooling_type="avg", view_flag=True, qtype="final"),
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.segment(x)
        return out
