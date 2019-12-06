import torch.nn as nn


class VisualModel(nn.Module):
    """
    use auto encoder for compressing the output dimension
    """
    def __init__(self, in_plane):
        """
        initialize model and weights
        :param in_plane: size of input plane 
        """
        super(VisualModel, self).__init__()
        self.fc_1 = nn.Linear(in_plane, 3)
        self.fc_2 = nn.Linear(3, in_plane)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        """
        forward procedure of model
        :param x: input feature maps
        :return: output feature maps and 3D position after compression
        """
        position = self.fc_1(x)
        out = self.fc_2(position)

        return out, position
