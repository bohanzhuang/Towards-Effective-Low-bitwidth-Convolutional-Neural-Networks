import torch
import torch.nn as nn
import math

from ipytorch.models.dqn.dorefa import QConv2d, QLinear, QReLU, QuantizationGradientFunction

__all__ = ["DoReFaNetSVHN"]

class DoReFaNetSVHN(nn.Module):
    def __init__(self, num_classes=10):
        super(DoReFaNetSVHN, self).__init__()
        self.conv1 = nn.Conv2d(3, 48, kernel_size=5, bias=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = QReLU(2, inplace=True)

        self.conv2 = QConv2d(48, 64, kernel_size=3, padding=1, k=1)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-4)

        self.conv3 = QConv2d(64, 64, kernel_size=3, padding=1, k=1)
        self.bn2 = nn.BatchNorm2d(64, eps=1e-4)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4 = QConv2d(64, 128, kernel_size=3, k=1)
        self.bn3 = nn.BatchNorm2d(128, eps=1e-4)

        self.conv5 = QConv2d(128, 128, kernel_size=3, padding=1, k=1)
        self.bn4 = nn.BatchNorm2d(128, eps=1e-4)

        self.conv6 = QConv2d(128, 128, kernel_size=3, k=1)
        self.bn5 = nn.BatchNorm2d(128, eps=1e-4)

        self.dropout = nn.Dropout()
        self.conv7 = QConv2d(128, 512, kernel_size=5, k=1)
        self.bn6 = nn.BatchNorm2d(512, eps=1e-4)

        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = QuantizationGradientFunction.apply(x, x.new([4]))
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = QuantizationGradientFunction.apply(x, x.new([4]))
        x = self.bn2(x)
        x = self.pool2(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = QuantizationGradientFunction.apply(x, x.new([4]))
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = QuantizationGradientFunction.apply(x, x.new([4]))
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv6(x)
        x = QuantizationGradientFunction.apply(x, x.new([4]))
        X = self.bn5(x)
        x = self.relu(x)

        x = self.dropout(x)
        x = self.conv7(x)
        x = QuantizationGradientFunction.apply(x, x.new([4]))
        x = self.bn6(x)
        x = torch.clamp(x, 0, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
