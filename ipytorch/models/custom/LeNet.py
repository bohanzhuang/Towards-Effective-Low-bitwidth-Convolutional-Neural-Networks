import torch
import torch.nn as nn
import torch.nn.init as nnInit
import math

__all__ = ["LeNet5", "LeNet500300"]
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=50 * 4 * 4, out_features=500),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=500, out_features=10)
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nnInit.xavier_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        return out


class LeNet500300(nn.Module):
    def __init__(self):
        super(LeNet500300, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(in_features=28 * 28, out_features=500, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=500, out_features=300, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=300, out_features=10, bias=False),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # nnInit.xavier_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.classifier(out)
        return out
