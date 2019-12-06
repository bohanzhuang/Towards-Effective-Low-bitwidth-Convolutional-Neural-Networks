from ipytorch.models.lqnet.lqnet_quant import *

# 32 --> 16 --> 8 --> 4
cfg = [128, 'BN', 128, 'M', 'BN', 256, 'BN', 256, 'M', 'BN', 512, 'BN', 512, 'M', 'BNNoQuant']

__all__ = ["VGG_SMALL"]


class VGG_SMALL(nn.Module):

    def __init__(self, qw, qa, num_classes=10):
        super(VGG_SMALL, self).__init__()
        self.cfg = cfg
        self.qw = qw
        self.qa = qa
        self.features = self.make_layers(self.cfg)
        self.classifier = nn.Linear(512 * 4 * 4, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layers(self, cfg):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v == 'BN':
                layers += [nn.BatchNorm2d(in_channels), QReLU(k=self.qa, inplace=True)]
            elif v == 'BNNoQuant':
                layers += [nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True)]
            else:
                if len(layers) == 0:
                    layers += [nn.Conv2d(in_channels, v, kernel_size=3, padding=1)]
                else:
                    layers += [QConv2d(in_channels, v, kernel_size=3, padding=1, k=self.qw)]
                in_channels = v

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
