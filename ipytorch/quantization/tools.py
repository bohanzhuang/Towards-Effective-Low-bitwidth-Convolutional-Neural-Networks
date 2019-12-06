from .TTQ import *


def modelquantize(layers):
    if isinstance(layers, qConv2d) or isinstance(layers, qLinear):
        layers.quantize()