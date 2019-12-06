#!/usr/bin/env bash
for i in {0..4}
do
    python main.py 189_cifar10_resnet_sgd.hocon $i
done