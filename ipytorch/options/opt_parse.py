import argparse
import torch

__all__ = ["Options"]
class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="PyTorch Experiment Package")

        # ------------- Data options -------------------------------------------
        self.parser.add_argument('--dataPath', type=str,
                                 default='/home/dataset',
                                 metavar='DIR', help='path to data set')

        self.parser.add_argument('--dataset', type=str,
                                 default='cifar10',
                                 metavar='DATA',
                                 help='options: imagenet | cifar10 | cifar100 | imagenet100 | mnist')

        self.parser.add_argument('--nThreads', type=int,
                                 default=4,
                                 metavar='N',
                                 help='number of data loader threads')

        # ------------- GPU options -------------------------------------------
        self.parser.add_argument('--nGPU', type=int,
                                 default=1,
                                 metavar='N',
                                 help='number of GPUs to use, default: 1')

        self.parser.add_argument('--GPU', type=int,
                                 default=0,
                                 metavar='N',
                                 help='index of master GPU, default: 0')

        self.parser.add_argument('--manualSeed', type=int,
                                 default=1,
                                 metavar='N',
                                 help='manually set RNG seed, default: 1')

        # ------------- Training options ---------------------------------------

        self.parser.add_argument('--testOnly', dest='test-only',
                                 action='store_true',
                                 help='run on validation set only')

        self.parser.add_argument('--tenCrop', dest='ten-crop',
                                 action='store_true',
                                 help='Ten-crop testing')

        # ---------- Optimization options --------------------------------------
        self.parser.add_argument('--nEpochs', type=int,
                                 default=1,
                                 metavar='N',
                                 help='number of total epochs to train, default: 1')

        self.parser.add_argument('--batchSize', type=int,
                                 default=64,
                                 metavar='N',
                                 help='mini-batch size, default: 64')

        self.parser.add_argument('--LR', type=float,
                                 default=0.01,
                                 metavar='N',
                                 help='initial learning rate, default: 1')

        self.parser.add_argument('--lrPolicy', type=str,
                                 default='multi_step',
                                 metavar='N',
                                 help='options: multi_step | linear | exp | fixed')

        self.parser.add_argument('--momentum', type=float,
                                 default=0.9,
                                 metavar='M',
                                 help='momentum')

        self.parser.add_argument('--weightDecay', type=float,
                                 default=1e-4,
                                 metavar='N',
                                 help='weight decay, default: 1')

        # ---------- Model options ---------------------------------------------
        self.parser.add_argument('--netType', type=float,
                                 default='PreResNet',
                                 metavar='N',
                                 help='options: ImageNetResNet | PreResNet | NIN | LeNet5 | PlainNet | InceptionResNetV2 | ImagenetResNet')

        self.parser.add_argument('--experimentID', type=float,
                                 default=1e-4,
                                 metavar='N',
                                 help='specific string to index experiment')

        # ---------- Resume or Retrain options ---------------------------------------------
        self.parser.add_argument('--retrain', type=str,
                                 metavar='path',
                                 help='path to model to retrain with')

        self.parser.add_argument('--resume', type=str,
                                 metavar='path',
                                 help='path to directory containing checkpoint')

        # ---------- Visualization options -------------------------------------
        self.parser.add_argument('--drawNetwork', dest='draw_network',
                                 action='store_true',
                                 help='draw graph of the network architecture')

        self.parser.add_argument('--drawInterval', type=int,
                                 default=30,
                                 metavar='N',
                                 help='interval of drawing experimental curves')

        self.extra_option()

        self.args = None
        self.save_path = None
        self.nClasses = 0
        self.depth = 10 # resnet depth: (n-2)%6==0
        self.wideFactor = 1  # wide factor for wide-resnet

        self.torch_version = torch.__version__
        torch_version_split = self.torch_version.split("_")
        self.torch_version = torch_version_split[0]

        self.get_args()

    def get_args(self):
        self.args = self.parser.parse_args()

    def params_check(self):
        assert self.args is not None, "no args"

        if self.torch_version != "0.2.0":
            self.args.drawNetwork = False
            print(("|===>DrawNetwork is supported by PyTorch with version: 0.2.0. The used version is ", self.torch_version))

        self.save_path = "log_%s_%s_bs%d_lr%0.3f_%s/" % (
            self.args.netType, self.args.dataset,
            self.args.batchSize, self.args.LR, self.args.experimentID)

        if self.args.dataset == "cifar10":
            self.nClasses = 10
        elif self.args.dataset == "cifar100":
            self.nClasses = 100
        elif self.args.dataset == "imagenet":
            self.nClasses = 1000
        elif self.args.dataset == "imagenet100":
            self.nClasses = 100

    def extra_option(self):
        pass
