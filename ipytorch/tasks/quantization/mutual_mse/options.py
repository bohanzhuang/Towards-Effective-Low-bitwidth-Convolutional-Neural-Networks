import os
import shutil

from pyhocon import ConfigFactory

from ipytorch.options import NetOption


class Option(NetOption):
    def __init__(self, conf_path):
        super(Option, self).__init__()
        self.conf = ConfigFactory.parse_file(conf_path)
        #  ------------ General options ----------------------------------------
        self.save_path = self.conf['save_path']
        self.dataPath = self.conf['dataPath']  # path for loading data set
        self.dataset = self.conf['dataset']  # options: imagenet | cifar10 | cifar100 | imagenet100 | mnist
        self.nGPU = self.conf['nGPU']  # number of GPUs to use by default
        self.GPU = self.conf['GPU']  # default gpu to use, options: range(nGPU)
        self.visible_devices = self.conf['visible_devices']

        # ------------- Data options -------------------------------------------
        self.nThreads = self.conf['nThreads']  # number of data loader threads

        # ---------- Optimization options --------------------------------------
        self.nEpochs = self.conf['nEpochs']  # number of total epochs to train 400
        self.batchSize = self.conf['batchSize']  # mini-batch size 128
        self.momentum = self.conf['momentum']  # momentum 0.9
        self.weightDecay = float(self.conf['weightDecay'])  # weight decay 1e-4
        self.ori_opt_type = self.conf['ori_opt_type']
        self.quan_opt_type = self.conf['quan_opt_type']

        # lr master for optimizer 1 (mask vector d)
        self.ori_lr = self.conf['ori_lr']  # initial learning rate
        self.quan_lr = self.conf['quan_lr']
        self.lrPolicy = self.conf['lrPolicy']  # options: multi_step | linear | exp | const | step
        self.power = self.conf['power']  # power for inv policy (lr_policy)
        self.step = self.conf['step']  # step for linear or exp learning rate policy
        self.decayRate = self.conf['decayRate']  # lr decay rate
        self.endlr = self.conf['endlr']

        # ---------- Model options ---------------------------------------------
        self.netType = self.conf[
            'netType']  # options: ResNet | PreResNet | GreedyNet | NIN | LeNet5 | LeNet500300 | DenseNet_Cifar | AlexNet
        self.experimentID = self.conf['experimentID']
        self.depth = self.conf['depth']  # resnet depth: (n-2)%6==0
        self.nClasses = self.conf['nClasses']  # number of classes in the dataset
        self.wideFactor = self.conf['wideFactor']  # wide factor for wide-resnet
        self.drawNetwork = self.conf['drawNetwork']

        # ---------- Quantization options ---------------------------------------------
        self.quantization_k = self.conf['quantization_k']
        self.alpha = float(self.conf['alpha'])
        self.beta = float(self.conf['beta'])
        self.gamma = float(self.conf['gamma'])

        # ---------- Resume or Retrain options ---------------------------------------------
        self.resume = None if len(self.conf['resume']) == 0 else self.conf['resume']  # "./checkpoint_064.pth"
        self.ori_retrain = None if len(self.conf['ori_retrain']) == 0 else self.conf['ori_retrain']
        self.quan_retrain = None if len(self.conf['quan_retrain']) == 0 else self.conf['quan_retrain']

    def set_save_path(self):
        if self.netType in ["PreResNet", "ResNet"]:
            self.save_path = self.save_path + "log_{}{:d}_{}_bs{:d}_orilr{:.6f}_quanlr{:.6f}_TELCNN_baseline_bit{}_epoch{}_alpha{}_beta{}_gamma{}_{}/".format(
                self.netType, self.depth, self.dataset,
                self.batchSize, self.ori_lr, self.quan_lr, self.quantization_k,
                self.nEpochs, self.alpha, self.beta, self.gamma, self.experimentID)
        else:

            self.save_path = self.save_path + "log_{}_{}_bs{:d}_orilr{:.6f}_quanlr{:.6f}_TELCNN_baseline_bit{}_epoch{}_alpha{}_beta{}_gamma{}_{}/".format(
                self.netType, self.dataset,
                self.batchSize, self.ori_lr, self.quan_lr, self.quantization_k,
                self.nEpochs, self.alpha, self.beta, self.gamma, self.experimentID)

        if os.path.exists(self.save_path):
            print("{} file exist!".format(self.save_path))
            action = input("Select Action: d (delete) / q (quit):").lower().strip()
            act = action
            if act == 'd':
                shutil.rmtree(self.save_path)
            else:
                raise OSError("Directory {} exits!".format(self.save_path))

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def paramscheck(self, logger):
        if self.torch_version != "0.2.0":
            self.drawNetwork = False
            logger.info("|===>DrawNetwork is supported by PyTorch with version: 0.2.0. The used version is {}".format(self.torch_version))

        if self.dataset in ["cifar10", "mnist"]:
            self.nClasses = 10
        elif self.dataset == "cifar100":
            self.nClasses = 100
        elif self.dataset == "imagenet" or "thi_imgnet":
            self.nClasses = 1000
        elif self.dataset == "imagenet100":
            self.nClasses = 100

        if self.depth >= 100:
            self.drawNetwork = False
            logger.info("|===>draw network with depth over 100 layers, skip this step")