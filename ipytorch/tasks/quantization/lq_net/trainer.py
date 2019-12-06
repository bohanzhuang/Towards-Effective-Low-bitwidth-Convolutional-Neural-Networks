"""
basic trainer
"""
import time

import torch.autograd
import torch.nn as nn

import ipytorch.utils as utils
from ipytorch.trainer import Trainer
from ipytorch.models.lqnet.lqnet_quant import *

__all__ = ["LQTrainer"]


class LQTrainer(Trainer):
    """
    trainer for training network, use SGD
    """

    def __init__(self, model, lr_master,
                 train_loader, test_loader,
                 settings, logger, tensorboard_logger=None,
                 opt_type="SGD", optimizer_state=None, run_count=0):
        """
        init trainer
        """
        super(LQTrainer, self).__init__(model, lr_master,
                 train_loader, test_loader,
                 settings, logger, tensorboard_logger=None,
                 opt_type="SGD", optimizer_state=None, run_count=0)

    def backward(self, loss):
        """
        backward propagation
        """
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for module in self.model.modules():
            if isinstance(module, (QConv2d, QReLU)):
                module.train_basis()
