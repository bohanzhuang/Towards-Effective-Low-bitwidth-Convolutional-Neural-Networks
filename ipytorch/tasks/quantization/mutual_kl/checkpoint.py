"""
TODO: write doc for this module
"""

import os

import torch
import torch.nn as nn

import ipytorch.utils as utils
from ipytorch.checkpoint import CheckPoint

'''
Notes:
# save and load entire model
torch.save(model, "model.pth")
model = torch.load("model.pth")
# save and load only the model parameters(recommended)
torch.save(model.state_dict(), "params.pth")
model.load_state_dict(torch.load("params.pth"))
'''
__all__ = ["CheckPoint"]
class GuidedCheckPoint(CheckPoint):
    """
    save model state to file
    check_point_params: ori_model, quan_model, ori_optimizer, quan_optimizer, epoch
    """

    def __init__(self, save_path, quantization_k, logger):

        self.save_path = os.path.join(save_path, "check_point")
        self.check_point_params = {'ori_model': None,
                                   'ori_optimizer': None,
                                   'epoch': None}
        self.quantization_k = quantization_k
        for i in range(len(quantization_k)):
            self.check_point_params['quan_model_{}bit'.format(quantization_k[i])] = None
            self.check_point_params['quan_optimizer_{}bit'.format(quantization_k[i])] = None
        self.logger = logger

        # make directory
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)

    def load_checkpoint(self, checkpoint_path):
        """
        load checkpoint file
        :params checkpoint_path: path to the checkpoint file
        :return: ori_model_state_dict, quan_model_state_dict, optimizer_state_dict, epoch
        """
        if os.path.isfile(checkpoint_path):
            self.logger.info("|===>Load resume check-point from: {}".format(checkpoint_path))
            self.check_point_params = torch.load(checkpoint_path)
            ori_model_state_dict = self.check_point_params['ori_model']
            quan_model_state_dict = self.check_point_params['quan_model']
            ori_optimizer_state_dict = self.check_point_params['ori_optimizer']
            quan_optimizer_state_dict = self.check_point_params['quan_optimizer']
            epoch = self.check_point_params['epoch']
            return ori_model_state_dict, quan_model_state_dict, ori_optimizer_state_dict, quan_optimizer_state_dict, epoch
        else:
            assert False, "file not exits" + checkpoint_path

    def save_checkpoint(self, ori_model, quan_model, ori_optimizer, quan_optimizer, epoch, index=0):
        """
        :params ori_model: original_model
        :params quan_model: quantized_model
        :params ori_optimizer: ori_optimizer
        :params quan_optimizer: quan_optimizer
        :params epoch: training epoch
        :params index: index of saved file, default: 0
        Note: if we add hook to the grad by using register_hook(hook), then the hook function
        can not be saved so we need to save state_dict() only. Although save state dictionary
        is recommended, some times we still need to save the whole model as it can save all
        the information of the trained model, and we do not need to create a new network in
        next time. However, the GPU information will be saved too, which leads to some issues
        when we use the model on different machine
        """

        # get state_dict from original model, quantized model and optimizer
        ori_model = utils.list2sequential(ori_model)
        if isinstance(ori_model, nn.DataParallel):
            ori_model = ori_model.module
        ori_model = ori_model.state_dict()
        save_quan_model_list = []
        save_quan_optimizer_list = []
        for i in range(len(self.quantization_k)):
            quan_model_temp = utils.list2sequential(quan_model[i])
            if isinstance(quan_model_temp, nn.DataParallel):
                quan_model_temp = quan_model_temp.module
            quan_model_temp = quan_model_temp.state_dict()
            save_quan_model_list.append(quan_model_temp)
            save_quan_optimizer_list.append(quan_optimizer[i].state_dict())
        ori_optimizer = ori_optimizer.state_dict()

        # save information to a dict
        self.check_point_params['ori_model'] = ori_model
        self.check_point_params['quan_model'] = save_quan_model_list
        self.check_point_params['ori_optimizer'] = ori_optimizer
        self.check_point_params['quan_optimizer'] = save_quan_optimizer_list
        self.check_point_params['epoch'] = epoch

        # save to file
        torch.save(self.check_point_params, os.path.join(
            self.save_path, "checkpoint_%03d.pth" % index))
