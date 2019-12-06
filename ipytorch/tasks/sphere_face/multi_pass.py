import torch
import sys
import os
# os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = "4, 5, 6, 7"
import datetime
import time
from termcolor import colored
from torch.autograd import Variable
import numpy as np

import torch.nn as nn
from dataloader import DataLoader
from checkpoint import CheckPoint
import models as md
import visualization as vs
from sphere_trainer import SphereTrainer as Trainer
import utils
import prune
from options import Option


class ExperimentDesign:
    def __init__(self, options=None):
        self.settings = options or Option()
        self.checkpoint = None
        self.data_loader = None
        self.model = None

        self.optimizer_state = None
        self.trainer = None
        self.start_epoch = 0

        self.model_analyse = None

        self.visualize = vs.Visualization(self.settings.save_path)
        self.logger = vs.Logger(self.settings.save_path)
        self.test_input = None
        self.lr_master = None
        self.prepare()

    def prepare(self):
        self._set_gpu()
        self._set_dataloader()
        self._set_model()
        self._set_checkpoint()
        self._set_parallel()
        self._set_lr_policy()
        self._set_trainer()
        self._draw_net()

    def _set_gpu(self):
        # set torch seed
        # init random seed
        torch.manual_seed(self.settings.manualSeed)
        torch.cuda.manual_seed(self.settings.manualSeed)
        assert self.settings.GPU <= torch.cuda.device_count() - 1, "Invalid GPU ID"
        torch.cuda.set_device(self.settings.GPU)
        print("|===>Set GPU done!")

    def _set_dataloader(self):
        # create data loader
        self.data_loader = DataLoader(dataset=self.settings.dataset,
                                      batch_size=self.settings.batchSize,
                                      data_path=self.settings.dataPath,
                                      n_threads=self.settings.nThreads,
                                      ten_crop=self.settings.tenCrop
                                      )
        print("|===>Set data loader done!")

    def _set_checkpoint(self):
        assert self.model is not None, "please create model first"

        self.checkpoint = CheckPoint(self.settings.save_path)
        if self.settings.retrain is not None:
            model_state = self.checkpoint.load_model(self.settings.retrain)
            self.model = self.checkpoint.load_state(self.model, model_state)

        if self.settings.resume is not None:
            model_state, optimizer_state, epoch = self.checkpoint.load_checkpoint(
                self.settings.resume)
            self.model = self.checkpoint.load_state(self.model, model_state)
            # self.start_epoch = epoch
            # self.optimizer_state = optimizer_state
        print("|===>Set checkpoint done!")

    def _set_model(self):
        if self.settings.dataset == "sphere":
            if self.settings.netType == "SphereNet":
                self.model = md.SphereNet(
                    depth=self.settings.depth,
                    num_features=self.settings.featureDim)

            elif self.settings.netType == "SphereNIN":
                self.model = md.SphereNIN(
                    num_features=self.settings.featureDim)

            elif self.settings.netType == "wcSphereNet":
                self.model = md.wcSphereNet(
                    depth=self.settings.depth,
                    num_features=self.settings.featureDim,
                    rate=self.settings.rate)
            else:
                assert False, "use %s data while network is %s" % (
                    self.settings.dataset, self.settings.netType)
        else:
            assert False, "unsupport data set: " + self.settings.dataset
        print("|===>Set model done!")

    def _set_parallel(self):
        self.model = utils.data_parallel(
            self.model, self.settings.nGPU, self.settings.GPU)

    def _set_lr_policy(self):
        self.lr_master = utils.LRPolicy(self.settings.lr,
                                        self.settings.nIters,
                                        self.settings.lrPolicy)
        params_dict = {
            'gamma': self.settings.gamma,
            'step': self.settings.step,
            'end_lr': self.settings.endlr,
            'decay_rate': self.settings.decayRate
        }

        self.lr_master.set_params(params_dict=params_dict)

    def _set_trainer(self):

        train_loader, test_loader = self.data_loader.getloader()
        self.trainer = Trainer(model=self.model,
                               lr_master=self.lr_master,
                               n_epochs=self.settings.nEpochs,
                               n_iters=self.settings.nIters,
                               train_loader=train_loader,
                               test_loader=test_loader,
                               feature_dim=self.settings.featureDim,
                               momentum=self.settings.momentum,
                               weight_decay=self.settings.weightDecay,
                               optimizer_state=self.optimizer_state,
                               logger=self.logger)

    def _draw_net(self):
        # visualize model

        if self.settings.dataset == "sphere":
            rand_input = torch.randn(1, 3, 112, 96)
        else:
            assert False, "invalid data set"
        rand_input = Variable(rand_input.cuda())
        self.test_input = rand_input

        if self.settings.drawNetwork:
            rand_output, _ = self.trainer.forward(rand_input)
            self.visualize.save_network(rand_output)
            print("|===>Draw network done!")

        self.visualize.write_settings(self.settings)

    def pruning(self, run_count=0):
        net_type = None
        if self.settings.dataset == "sphere":
            if self.settings.netType == "wcSphereNet":
                net_type = "SphereNet"

        assert net_type is not None, "net_type for prune is NoneType"

        self.trainer.test()

        if isinstance(self.model, nn.DataParallel):
            model = self.model.module
        else:
            model = self.model

        if net_type == "SphereNet":
            model_prune = prune.SpherePrune(model)
        model_prune.run()
        self.trainer.reset_model(model_prune.model)
        self.model = self.trainer.model

        self.trainer.test()
        self.checkpoint.save_model(
            self.trainer.model, index=run_count, tag="pruning")

        # analyse model
        self.model_analyse = utils.ModelAnalyse(
            self.trainer.model, self.visualize)
        params_num = self.model_analyse.params_count()
        self.model_analyse.flops_compute(self.test_input)

    def fine_tuning(self, run_count=0):
        # set lr
        self.settings.lr = 0.01#  0.1
        self.settings.nIters = 12000# 28000
        self.settings.lrPolicy = "multi_step"
        self.settings.decayRate = 0.1
        self.settings.step = [6000]#  [16000, 24000]

        self._set_lr_policy()
        self.trainer.reset_lr(self.lr_master, self.settings.nIters)

        # run fine-tuning
        self.training(run_count, tag="fine-tuning")

    def retrain(self, run_count=0):
        self.settings.lr = 0.1
        self.settings.nIters = 28000
        self.settings.lrPolicy = "multi_step"
        self.settings.decayRate = 0.1
        self.settings.step = [16000, 24000]
        self._set_lr_policy()
        self.trainer.reset_lr(self.lr_master, self.settings.nIters)

        # run retrain
        self.training(run_count, tag="training")

    def run(self, run_count=0):
        """
        if run_count == 0:
            print "|===> training"
            self.retrain(run_count)
        else:     
            print "|===> fine-tuning"
            self.fine_tuning(run_count)
        """
        if run_count >= 1:
            print("|===> training")
            self.retrain(run_count)

            self.trainer.reset_model(self.model)
            print("|===> fine-tuning")
            self.fine_tuning(run_count)

        print("|===> pruning")
        self.pruning(run_count)

        # keep margin_linear static
        layer_count = 0
        for layer in self.model.modules():
            if isinstance(layer, md.MarginLinear):
                layer.iteration.fill_(0)
                layer.margin_type.data.fill_(1) 
                layer.weight.requires_grad = False
            
            elif isinstance(layer, nn.Linear):
                layer.weight.requires_grad = False
                layer.bias.requires_grad = False

            elif isinstance(layer, nn.Conv2d):
                if layer.bias is not None:
                    bias_flag = True
                else:
                    bias_flag = False
                new_layer = prune.wcConv2d(layer.weight.size(1), layer.weight.size(0),
                                           kernel_size=layer.kernel_size,
                                           stride=layer.stride,
                                           padding=layer.padding,
                                           bias=bias_flag,
                                           rate=self.settings.rate)
                new_layer.weight.data.copy_(layer.weight.data)
                if layer.bias is not None:
                    new_layer.bias.data.copy_(layer.bias.data)
                if layer_count == 1:
                    self.model.conv2 = new_layer
                elif layer_count == 2:
                    self.model.conv3 = new_layer
                elif layer_count == 3:
                    self.model.conv4 = new_layer
                layer_count += 1
    
        print(self.model)
        self.trainer.reset_model(self.model)
        # assert False

    def training(self, run_count=0, tag="training"):
        best_top1 = 100
        # start_time = time.time()
        self.trainer.test()
        # assert False
        for epoch in range(self.start_epoch, self.settings.nEpochs):
            if self.trainer.iteration >= self.trainer.n_iters:
                break
            start_epoch = 0
            # training and testing
            train_error, train_loss, train5_error = self.trainer.train(
                epoch=epoch)
            acc_mean, acc_std, acc_all = self.trainer.test()

            test_error_mean = 100 - acc_mean * 100
            # write and print result
            log_str = "%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t" % (epoch, train_error,
                                                              train_loss, train5_error,
                                                              acc_mean, acc_std)
            for acc in acc_all:
                log_str += "%.4f\t" % acc

            self.visualize.write_log(log_str)
            best_flag = False
            if best_top1 >= test_error_mean:
                best_top1 = test_error_mean
                best_flag = True
                print(colored("# %d ==>Best Result is: Top1 Error: %f\n" % (
                    run_count, best_top1), "red"))
            else:
                print(colored("# %d ==>Best Result is: Top1 Error: %f\n" % (
                    run_count, best_top1), "blue"))

            self.checkpoint.save_checkpoint(
                self.model, self.trainer.optimizer, epoch)

            if best_flag:
                self.checkpoint.save_model(
                    self.model, best_flag=best_flag, tag="%s_%d" % (tag, run_count))

            if (epoch + 1) % self.settings.drawInterval == 0:
                self.visualize.draw_curves()
            
            for name, value in self.model.named_parameters():
                if 'weight' in name:
                    name = name.replace('.', '/')
                    self.logger.histo_summary(
                    name, value.data.cpu().numpy(), run_count * self.settings.nEpochs + epoch + 1)
                    if value.grad is not None:
                        self.logger.histo_summary(
                            name + "/grad", value.grad.data.cpu().numpy(), run_count * self.settings.nEpochs + epoch + 1)

        # end_time = time.time()

        if isinstance(self.model, nn.DataParallel):
            self.model = self.model.module
        # draw experimental curves
        self.visualize.draw_curves()

        # compute cost time
        # time_interval = end_time - start_time
        # t_string = "Running Time is: " + \
        #    str(datetime.timedelta(seconds=time_interval)) + "\n"
        # print(t_string)
        # write cost time to file
        # self.visualize.write_readme(t_string)

        # save experimental results
        self.model_analyse = utils.ModelAnalyse(
            self.trainer.model, self.visualize)
        self.visualize.write_readme(
            "Best Result of all is: Top1 Error: %f\n" % best_top1)

        # analyse model
        params_num = self.model_analyse.params_count()

        # save analyse result to file
        self.visualize.write_readme(
            "Number of parameters is: %d" % (params_num))
        self.model_analyse.prune_rate()

        self.model_analyse.flops_compute(self.test_input)

        return best_top1


def main():
    experiment = ExperimentDesign()
    for i in range(10):
        experiment.run(i)


if __name__ == '__main__':
    main()
