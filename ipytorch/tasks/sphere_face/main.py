import argparse
import datetime
import logging
import os
import sys
import time

import models as md
import torch
import torch.nn as nn
import utils
import visualization as vs
from checkpoint import CheckPoint
from dataloader import DataLoader
from options import Option
from sphere_trainer import SphereTrainer as Trainer
from torch.autograd import Variable
from torch.backends import cudnn

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "2"


class ExperimentDesign:
    def __init__(self, options=None):
        self.settings = options or Option()
        self.checkpoint = None
        self.train_loader = None
        self.test_loader = None
        self.model = None

        self.optimizer_state = None
        self.trainer = None
        self.start_epoch = 0
        self.test_input = None
        self.model_analyse = None

        os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = self.settings.visible_devices

        self.settings.set_save_path()
        self.logger = self.set_logger()
        self.settings.paramscheck(self.logger)
        self.visualize = vs.Visualization(self.settings.save_path, self.logger)
        self.tensorboard_logger = vs.Logger(self.settings.save_path)

        self.prepare()

    def set_logger(self):
        logger = logging.getLogger('sphereface')
        file_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
        console_formatter = logging.Formatter('%(message)s')
        # file log
        file_handler = logging.FileHandler(os.path.join(self.settings.save_path, "train_test.log"))
        file_handler.setFormatter(file_formatter)

        # console log
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        logger.setLevel(logging.INFO)
        return logger

    def prepare(self):
        self._set_gpu()
        self._set_dataloader()
        self._set_model()
        self._set_checkpoint()
        self._set_trainer()
        self._draw_net()

    def _set_gpu(self):
        # set torch seed
        # init random seed
        torch.manual_seed(self.settings.manualSeed)
        torch.cuda.manual_seed(self.settings.manualSeed)
        assert self.settings.GPU <= torch.cuda.device_count() - 1, "Invalid GPU ID"
        torch.cuda.set_device(self.settings.GPU)
        cudnn.benchmark = True

    def _set_dataloader(self):
        # create data loader
        data_loader = DataLoader(dataset=self.settings.dataset,
                                 batch_size=self.settings.batchSize,
                                 data_path=self.settings.dataPath,
                                 n_threads=self.settings.nThreads,
                                 ten_crop=self.settings.tenCrop,
                                 logger=self.logger)
        self.train_loader, self.test_loader = data_loader.getloader()

    def _set_checkpoint(self):
        assert self.model is not None, "please create model first"

        self.checkpoint = CheckPoint(self.settings.save_path, self.logger)
        if self.settings.retrain is not None:
            model_state = self.checkpoint.load_model(self.settings.retrain)
            self.model = self.checkpoint.load_state(self.model, model_state)

        if self.settings.resume is not None:
            model_state, optimizer_state, epoch = self.checkpoint.load_checkpoint(
                self.settings.resume)
            self.model = self.checkpoint.load_state(self.model, model_state)
            self.start_epoch = epoch
            self.optimizer_state = optimizer_state

    def _set_model(self):
        if self.settings.dataset in ["sphere", "sphere_large"]:
            self.test_input = Variable(torch.randn(1, 3, 112, 96).cuda())
            if self.settings.netType == "SphereNet":
                self.model = md.SphereNet(
                    depth=self.settings.depth,
                    num_features=self.settings.featureDim)

            elif self.settings.netType == "SphereNIN":
                self.model = md.SphereNIN(
                    num_features=self.settings.featureDim)

            elif self.settings.netType == "SphereMobileNet_v2":
                self.model = md.SphereMobleNet_v2(
                    num_features=self.settings.featureDim)
            else:
                assert False, "use %s data while network is %s" % (
                    self.settings.dataset, self.settings.netType)
        else:
            assert False, "unsupport data set: " + self.settings.dataset

    def _set_trainer(self):
        lr_master = utils.LRPolicy(self.settings.lr,
                                   self.settings.nEpochs,
                                   self.settings.lrPolicy)
        params_dict = {
            'power': self.settings.power,
            'step': self.settings.step,
            'end_lr': self.settings.endlr,
            'decay_rate': self.settings.decayRate
        }

        lr_master.set_params(params_dict=params_dict)

        self.trainer = Trainer(model=self.model,
                               train_loader=self.train_loader,
                               test_loader=self.test_loader,
                               lr_master=lr_master,
                               settings=self.settings,
                               logger=self.logger,
                               tensorboard_logger=self.tensorboard_logger,
                               optimizer_state=self.optimizer_state)

    def _draw_net(self):
        # visualize model
        if self.settings.drawNetwork:
            rand_output, _ = self.trainer.forward(self.test_input)
            self.visualize.save_network(rand_output)
            self.logger.info("|===>Draw network done!")

        self.visualize.write_settings(self.settings)

    def _model_analyse(self, model):
        # analyse model
        model_analyse = utils.ModelAnalyse(model, self.visualize)
        params_num = model_analyse.params_count()
        zero_num = model_analyse.zero_count()
        zero_rate = zero_num * 1.0 / params_num
        self.logger.info("zero rate is: {}".format(zero_rate))

        # save analyse result to file
        self.visualize.write_readme(
            "Number of parameters is: %d, number of zeros is: %d, zero rate is: %f" % (params_num, zero_num, zero_rate))

        # model_analyse.flops_compute(self.test_input)
        model_analyse.madds_compute(self.test_input)

    def run(self, run_count=0):
        self.logger.info("|===>Start training")
        best_top1 = 100
        start_time = time.time()
        # self._model_analyse(self.model)
        # assert False
        self.trainer.test()
        # assert False
        for epoch in range(self.start_epoch, self.settings.nEpochs):
            if self.trainer.iteration >= self.settings.nIters:
                break
            self.start_epoch = 0
            # training and testing
            train_error, train_loss, train5_error = self.trainer.train(epoch=epoch)
            acc_mean, acc_std, acc_all = self.trainer.test()

            test_error_mean = 100 - acc_mean * 100
            # write and print result
            log_str = "{:d}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t".format(
                epoch, train_error,
                train_loss, train5_error,
                acc_mean, acc_std)
            for acc in acc_all:
                log_str += "%.4f\t" % acc

            self.visualize.write_log(log_str)
            best_flag = False
            if best_top1 >= test_error_mean:
                best_top1 = test_error_mean
                best_flag = True
                self.logger.info("# {:d} ==>Best Result is: Top1 Error: {:f}\n".format(
                    run_count, best_top1))
            else:
                self.logger.info("# {:d} ==>Best Result is: Top1 Error: {:f}\n".format(
                    run_count, best_top1))

            self.checkpoint.save_checkpoint(
                self.model, self.trainer.optimizer, epoch)

            if best_flag:
                self.checkpoint.save_model(self.model, best_flag=best_flag)

            if (epoch + 1) % self.settings.drawInterval == 0:
                self.visualize.draw_curves()

        end_time = time.time()

        if isinstance(self.model, nn.DataParallel):
            self.model = self.model.module
        # draw experimental curves
        self.visualize.draw_curves()

        # compute cost time
        time_interval = end_time - start_time
        t_string = "Running Time is: " + \
                   str(datetime.timedelta(seconds=time_interval)) + "\n"
        self.logger.info(t_string)
        # write cost time to file
        self.visualize.write_readme(t_string)
        # analyse model
        self._model_analyse(self.model)

        return best_top1


def main():
    parser = argparse.ArgumentParser(description='Baseline')
    parser.add_argument('conf_path', type=str, metavar='conf_path',
                        help='input batch size for training (default: 64)')
    args = parser.parse_args()
    option = Option(args.conf_path)

    experiment = ExperimentDesign(option)
    experiment.run()


if __name__ == '__main__':
    main()
