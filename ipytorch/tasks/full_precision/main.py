import argparse
import datetime
import logging
import os
import sys
import time
import traceback

import torch
import torch.backends.cudnn as cudnn
# option file should be modified according to your expriment
from options import Option
from torchvision import transforms

import ipytorch.models as md
import ipytorch.utils as utils
import ipytorch.visualization as vs
from ipytorch.utils.ifeige import IFeige, Notification
from ipytorch.checkpoint import CheckPoint
from ipytorch.dataloader import DataLoader
from ipytorch.models.dqn.quantization import ClipReLU
from ipytorch.trainer import Trainer


class ExperimentDesign:
    def __init__(self, options=None, conf_path=None):
        self.settings = options or Option(conf_path)
        self.checkpoint = None
        self.train_loader = None
        self.test_loader = None
        self.model = None

        self.optimizer_state = None
        self.trainer = None
        self.start_epoch = 0
        self.test_input = None

        os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = self.settings.visible_devices

        self.settings.set_save_path()
        self.logger = self.set_logger()
        self.settings.paramscheck(self.logger)
        self.visualize = vs.Visualization(self.settings.save_path, self.logger)
        self.tensorboard_logger = vs.Logger(self.settings.save_path)
        self.ifeige = IFeige()

        self.prepare()

    def set_logger(self):
        logger = logging.getLogger('Baseline')
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
        # file log
        file_handler = logging.FileHandler(os.path.join(self.settings.save_path, "train_test.log"))
        file_handler.setFormatter(formatter)

        # console log
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        logger.setLevel(logging.INFO)
        return logger

    def prepare(self):
        self._set_gpu()
        self._set_dataloader()
        self._set_model()
        self._set_checkpoint()
        self.logger.info(self.model)
        # assert False
        # self._replace_clip_activation()
        # print(self.model)
        # assert  False
        self._set_trainer()
        # self._draw_net()

    def _set_gpu(self):
        # set torch seed
        # init random seed
        torch.backends.cudnn.deterministic = True
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

        if self.settings.netType == "AlexNet" and self.settings.dataset == "cifar100":
            if self.settings.dataset == "cifar10":
                norm_mean = [0.49139968, 0.48215827, 0.44653124]
                norm_std = [0.24703233, 0.24348505, 0.26158768]
            elif self.settings.dataset == "cifar100":
                norm_mean = [0.50705882, 0.48666667, 0.44078431]
                norm_std = [0.26745098, 0.25568627, 0.27607843]

            train_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                # transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std)])
            test_transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std)])

            self.train_loader.dataset.transform = train_transform
            self.test_loader.dataset.transform = test_transform

    def _set_checkpoint(self):

        assert self.model is not None, "please create model first"

        self.checkpoint = CheckPoint(self.settings.save_path, self.logger)

        if self.settings.retrain is not None:
            model_state = self.checkpoint.load_model(self.settings.retrain)
            model_state = model_state['model']
            # new_model_state_dict = {}
            # for key, value in model_state.items():
            #     key = key.replace('module.', '')
            #     new_model_state_dict[key] = value
            # self.model = self.checkpoint.load_state(self.model, new_model_state_dict)
            self.model = self.checkpoint.load_state(self.model, model_state)

        if self.settings.resume is not None:
            model_state, optimizer_state, epoch = self.checkpoint.load_checkpoint(
                self.settings.resume)
            self.model = self.checkpoint.load_state(self.model, model_state)
            self.start_epoch = epoch
            self.optimizer_state = optimizer_state

    def _set_model(self):
        if self.settings.dataset in ["cifar10", "cifar100"]:
            self.test_input = torch.randn(1, 3, 32, 32).cuda()
            if self.settings.netType == "PreResNet":
                self.model = md.official.PreResNet(depth=self.settings.depth,
                                                   num_classes=self.settings.nClasses,
                                                   wide_factor=self.settings.wideFactor)

            elif self.settings.netType == "PreResNet_Test":
                self.model = md.PreResNet_Test(depth=self.settings.depth,
                                               num_classes=self.settings.nClasses,
                                               wide_factor=self.settings.wideFactor,
                                               max_conv=10)

            elif self.settings.netType == "ResNet":
                self.model = md.custom.ResNetCifar(depth=self.settings.depth,
                                                   num_classes=self.settings.nClasses,
                                                   wide_factor=self.settings.wideFactor)

            elif self.settings.netType == "DenseNet_Cifar":
                self.model = md.official.DenseNet_Cifar(depth=self.settings.depth,
                                                        num_classes=self.settings.nClasses,
                                                        reduction=1.0,
                                                        bottleneck=False)

            elif self.settings.netType == "NetworkInNetwork":
                self.model = md.NetworkInNetwork()

            elif self.settings.netType == "AlexNet":
                self.model = md.custom.AlexNetBNCifar(num_classes=self.settings.nClasses)

            elif self.settings.netType == "VGG":
                self.model = md.VGG_CIFAR(
                    self.settings.depth, num_classes=self.settings.nClasses)
            else:
                self.logger.error(
                    "use {} data while network is {}".format(self.settings.dataset, self.settings.netType))
                assert False

        elif self.settings.dataset == "mnist":
            self.test_input = torch.randn(1, 1, 28, 28).cuda()
            if self.settings.netType == "LeNet5":
                self.model = md.LeNet5()
            elif self.settings.netType == "LeNet500300":
                self.model = md.LeNet500300()
            else:
                self.logger.error("use mnist data while network is: {}".format(self.settings.netType))
                assert False

        elif self.settings.dataset in ["imagenet", "imagenet100", "imagenet_mio"]:
            if self.settings.netType == "ResNet":
                self.model = md.official.ResNet(self.settings.depth, self.settings.nClasses)
            elif self.settings.netType == "resnet18":
                self.model = md.resnet18()
            elif self.settings.netType == "resnet34":
                self.model = md.resnet34()
            elif self.settings.netType == "resnet50":
                self.model = md.resnet50()
            elif self.settings.netType == "resnet101":
                self.model = md.resnet101()
            elif self.settings.netType == "resnet152":
                self.model = md.resnet152()
            elif self.settings.netType == "VGG":
                self.model = md.VGG(
                    depth=self.settings.depth, bn_flag=False, num_classes=self.settings.nClasses)
            elif self.settings.netType == "VGG_GAP":
                self.model = md.VGG_GAP(
                    depth=self.settings.depth, bn_flag=False, num_classes=self.settings.nClasses)
            elif self.settings.netType == "Inception3":
                self.model = md.Inception3(num_classes=self.settings.nClasses)
            elif self.settings.netType == "MobileNet_v2":
                self.model = md.MobileNet_v2(
                    num_classes=self.settings.nClasses,
                )  # wide_scale=1.4)
            elif self.settings.netType == "MobileNet":
                self.model = md.custom.MobileNetV1()
            elif self.settings.netType == "ThinMobileNet":
                self.model = md.custom.ThinMobileNetV1(
                    num_classes=self.settings.nClasses,
                    wide_scale=self.settings.wideFactor,
                    width=self.settings.width
                )
            elif self.settings.netType == "AlexNet":
                self.model = md.AlexNetBN(num_classes=self.settings.nClasses)
            else:
                self.logger.error(
                    "use {} data while network is {}".format(self.settings.dataset, self.settings.netType))
                assert False

            if self.settings.netType in ["InceptionResNetV2", "Inception3"]:
                self.test_input = torch.randn(1, 3, 299, 299).cuda()
            else:
                self.test_input = torch.randn(1, 3, 224, 224).cuda()
        else:
            self.logger.error("unsupport data set: {}".format(self.settings.dataset))
            assert False

    def _replace_clip_activation(self):
        if self.settings.netType == "PreResNet":
            for module in self.model.modules():
                if isinstance(module, (md.official.PreResNet)):
                    module.relu = ClipReLU(module.relu.inplace)
                elif isinstance(module, (md.official.PreBasicBlock)):
                    module.relu = ClipReLU(module.relu.inplace)
        elif self.settings.netType == "ResNet":
            for module in self.model.modules():
                if isinstance(module, (md.custom.ResNetCifar)):
                    module.relu = ClipReLU(module.relu.inplace)
                elif isinstance(module, (md.custom.ResidualBlock)):
                    module.relu = ClipReLU(module.relu.inplace)
        elif self.settings.netType == "AlexNet":
            for module in self.model.modules():
                if isinstance(module, torch.nn.Sequential) and len(module) == 18:
                    module[2] = ClipReLU(module[2].inplace)
                    module[6] = ClipReLU(module[6].inplace)
                    module[10] = ClipReLU(module[10].inplace)
                    module[13] = ClipReLU(module[13].inplace)
                    module[16] = ClipReLU(module[16].inplace)
                    # module[2] = CabsReLU(module[2].inplace)
                    # module[6] = CabsReLU(module[6].inplace)
                    # module[10] = CabsReLU(module[10].inplace)
                    # module[13] = CabsReLU(module[13].inplace)
                    # module[16] = CabsReLU(module[16].inplace)
                elif isinstance(module, torch.nn.Sequential) and len(module) == 7:
                    module[2] = ClipReLU(module[2].inplace)
                    module[5] = ClipReLU(module[5].inplace)
                    # module[2] = CabsReLU(module[2].inplace)
                    # module[5] = CabsReLU(module[5].inplace)

    def _set_trainer(self):
        # set lr master
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
        # set trainer
        self.trainer = Trainer(
            model=self.model,
            train_loader=self.train_loader,
            test_loader=self.test_loader,
            lr_master=lr_master,
            settings=self.settings,
            logger=self.logger,
            tensorboard_logger=self.tensorboard_logger,
            opt_type=self.settings.opt_type,
            optimizer_state=self.optimizer_state
        )
        # self.trainer.reset_optimizer(opt_type="RMSProp")

    def _draw_net(self):
        if self.settings.drawNetwork:
            rand_output, _ = self.trainer.forward(self.test_input)
            self.visualize.save_network(rand_output)
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
        best_top1 = 100
        best_top5 = 100
        start_time = time.time()
        # self.trainer.test(0)
        # assert False
        # self.logger.info(self.model)
        # self._model_analyse(self.model)
        # assert False
        # test_error, test_loss, test5_error = self.trainer.test(0)
        # assert False
        try:
            for epoch in range(self.start_epoch, self.settings.nEpochs):
                self.epoch = epoch
                self.start_epoch = 0
                # training and testing
                train_error, train_loss, train5_error = self.trainer.train(
                    epoch=epoch)
                test_error, test_loss, test5_error = self.trainer.test(
                    epoch=epoch)
                # self.trainer.model.apply(utils.SVB)
                # self.trainer.model.apply(utils.BBN)

                # write and print result
                log_str = "%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t" % (
                    epoch, train_error, train_loss, test_error,
                    test_loss, train5_error, test5_error)

                self.visualize.write_log(log_str)
                best_flag = False
                if best_top1 >= test_error:
                    best_top1 = test_error
                    best_top5 = test5_error
                    best_flag = True

                self.logger.info("#==>Best Result is: Top1 Error: {:f}, Top5 Error: {:f}".format(best_top1, best_top5))
                self.logger.info("#==>Best Result is: Top1 Accuracy: {:f}, Top5 Accuracy: {:f}".format(100 - best_top1,
                                                                                                       100 - best_top5))

                if self.settings.dataset in ["imagenet", "imagenet100", "imagenet_mio"]:
                    self.checkpoint.save_checkpoint(
                        self.model, self.trainer.optimizer, epoch, epoch)
                else:
                    self.checkpoint.save_checkpoint(
                        self.model, self.trainer.optimizer, epoch)

                if best_flag:
                    self.checkpoint.save_model(self.model, best_flag=best_flag)

                # if (epoch + 1) % self.settings.drawInterval == 0:
                # self.visualize.draw_curves()
        except BaseException as e:
            self.logger.error("Training is terminating due to exception: {}".format(str(e)))
            traceback.print_exc()
            self.checkpoint.save_checkpoint(
                self.model, self.trainer.optimizer, self.epoch, self.epoch)

        end_time = time.time()
        time_interval = end_time - start_time
        t_string = "Running Time is: " + \
                   str(datetime.timedelta(seconds=time_interval)) + "\n"
        self.logger.info(t_string)

        self.visualize.write_settings(self.settings)
        # save experimental results
        self.visualize.write_readme(
            "Best Result of all is: Top1 Error: {:f}, Top5 Error: {:f}\n".format(best_top1, best_top5))
        self.visualize.write_readme(
            "Best Result of all is: Top1 Accuracy: {:f}, Top5 Accuracy: {:f}\n".format(100 - best_top1,
                                                                                       100 - best_top5))
        self.ifeige.send_msg_to_user(
            username="Key",
            key=Notification.NOTICE,
            title="{} experiment complete\n".format(self.settings.save_path.split('/')[-1]),
            content="Top1 Accuracy: {:f}, Top5 Accuracy: {:f}".format(100 - best_top1, 100 - best_top5),
            remark=""
        )

        self.visualize.draw_curves()

        # analyse model
        # self._model_analyse(self.model)
        return best_top1


# ---------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description='Baseline')
    parser.add_argument('conf_path', type=str, metavar='conf_path',
                        help='input batch size for training (default: 64)')
    parser.add_argument('id', type=int, metavar='experiment_id',
                        help='Experiment ID')
    args = parser.parse_args()

    option = Option(args.conf_path)
    option.manualSeed = args.id + 1
    option.experimentID = option.experimentID + "{:0>2d}_repeat".format(args.id + 1)

    experiment = ExperimentDesign(option)
    experiment.run()


if __name__ == '__main__':
    main()
