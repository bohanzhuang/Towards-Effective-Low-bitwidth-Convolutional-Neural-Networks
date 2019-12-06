"""
basic trainer
"""
import time

import torch.autograd
import torch.nn as nn

import ipytorch.utils as utils
from ipytorch.trainer import Trainer

__all__ = ["GuidedTrainer"]


class GuidedTrainer(Trainer):
    """
    trainer for training network, use SGD
    """

    def __init__(self, ori_model, quan_model, ori_lr_master, quan_lr_master,
                 train_loader, test_loader,
                 settings, logger, tensorboard_logger=None, ori_opt_type="SGD", quan_opt_type="Adam",
                 ori_optimizer_state=None, quan_optimizer_state=None, run_count=0):
        """
        init trainer
        """

        self.settings = settings

        self.ori_model = utils.data_parallel(
            ori_model, self.settings.nGPU, self.settings.GPU)
        self.quan_model = []
        self.num_model = len(quan_model)
        for j in range(len(quan_model)):
            self.quan_model.append(
                utils.data_parallel(quan_model[j], self.settings.nGPU, self.settings.GPU))
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.log_softmax = nn.LogSoftmax().cuda()
        self.softmax = nn.Softmax().cuda()
        self.criterion_nll = nn.NLLLoss().cuda()
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.criterion_kl = nn.KLDivLoss(size_average=False).cuda()
        self.ori_lr_master = ori_lr_master
        self.quan_lr_master = quan_lr_master
        self.ori_opt_type = ori_opt_type
        self.quan_opt_type = quan_opt_type

        self.quan_optimizer = []
        if self.ori_opt_type == "SGD":
            self.ori_optimizer = torch.optim.SGD(
                params=self.ori_model.parameters(),
                lr=self.ori_lr_master.lr,
                momentum=self.settings.momentum,
                weight_decay=self.settings.weightDecay,
                nesterov=True)
        elif self.ori_opt_type == "RMSProp":
            self.ori_optimizer = torch.optim.RMSprop(
                params=self.ori_model.parameters(),
                lr=self.ori_lr_master.lr,
                eps=1.0,
                weight_decay=self.settings.weightDecay,
                momentum=self.settings.momentum,
                alpha=self.settings.momentum)
        elif self.ori_opt_type == "Adam":
            self.ori_optimizer = torch.optim.Adam(
                params=self.ori_model.parameters(),
                lr=self.ori_lr_master.lr,
                eps=1e-5,
                weight_decay=self.settings.weightDecay)
        else:
            assert False, "invalid type: %d" % self.ori_opt_type

        if self.quan_opt_type == "SGD":
            for j in range(len(self.quan_model)):
                self.quan_optimizer.append(
                    torch.optim.SGD(
                        params=self.quan_model[j].parameters(),
                        lr=self.quan_lr_master.lr,
                        momentum=self.settings.momentum,
                        weight_decay=self.settings.weightDecay,
                        nesterov=True))
        elif self.quan_opt_type == "RMSProp":
            for j in range(len(self.quan_model)):
                self.quan_optimizer.append(
                    torch.optim.RMSprop(
                        params=self.quan_model[j].parameters(),
                        lr=self.quan_lr_master.lr,
                        eps=1.0,
                        weight_decay=self.settings.weightDecay,
                        momentum=self.settings.momentum,
                        alpha=self.settings.momentum))
        elif self.quan_opt_type == "Adam":
            for j in range(len(self.quan_model)):
                self.quan_optimizer.append(
                    torch.optim.Adam(
                        params=self.quan_model[j].parameters(),
                        lr=self.quan_lr_master.lr,
                        eps=1e-5,
                        weight_decay=self.settings.weightDecay))
        else:
            assert False, "invalid type: %d" % self.quan_opt_type

        if ori_optimizer_state is not None:
            self.ori_optimizer.load_state_dict(ori_optimizer_state)
        if len(quan_optimizer_state) != 0:
            for j in range(len(self.quan_model)):
                self.quan_optimizer[j].load_state_dict(quan_optimizer_state[j])
        self.logger = logger
        self.tensorboard_logger = tensorboard_logger
        self.run_count = run_count
        self.scalar_info = {}

    def reset_optimizer(self, opt_type="SGD"):
        if self.ori_opt_type == "SGD":
            self.ori_optimizer = torch.optim.SGD(
                params=self.ori_model.parameters(),
                lr=self.ori_lr_master.lr,
                momentum=self.settings.momentum,
                weight_decay=self.settings.weightDecay,
                nesterov=True)
        elif self.ori_opt_type == "RMSProp":
            self.ori_optimizer = torch.optim.RMSprop(
                params=self.ori_model.parameters(),
                lr=self.ori_lr_master.lr,
                eps=1.0,
                weight_decay=self.settings.weightDecay,
                momentum=self.settings.momentum,
                alpha=self.settings.momentum)
        elif self.ori_opt_type == "Adam":
            self.ori_optimizer = torch.optim.Adam(
                params=self.ori_model.parameters(),
                lr=self.ori_lr_master.lr,
                eps=1e-5,
                weight_decay=self.settings.weightDecay)
        else:
            assert False, "invalid type: %d" % self.ori_opt_type

        if self.quan_opt_type == "SGD":
            for j in range(len(self.quan_model)):
                self.quan_optimizer.append(
                    torch.optim.SGD(
                        params=self.quan_model[j].parameters(),
                        lr=self.quan_lr_master.lr,
                        momentum=self.settings.momentum,
                        weight_decay=self.settings.weightDecay,
                        nesterov=True))
        elif self.quan_opt_type == "RMSProp":
            for j in range(len(self.quan_model)):
                self.quan_optimizer.append(
                    torch.optim.RMSprop(
                        params=self.quan_model[j].parameters(),
                        lr=self.quan_lr_master.lr,
                        eps=1.0,
                        weight_decay=self.settings.weightDecay,
                        momentum=self.settings.momentum,
                        alpha=self.settings.momentum))
        elif self.quan_opt_type == "Adam":
            for j in range(len(self.quan_model)):
                self.quan_optimizer.append(
                    torch.optim.Adam(
                        params=self.quan_model[j].parameters(),
                        lr=self.quan_lr_master.lr,
                        eps=1e-5,
                        weight_decay=self.settings.weightDecay))
        else:
            assert False, "invalid type: %d" % self.quan_opt_type

    def update_ori_lr(self, epoch):
        """
        update learning rate of optimizers
        :param epoch: current training epoch
        """
        lr = self.ori_lr_master.get_lr(epoch)
        # update learning rate of model optimizer
        for param_group in self.ori_optimizer.param_groups:
            param_group['lr'] = lr

    def update_quan_lr(self, epoch):
        """
        update learning rate of optimizers
        :param epoch: current training epoch
        """
        lr = self.quan_lr_master.get_lr(epoch)
        # update learning rate of model optimizer
        for j in range(len(self.quan_optimizer)):
            for param_group in self.quan_optimizer[j].param_groups:
                param_group['lr'] = lr

    @staticmethod
    def _concat_gpu_data(data):
        data_cat = data["0"]
        # for k in data.keys():
        #     print "gpu id:", k
        for i in range(1, len(data)):
            data_cat = torch.cat((data_cat, data[str(i)].cuda(0)))
        return data_cat

    def forward(self, model, images, labels=None):
        """
        forward propagation
        """
        output = model(images)

        if labels is not None:
            loss = self.criterion(output, labels)
            return output, loss
        else:
            return output, None

    def backward(self, optimizer, loss):
        """
        backward propagation
        """
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def train(self, epoch):
        """
        training
        """
        ori_top1_error = utils.AverageMeter()
        ori_top1_loss = utils.AverageMeter()
        ori_kl_loss = utils.AverageMeter()
        ori_total_loss = utils.AverageMeter()
        ori_top5_error = utils.AverageMeter()

        quan_top1_error = []
        quan_top1_loss = []
        quan_kl_loss = []
        quan_total_loss = []
        quan_top5_error = []
        for i in range(len(self.quan_model)):
            quan_top1_error.append(utils.AverageMeter())
            quan_top1_loss.append(utils.AverageMeter())
            quan_kl_loss.append(utils.AverageMeter())
            quan_total_loss.append(utils.AverageMeter())
            quan_top5_error.append(utils.AverageMeter())

        iters = len(self.train_loader)
        self.update_ori_lr(epoch)
        self.update_quan_lr(epoch)
        # Switch to train mode
        self.ori_model.train()
        for i in range(len(self.quan_model)):
            self.quan_model[i].train()

        start_time = time.time()
        end_time = start_time

        for i, (images, labels) in enumerate(self.train_loader):
            start_time = time.time()
            data_time = start_time - end_time

            # if we use multi-gpu, its more efficient to send input
            # to different gpu, instead of send it to the master gpu.

            if self.settings.nGPU == 1:
                images = images.cuda()
            labels = labels.cuda()

            quan_output_list = []
            ori_output, _ = self.forward(self.ori_model, images)

            for j in range(self.num_model):
                quan_output, _ = self.forward(self.quan_model[j], images)
                quan_loss = self.criterion(quan_output, labels)
                quan_kl_loss_ = (self.settings.T[j] * self.settings.T[j]) * \
                                (self.criterion_kl(self.log_softmax(quan_output / self.settings.T[j]),
                                self.softmax(ori_output.detach() / self.settings.T[j])) / quan_output.shape[0])
                quan_total_loss_ = self.settings.student_lambda * quan_loss + self.settings.sloss_lambda * quan_kl_loss_
                self.quan_optimizer[j].zero_grad()
                quan_total_loss_.backward()
                self.quan_optimizer[j].step()

                quan_output_list.append(quan_output.detach())

                quan_single_error, quan_single_loss, quan_single5_error = utils.compute_singlecrop(
                    outputs=quan_output, labels=labels,
                    loss=quan_loss, top5_flag=True, mean_flag=True)
                quan_top1_error[j].update(quan_single_error, images.size(0))
                quan_top1_loss[j].update(quan_single_loss, images.size(0))
                quan_kl_loss[j].update(quan_kl_loss_.item(), images.size(0))
                quan_total_loss[j].update(quan_total_loss_.item(), images.size(0))
                quan_top5_error[j].update(quan_single5_error, images.size(0))

            # avg_p = 0
            # for j in range(len(self.quan_model)):
            #     avg_p += self.softmax(quan_output_list[j])
            # avg_p = avg_p / self.num_model
            # ori_log_p = self.log_softmax(ori_output)
            # ori_loss = self.criterion(ori_output, labels)
            # ori_kl_loss_ = self.settings.loss_lambda * \
            #     self.criterion_kl(ori_log_p, self.softmax(avg_p.detach()))

            ori_kl_loss_ = 0
            ori_loss = self.criterion(ori_output, labels)
            for j in range(self.num_model):
                ori_kl_loss_ += (self.settings.T[j] * self.settings.T[j]) * \
                                (self.criterion_kl(self.log_softmax(ori_output / self.settings.T[j]),
                                                   self.softmax(quan_output_list[j].detach() / self.settings.T[j])) / ori_output.shape[0])
            # Add in 2018.12.30
            ori_kl_loss_ = ori_kl_loss_ / self.num_model
            ori_total_loss_ = self.settings.teacher_lambda * ori_loss + self.settings.tloss_lambda * ori_kl_loss_
            self.ori_optimizer.zero_grad()
            ori_total_loss_.backward()
            self.ori_optimizer.step()

            ori_single_error, ori_single_loss, ori_single5_error = utils.compute_singlecrop(
                outputs=ori_output, labels=labels,
                loss=ori_loss, top5_flag=True, mean_flag=True)
            ori_top1_error.update(ori_single_error, images.size(0))
            ori_top1_loss.update(ori_single_loss, images.size(0))
            ori_top5_error.update(ori_single5_error, images.size(0))
            ori_kl_loss.update(ori_kl_loss_.item(), images.size(0))
            ori_total_loss.update(ori_total_loss_.item(), images.size(0))

            end_time = time.time()
            iter_time = end_time - start_time

            log_str = ">>> {}: [{:0>3d}|{:0>3d}], Iter: [{:0>3d}|{:0>3d}], LR: {:.6f}, DataTime: {:.4f}, IterTime: {:.4f}, ".format(
                "Train", epoch + 1, self.settings.nEpochs, i + 1, iters, self.ori_lr_master.lr, data_time, iter_time)
            log_str += "KLD Loss: {:.4f}, Softmax Loss: {:.4f}, Total Loss: {:.4f}, Error: {:.4f}, Top5 Error: {:.4f}, ".format(
                ori_kl_loss_.item(), ori_single_loss, ori_total_loss_.item(), ori_single_error, ori_single5_error)
            time_str, _, _ = utils.compute_remain_time(epoch, self.settings.nEpochs, i + 1, iters, data_time, iter_time,
                                                       mode="Train")
            self.logger.info(log_str + time_str)

            log_str = ">>> {}: [{:0>3d}|{:0>3d}], Iter: [{:0>3d}|{:0>3d}], LR: {:.6f}, DataTime: {:.4f}, IterTime: {:.4f}, ".format(
                "Train", epoch + 1, self.settings.nEpochs, i + 1, iters, self.quan_lr_master.lr, data_time, iter_time)
            for j in range(len(self.quan_model)):
                log_str += "KLD Loss_{:d}: {:.4f}, Softmax Loss_{:d}: {:.4f}, Total Loss_{:d}: {:.4f}, Error_{:d}: {:.4f}, Top5 Error_{:d}: {:.4f}, ".format(
                    j, quan_kl_loss[j].val, j, quan_top1_loss[j].val, j, quan_total_loss[j].val,
                    j, quan_top1_error[j].val, j, quan_top5_error[j].val)
            self.logger.info(log_str)

            if self.settings.nEpochs == 1 and i + 1 >= 50:
                self.logger.info("|===>Program testing for only 50 iterations")
                break

        self.scalar_info['ori_training_top1error'] = ori_top1_error.avg
        self.scalar_info['ori_training_top5error'] = ori_top5_error.avg
        self.scalar_info['ori_training_loss'] = ori_top1_loss.avg
        self.scalar_info['ori_total_loss'] = ori_total_loss.avg
        self.scalar_info['ori_kl_loss'] = ori_kl_loss.avg
        for j in range(len(self.quan_model)):
            self.scalar_info['quan_training_top1error_{}'.format(j)] = quan_top1_error[j].avg
            self.scalar_info['quan_training_top5error_{}'.format(j)] = quan_top5_error[j].avg
            self.scalar_info['quan_training_loss_{}'.format(j)] = quan_top1_loss[j].avg
            self.scalar_info['quan_total_loss_{}'.format(j)] = quan_total_loss[j].avg
            self.scalar_info['quan_kl_loss_{}'.format(j)] = quan_kl_loss[j].avg

        if self.tensorboard_logger is not None:
            for tag, value in list(self.scalar_info.items()):
                self.tensorboard_logger.scalar_summary(tag, value, self.run_count)
            self.scalar_info = {}

        self.logger.info(
            "|===>Original Model Training Error: {:.4f}, Loss: {:.4f}, Top5 Error: {:.4f}".format(ori_top1_error.avg,
                                                                                                  ori_top1_loss.avg,
                                                                                                  ori_top5_error.avg))

        log_str = "|===>Quantized Model "
        for j in range(len(self.quan_model)):
            log_str += "Training Error_{}: {:.4f}, Loss_{}: {:.4f}, Top5 Error_{}: {:.4f}, ".format(j, quan_top1_error[
                j].avg,
                                                                                                    j, quan_top1_loss[
                                                                                                        j].avg,
                                                                                                    j, quan_top5_error[
                                                                                                        j].avg)
        log_str += "\n"
        self.logger.info(log_str)

        return ori_top1_error.avg, ori_top1_loss.avg, ori_top5_error.avg, quan_top1_error, quan_top1_loss, quan_top5_error

    def test(self, epoch):
        """
        testing
        """
        ori_top1_error = utils.AverageMeter()
        ori_top1_loss = utils.AverageMeter()
        ori_top5_error = utils.AverageMeter()

        quan_top1_error = []
        quan_top1_loss = []
        quan_top5_error = []
        for j in range(len(self.quan_model)):
            quan_top1_error.append(utils.AverageMeter())
            quan_top1_loss.append(utils.AverageMeter())
            quan_top5_error.append(utils.AverageMeter())
            self.quan_model[j].eval()
        self.ori_model.eval()

        iters = len(self.test_loader)
        start_time = time.time()
        end_time = start_time

        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test_loader):
                start_time = time.time()
                data_time = start_time - end_time

                labels = labels.cuda()
                if self.settings.nGPU == 1:
                    images = images.cuda()
                ori_output, ori_loss = self.forward(self.ori_model, images, labels)
                ori_single_error, ori_single_loss, ori_single5_error = utils.compute_singlecrop(
                    outputs=ori_output, loss=ori_loss,
                    labels=labels, top5_flag=True, mean_flag=True)
                for j in range(len(self.quan_model)):
                    quan_output, quan_loss = self.forward(self.quan_model[j], images, labels)
                    quan_single_error, quan_single_loss, quan_single5_error = utils.compute_singlecrop(
                        outputs=quan_output, loss=quan_loss,
                        labels=labels, top5_flag=True, mean_flag=True)

                    quan_top1_error[j].update(quan_single_error, images.size(0))
                    quan_top1_loss[j].update(quan_single_loss, images.size(0))
                    quan_top5_error[j].update(quan_single5_error, images.size(0))

                ori_top1_error.update(ori_single_error, images.size(0))
                ori_top1_loss.update(ori_single_loss, images.size(0))
                ori_top5_error.update(ori_single5_error, images.size(0))

                end_time = time.time()
                iter_time = end_time - start_time

                log_str = ">>> {}: [{:0>3d}|{:0>3d}], Iter: [{:0>3d}|{:0>3d}], LR: {:.6f}, DataTime: {:.4f}, IterTime: {:.4f}, ".format(
                    "Test", epoch + 1, self.settings.nEpochs, i + 1, iters, self.ori_lr_master.lr, data_time,
                    iter_time)
                log_str += "Loss: {:.4f}, Error: {:.4f}, Top5 Error: {:.4f}, ".format(
                    ori_single_loss, ori_single_error, ori_single5_error)
                time_str, _, _ = utils.compute_remain_time(epoch, self.settings.nEpochs, i + 1, iters, data_time,
                                                           iter_time,
                                                           mode="Train")
                self.logger.info(log_str + time_str)

                log_str = ">>> {}: [{:0>3d}|{:0>3d}], Iter: [{:0>3d}|{:0>3d}], LR: {:.6f}, DataTime: {:.4f}, IterTime: {:.4f}, ".format(
                    "Test", epoch + 1, self.settings.nEpochs, i + 1, iters, self.quan_lr_master.lr, data_time,
                    iter_time)
                for j in range(len(self.quan_model)):
                    log_str += "Loss_{:d}: {:.4f}, Error_{:d}: {:.4f}, Top5 Error_{:d}: {:.4f}, ".format(
                        j, quan_top1_loss[j].val,
                        j, quan_top1_error[j].val,
                        j, quan_top5_error[j].val)
                self.logger.info(log_str)

                if self.settings.nEpochs == 1 and i + 1 >= 50:
                    self.logger.info("|===>Program testing for only 50 iterations")
                    break

        self.scalar_info['ori_testing_top1error'] = ori_top1_error.avg
        self.scalar_info['ori_testing_top5error'] = ori_top5_error.avg
        self.scalar_info['ori_testing_loss'] = ori_top1_loss.avg
        for j in range(len(self.quan_model)):
            self.scalar_info['quan_testing_top1error_{}'.format(j)] = quan_top1_error[j].avg
            self.scalar_info['quan_testing_top5error_{}'.format(j)] = quan_top5_error[j].avg
            self.scalar_info['quan_testing_loss_{}'.format(j)] = quan_top1_loss[j].avg
        if self.tensorboard_logger is not None:
            for tag, value in self.scalar_info.items():
                self.tensorboard_logger.scalar_summary(tag, value, self.run_count)
            self.scalar_info = {}
        self.run_count += 1
        self.logger.info(
            "|===>Original Model Testing Error: {:.4f} Loss: {:.4f}, Top5 Error: {:.4f}".format(ori_top1_error.avg,
                                                                                                ori_top1_loss.avg,
                                                                                                ori_top5_error.avg))
        log_str = "|===>Quantized Model "
        for j in range(len(self.quan_model)):
            log_str += "Training Error_{}: {:.4f}, Loss_{}: {:.4f}, Top5 Error_{}: {:.4f}, ".format(j, quan_top1_error[
                j].avg,
                                                                                                    j, quan_top1_loss[
                                                                                                        j].avg,
                                                                                                    j, quan_top5_error[
                                                                                                        j].avg)
        self.logger.info(log_str)
        return ori_top1_error.avg, ori_top1_loss.avg, ori_top5_error.avg, quan_top1_error, quan_top1_loss, quan_top5_error