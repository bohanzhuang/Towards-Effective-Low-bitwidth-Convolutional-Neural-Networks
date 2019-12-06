import time
import torch.autograd
import utils
import torch.nn as nn
from torch.autograd import Variable
import models as MD


__all__ = ["Trainer"]

class Trainer(object):
    """
    trainer for training network, use SGD
    """

    def __init__(self, model, lr_master, n_epoch, train_loader, test_loader,
                 momentum=0.9, weight_decay=1e-4, optimizer_state=None, tencrop=False,
                 logger=None, ngpu=1, gpu=0):
        """
        init trainer
        """

        self.model = utils.data_parallel(model, ngpu, gpu)

        self.n_epoch = n_epoch
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.ten_crop = False
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.lr_master = lr_master
        self.optimizer = torch.optim.SGD(params=self.model.parameters(),
                                         lr=self.lr_master.lr,
                                         momentum=momentum,
                                         weight_decay=weight_decay,
                                         nesterov=True,
                                         )
        self.logger = logger
        self.run_count = 0
        self.scalar_info = {}
        self.ngpu = ngpu
        self.gpu = gpu
        self.momentum = momentum
        self.weight_decay = weight_decay

    def reset_model(self, model):
        self.model = utils.data_parallel(model, self.ngpu, self.gpu)
        self.optimizer = torch.optim.SGD(params=self.model.parameters(),
                                         lr=self.lr_master.lr,
                                         momentum=self.momentum,
                                         weight_decay=self.weight_decay,
                                         nesterov=True,
                                         )

    def update_lr(self, epoch):
        """
        update learning rate of optimizers
        :param epoch: current training epoch
        """
        lr = self.lr_master.get_lr(epoch)
        # update learning rate of model optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def forward(self, images, labels=None):
        # forward and backward and optimize
        output = self.model(images)

        if labels is not None:
            loss = self.criterion(output, labels)
            return output, loss
        else:
            return output, None

    def backward(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, epoch):

        top1_error = 0
        top1_loss = 0
        top5_error = 0
        images_count = 0
        iters = len(self.train_loader)

        self.update_lr(epoch)
        self.model.train()

        start_time = time.time()
        end_time = start_time

        for i, (images, labels) in enumerate(self.train_loader):
            start_time = time.time()
            data_time = start_time - end_time
            
            # if we use multi-gpu, its more efficient to send input to different gpu, instead of send it to the master gpu.
            if self.ngpu == 1:
                images = images.cuda()
            labels = labels.cuda()
            images_var = Variable(images)
            labels_var = Variable(labels)

            # forward_ts = time.time()
            output, loss = self.forward(images_var, labels_var)
            # forward_te = time.time()
            self.backward(loss)
            # backward_te = time.time()
            # print "forward time: %f, backward time: %f"%(forward_te-forward_ts, backward_te-forward_te)

            single_error, single_loss, single5_error = utils.compute_singlecrop(outputs=output, labels=labels_var,
                                                                                loss=loss, top5_flag=True)
            top1_error += single_error
            top1_loss += single_loss
            top5_error += single5_error
            end_time = time.time()
            iter_time = end_time - start_time

            images_count += images.size(0)

            total_time, left_time = utils.print_result(epoch, self.n_epoch, i + 1,
                                                       iters, self.lr_master.lr, data_time, iter_time,
                                                       single_error /
                                                       images.size(0),
                                                       single_loss, top5error=single5_error /
                                                       images.size(0),
                                                       mode="Train")

            if self.n_epoch == 1 and i + 1 >= 50:
                print("|===>Program testing for only 50 iterations")
                break

        top1_loss /= iters
        top1_error /= images_count
        top5_error /= images_count

        self.scalar_info['training_top1error'] = top1_error
        self.scalar_info['training_top5error'] = top5_error
        self.scalar_info['training_loss'] = top1_loss

        if self.logger is not None:
            for tag, value in self.scalar_info.items():
                self.logger.scalar_summary(tag, value, self.run_count)
            self.scalar_info = {}

        print("|===>Training Error: %.4f Loss: %.4f, Top5 Error:%.4f" % (top1_error, top1_loss, top5_error))
        return top1_error, top1_loss, top5_error

    def test(self, epoch):

        top1_error = 0
        top1_loss = 0
        top5_error = 0
        images_count = 0

        self.model.eval()

        iters = len(self.test_loader)
        start_time = time.time()
        end_time = start_time

        for i, (images, labels) in enumerate(self.test_loader):
            start_time = time.time()
            data_time = start_time - end_time

            labels = labels.cuda()
            labels_var = Variable(labels, volatile=True)
            if self.ten_crop:
                image_size = images.size()
                images = images.view(
                    image_size[0] * 10, image_size[1] / 10, image_size[2], image_size[3])
                images_tuple = images.split(image_size[0])
                output = None
                for img in images_tuple:
                    if self.ngpu == 1:
                        img = img.cuda()
                    img_var = Variable(img, volatile=True)
                    temp_output, _ = self.forward(img_var)
                    if output is None:
                        output = temp_output.data
                    else:
                        output = torch.cat((output, temp_output.data))
                single_error, single_loss, single5_error = computetencrop(
                    outputs=output, labels=labels_var)
            else:
                if self.ngpu == 1:
                    images = images.cuda()
                images_var = Variable(images, volatile=True)
                output, loss = self.forward(images_var, labels_var)

                single_error, single_loss, single5_error = utils.compute_singlecrop(outputs=output, loss=loss,
                                                                                    labels=labels_var, top5_flag=True)
            images_count += images.size(0)

            top1_loss += single_loss
            top1_error += single_error
            top5_error += single5_error

            end_time = time.time()
            iter_time = end_time - start_time

            total_time, left_time = utils.print_result(epoch, self.n_epoch, i + 1,
                                                       iters, self.lr_master.lr, data_time, iter_time,
                                                       single_error /
                                                       images.size(
                                                           0), single_loss,
                                                       top5error=single5_error /
                                                       images.size(0),
                                                       mode="Test")

            if self.n_epoch == 1 and i + 1 >= 50:
                print("|===>Program testing for only 50 iterations")
                break

        top1_loss /= iters
        top1_error /= images_count
        top5_error /= images_count

        self.scalar_info['testing_top1error'] = top1_error
        self.scalar_info['testing_top5error'] = top5_error
        self.scalar_info['testing_loss'] = top1_loss
        if self.logger is not None:
            for tag, value in self.scalar_info.items():
                self.logger.scalar_summary(tag, value, self.run_count)
            self.scalar_info = {}
        self.run_count += 1
        print("|===>Testing Error: %.4f Loss: %.4f, Top5 Error: %.4f" % (top1_error, top1_loss, top5_error))
        return top1_error, top1_loss, top5_error
