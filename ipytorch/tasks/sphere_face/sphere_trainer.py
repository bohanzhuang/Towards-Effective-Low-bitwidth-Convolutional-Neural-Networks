import time

import numpy as np
import torch
import torch.autograd
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.parallel

import utils

# weighted channel package

__all__ = ["SphereTrainer"]


class SphereTrainer(object):
    """Trainer class

    Trainer takes charge of network training, lfw evaluation and network saving.

    Attributes:
        model: Network model.
        train_loader: Training dataset loader.
        test_loader: LFW dataset loader.
        feature_dim: The dimension of face feature.
        settings: Settings of the experiments
        optimizer_state: optimizer_state from resume file
        logger: Tensorboard logger.
    """

    def __init__(self, model, lr_master,
                 train_loader, test_loader,
                 settings, logger, tensorboard_logger=None,
                 optimizer_state=None, run_count=0):

        self.settings = settings
        self.model = utils.data_parallel(model, self.settings.nGPU, self.settings.GPU)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr_master = lr_master
        self.logger = logger
        self.criterion = nn.CrossEntropyLoss().cuda()

        self.optimizer = torch.optim.SGD(params=self.model.parameters(),
                                         lr=self.lr_master.lr,
                                         momentum=self.settings.momentum,
                                         weight_decay=self.settings.weightDecay,
                                         nesterov=True)
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)

        self.iteration = 0
        self.criterion = nn.CrossEntropyLoss()
        self.logger = logger
        self.tensorboard_logger = tensorboard_logger
        self.run_count = run_count

    def update_lr(self, iters):
        """
        update learning rate of optimizers
        :param iters: current training iteration
        """
        lr = self.lr_master.get_lr(iters)
        # update learning rate of model optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.tensorboard_logger.scalar_summary('learning_rate', lr, iters)

    def update_model(self, model):
        self.model = utils.data_parallel(model=model,
                                         ngpus=self.settings.nGPU,
                                         gpu0=self.settings.GPU)

        parameters = filter(lambda p: p.requires_grad,
                            self.model.parameters())

        self.optimizer = torch.optim.SGD(params=parameters,
                                         lr=self.lr_master.lr,
                                         momentum=self.settings.momentum,
                                         weight_decay=self.settings.weightDecay,
                                         nesterov=True)

    def backward(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        """
        we constrain the grad_norm in the range of 0~2, otherwise the gradient will explode
        """
        # torch.nn.utils.clip_grad_norm(self.model.parameters(), 2.)
        self.optimizer.step()

    def forward(self, images, labels=None):
        # forward and backward and optimize
        if labels is not None:
            output = self.model(images, labels)
            loss = self.criterion(output, labels)
            return output, loss
        else:
            output = self.model(images)
            return output, None

    def extract_deep_feature(self, img):
        """Extract face image and horizontally flipped face image features and
        concatenating them together to get final face representation."""
        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()
        img_flip = torch.index_select(img.cpu(), 3, inv_idx)

        img_var = img.cuda()
        img_flip_var = img_flip.cuda()

        extract_time = time.time()
        with torch.no_grad():
            res = self.model(img_var)
            res_ = self.model(img_flip_var)
        extract_time = time.time() - extract_time
        feature = torch.cat((res, res_), 1)
        return feature.double(), extract_time

    def lfw_get_threshold(self, scores, target_array, threshold_num):
        """Get the best threshold with respect to current set of scores."""
        accuracys = np.zeros((2 * threshold_num + 1,))
        thresholds = np.arange(-threshold_num,
                               threshold_num + 1) / float(threshold_num)
        for i in range(2 * threshold_num + 1):
            accuracys[i] = self.lfw_get_accuracy(
                scores, target_array, thresholds[i])

        best_threshold = np.mean(
            thresholds[np.where(accuracys == np.max(accuracys))])
        return best_threshold

    def lfw_get_accuracy(self, scores, target_array, threshold):
        """Get lfw accuracy with respect to threshold."""
        accuracy = (len(np.where(scores[np.where(target_array == 1)] > threshold)[
                            0]) + len(np.where(scores[np.where(target_array != 1)] < threshold)[0])) / float(
            len(scores))
        return accuracy

    def test(self):
        """Evaluate on lfw dataset."""
        self.model.eval()

        lfw_batch_size = 100
        feature_dim = self.settings.featureDim
        nrof_pair = len(self.test_loader) * lfw_batch_size
        feature_left_array = np.zeros((nrof_pair, feature_dim * 2))
        feature_right_array = np.zeros((nrof_pair, feature_dim * 2))
        target_array = np.zeros((nrof_pair,))
        fold_array = np.zeros((nrof_pair,))

        accuracy = np.zeros(10, )
        forward_time = 0
        for batch_idx, (img_l, img_r, target, fold) in enumerate(self.test_loader):
            img_l, img_r = img_l.cuda(), img_r.cuda()

            target_numpy = target.numpy()
            fold_numpy = fold.numpy()
            feature_l, single_time = self.extract_deep_feature(img_l)
            forward_time += single_time
            feature_r, single_time = self.extract_deep_feature(img_r)
            forward_time += single_time
            feature_left_array[
            batch_idx * lfw_batch_size:(batch_idx + 1) * lfw_batch_size] = feature_l.data.cpu().numpy()
            feature_right_array[
            batch_idx * lfw_batch_size:(batch_idx + 1) * lfw_batch_size] = feature_r.data.cpu().numpy()
            target_array[batch_idx * lfw_batch_size:(batch_idx + 1) * lfw_batch_size] = target_numpy
            fold_array[batch_idx * lfw_batch_size:(batch_idx + 1) * lfw_batch_size] = fold_numpy

        self.logger.info("forward time {}".format(forward_time))
        # print fold_array

        for i in range(10):
            # split 10 filds into val & test set
            val_fold_index = np.where(fold_array != i)
            test_fold_index = np.where(fold_array == i)
            # get normalized feature
            feature = np.concatenate(
                (feature_left_array[val_fold_index], feature_right_array[val_fold_index]))
            mu = np.mean(feature, 0)

            feature_l = np.copy(feature_left_array) - mu
            feature_r = np.copy(feature_right_array) - mu
            feature_l = feature_l / np.repeat(np.expand_dims(
                np.sqrt(np.sum(np.square(feature_l), 1)), axis=1), feature_dim * 2, axis=1)
            feature_r = feature_r / np.repeat(np.expand_dims(
                np.sqrt(np.sum(np.square(feature_r), 1)), axis=1), feature_dim * 2, axis=1)

            # get accuracy of the ith fold using cosine similarity
            scores = np.sum(np.multiply(feature_l, feature_r), 1)
            threshold = self.lfw_get_threshold(
                scores[val_fold_index], target_array[val_fold_index], 10000)
            accuracy[i] = self.lfw_get_accuracy(
                scores[test_fold_index], target_array[test_fold_index], threshold)

            self.logger.info(('[{:d}] fold accuracy: {:1.4f}, threshold: {:1.4f}'.format(i, accuracy[i], threshold)))

        mean_accuracy = np.mean(accuracy)
        std = np.std(accuracy)
        self.logger.info(('Accuracy: {:1.4f}+-{:1.4f}'.format(mean_accuracy, std)))
        self.tensorboard_logger.scalar_summary('lfw_accuracy', mean_accuracy, self.iteration)

        return mean_accuracy, std, accuracy

    def train(self, epoch):
        """Train a epoch"""
        if self.iteration >= self.settings.nIters:
            return
        batch_time = utils.AverageMeter()
        data_load_time = utils.AverageMeter()
        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()
        iters = len(self.train_loader)
        # switch to train mode
        self.model.train()

        end = time.time()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.update_lr(self.iteration)

            # measure data loading time
            data_load_time.update(time.time() - end)
            end = time.time()
            data, target = data.cuda(), target.cuda()
            output, loss = self.forward(data, target)
            # compute gradient and do SGD step
            self.backward(loss)
            # measure accuracy and record loss
            single_error, single_loss, single5_error = utils.compute_singlecrop(
                outputs=output, labels=target,
                loss=loss, top5_flag=True)
            single_error /= target.size(0)
            single5_error /= target.size(0)

            losses.update(single_loss, data.size(0))
            top1.update(single_error, data.size(0))
            top5.update(single5_error, data.size(0))

            self.tensorboard_logger.scalar_summary(
                'train_loss', single_loss, self.iteration)
            self.tensorboard_logger.scalar_summary(
                'train_top1error', single_error, self.iteration)
            self.tensorboard_logger.scalar_summary(
                'train_top5error', single5_error, self.iteration)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            self.iteration = self.iteration + 1

            utils.print_result(epoch, self.settings.nEpochs, batch_idx + 1,
                               iters, self.lr_master.lr, data_load_time.avg, batch_time.avg,
                               single_error, single_loss, top5error=single5_error,
                               mode="Train",
                               logger=self.logger)

            if self.iteration >= self.settings.nIters or self.settings.nEpochs == 1:
                break

        self.logger.info(
            "|===>Training Error: {:.4f} Loss: {:.4f}, Top5 Error: {:.4f}".format(top1.avg, losses.avg, top5.avg))
        return top1.avg, losses.avg, top5.avg
