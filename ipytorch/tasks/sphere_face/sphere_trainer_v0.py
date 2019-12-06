import time
import torch
import torch.autograd
import torch.nn.functional as F
import torch.nn.parallel
import torch.nn as nn
from torch.autograd import Variable
from PIL import Image
from torchvision import transforms
import utils
import itertools
import os
import shutil
import numpy as np

# weighted channel package
from prune import wcConv2d, wcLinear


__all__ = ["SphereTrainer"]


class SphereTrainer(object):
    """Trainer class

    Trainer takes charge of network training, lfw evaluation and network saving.

    Attributes:
        cuda: Whether to run on GPU.
        model: Network model.
        feature_dim: The dimension of face feature.
        optim: Optimizer (default SGD).
        train_loader: Training dataset loader.
        test_loader: LFW dataset loader.
        save_path: Path to save traing log.
        epoch: Current epoch during training.
        iteration: Current iteration during training.
        max_iter: Max number of iteration.
        criterion: Criterion that measures the error between the predicted value and the ground truth.
        best_lfw_accuracy: The best accuracy during evaluation on lfw dataset.
        logger: Tensorboard logger.
        ToPILImage: Converts a torch.*Tensor to PIL.Image function.
        ToTensor: Converts a PIL.Image to torch.*Tensor function.
    """

    def __init__(self, model, lr_master, n_epochs, n_iters, train_loader, test_loader,
                 feature_dim,
                 momentum=0.9, weight_decay=1e-4, optimizer_state=None,
                 logger=None, ngpu=1, gpu=0):
        self.model = utils.data_parallel(model, ngpu, gpu)

        self.n_iters = n_iters
        self.n_epochs = n_epochs
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.lr_master = lr_master
        self.optimizer = torch.optim.SGD(params=self.model.parameters(),
                                         lr=self.lr_master.lr,
                                         momentum=momentum,
                                         weight_decay=weight_decay,
                                         nesterov=True,
                                         )
        weight_params = []
        bias_params = []

        self.logger = logger
        self.ngpu = ngpu
        self.gpu = gpu
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.feature_dim = feature_dim
        self.iteration = 0
        self.criterion = F.cross_entropy
        self.ToPILImage = transforms.ToPILImage()
        self.ToTensor = transforms.ToTensor()

    def reset_lr(self, lr_master, n_iters):
        self.lr_master = lr_master
        self.n_iters = n_iters
        self.iteration = 0

    def update_lr(self, iters):
        """
        update learning rate of optimizers
        :param iters: current training iteration
        """
        lr = self.lr_master.get_lr(iters)
        # update learning rate of model optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        self.logger.scalar_summary('learning_rate', lr, iters)

    def reset_model(self, model):
        self.model = utils.data_parallel(model, self.ngpu, self.gpu)
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.SGD(params=parameters,
                                         lr=self.lr_master.lr,
                                         momentum=self.momentum,
                                         weight_decay=self.weight_decay,
                                         nesterov=True,
                                         )
    def compute_num_channels(self):
        self.num_channels = 0
        for layer in self.model.modules():
            if isinstance(layer, wcConv2d):
                self.num_channels += layer.binary_weight.size(0)
                                                     
    def compute_ratio(self, final_r, epoch, n_epoch):
        return min(epoch, n_epoch)*final_r/n_epoch

    def update_threshold(self, ratio=0):
        # compute number of select channels
        select_channels = min(int(self.num_channels*ratio), self.num_channels-1)
        if select_channels == 0:
            return
        
        # get all float_weight
        float_weight_cache = None
        for layer in self.model.modules():
            if isinstance(layer, (wcConv2d, wcLinear)) and layer.weight.size(1) > 3:
                if float_weight_cache is None:
                    float_weight_cache = layer.float_weight.data.clone()
                else:
                    float_weight_cache = torch.cat((float_weight_cache, layer.float_weight.data))
        float_weight_cache = float_weight_cache.view(-1)
        # print self.num_channels, select_channels, float_weight_cache.size()

        # compute threshold
        r = float_weight_cache.topk(select_channels, dim=0, largest=False)[0].max()
        
        for layer in self.model.modules():
            if isinstance(layer, wcConv2d) and layer.weight.size(1) > 3:
                layer.rate.fill_(r)

    def backward(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        """
        we constrain the grad_norm in the range of 0~2, otherwise the gradient will explode
        """
        torch.nn.utils.clip_grad_norm(self.model.parameters(), 2.)
        if self.iteration < 28000:
            for layer in self.model.modules():
                # print type(layer)
                if isinstance(layer, (wcConv2d, wcLinear)):
                    # print type(layer)
                    layer.compute_grad()         
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
        inv_idx = torch.arange(img.size(3)-1, -1, -1).long()
        img_flip = torch.index_select(img.cpu(), 3, inv_idx)

        img_var = Variable(img, volatile=True).cuda()
        img_flip_var = Variable(img_flip, volatile=True).cuda()

        extract_time = time.time()
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
                    0]) + len(np.where(scores[np.where(target_array != 1)] < threshold)[0])) / float(len(scores))
        return accuracy

    def test(self):
        """Evaluate on lfw dataset."""
        self.model.eval()
        lfw_batch_size = 100
        feature_dim = self.feature_dim
        nrof_pair = len(self.test_loader) * lfw_batch_size
        feature_left_array = np.zeros((nrof_pair, feature_dim * 2))
        feature_right_array = np.zeros((nrof_pair, feature_dim * 2))
        target_array = np.zeros((nrof_pair,))
        fold_array = np.zeros((nrof_pair,))

        accuracy = np.zeros(10,)
        forward_time = 0
        for batch_idx, (img_l, img_r, target, fold) in enumerate(self.test_loader):
            img_l, img_r = img_l.cuda(), img_r.cuda()

            target_numpy = target.numpy()
            fold_numpy = fold.numpy()
            feature_l, single_time = self.extract_deep_feature(img_l)
            forward_time += single_time
            feature_r, single_time = self.extract_deep_feature(img_r)
            forward_time += single_time
            feature_left_array[batch_idx * lfw_batch_size:(
                batch_idx + 1) * lfw_batch_size] = feature_l.data.cpu().numpy()
            feature_right_array[batch_idx * lfw_batch_size:(
                batch_idx + 1) * lfw_batch_size] = feature_r.data.cpu().numpy()
            target_array[batch_idx *
                         lfw_batch_size:(batch_idx + 1) * lfw_batch_size] = target_numpy
            fold_array[batch_idx *
                       lfw_batch_size:(batch_idx + 1) * lfw_batch_size] = fold_numpy

        print("forward time", forward_time)
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

            print(('[%d] fold accuracy: %1.4f, threshold: %1.4f' %
                  (i, accuracy[i], threshold)))

        mean_accuracy = np.mean(accuracy)
        std = np.std(accuracy)
        print(('Accuracy: %1.4f+-%1.4f' % (mean_accuracy, std)))
        self.logger.scalar_summary(
            'lfw_accuracy', mean_accuracy, self.iteration)

        return mean_accuracy, std, accuracy

    def train(self, epoch):
        """Train a epoch"""
        if self.iteration >= self.n_iters:
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
            data_var, target_var = Variable(
            data.cuda()), Variable(target.cuda())
            output, loss = self.forward(data_var, target_var)
            # compute gradient and do SGD step
            self.backward(loss)
            # measure accuracy and record loss
            single_error, single_loss, single5_error = utils.compute_singlecrop(outputs=output, labels=target_var,
                                                                                loss=loss, top5_flag=True)
            single_error /= target_var.size(0)
            single5_error /= target_var.size(0)

            losses.update(single_loss, data.size(0))
            top1.update(single_error, data.size(0))
            top5.update(single5_error, data.size(0))

            self.logger.scalar_summary(
                'train_loss', single_loss, self.iteration)
            self.logger.scalar_summary(
                'train_top1error', single_error, self.iteration)
            self.logger.scalar_summary(
                'train_top5error', single5_error, self.iteration)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            self.iteration = self.iteration + 1

            utils.print_result(epoch, self.n_epochs, batch_idx + 1,
                               iters, self.lr_master.lr, data_load_time.avg, batch_time.avg,
                               single_error, single_loss, top5error=single5_error,
                               mode="Train")

            if self.iteration >= self.n_iters or self.n_epochs == 1:
                break
            
            r = self.compute_ratio(0.9, epoch*iters+batch_idx+1, 16*iters)
            self.update_threshold(r)

        print("|===>Training Error: %.4f Loss: %.4f, Top5 Error:%.4f" % (top1.avg, losses.avg, top5.avg))
        return top1.avg, losses.avg, top5.avg
