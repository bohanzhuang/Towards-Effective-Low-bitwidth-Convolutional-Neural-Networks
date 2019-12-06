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
        self.optimizer = torch.optim.SGD(params=self.model.parameters(),
                                         lr=self.lr_master.lr,
                                         momentum=self.momentum,
                                         weight_decay=self.weight_decay,
                                         nesterov=True,
                                         )

    def backward(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        for layer in self.model.modules():
            # print type(layer)
            if isinstance(layer, wcConv2d) or isinstance(layer, wcLinear):
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
        img_clone = img.clone()
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        for t, m, s in zip(img_clone, mean, std):
            t.mul(s).add(m)
        img_clone = img_clone.cpu()

        img_flip_list = []
        for img_c in img_clone:
            img_pil = self.ToPILImage(img_c)
            img_pil_flip = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
            img_flip = self.ToTensor(img_pil_flip)
            for t, m, s in zip(img_flip, mean, std):
                t.sub(m).div(s)
            img_flip_list.append(img_flip)
        img_flip_list = torch.stack(img_flip_list)

        img_var = Variable(img, volatile=True).cuda()
        img_flip_var = Variable(img_flip_list, volatile=True).cuda()

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
        scores_eq = torch.gather(
            scores, 0, torch.nonzero(target_array.eq(1)).squeeze()).cpu()
        scores_ne = torch.gather(
            scores, 0, torch.nonzero(target_array.ne(1)).squeeze()).cpu()
        for i in range(2 * threshold_num + 1):
            accuracys[i] = (scores_eq.gt(
                thresholds[i]) + scores_ne.lt(thresholds[i])).sum()
        accuracys /= float(scores.size(0))
        best_threshold = np.mean(
            thresholds[np.where(accuracys == np.max(accuracys))])
        return best_threshold

    def lfw_get_accuracy(self, scores, target_array, threshold):
        """Get lfw accuracy with respect to threshold."""
        accuracy = (torch.gather(scores, 0, torch.nonzero(target_array.eq(1)).squeeze()).gt(threshold) +
                    torch.gather(scores, 0, torch.nonzero(target_array.ne(1)).squeeze()).lt(threshold)).sum() / float(scores.size(0))
        # accuracy = (torch.masked_select(scores, target_array.eq(1)).gt(threshold).sum() +
        #             torch.masked_select(scores, target_array.ne(1)).lt(threshold).sum()) * 1.0 / scores.size(0)
        # accuracy = (len(np.where(scores[np.where(target_array == 1)] > threshold)[
        #             0]) + len(np.where(scores[np.where(target_array != 1)] < threshold)[0])) / float(len(scores))
        return accuracy

    def test(self):
        """Evaluate on lfw dataset."""
        self.model.eval()
        feature_dim = self.feature_dim
        feature_left_array = tuple()
        feature_right_array = tuple()
        target_array = tuple()
        fold_array = tuple()

        accuracy = np.zeros(10,)
        forward_time = 0
        for batch_idx, (img_l, img_r, target, fold) in enumerate(self.test_loader):
            img_l, img_r = img_l.cuda(), img_r.cuda()

            feature_l, single_time = self.extract_deep_feature(img_l)
            forward_time += single_time
            feature_r, single_time = self.extract_deep_feature(img_r)
            forward_time += single_time

            feature_left_array += (feature_l.data, )
            feature_right_array = feature_right_array + (feature_r.data, )
            target_array = target_array + (target, )
            fold_array = fold_array + (fold, )

        feature_left_array = torch.cat(feature_left_array, 0)
        feature_right_array = torch.cat(feature_right_array, 0)
        target_array = torch.cat(target_array).cuda()
        fold_array = torch.cat(fold_array).cuda()
        # print feature_left_array.size()
        # print target_array.size()
        print("forward time", forward_time)
        # print fold_array
        for i in range(10):
            # split 10 filds into val & test set

            val_fold_mask = torch.nonzero(fold_array.ne(i))
            test_fold_mask = torch.nonzero(fold_array.eq(i))
            feature_ls = feature_left_array.clone()
            feature_rs = feature_right_array.clone()
            # get normalized feature
            feature = torch.cat((torch.gather(feature_ls, 0, val_fold_mask.expand(
                val_fold_mask.size(0), feature_ls.size(1))),
                torch.gather(feature_rs, 0, val_fold_mask.expand(
                    val_fold_mask.size(0), feature_rs.size(1)))
            ))
            # feature = torch.cat((torch.masked_select(feature_ls, val_fold_mask.unsqueeze(1)).view(val_fold_mask.sum(), -1),
            #                      torch.masked_select(feature_rs, val_fold_mask.unsqueeze(1)).view(val_fold_mask.sum(), -1)))
            # print feature.size()

            mu = feature.mean(0)
            # print mu.size()

            feature_ls -= mu.unsqueeze(0)
            feature_rs -= mu.unsqueeze(0)
            feature_ls /= feature_ls.pow(2).sum(1).sqrt().unsqueeze(1)
            feature_rs /= feature_rs.pow(2).sum(1).sqrt().unsqueeze(1)

            # get accuracy of the ith fold using cosine similarity
            scores = (feature_ls * feature_rs).sum(1)
            threshold = self.lfw_get_threshold(torch.gather(scores, 0, val_fold_mask.squeeze()),
                                               torch.gather(
                                                   target_array, 0, val_fold_mask.squeeze()),
                                               10000)
            # threshold = self.lfw_get_threshold(torch.masked_select(scores, val_fold_mask),
            #                                    torch.masked_select(target_array, val_fold_mask), 10000)
            # print threshold

            accuracy[i] = self.lfw_get_accuracy(torch.gather(scores, 0, test_fold_mask.squeeze()),
                                                torch.gather(
                                                    target_array, 0, test_fold_mask.squeeze()),
                                                threshold)
            # accuracy[i] = self.lfw_get_accuracy(
            #     torch.masked_select(scores, test_fold_mask),
            #     torch.masked_select(target_array, test_fold_mask),
            #     threshold)

            print('[%d] fold accuracy: %1.4f, threshold: %1.4f' %
                  (i, accuracy[i], threshold))

        mean_accuracy = np.mean(accuracy)
        std = np.std(accuracy)
        print('Accuracy: %1.4f+-%1.4f' % (mean_accuracy, std))
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
        print("|===>Training Error: %.4f Loss: %.4f, Top5 Error:%.4f" % (top1.avg, losses.avg, top5.avg))
        return top1.avg, losses.avg, top5.avg
