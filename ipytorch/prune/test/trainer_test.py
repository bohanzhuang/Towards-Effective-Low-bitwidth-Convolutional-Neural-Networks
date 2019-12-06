import time
import torch.autograd
from utils import *
from torch.autograd import Variable
import models as MD

class Trainer(object):
    """
    trainer for training network, use SGD
    """
    def __init__(self, model, opt, optimizer=None):
        """
        init trainer
        :param model: <list>type network model 
        :param opt: option parameters define by users
        :param optimizer: optimizer to update parameters of network, if exists
        """
        self.opt = opt
        self.model = model
        
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.lr = self.opt.LR

        if isinstance(self.model, list):
            optim_list = []
            for i in range(len(self.model)):
                optim_list.append({'params': self.model[i].parameters(), 'lr': self.lr})
        else:
            optim_list = [{'params': self.model.parameters(), 'lr': self.lr}]

        self.optimzer = optimizer or torch.optim.SGD(params=optim_list,
                                                     lr=self.lr,
                                                     momentum=self.opt.momentum,
                                                     weight_decay=self.opt.weightDecay,
                                                     nesterov=True,
                                                     )

        self.scalar_info = {}

    def updateopts(self):
        """
        update optimizers
        :return: no return parameters
        """
        if isinstance(self.model, list):
            optim_list = []
            for i in range(len(self.model)):
                optim_list.append({'params': self.model[i].parameters(), 'lr': self.lr})
            self.optimzer = torch.optim.SGD(params=optim_list,
                                            lr=self.lr,
                                            momentum=self.opt.momentum,
                                            weight_decay=self.opt.weightDecay,
                                            nesterov=True)
        else:
            self.optimzer = torch.optim.SGD(params=self.model.parameters(),
                                            lr=self.lr,
                                            momentum=self.opt.momentum,
                                            weight_decay=self.opt.weightDecay,
                                            nesterov=True)

    def updatelearningrate(self, epoch):
        """
        update learning rate of optimizers
        :param epoch: current training epoch
        :return: no parameters to return
        """
        self.lr = getlearningrate(epoch=epoch, opt=self.opt)
        # update learning rate of model optimizer
        for param_group in self.optimzer.param_groups:
            param_group['lr'] = self.lr

    def forward(self, images, labels=None):
        # forward and backward and optimize
        if isinstance(self.model, list):
            output = self.model[0](images)
            for i in range(1, len(self.model)):
                output = self.model[i](output)
        else:
            output = self.model(images)

        if labels is not None:    
            loss = self.criterion(output, labels)
            return output, loss
        else:
            return output

    def backward(self, loss):
        self.optimzer.zero_grad()
        loss.backward()
        self.optimzer.step()

    def train(self, epoch, train_loader):
        
        top1_error = 0
        top1_loss = 0
        top5_error = 0
        images_count = 0
        iters = len(train_loader)

        self.updatelearningrate(epoch)

        if isinstance(self.model, list):
            for i in range(len(self.model)):
                self.model[i].train()
        else:
            self.model.train()

        start_time = time.time()
        end_time = start_time

        for i, (images, labels) in enumerate(train_loader):
            start_time = time.time()
            data_time = start_time - end_time

            images = images.cuda()
            labels = labels.cuda()
            images_var = Variable(images)
            labels_var = Variable(labels)

            output, loss = self.forward(images_var, labels_var)
            self.backward(loss)

            single_error, single_loss, single5_error = computeresult(outputs=output, labels=labels_var,
                                                                     loss=loss, top5_flag=True)
            top1_error += single_error
            top1_loss += single_loss
            top5_error += single5_error
            end_time = time.time()
            iter_time = end_time - start_time

            images_count += images.size(0)

            total_time, left_time = printresult(epoch, self.opt.nEpochs, i+1, 
                                                iters, self.lr, data_time, iter_time,
                                                single_error/images.size(0), 
                                                single_loss, top5error=single5_error/images.size(0),
                                                mode="Train")

            if self.opt.nEpochs == 1 and i+1 >= 50:
                print("|===>Program testing for only 50 iterations")
                break

        top1_loss /= iters
        top1_error /= images_count
        top5_error /= images_count
        
        self.scalar_info['training_top1error'] = top1_error
        self.scalar_info['training_top5error'] = top5_error
        self.scalar_info['training_loss'] = top1_loss

        print(("|===>Training Error: %.4f Loss: %.4f, Top5 Error:%.4f" % (top1_error, top1_loss, top5_error)))
        return top1_error, top1_loss, top5_error

    def test(self, epoch, test_loader):

        top1_error = 0
        top1_loss = 0
        top5_error = 0
        images_count = 0
        if isinstance(self.model, list):
            for i in range(len(self.model)):
                self.model[i].eval()
        else:
            self.model.eval()

        iters = len(test_loader)

        start_time = time.time()
        end_time = start_time
        for i, (images, labels) in enumerate(test_loader):
            start_time = time.time()
            data_time = start_time - end_time

            labels = labels.cuda()
            labels_var = Variable(labels, volatile=True)
            if self.opt.tenCrop:
                image_size = images.size()
                images = images.view(image_size[0]*10, image_size[1]/10, image_size[2], image_size[3])
                images_tuple = images.split(image_size[0])
                output = None
                for img in images_tuple:
                    img = img.cuda()
                    img_var = Variable(img, volatile=True)
                    temp_output, _ = self.forward(img_var)
                    if output is None:
                        output = temp_output.data
                    else:
                        output = torch.cat((output, temp_output.data))
                single_error, single_loss, single5_error = computetencrop(outputs=output, labels=labels_var)
            else:
                images = images.cuda()
                images_var = Variable(images, volatile=True)
                # output, loss, _, position = self.forward(images_var, labels_var)
                output, loss = self.forward(images_var, labels_var)

                # print all_label.shape
                single_error, single_loss, single5_error = computeresult(outputs=output, loss=loss,
                                                                         labels=labels_var, top5_flag=True)
            images_count += images.size(0)
            
            top1_loss += single_loss
            top1_error += single_error
            top5_error += single5_error

            end_time = time.time()
            iter_time = end_time - start_time

            total_time, left_time = printresult(epoch, self.opt.nEpochs, i+1, 
                                                iters, self.lr, data_time, iter_time,
                                                single_error/images.size(0), single_loss, 
                                                top5error=single5_error/images.size(0), 
                                                mode="Test")

            if self.opt.nEpochs == 1 and i+1 >= 50:
                print("|===>Program testing for only 50 iterations")
                break

        top1_loss /= iters
        top1_error /= images_count
        top5_error /= images_count

        self.scalar_info['testing_top1error'] = top1_error
        self.scalar_info['testing_top5error'] = top5_error
        self.scalar_info['testing_loss'] = top1_loss

        print(("|===>Testing Error: %.4f Loss: %.4f, Top5 Error: %.4f" % (top1_error, top1_loss, top5_error)))
        return top1_error, top1_loss, top5_error
