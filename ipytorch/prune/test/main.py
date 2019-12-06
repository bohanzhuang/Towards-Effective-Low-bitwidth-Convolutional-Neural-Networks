import sys
sys.path.insert(0, '../')
from dataloader import *
from utils import *
from opt import *
import models as MD
from checkpoint import *

from trainer import *
from prune import *
import torch
import torch.nn


def main():
    opt = NetOption()

    # create data loader
    data_loader = DataLoader(dataset=opt.data_set, batch_size=opt.batchSize, data_path=opt.dataPath,
                             n_threads=opt.nThreads, ten_crop=opt.tenCrop, dataset_ratio=opt.datasetRatio)
    train_loader, test_loader = data_loader.getloader()

    # define check point
    check_point = CheckPoint(opt=opt)
    # create residual network model
    if opt.retrain:
        check_point_params = check_point.retrainmodel()

    """if opt.netType == "LeNet5":
        model = MD.LeNet5()
    elif opt.netType == "PreResNet":
        model = MD.PreResNet(opt.depth)
    else:
        assert False, "testing model"

    if check_point_params['model'] is not None:
        previous_model_dict = check_point_params['model']
        # model.load_state_dict(check_point_params['model'])
        model_dict = model.state_dict()
        for key, value in previous_model_dict.items():
            if key in model_dict.keys():
                model_dict[key] = value
        model.load_state_dict(model_dict)
    # model = dataparallel(model, opt.nGPU, opt.GPU)
    model.cuda()"""
    model = MD.resnet50()
    model.cuda()

    # testing original model
    trainer = Trainer(model=model, opt=opt)
    # trainer.test(epoch=0, test_loader=test_loader)

    # filter level prune
    # prune lenet5
    print(("model structure:", model))
    print("--------------------------------------")

    model_prune = ModelPrune(model, train_loader, test_loader, trainer, net_type="resnet_bottleneck")
    model_prune()

if __name__ == '__main__':
    main()
