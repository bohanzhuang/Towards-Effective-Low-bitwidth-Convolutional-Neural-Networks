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


def segment_prune(model, train_loader, test_loader, trainer, opt,
                  segment_name="features", previous_record=None,):
    """
    :params model: pruned model
    :params segment_name: name of segment, value: <features> or <classifier> only, default: <features>
    """

    if segment_name == "features":
        segment = model.features
    elif segment_name == "classifier":
        segment = model.classifier
    else:
        assert False, "unsupport segment" + segment_name

    prune_id = 0
    last_prune_id = 0
    bn_id = None
    layer_count = 0

    prune_record = {}
    result_record = {}
    for layer in segment:
        print(layer_count)
        # prune conv and linear layers
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            prune_id = layer_count
            print(("prune_id: %d, last_prune_id: %d" % (prune_id, last_prune_id)))
            if segment_name == "classifier" or prune_id > 0:
                if segment_name == "classifier" and prune_id == 0:
                    layer_prune = FilterPrune(4, 4)
                    print(model)
                else:
                    layer_prune = FilterPrune(1, 1)
                    print(model)
                
                # features extraction
                for i, (images, labels) in enumerate(train_loader):
                    images = images.cuda()
                    images_var = Variable(images)

                    if segment_name == "features":
                        output = images_var
                    else:
                        output = model.features(images_var)

                    for l in range(0, layer_count):
                        if isinstance(segment[l], nn.Linear) and output.dim() != 2:
                            output = output.view(output.size(0), -1)
                        output = segment[l](output)

                    if isinstance(segment[layer_count], nn.Linear) and output.dim() != 2:
                        output = output.view(output.size(0), -1)
                        
                    # print output.size()
                    y = segment[layer_count](output)

                    layer_prune.feature_extract(output, y, segment[layer_count])
                    if i * opt.batchSize >= 1e4:
                        print(("break: ", i))
                        break

                # channel selection
                w_hat = layer_prune.channel_select(segment[prune_id])
                prune_record[prune_id] = layer_prune.select_channels

                # replace current layer
                thin_weight, thin_bias = layer_prune.get_thin_params(segment[prune_id], 1)
                segment = replace_layer(segment,
                                        prune_id,
                                        thin_weight,
                                        thin_bias,
                                        w_hat)

                if prune_id != last_prune_id:
                    # prune and replace last layer
                    thin_weight, thin_bias = layer_prune.get_thin_params(segment[last_prune_id], 0)
                    segment = replace_layer(segment,
                                            last_prune_id,
                                            thin_weight,
                                            thin_bias)

                # prune and replace batchnorm layer if layer exists
                if bn_id is not None:
                    thin_weight, thin_bias = layer_prune.get_thin_params(segment[bn_id], 0)
                    segment = replace_layer(segment,
                                            bn_id,
                                            thin_weight,
                                            thin_bias)
                    bn_id = None

                if segment_name == "features":
                    model.features = segment
                else:
                    model.classifier = segment
                    if prune_id == 0:
                        thin_weight, thin_bias = layer_prune.get_thin_params(model.features[list(previous_record.keys())[-1]], 0)
                        model.features = replace_layer(model.features,
                                                      list(previous_record.keys(
                                                      ))[-1],
                                                      thin_weight,
                                                      thin_bias)

                print(("new structure of model:", model))
                print("---------------------------------------")
                model.cuda()
                trainer.model = model
                training_top1error, _, _ = trainer.train(epoch=0, train_loader=train_loader)
                testing_top1error, _, _ = trainer.test(epoch=0, test_loader=test_loader)
                result_record[prune_id] = [training_top1error, testing_top1error]

            last_prune_id = prune_id
        elif isinstance(layer, nn.BatchNorm2d):
            bn_id = layer_count
        layer_count += 1

    return model, prune_record, result_record


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

    if opt.netType == "LeNet5":
        model = MD.LeNet5()
    else:
        assert False, "testing model"

    if check_point_params['model'] is not None:
        previous_model_dict = check_point_params['model']
        # model.load_state_dict(check_point_params['model'])
        model_dict = model.state_dict()
        for key, value in list(previous_model_dict.items()):
            if key in list(model_dict.keys()):
                model_dict[key] = value
        model.load_state_dict(model_dict)
    # model = dataparallel(model, opt.nGPU, opt.GPU)
    model.cuda()

    # testing original model
    trainer = Trainer(model=model, opt=opt)
    # trainer.test(epoch=0, test_loader=test_loader)

    # filter level prune
    # prune lenet5
    print(("model structure:", model))
    print("--------------------------------------")

    result_record = []
    model, prune_record, result = segment_prune(
        model, train_loader, test_loader, trainer, opt)
    result_record.append(result)
    model, prune_record, result = segment_prune(model, train_loader, test_loader, trainer, opt,
                                        "classifier", prune_record)
    result_record.append(result)
    print("======================================")
    print(result_record)


if __name__ == '__main__':
    main()
