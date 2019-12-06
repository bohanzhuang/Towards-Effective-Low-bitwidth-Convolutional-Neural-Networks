from ipytorch.models.custom.Mobilenet import MobileNet
from ipytorch.models.custom.MobilenetV1 import MobileNetV1
import torch
import torch.nn as nn

def main():
    origin_model = MobileNet()
    origin_checkpoint_param = torch.load("/home/liujing/Experiments/ChannelSelection/log_MobileNet_imagenet_bs128_lr0.010_auxMobileNet-b17-[12,24,27]-channel-selection-r0.1-0729-01/check_point/model_000.pth")
    origin_state_dict = origin_checkpoint_param['model']
    for key in origin_state_dict.keys():
        print(key)
    new_model_state_dict = origin_state_dict
    # new_model_state_dict = {}
    # for key, value in origin_state_dict.items():
    #     key = key.replace('module.', '')
    #     new_model_state_dict[key] = value
    tmp_state_dict = origin_model.state_dict()
    print("---------------------")
    for key in tmp_state_dict.keys():
        print(key)
    origin_model.load_state_dict(new_model_state_dict)
    # origin_model = origin_model.module
    layer_list = []
    for layer in origin_model.model.modules():
        if isinstance(layer, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.AvgPool2d)):
            layer_list.append(layer)
    origin_feature = nn.Sequential(*layer_list)
    new_model = MobileNetV1()
    new_model.features.load_state_dict(origin_feature.state_dict())
    new_model.classifier.load_state_dict(origin_model.fc.state_dict())
    checkpoint_param = {}
    checkpoint_param["model"] = new_model.state_dict()
    checkpoint_param["aux_fc"] = origin_checkpoint_param["aux_fc"]
    # checkpoint_param["seg_opt"] = None
    # checkpoint_param["fc_opt"]  = None
    # torch.save(new_model.state_dict(), "ImageNet-MobileNet_v1-baseline-01.pth")
    torch.save(checkpoint_param, "/home/liujing/Experiments/ChannelSelection/log_MobileNet_imagenet_bs128_lr0.010_auxMobileNet-b17-[12,24,27]-channel-selection-r0.1-0729-01/check_point/model_000_MobileNetv1.pth")
    print("program done!")

if __name__ == '__main__':
    main()