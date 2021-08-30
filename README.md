# Towards effective low-bitwidth convolutional neural networks

This project hosts the code for implementing the algorithms as presented in our papers:

````
@article{zhuang2021effective,
  title={Effective Training of Convolutional Neural Networks with Low-bitwidth Weights and Activations},
  author={Zhuang, Bohan and Liu, Jing and Tan, Mingkui and Liu, Lingqiao and Reid, Ian and Shen, Chunhua},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2021}
}

@inproceedings{zhuang2018towards,
  title={Towards effective low-bitwidth convolutional neural networks},
  author={Zhuang, Bohan and Shen, Chunhua and Tan, Mingkui and Liu, Lingqiao and Reid, Ian},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={7920--7928},
  year={2018}
}

````

## ipytorch

ipytorch is a self-implemented package for running experiments on pytorch

## Requirements
```
pip install git+https://github.com/chenyaofo/torchlearning.git@master
pip install pydot termcolor
pip install tensorflow
pip instal pytorch
pip install torchvision
pip install pydot
```
## Training and testing

For joint knowledge distillation on quantization, run python ./ipytorch/tasks/quantization/mutual_kl/trainer.py --conf_path imagenet_[2]_lambda1_T1.hocon --id 1


## Copyright

Copyright (c) Jing Liu. 2019

** This code is for non-commercial purposes only. For commerical purposes,
please contact Jing Liu <seliujing@@mail.scut.edu.cn> **

This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
