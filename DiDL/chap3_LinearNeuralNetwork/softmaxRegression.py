"""
@File: softmaxRegression.py
@Author: zhang
@Time: 10/6/22 10:30 PM
"""

import torch
import torchvision
from d2l.torch import Animator, Accumulator, accuracy
from torch import nn
from d2l import torch as d2l
from torch.utils import data
from torchvision.transforms import transforms

batch_size = 256

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

"""初始化模型参数"""
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


net.apply(init_weights)

"""损失函数"""
loss = nn.CrossEntropyLoss(reduction='none')

"""优化算法"""
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

"""训练"""
num_epochs = 10

d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
