"""
@File: GRU.py
@Author: zhang
@Time: 10/18/22 3:37 PM
"""
import torch
from torch import nn
from d2l import torch as d2l

""""
加载数据集
"""
batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
"""

"""
num_inputs = len(vocab)
gru_layer = nn.GRU(num_inputs, num_hiddens)
model = d2l.RNNModel(gru_layer, vocab_size)
model = model.to(device)
num_epochs, lr = 500, 1
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
d2l.plt.show()

