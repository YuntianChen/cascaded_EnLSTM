# -*- coding: utf-8 -*-
"""
Created on Thu May 24 21:34:10 2018

@author: Yuntian Chen
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May  9 22:34:42 2018

@author: Yuntian Chen
"""
"""
Created on Thu Apr 12 10:51:53 2018

@author: lthpc
定义RNN(LSTM)网络
"""


import torch as t
import torch.nn as nn
from torch.autograd import Variable
from configuration import config
import torch.nn.functional as F  

class netLSTM(nn.Module):
    def __init__(self):
        super(netLSTM, self).__init__()
        self.lstm = nn.LSTM(config.input_dim, config.hid_dim, 
                            config.num_layer, batch_first=True, dropout=config.drop_out)
        # 全连接至预测的测井曲线
        # self.fc1 = nn.Linear(config.hid_dim, config.hid_dim)
        self.fc2 = nn.Linear(config.hid_dim, int(config.hid_dim/2))
        self.fc3 = nn.Linear(int(config.hid_dim/2), config.output_dim)

    def forward(self, x, hs=None, use_gpu=config.use_gpu):
        # batch_size = config.batch_size
        batch_size = x.size(0) # 不能用batch_size = config.batch_size，因为从第二个epoch开始，dataloder导入的数据batch_size变为了2，如果用config.batch_size,那么hs维度和输入的x会不匹配。
        if hs is None:
            h = Variable(t.zeros(config.num_layer, batch_size, config.hid_dim))
            c = Variable(t.zeros(config.num_layer, batch_size, config.hid_dim))
            hs = (h, c)
        if use_gpu:
            hs = (hs[0].cuda(), hs[1].cuda())
        self.lstm.flatten_parameters()
        out, hs_0 = self.lstm(x, hs)  # 输入：batch_size * train_len * input_dim；输出：batch_size * train_len * hid_dim
        out = out[:, -10:, :]
        self.lstm.flatten_parameters()
        out = out.contiguous()
        out = out.view(-1, config.hid_dim)  # 相当于reshape成(batch_size * train_len) * hid_dim的二维矩阵
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out, hs_0


class netLSTM_withbn(nn.Module):
    def __init__(self):
        super(netLSTM_withbn, self).__init__()
        self.lstm = nn.LSTM(config.input_dim, config.hid_dim,
                            config.num_layer, batch_first=True, dropout=config.drop_out)
        # 全连接至预测的测井曲线
        # self.fc1 = nn.Linear(config.hid_dim, config.hid_dim)
        self.fc2 = nn.Linear(config.hid_dim, int(config.hid_dim / 2))
        self.fc3 = nn.Linear(int(config.hid_dim / 2), config.output_dim)
        # self.fc4 = nn.Linear(config.hid_dim, config.hid_dim)
        # self.fc = nn.Linear(config.hid_dim, config.output_dim)
        self.bn = nn.BatchNorm1d(int(config.hid_dim / 2))

    def forward(self, x, hs=None, use_gpu=config.use_gpu):
        # batch_size = config.batch_size
        batch_size = x.size(0)
        # 不能用batch_size = config.batch_size，因为从第二个epoch开始，dataloder导入的数据batch_size变为了2，如果用config.batch_size,那么hs维度和输入的x会不匹配。
        if hs is None:
            h = Variable(t.zeros(config.num_layer, batch_size, config.hid_dim))
            c = Variable(t.zeros(config.num_layer, batch_size, config.hid_dim))
            hs = (h, c)
        if use_gpu:
            hs = (hs[0].cuda(), hs[1].cuda())
        out, hs_0 = self.lstm(x, hs)  # 输入：batch_size * train_len * input_dim；输出：batch_size * train_len * hid_dim
        # out = out[:, -10:, :]
        out = out.contiguous()
        out = out.view(-1, config.hid_dim)  # 相当于reshape成(batch_size * train_len) * hid_dim的二维矩阵
        # out = self.fc1(out) #batch_size*train_len, output_dim
        # out = self.fc2(out)
        out = F.relu(self.bn(self.fc2(out)))
        out = self.fc3(out)
        return out, hs_0

