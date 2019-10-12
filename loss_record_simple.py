# -*- coding: utf-8 -*-
# @Time    : 8/23/2019 15:59
# @Author  : yuanqi
# @File    : loss_record_simple.py
# @Software: PyCharm

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

experiment_num = 1 # random repeated experiments
epoch_num = 1 # total number of the epoches
epoch_ID = 0 #No 5 epoch as the result ID

# 设置读取的文件夹
loss_file_fmt = 'test_loss_{}.txt'
feature_list = ['HAC', 'BHC', 'DEN']
# Making the experiments list
experiment_fmt = "{:0>3d}"
'''
experiment_list = []
for i in list(range(171, 177)):
    experiment_list.append(i)
experiment_list = [experiment_fmt.format(i) for i in experiment_list]

'''
experiment_list = [51, 52, 32, 53, 56, 55]
experiment_list = [experiment_fmt.format(i) for i in experiment_list]



# 读取文件夹数据并合并
all_data = []
a1_a6_data = []
for i, experiment in enumerate(experiment_list):
    experiment_data = []
    for feature in feature_list:
        feature_file = os.path.join('Experiments', experiment, loss_file_fmt.format(feature))
        loss_test = pd.read_csv(feature_file, header=None).iloc[:, 1]
        if len(loss_test) > epoch_num:
            loss_test = loss_test[:epoch_num]
        #print(loss_test.values)
        experiment_data.append(loss_test.values)
    a1_a6_data.append(experiment_data)
    if i % 6 == 5:
        all_data.append(a1_a6_data)
        a1_a6_data = []
print(np.array(experiment_list).shape)
print(np.array(all_data).shape)

data_lastepoch = np.array(all_data).reshape((-1, epoch_num))[:, epoch_ID]
print(data_lastepoch)
data_lastepoch_mean = np.mean(data_lastepoch.reshape((3, 6)), axis=1)
print(data_lastepoch_mean)
data_lastepoch_std = np.std(data_lastepoch.reshape((3, 6)), axis=1)
print(data_lastepoch_std)
data_lastepoch_std = np.std(data_lastepoch.reshape((-1)))
print(data_lastepoch_std)
print(data_lastepoch.reshape((-1)))