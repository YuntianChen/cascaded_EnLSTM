# -*- coding: utf-8 -*-
# @Time    : 8/23/2019 15:59
# @Author  : yuanqi
# @File    : loss_record_simple.py
# @Software: PyCharm

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

exp = 5 # the total number of experments
epoch = 3 # the total number of epoches

exp_data_summary = [] # 存下来不同epoch的平均结果的汇总
exp_data_all = [] # 全部实验数据拼接
exp_data_summary_std = []
print(type(exp_data_all))
for j in range(epoch):
    exp_data_mean_list = []
    for i in range(exp):
        exp_data = [] # 存下来每个实验每个epoch的每个曲线的结果
        exp_ID = i + 1
        epoch_ID = j
        # 设置读取的文件夹
        loss_file_fmt = 'evalute_losses.csv'
        feature_list = ['E_HORZ', 'E_VERT','COHESION', 'UCS', 'DEN', 
                        'ST', 'BRITTLE_HORZ', 'BRITTLE_VERT', 'PR_HORZ', 
                        'PR_VERT', 'NPRL', 'TOC', 'mean']
        # Making the experiments list
        experiment_fmt = "{:0>3d}"
        experiment_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        print('exp_ID = %d' % exp_ID)
        experiment_list = [i + exp_ID * 1000 for i in experiment_list]
        experiment_list = [experiment_fmt.format(i) for i in experiment_list]

        # 读取文件夹数据并合并
        for k, experiment in enumerate(experiment_list):
            feature_file = os.path.join('Experiments', experiment, loss_file_fmt)
            loss_test = pd.read_csv(feature_file).iloc[epoch_ID]    
            exp_data.append(loss_test.values)
        exp_data = np.delete(exp_data, 0, axis=1) # 每个实验的不同井的预测结果
        np.savetxt('loss_exp%d_epoch%d.csv' % (exp_ID, epoch_ID), exp_data, delimiter = ',')  # 存下来每个实验每个epoch的每个曲线的结果
        exp_data_mean = np.mean(exp_data,0) # 每个实验内部不同井的平均结果
        exp_data_mean_list.append(exp_data_mean) # 记录每个重复实验中，不同井的平均结果
   
    exp_data_mean_ave = np.mean(exp_data_mean_list, 0) # 不同实验的平均结果
    exp_data_mean_std = np.std(exp_data_mean_list, 0) # 不同实验的std
    np.savetxt('loss_ave_epoch%d.csv' % epoch_ID, exp_data_mean_ave, delimiter = ',') # 存下来每个epoch的平均结果，对于实验数目求平均
    np.savetxt('loss_std_epoch%d.csv' % epoch_ID, exp_data_mean_std, delimiter = ',') # 存下来每个epoch的std(此时求std的对象都是不同井的均值，相当于5个样本求std)
    exp_data_summary.append(exp_data_mean_ave)
    exp_data_summary_std.append(exp_data_mean_std)      
np.savetxt('loss_summary.csv', exp_data_summary, delimiter = ',') # 存下来不同epoch的平均结果的汇总
np.savetxt('loss_summary_std.csv', exp_data_summary_std, delimiter = ',') # 存下来不同epoch的std的汇总


