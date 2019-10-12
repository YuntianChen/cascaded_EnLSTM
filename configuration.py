# -*- coding: utf-8 -*-
"""
Created on Wed May 23 19:15:00 2018

@author: Yuntian Chen
"""
import torch


class DefaultConfiguration:

    def __init__(self):
        # 数据相关设置
        self.well_num = 6
        self.train_len = 130
        self.shrink_len = 1
        self.window_step = 10
        self.head = ['DEPT', 'RMG', 'RMN', 'RMN-RMG', 'CAL', 'SP', 'GR', 'HAC', 'BHC', 'DEN']
        self.columns = ['DEPT', 'RMN-RMG', 'CAL', 'SP', 'GR', 'HAC', 'BHC', 'DEN']
        self.columns_input = ['DEPT', 'RMN-RMG', 'CAL', 'SP', 'GR']
        self.columns_target = ['HAC', 'BHC', 'DEN']
        # 神经网络设置
        self.ERROR_PER = 0.02 # 0.02
        self.drop_last = False
        self.input_dim = 5
        self.hid_dim = 30
        self.num_layer = 1
        self.drop_out = 0.3
        self.output_dim = 3 # cascaded EnLSTM
        # 训练设置
        self.ne = 100
        self.T = 1
        self.batch_size = 64
        self.num_workers = 1
        self.epoch = 5
        self.GAMMA = 10
        # 实验参数设置
        '''
        self.train_ID = [2, 3, 4, 5, 6]
        self.test_ID = [1, ]
        self.data_prefix = 'data/vertical_all_A{}.csv'
        self.well_ID = 'A1'
        self.experiment_ID = '141' \
                             ''
        self.deviceID = 1
        '''
        self.test_ID = [6]
        self.train_ID = [i for i in range(1, self.well_num+1) if i not in self.test_ID]
        self.data_prefix = 'data/vertical_all_A{}.csv'
        self.well_ID = 'A{}'.format(self.test_ID[0])
        self.experiment_ID = '156'
        # self.deviceID = 0
        self.deviceID = int(self.experiment_ID) % 2






        self.path = 'Experiments/{}'.format(self.experiment_ID)
        self.info = "Exp{}-{}-{}-{}-{}".format(self.experiment_ID, str(self.hid_dim),
                                               str(self.batch_size), str(self.ne), str(self.ERROR_PER))
        if torch.cuda.is_available():
            self.use_gpu = True
        else:
            self.use_gpu = False
        self.text_path = 'data/vertical_all_{}.xls'.format(self.well_ID)


config = DefaultConfiguration()
