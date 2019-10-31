import numpy as np
import torch.utils.data
import pandas as pd
from sklearn import preprocessing
import torch

# NOTE:
#   Record TRAIN_ID and TEST_ID if you change them
WELL = 6
HEAD = ['DEPT', 'RMG', 'RMN', 'RMN-RMG', 'CAL', 'SP', 'GR', 'HAC', 'BHC', 'DEN']
COLUMNS = ['DEPT', 'RMN-RMG', 'CAL', 'SP', 'GR', 'HAC', 'BHC', 'DEN']
COLUMNS_TARGET = ['HAC', 'BHC', 'DEN']

TRAIN_LEN = 130

file_prefix = 'data/vertical_all_A{}.csv'

# read file and change the head name
def read_file(path):
    df = pd.read_csv(path)
    df.columns = HEAD
    return df


# make dataset using moving window with the step of -> window_step
def make_dataset(data, window_size):
    i = 0
    while i + window_size - 1 < len(data):
        yield data[i:i+window_size]
        i += 10 # set windows step here


def normalize(x):
    scaler = preprocessing.StandardScaler().fit(x)
    return scaler.transform(x), scaler


class WelllogDataset(torch.utils.data.Dataset):
    # scaler = None # the output scaler
    dataset_scaler = {} # save all scaler

    def __init__(self, input_dim, output_dim, train_id):  # 默认excel中前5口井训练，第6口检测
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.train_id = train_id
        self.data_all = [] # save all the well log data
        # add all well log data
        for i in range(WELL):
            # the test and train data will be the unnormalized data
            filename = file_prefix.format(i+1)
            df = read_file(filename)
            df['DEPT'] = np.arange(1, len(df)+1)
            self.data_all.append(df)
            
        # combine all well log data
        self.dataset = pd.concat(self.data_all, axis=0, ignore_index=True)
        # save scaler
        for feature in COLUMNS:
            self.dataset_scaler[feature] = preprocessing.StandardScaler().fit(self.dataset[feature].values.reshape(-1, 1))
        self.target_scaler = preprocessing.StandardScaler().fit(self.dataset[COLUMNS_TARGET].values)
        # get train dataset
        self.input_data, self.target_data = self.train_dataset()
        self.line_num = len(self.input_data)
    # reset train dataset
    def reset_dataset(self, input_dim, output_dim):
        self.input_dim, self.output_dim = input_dim, output_dim
        self.input_data, self.target_data = self.train_dataset()
        self.line_num = len(self.input_data)
    # Returen input and target as numpy array
    def train_dataset(self):
        input_data = []
        target_data = []
        for items in self.train_id:
            data = self.data_all[items-1]
            input_ = np.array(list(make_dataset(
                normalize(data[COLUMNS[:self.input_dim]].values)[0], TRAIN_LEN)))
            target_ = []
            for feature in COLUMNS[self.input_dim:self.input_dim+self.output_dim]:
                target_.append(self.dataset_scaler[feature].transform(data[feature].values.reshape(-1, 1)))
            target_ = np.concatenate(target_, axis=1)
            target_ = np.array(list(make_dataset(target_, TRAIN_LEN)))
            input_data.append(input_)
            target_data.append(target_)
        # concat all data
        return torch.from_numpy(np.concatenate(input_data)).float(), torch.from_numpy(np.concatenate(target_data)).float()

    def test_dataset(self, index):
        data = self.data_all[index-1]
        # input data
        input_ = normalize(data[COLUMNS[:-len(COLUMNS_TARGET)]].values)[0]
        # target data
        # target_ = []
        # for feature in COLUMNS[self.input_dim:self.input_dim+self.output_dim]:
        #     target_.append(self.dataset_scaler[feature].transform(data[feature].values.reshape(-1, 1)))
        # target_ = np.concatenate(target_, axis=1)
        # save target scaler for inversing
        # self.scaler = preprocessing.StandardScaler().fit(self.dataset[COLUMNS[self.input_dim:self.input_dim+self.output_dim]].values)
        target_ = self.target_scaler.transform(data[COLUMNS_TARGET].values)
        return input_, target_

    def inverse_normalize(self, x):
        # this feature is only used for inverse normalization of target
        return self.target_scaler.inverse_transform(x)

    def __getitem__(self, index):
        return self.input_data[index], self.target_data[index]
    
    def __len__(self):
        return self.line_num
