import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

experiment_num = 15
epoch_num = 5

# 设置读取的文件夹
loss_file_fmt = 'test_loss_{}.txt'
feature_list = ['HAC', 'BHC', 'DEN']
# Making the experiments list
experiment_fmt = "{:0>3d}"
experiment_list = [51, 52, 32]
for i in list(range(53, 56)):
    experiment_list.append(i)
for i in list(range(57, 141)):
    experiment_list.append(i)
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

# 根据epoch对数据进行重组，并且求experiment_num次实验结果的中位数
data_epoch = []
data_epoch_median = []
for m in range(epoch_num):
    epoch = np.zeros((6, 3, experiment_num))
    for i in range(experiment_num):
        for j in range(6):
            for k in range(3):
                epoch[j][k][i] = all_data[i][j][k][m]
    epoch_median = np.zeros((6, 3, 1))
    for i in range(6):
        for j in range(3):
            epoch_median[i][j][0] = np.median(epoch[i][j])
    data_epoch.append(epoch)
    data_epoch_median.append(epoch_median)
print(np.array(data_epoch).shape)
#print(data_epoch[0])
#print(data_epoch[1])
#print(data_epoch_median[1])

# 计算多次实验的中位数
data_epoch_median_array = np.zeros((epoch_num, 3*6))
for i in range(epoch_num):
    data_epoch_median_array[i] = data_epoch_median[i].reshape(3*6)
#print(data_epoch_median_array.transpose())
#print(data_epoch_median_array.transpose().shape)

baseline = (0.33, 0.24, 0.75,
            0.94, 1.02, 1.34,
            0.32, 0.24, 0.62,
            0.45, 0.45, 0.75,
            0.44, 0.36, 0.89,
            0.36, 0.29, 1.19)
baseline = np.array(baseline)
baseline = baseline.reshape((3*6, 1))
baseline_ = baseline.reshape(6, 3)  # baseline_ for drawing
#print(baseline.shape)
compare_median = np.hstack((baseline, data_epoch_median_array.transpose()))
print(compare_median)
compare_median = pd.SparseDataFrame(compare_median)
compare_median.to_csv('compare_median')



# 提取绘制箱型图所需数据，生成的data_box_list的3个元素对应HAC，BHC，DEN。每个元素为experiment_num * 6维
data = data_epoch[4]
data_median = data_epoch_median[0]
data_median_mean = np.mean(data_median, axis=0)
print(data.shape)
print(data_median.shape)
print(data_median_mean)
data_box_list = []
for k in range(3):
    data_box = np.zeros((15, 6))
    for i in range(6):
        for j in range(15):
            data_box[j][i] = data[i][k][j]
    data_box_list.append(data_box)

marker_style = dict(color='red', marker='*', s=50)

fig, axes = plt.subplots(3, 1, figsize = (10,15))
df_0 = pd.DataFrame(data_box_list[0], columns = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6'])
df_0.plot.box(grid = True,
              ax = axes[0],
              showfliers=False,
              showmeans=False)
#plt.text(3.3,2.6,'HAC',fontsize=15)
axes[0].set_title('HAC')
axes[0].scatter(list(range(1, 7)), baseline_[:, 0], **marker_style)
df_1 = pd.DataFrame(data_box_list[1], columns = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6'])
df_1.plot.box(grid = True,
              ax = axes[1],
              showfliers=False,
              showmeans=False)
#plt.text(3.3,1.82,'BHC',fontsize=15)
axes[1].set_title('BHC')
axes[1].scatter(list(range(1, 7)), baseline_[:, 1], **marker_style)

df_2 = pd.DataFrame(data_box_list[2], columns = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6'])
df_2.plot.box(grid = True,
              ax = axes[2],
              showfliers=False,
              showmeans=False)
#plt.text(3.3,1.045,'DEN',fontsize=15)
axes[2].set_title('DEN')
axes[2].scatter(list(range(1, 7)), baseline_[:, 2], **marker_style)

#fig.suptitle('Figure')
plt.show()