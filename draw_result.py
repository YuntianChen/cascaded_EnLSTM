import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


label = ['HAC', 'DEN', 'BHC']

lstm = pd.read_csv('lstm_a3.csv')

experiment_ID = '089'
#plt.figure(figsize=(6, 6.5))
fig, axs = plt.subplots(6, 1)
fig.set_figheight(50)
fig.set_figwidth(15)

for i in range(6):
    target_file = 'result/e{}_target_{}.csv'.format(experiment_ID, label[i // 2])
    target = np.loadtxt(target_file, delimiter=',')
    if i % 2 == 0:
        predict_file = 'result/e{}_pred_{}.csv'.format(experiment_ID, label[i//2])
        predict = np.loadtxt(predict_file, delimiter=',')
        x = np.arange(len(target))
        std = 3 * np.array(predict.std(1))
        axs[i].plot(target, label='target', color='black', alpha=0.4)
        axs[i].errorbar(x, predict.mean(1), label='predict', yerr=std, color='red', alpha=0.7)
        axs[i].set_title(label[i//2])
        axs[i].legend()
    else:
        axs[i].plot(target, label='target', color='black', alpha=0.4)
        axs[i].plot(lstm[label[i//2]], label='predict', color='blue', alpha=0.7)
        axs[i].legend()
#plt.tight_layout()
plt.show()
