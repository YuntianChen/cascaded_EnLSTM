import pandas as pd
import os
import matplotlib.pyplot as plt
from configuration import config
import numpy as np


folder = config.path


def draw_loss():
    fig, axs = plt.subplots(len(config.columns_target), 1)
    # plt.tight_layout()
    for i, feature_name in enumerate(config.columns_target):
        filename_train = 'loss_{}.txt'.format(feature_name)
        filename_test = 'test_loss_{}.txt'.format(feature_name)
        loss_train = pd.read_csv(os.path.join(folder, filename_train), header=None).iloc[:, 1]
        loss_test = pd.read_csv(os.path.join(folder, filename_test), header=None).iloc[:, 1]
        x_train = np.arange(len(loss_train))
        x_test = np.arange(1, len(loss_test)+1)
        x_test *= int(len(loss_train)/len(loss_test))
        print(len(loss_train), len(loss_test))
        axs[i].plot(x_train, loss_train, label='train')
        axs[i].plot(x_test, loss_test, label='test')
        axs[i].set_ylabel(feature_name)
        axs[i].legend()
        axs[i].grid(axis='y', color='black', linestyle='-', linewidth=0.2)
    plt.savefig(os.path.join(folder, 'loss.png'))
    plt.show()


if __name__ == "__main__":
    draw_loss()

