import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
import PIL
from configuration import config
import threading



feature_name = 'HAC'
folder = config.path
params_list = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith('{}_params'.format(feature_name))]


def get_data(li, num):
    mean = []
    std = []
    for file in li:
        data = np.array(pickle.load(open(file, 'rb')))
        mean.append(data.mean(1)[num])
        std.append(data.std(1)[num])
    return np.array(mean), np.array(std)


def draw_parameters():
    mean, std = get_data(params_list, 0)
    print("Record num: {}".format(len(mean)))
    for i in range(100):
        print("Working on No.{}".format(i))
        mean, std = get_data(params_list, i*50)
        plt.plot(mean, label='mean')
        plt.plot(mean-std, label='-std')
        plt.plot(mean+std, label='+std')
        plt.legend()
        plt.title("No.{}".format(i*50))
        plt.savefig('{}/{}_{}.png'.format(folder, feature_name, i))
        plt.close()

    img = PIL.Image.open('{}/{}_{}.png'.format(folder, feature_name, 0))
    size = img.size
    target = PIL.Image.new('RGB', (10 * size[0], 10 * size[1]))
    d = 0
    for i in range(10):
        for j in range(10):
            filename = '{}/{}_{}.png'.format(folder, feature_name, d)
            data = PIL.Image.open(filename)
            target.paste(data, (i * size[0], j*size[1]))
            d += 1
    target.save('{}/{}_target.png'.format(folder, feature_name))


if __name__ == "__main__":
    draw_parameters()
