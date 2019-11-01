import matplotlib.pyplot as plt
import numpy as np
import os


def draw_comparing_diagram(pred, pred_std, target, ylabel=None, title=None, save_path=None):
    x = np.arange(len(target))
    plt.figure(figsize=(18, 5))
    plt.plot(target, label='target', color='black', alpha=0.4)
    plt.errorbar(x, pred, yerr=pred_std, color='red', alpha=0.7)
    plt.title('')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, '{}.png'.format(title))) 