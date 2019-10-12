import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
f, axes = plt.subplots(3, 2)


def draw_scatter(experiment_ID, feature_name, axes):
    pred_filename = 'result/e{}_pred_{}.csv'.format(experiment_ID, feature_name)
    target_filename = 'result/e{}_target_{}.csv'.format(experiment_ID, feature_name)
    pred_all = np.loadtxt(pred_filename, delimiter=',')
    pred = pred_all.mean(1)
    target = np.loadtxt(target_filename, delimiter=',')
    axes.scatter(pred, target, alpha=1, s=10)
    l = np.linspace(pred.min(), pred.max(), 100)
    axes.plot(l, l, color='red')
    axes.set_title(feature_name)


def dist(experiment_ID, feature_name, axes):
    target_filename = 'result/e{}_target_{}.csv'.format(experiment_ID, feature_name)
    target = np.loadtxt(target_filename, delimiter=',')
    sns.distplot(target, ax=axes)
    axes.set_title(feature_name + " distribution")


features = ['HAC', 'DEN', 'BHC']
for i in range(3):
    dist('051', features[i], axes[i, 0])
    draw_scatter('032', features[i], axes[i, 1])

plt.show()
