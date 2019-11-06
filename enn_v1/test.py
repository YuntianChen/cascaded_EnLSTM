import pandas as pd 
import unittest
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import torch
import torch.nn as nn
from collections import OrderedDict
from ENN_forward import ENN_forward, WEIGHT_CODER
from ENN_backward import ENN_backward
from net import netLSTM_withbn


def plot_decision_regions(ax, X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    ax.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    ax.set_xlim(xx1.min(), xx1.max())
    ax.set_ylim(xx2.min(), xx2.max())
    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        ax.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)


class TrainingTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv('https://archive.ics.uci.edu/ml/m'
            'achine-learning-databases/iris/iris.data', header=None)
        cls.size = 100
        cls.y = cls.df.iloc[:cls.size, 4].values
        cls.y = np.where(cls.y == 'Iris-setosa', -1, 1)
        cls.X = cls.df.iloc[:cls.size, [0, 2]].values
    
    @classmethod
    def tearDownClass(cls):
        plt.show()

    def test_reading(self):
        print(self.df.tail(10))

    def test_data(self):
        plt.figure()
        half_size = int(self.size * 0.5)
        plt.scatter(self.X[:half_size, 0], self.X[:half_size, 1], 
                    color='red', marker='o', label='setosa')
        plt.scatter(self.X[-half_size:, 0], self.X[-half_size:, 1], 
                    color='blue', marker='x', label='versicolor')
        plt.xlabel('petal length')
        plt.ylabel('sepal length')
        plt.legend(loc='upper left')
        plt.title('test_data')

    def test_enn(self):
        pass

    
    def test_enn_classifier(self):
        pass

class TorchTest(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.net = netLSTM_withbn(2, 2, 10, 1, 0)
        print(cls.net.state_dict())

    def test_set_weights(self):
        layers = list(self.net.state_dict())
        
        weights_name = layers[0]
        size = self.net.state_dict()[weights_name].shape
        new_weights = torch.randn(size)
        new_state_dict = OrderedDict({weights_name: new_weights})
        self.net.load_state_dict(new_state_dict, strict=False)
        print(self.net.state_dict())


class WEIGHT_CODER_Test(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.net = netLSTM_withbn(2, 2, 10, 1, 0)
        print(cls.net.state_dict())

    def test_all(self):
        coder = WEIGHT_CODER(self.net)
        parameters = torch.randn(coder.weights_count())
        net = coder.decoder(parameters)
        print(net.state_dict())

class ENN_Test(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.net = netLSTM_withbn(2, 2, 10, 2, 0.3)
        cls.net = cls.net.cuda()
        cls.enn_net = ENN_forward(cls.net, ensemble_size=10) 
        cls.input_ = torch.randn(2, 2, 2).cuda()
        cls.target = torch.randn(2, 2, 2).cuda()
        cls.target = cls.target.reshape(-1, 2)

    def test_all(self):
        output = self.enn_net(self.input_)
        self.assertEqual(output.shape, torch.Size([10, 4, 2]))

    def test_fit(self):
        initial_weights = self.enn_net.weights
        output = self.enn_net(self.input_)
        loss_fn = torch.nn.MSELoss()
        # print(self.enn_net.weights)
        optimizer = ENN_backward(self.enn_net, 0.003, 2, loss_fn)
        optimizer.fit(self.input_, self.target)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(ENN_Test('test_all'))
    suite.addTest(ENN_Test('test_fit'))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
