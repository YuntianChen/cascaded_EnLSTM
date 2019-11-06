import torch
import numpy as np
from numpy.random import seed
from collections import OrderedDict


DEFAULT_DTYPE = torch.float32


class ENN_forward:

    def __init__(self, net, ensemble_size=10):
        self.net = net
        self.ensemble_size = ensemble_size
        self.coder = WEIGHT_CODER(net)
        self.parameter_size = self.coder.weights_count()
        self._reset_weights()

    def __call__(self, input):
        return self.forward(input)

    def forward(self, input):
        enn_output = []
        for weights in self._w.t():
            net = self.coder.decoder(weights)
            # output, *_ = net(input)
            with torch.no_grad():
                output, _ = net(input)
            enn_output.append(output)
        enn_output = torch.stack(enn_output)
        return enn_output

    def _reset_weights(self):
        self._w = torch.randn(self.parameter_size, self.ensemble_size,
                              requires_grad=False, dtype=DEFAULT_DTYPE)

    def load_weights(self, new_weights):
        """
        new_weights: dtype: float32 or float16
        """
        self._w = new_weights

    @property
    def weights(self):
        return self._w
    
    @property
    def weight_size(self):
        return self.ensemble_size, self.parameter_size


class WEIGHT_CODER(object):

    def __init__(self, net):
        self.net = net
        self._state_dict = self.net.state_dict()
        self.layers = list(self._state_dict)
        self.size = OrderedDict()
        self._count = OrderedDict()
        for layer in self.layers:
            shape = self._state_dict[layer].shape
            self.size[layer] = shape
            self._count[layer] = np.prod(shape)
        
    def encoder(self):
        """
        Flattern the selected weights
        """
        pass

    def weights_count(self):
        sum_weight = 0
        for layer in self.layers:
            sum_weight += self._count[layer]
        return int(sum_weight)

    def decoder(self, weights):
        """
        Reshpae the 1-D array
        match the shape the model weight
        """
        new_state_dict = OrderedDict()
        weights_pointer = 0
        for layer in self.layers:
            count = int(self._count[layer])
            new_weight = weights[weights_pointer: weights_pointer+count]
            new_weight = new_weight.reshape(self.size[layer])
            weights_pointer += count
            new_state_dict[layer] = new_weight
        self.net.load_state_dict(new_state_dict, strict=False)
        return self.net


class ProgressMeter(object):

    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
