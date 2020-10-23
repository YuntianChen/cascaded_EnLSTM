# -*- coding: utf-8 -*-
"""
Script for Reading Save Enn Model.
*********************************************

"""
import torch
import torch.nn as nn
from collections import OrderedDict
from torch.autograd import Variable


###############################################################################
# stor the input and output size for the cascaded models
###############################################################################
CASCADING = OrderedDict()
CASCADING['model_1'] = (11, 2)
CASCADING['model_2'] = (13, 2)
CASCADING['model_3'] = (15, 2)
CASCADING['model_4'] = (17, 2)
CASCADING['model_5'] = (19, 2)
CASCADING['model_6'] = (21, 2)
# e. g. for model_1 the input size is 11, the output size is 2.'


######################################################################
# Creating the Network
###############################################################################
class netLSTM_withbn(nn.Module):
    def __init__(self, input_size, output_size,
                hidden_size=30, layers=1):
        super(netLSTM_withbn, self).__init__()
        self.drop = 0.3
        self.layers = layers
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(input_size, self.hidden_size,
                            self.layers, batch_first=True,
                            dropout=self.drop)
        self.fc2 = nn.Linear(self.hidden_size,
                             int(self.hidden_size / 2))
        self.fc3 = nn.Linear(int(self.hidden_size / 2), 
                            output_size)
        self.bn = nn.BatchNorm1d(int(self.hidden_size / 2))

    def forward(self, input, hs=None):
        batch_size = input.size(0)
        if hs is None:
            h = Variable(torch.zeros(self.layers, batch_size,
                                     self.hidden_size))
            c = Variable(torch.zeros(self.layers, batch_size,
                                     self.hidden_size))
            if input.device.type == 'gpu':
                h = h.cuda()
                c = c.cuda()
            hs = (h, c)
        output, hs_0 = self.lstm(input, hs)
        output = output.contiguous()
        output = output.view(-1, self.hidden_size)
        output = nn.functional.relu(self.fc2(output))
        output = self.fc3(output)
        return output, hs_0


def enn(model, state_dicts, input):
    r"""
    Args:
        model: the network
        state_dicts: list of network state_dict
        input: the sequence fed to the network
    Shape:
        input: [sequence length, input dim, batch size]
        output: [ensemble size, sequence length*batch size, output dim]
    Examples:
        >>> output = enn(model, state_dicts, input)
    """
    enn_output = []
    for model_state_dict in state_dicts:
        # load saved network state
        model.load_state_dict(model_state_dict, strict=False)
        # make prediction
        with torch.no_grad():
            output, _ = model(input)
            
        enn_output.append(output)
    return torch.stack(enn_output)


def cascaded_output(cascaded_model, input):
    r"""
    Args:
        cascaded_model: the cascaded enn model (required)
        input: the sequence fed to the cascaded model (required)
    Shape:
        input: [sequence length, input dim]
        output: [ensemble size, sequence length, output dim]
    Examples:
        >>> output = cascaded_output(cascaded_model, input)
    """
    input_ = input
    # cascading
    cascaded_pred = []
    for name in cascaded_model:
        # get the cascaded model
        state_dicts = cascaded_model[name]
        # build the model
        model = netLSTM_withbn(*CASCADING[name])
        # make prediction
        pred = enn(model, state_dicts, input_.reshape(1, len(input_), -1))
        # add predict result to the input for the next cascading
        input_ = torch.cat([input_, pred.mean(0)], 1)
        
        cascaded_pred.append(pred)
    return torch.cat(cascaded_pred, 2)


# load the saved model
pth = 'result/5001.pth.tar'
cascaded_model = torch.load(pth)

# generate random input
input = torch.rand(100, 11)

# print the predict result
print(cascaded_output(cascaded_model, input))
        
    