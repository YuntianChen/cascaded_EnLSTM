import torch
from collections import OrderedDict
import torch.nn as nn
from torch.autograd import Variable


CASCADING = OrderedDict()
CASCADING['model_1'] = (11, 2)
CASCADING['model_2'] = (13, 2)
CASCADING['model_3'] = (15, 2)
CASCADING['model_4'] = (17, 2)
CASCADING['model_5'] = (19, 2)
CASCADING['model_6'] = (21, 2)


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
            if x.device.type == 'gpu':
                h = h.cuda()
                c = c.cuda()
            hs = (h, c)
        output, hs_0 = self.lstm(input, hs)
        output = output.contiguous()
        output = output.view(-1, self.hidden_size)
        output = nn.functional.relu(self.fc2(output))
        output = self.fc3(output)
        return output, hs_0

        
def enn_output(models, input):
    output = []
    for model in models:
        output.append(model(input))
    return torch.stack(output)


def format_model(cascaded_model, transfer_mode):
    result = OrderedDict()
    for key in cascaded_model:
        input_size, output_size = CASCADING[key]
        torch_model = []
        model = netLSTM_withbn(input_size, output_size)
        if transfer_mode == 'ennv1':
            state_dicts = ennv1_to_torch_model(cascaded_model[key])   
        else:
            state_dicts = enn_to_torch_model(cascaded_model[key])
        """
        for state_dict in state_dicts:
            if transfer_mode == 'ennv1':
                model.load_state_dict(state_dict)
            else:
                for layers in state_dict:
                    model.state_dict()[layers] -= model.state_dict()[layers] - state_dict[layers]
            torch_model.append(model)
        result[key] = torch_model
        """
        result[key] = state_dicts
    return result
        
def enn_to_torch_model(enn_model):
    torch_model = []
    for param in enn_model.parameters.t():
        state_dict = OrderedDict()
        for i, name in enumerate(enn_model.param_list):
            head = enn_model.param_index[i][0]
            tail = enn_model.param_index[i][1]
            size = enn_model.param_size[i]
            state_dict[name] = param[head:tail].reshape(size)
        torch_model.append(state_dict)
    return torch_model
    
def ennv1_to_torch_model(ennv1_model):
    torch_model = []
    for param in ennv1_model.weights.t():
        new_state_dict = OrderedDict()
        state_dict = ennv1_model.coder.decoder(param).state_dict()
        for key in list(state_dict.keys())[:-3]:
            new_state_dict[key] = state_dict[key].cpu()
        torch_model.append(new_state_dict)
    return torch_model

transfer = {
    'enn': enn_to_torch_model,
    'ennv1': ennv1_to_torch_model
}

def run():
    for i in range(1, 6):
        for j in range(1, 15):
            experiment_id = i*1000 + j
            model_pth = 'Experiments/{}/epoch_2.pth.tar'.format(experiment_id)
            cascaded_model = torch.load(model_pth)
            if i < 4:
                cascaded_torch_model = format_model(cascaded_model, 'enn')
            else:
                cascaded_torch_model = format_model(cascaded_model, 'ennv1')
            torch.save(cascaded_torch_model,
                    'result/{}.pth.tar'.format(experiment_id))
            print('saving {}'.format(experiment_id))

if __name__ == '__main__':
    run()
        
        
        