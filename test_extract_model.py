import torch
from collections import OrderedDict
import torch.nn as nn
from torch.autograd import Variable
from extract_model import format_model
from evaluate import cascaded_output


CASCADING = OrderedDict()
CASCADING['model_1'] = (11, 2)
CASCADING['model_2'] = (13, 2)
CASCADING['model_3'] = (15, 2)
CASCADING['model_4'] = (17, 2)
CASCADING['model_5'] = (19, 2)
CASCADING['model_6'] = (21, 2)

test_enn_model = torch.load(
    'Experiments/1001/epoch_2.pth.tar')

test_ennv1_model = torch.load(
    'Experiments/5001/epoch_2.pth.tar')

def test_format_model():       
    print(format_model(test_enn_model, 'enn')['model_1'][0])
    print(format_model(test_ennv1_model, 'ennv1')['model_1'][0])

def test_evaluate():
    for i in range(1, 6):
        for j in range(1, 15):
            experiment_id = i*1000 + j
            model_pth = 'result/{}.pth.tar'.format(experiment_id)
            input = torch.rand(100, 11)
            model = torch.load(model_pth)
            print(cascaded_output(model, input).shape)
            print('{} checked'.format(experiment_id))

test_evaluate()

