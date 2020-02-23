import torch

for i in range(1, 6):
    for j in range(1, 15):
        experiment_id = i*1000 + j
        model_pth = 'Experiments/{}/epoch_2.pth.tar'.format(experiment_id)
        


enn_model = []

def format_model(cascaded_model):
    for key in cascaded_model:
        enn_model = cascaded_model[key]
    