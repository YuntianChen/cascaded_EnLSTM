import logging
import argparse
import time
import os

import numpy as np
import pandas as pd
import torch
import pickle
import json

from torch.autograd import Variable
from torch.utils.data import DataLoader
from collections import OrderedDict
from tqdm import tqdm

import util
from enn import enn, enrml, lamuda
from enn_v1.ENN_backward import ENN_backward
from enn_v1.ENN_forward import ENN_forward
from net import netLSTM_withbn
from data import WelllogDataset, COLUMNS_TARGET, WELL
from util import Record, save_var, get_file_list, list_to_csv, shrink, save_txt
from plotting import draw_comparing_diagram

"""
Cascading: use an OrderedDict of (input_dim, output_dim) to define the cascading scheam
    e.g.
        {
            'model_1': (2, 1),
            'model_2': (3, 1)
        }
"""

parser = argparse.ArgumentParser()
parser.add_argument("--evaluate", default=False,
                    help='Set True to evaluate')
parser.add_argument("--resotre_file", default=None,
                    help='Optional, name fo the file to reload before training')

#Store model
CASCADING_MODEL = OrderedDict()

# Define model input and output dimension
CASCADING = OrderedDict()
CASCADING['model_1'] = (11, 2)
CASCADING['model_2'] = (13, 2)
CASCADING['model_3'] = (15, 2)
CASCADING['model_4'] = (17, 2)
CASCADING['model_5'] = (19, 2)
CASCADING['model_6'] = (21, 2)
# CASCADING['model_3'] = (7, 1)

# Save losses
LOSSES = OrderedDict((model_name, []) for model_name in CASCADING)
TEST_LOSSES = OrderedDict((feature, []) for feature in COLUMNS_TARGET)


def evaluate(cascaded_model, loss_fn, evaluate_dataset, params, drawing_result=False):
    
    # Evaluate for one well log validation set
    for i in params.TEST_ID:

        # define evaluate dataset
        input_, target = evaluate_dataset.test_dataset(i) 

        # convert to torch variable
        input_ = torch.from_numpy(input_).float()
        target = torch.from_numpy(target).float()
        input_, target = map(torch.autograd.Variable, (input_, target))
        
        # move to GPU if avriable
        if params.cuda:
            input_ = input_.cuda()
            target = target.cuda()

        # cascading
        cascaded_pred = []
        for model_name in cascaded_model:
            model = cascaded_model[model_name]
                 
            # make prediction
            pred = model(input_.reshape(1, len(input_), -1))
            input_ = torch.cat([input_, pred.mean(0)], 1)
            cascaded_pred.append(pred)
            
        # calculate loss seperately
        cascaded_pred = torch.cat(cascaded_pred, 2)
        for i, feature in enumerate(COLUMNS_TARGET):
            loss = loss_fn(cascaded_pred.mean(0)[:, i], target[:, i])
            TEST_LOSSES[feature].append(loss.item())
                
        # plot the pred and target
        if drawing_result:
            
            # save prediction and target
            torch.save(cascaded_pred, os.path.join(params.model_dir, 'pred.pth.tar'))
            torch.save(evaluate_dataset.inverse_normalize(cascaded_pred), os.path.join(params.model_dir, 'pred_unnormalized.pth.tar'))
            np.savetxt(os.path.join(params.model_dir, 'e{}_pred.csv'.format(params.experiments_id)),
                    np.array(cascaded_pred.mean(0)), delimiter=',')
            np.savetxt(os.path.join(params.model_dir, 'e{}_pred_unnormalized.csv'.format(params.experiments_id)),
                    np.array(evaluate_dataset.inverse_normalize(cascaded_pred.mean(0))), delimiter=',')
            np.savetxt(os.path.join(params.model_dir, 'e{}_target.csv'.format(params.experiments_id)),
                    np.array(target), delimiter=',')
            np.savetxt(os.path.join(params.model_dir, 'e{}_target_unnormalized.csv'.format(params.experiments_id)),
                    np.array(evaluate_dataset.inverse_normalize(target)), delimiter=',')
            
            # inverse normalization
            cascaded_pred_std = evaluate_dataset.inverse_normalize(cascaded_pred).std(0)
            cascaded_pred_mean, target = map(evaluate_dataset.inverse_normalize, 
                                             (cascaded_pred.mean(0), target))
            # plotting feature seperatly
            for i, feature in enumerate(COLUMNS_TARGET):
                draw_comparing_diagram(cascaded_pred_mean[:, i],
                                       cascaded_pred_std[:, i],
                                       target[:, i],
                                       ylabel=feature,
                                       title=feature,
                                       save_path=params.model_dir)


def train(model, optimizer, loss_fn, dataloader, params, name):
    
    # runing average object for loss
    loss_avg = util.RunningAverage()
    optimizer = ENN_backward(model, params.ERROR_PER, params.T, loss_fn)
    
    # use tqdm for pregress bar
    with tqdm(total=len(dataloader)) as t:
        for i,  (in_feature, target) in enumerate(dataloader):
            
            # convert to troch Variables
            in_feature, target = map(Variable, (in_feature, target))

            # flattern target
            target = target.reshape(-1, params.output_dim)
            
            # move to GPU if avriable
            if params.cuda:
                in_feature = in_feature.cuda()
                target = target.cuda()

            # compute the model output and loss
            loss = optimizer.fit(in_feature, target)

            # save model weights
            # torch.save(weights, os.path.join(params.model_dir, 'weights'))
            
            # update the average loss
            loss_avg.update(np.average(loss))
            LOSSES[name].append(np.average(loss))

            t.set_postfix(loss='{:05.3f} avg: {:05.3f}'.format(np.average(loss), loss_avg()))
            t.update()
    
    return optimizer.model


def train_and_evaluate(dataset, optimizer, loss_fn, params):
    """
    Train the model and evaluate every epoch,

    Args:
        model: (torch.nn.Module) the neural network
        dataset: (DataLoader) a torch.utils.data.DataLoader object that fetches training and testing data
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch 
        params: (Params) hyperparameters
        restore_file: (string) optional - name of file to resotre from (without its extension .pth.tar)
    """
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    for epoch in range(params.num_epochs): 
        
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch+1, params.num_epochs))  
             
        # Cascading
        for i, model_name in enumerate(CASCADING):
            
            # reading model input and output dimension
            input_dim, output_dim = CASCADING[model_name]
            params.input_dim, params.output_dim = input_dim, output_dim
            logging.info("CASCADING: {} {} -> {}".format(model_name, input_dim, output_dim))
            
            # reset dataset to fit the train data dimension
            dataset.reset_dataset(input_dim, output_dim)
           
            # define train dataloader
            train_dl = DataLoader(dataset, batch_size=params.batch_size, shuffle=True,
                                num_workers=4, drop_last=True)

            # define the model
            net = netLSTM_withbn(params)

            # move model to CUDA
            with torch.no_grad():
                net = net.cuda()

            # define ENN model
            if epoch == 0:
                model = ENN_forward(net, ensemble_size=params.ensemble_size)
            else:
                model = CASCADING_MODEL[model_name]

            # train the model
            model = train(model, optimizer, loss_fn, train_dl, params, name=model_name)

            # save model
            CASCADING_MODEL[model_name] = model

        # save checkpoint
        torch.save(CASCADING_MODEL, os.path.join(params.model_dir, 'epoch_{}.pth.tar'.format(epoch)))
        
        # Evaluate
        evaluate(CASCADING_MODEL, loss_fn, dataset, params, drawing_result=bool(epoch == params.num_epochs-1))
        
        # save losses
        LOSSES_DF = pd.DataFrame.from_dict(LOSSES)
        LOSSES_DF.to_csv(os.path.join(params.model_dir, 'losses.csv'))
        TEST_LOSSES_DF = pd.DataFrame.from_dict(TEST_LOSSES)
        TEST_LOSSES_DF['mean'] = TEST_LOSSES_DF.mean(1)
        TEST_LOSSES_DF.to_csv(os.path.join(params.model_dir, 'evalute_losses.csv'))
        

if __name__ == '__main__':
    
    # os.chdir('E:\\CYQ\zj_well-log-cascaded')
    
    # Load the parameters from json file
    args = parser.parse_args()
    json_path = 'params.json'
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = util.Params(json_path)

    # Defind experiment folder
    params.model_dir = 'Experiments/{}'.format(params.experiments_id)

    # use GPU if available 
    params.cuda = torch.cuda.is_available()

    # set GPU
    if params.cuda:
        device_id = int(params.experiments_id) % 2
        torch.cuda.set_device(device_id)

    # set the random seed for reproducible experiments
    torch.manual_seed(params.random_seed)
    if params.cuda: torch.cuda.manual_seed(params.random_seed)

    # create a new folder for training
    if not os.path.exists(params.model_dir):
        os.makedirs(params.model_dir)
    
    # write parameters in json file
    params.save(os.path.join(params.model_dir, 'params.json'))

    # set the logger
    util.set_logger(os.path.join(params.model_dir, 'train.log'))

    # create the input data
    logging.info("Experiment {}".format(params.experiments_id))
    logging.info("Loading the datasets...")
    
    # fetch dataloaders
    train_id = [i for i in range(1, WELL+1) if i not in params.TEST_ID]
    dataset = WelllogDataset(*CASCADING['model_1'], train_id)
    logging.info("- done.")
    
    # define optimizer
    optimizer = ENN_backward
    
    # fetch the loss function
    loss_fn = torch.nn.MSELoss()

    if not args.evaluate:
        # Train the and evaluate the model
        train_and_evaluate(dataset, optimizer, loss_fn, params)
    else:
        # evaluate the model
        CASCADING_MODEL = torch.load(os.path.join(params.model_dir, 'epoch_{}.pth.tar'.format(params.num_epochs-1)))
        evaluate(CASCADING_MODEL, loss_fn, dataset, params, drawing_result=True)
