import logging
import argparse
import time
import os

import numpy as np
import torch
import pickle
import json

from torch.autograd import Variable
from torch.utils.data import DataLoader
from collections import OrderedDict
from tqdm import tqdm

import util
from enn import enn, enrml, lamuda
from net import netLSTM_withbn
from data import WelllogDataset, TEST_ID, COLUMNS_TARGET
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
parser.add_argument("--resotre_file", default=None,
                    help='Optional, name fo the file to reload before training')

#Store model
CASCADING_MODEL = OrderedDict()

# Define model input and output dimension
CASCADING = OrderedDict()
CASCADING['model_1'] = (5, 2)
CASCADING['model_2'] = (7, 1)
# CASCADING['model_3'] = (7, 1)


def enn_optimizer(model, input_, target, loss_fn, params, cascading=''):
    net_enn = model
    dstb_y = lamuda.Lamuda(target, params.ensemble_size, params.ERROR_PER)
    train_losses = Record()
    losses = Record()
    lamuda_history = Record()
    std_history = Record()
    pred_history = Record()

    initial_parameters = net_enn.initial_parameters
    initial_pred = net_enn.output(input_)
    train_losses.update(loss_fn(initial_pred.mean(0), target).tolist())
    losses.update(loss_fn(initial_pred.mean(0), target).tolist())
    std_history.update(dstb_y.std(initial_pred))
    pred_history.update(initial_pred)
    lamuda_history.update(dstb_y.lamuda(initial_pred))

    for _ in range(params.T):
        torch.cuda.empty_cache()
        parameters = net_enn.get_parameter()
        dstb_y.update()
        # time_ = time.strftime('%Y%m%d_%H_%M_%S')
        delta = enrml.EnRML(pred_history.get_latest(mean=False), parameters, initial_parameters,
                            lamuda_history.get_latest(mean=False), dstb_y.dstb, params.ERROR_PER)
        params_raw = net_enn.update_parameter(delta)
        torch.cuda.empty_cache()
        pred = net_enn.output(input_)
        loss_new = loss_fn(pred.mean(0), target).tolist()
        bigger = train_losses.check(loss_new)
        record_while = 0
        while bigger:
            record_while += 1
            lamuda_history.update(lamuda_history.get_latest(mean=False) * params.GAMMA)
            if lamuda_history.get_latest(mean=False) > params.GAMMA ** 10:
                lamuda_history.update(lamuda_history.data[0])
                # print('abandon current iteration')
                logging.info("Abandon current batch")
                net_enn.set_parameter(parameters)
                loss_new = train_losses.get_latest()
                dstb_y.update()
                params_raw = parameters
                break
            dstb_y.update()
            net_enn.set_parameter(parameters)
            delta = enrml.EnRML(pred_history.get_latest(mean=False), parameters, initial_parameters,
                                lamuda_history.get_latest(mean=False), dstb_y.dstb, params.ERROR_PER)
            params_raw = net_enn.update_parameter(delta)
            torch.cuda.empty_cache()
            pred = net_enn.output(input_)
            loss_new = loss_fn(pred.mean(0), target).tolist()
            # print('update losses, new loss:{}'.format(loss_new))
            bigger = train_losses.check(loss_new)
        train_losses.update(loss_new)
        # save_var(params_raw, '{}/{}_{}_params'.format(PATH, time_, cascading))
        # print("iteration:{} \t current train losses:{}".format(j, train_losses.get_latest(mean=True)))
        # save_txt('{}/loss_{}.txt'.format(PATH, cascading), time.strftime('%Y%m%d_%H_%M_%S')+','+str(train_losses.get_latest(mean=True))+',\n')
        pred_history.update(pred)
        std_history.update(dstb_y.std(pred))
        if std_history.bigger():
            lamuda_history.update(lamuda_history.get_latest(mean=False))
        else:
            lamuda_tmp = lamuda_history.get_latest(mean=False) / params.GAMMA
            if lamuda_tmp < 0.005:
                lamuda_tmp = 0.005
            lamuda_history.update(lamuda_tmp)
    return net_enn, params_raw, train_losses.get_latest(mean=True), pred_history.get_latest(mean=False)


def evaluate(cascaded_model, loss_fn, evaluate_dataset, params, drawing_result=False):
    
    # Evaluate for one well log validation set
    for i in TEST_ID:

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
            pred = model.output(input_.reshape(1, len(input_), -1))
            input_ = torch.cat([input_, pred.mean(0)], 1)
            cascaded_pred.append(pred)
            
        # calculate loss
        cascaded_pred = torch.cat(cascaded_pred, 2)
        loss = loss_fn(cascaded_pred.mean(0), target)
        
        # plot the pred and target
        if drawing_result:
            # inverse normalization
            cascaded_pred_mean, cascaded__pred_std, target = map(evaluate_dataset.inverse_normalize, 
                                                                 (cascaded_pred.mean(0),
                                                                 cascaded_pred.std(0),
                                                                 target))
            # plotting feature seperatly
            for i, feature in enumerate(COLUMNS_TARGET):
                draw_comparing_diagram(cascaded_pred_mean[:, i],
                                       cascaded__pred_std[:, i],
                                       target[:, i],
                                       ylabel=feature,
                                       title=feature,
                                       save_path=params.model_dir)


def train(model, optimizer, loss_fn, dataloader, params, name):
    
    # runing average object for loss
    loss_avg = util.RunningAverage()
    
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
            model, weights, loss, pred = optimizer(model, in_feature, target, loss_fn, params, cascading=name)

            # save model weights
            # torch.save(weights, os.path.join(params.model_dir, 'weights'))
            
            # update the average loss
            loss_avg.update(np.average(loss))

            t.set_postfix(loss='{:05.3f} avg: {:05.3f}'.format(np.average(loss), loss_avg()))
            t.update()
    
    return model


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
            model = enn.ENN(net, params.ensemble_size)

            # train the model
            model = train(model, optimizer, loss_fn, train_dl, params, name=model_name)

            # save model
            CASCADING_MODEL[model_name] = model
        
        # save checkpoint
        torch.save(CASCADING_MODEL, os.path.join(params.model_dir, 'epoch_{}.pth.tar'.format(epoch)))
        
        # Evaluate
        evaluate(CASCADING_MODEL, loss_fn, dataset, params, drawing_result=bool(epoch == params.num_epochs-1))
        

if __name__ == '__main__':
    
    # Load the parameters from json file
    args = parser.parse_args()
    json_path = 'params.json'
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = util.Params(json_path)

    # use GPU if available 
    params.cuda = torch.cuda.is_available()

    # set GPU
    if params.cuda:
        torch.cuda.set_device(params.device_id)


    # set the random seed for reproducible experiments
    torch.manual_seed(666)
    if params.cuda: torch.cuda.manual_seed(666)

    # create a new folder for training
    if not os.path.exists(params.model_dir):
        os.makedirs(params.model_dir)
    
    # write parameters in json file
    params.save(os.path.join(params.model_dir, 'params.json'))

    # set the logger
    util.set_logger(os.path.join(params.model_dir, 'train.log'))

    # create the input data
    logging.info("Loading the datasets...")
    
    # fetch dataloaders
    dataset = WelllogDataset(*CASCADING['model_1'])
    logging.info("- done.")
    
    # define optimizer
    optimizer = enn_optimizer
    
    # fetch the loss function
    loss_fn = torch.nn.MSELoss()

    # Train the and evaluate the model
    train_and_evaluate(dataset, optimizer, loss_fn, params)

    # uncomment to evaluate the model
    # CASCADING_MODEL = torch.load(os.path.join(params.model_dir, 'epoch_{}.pth.tar'.format(params.num_epochs-1)))
    # evaluate(CASCADING_MODEL, loss_fn, dataset, params, drawing_result=True)
