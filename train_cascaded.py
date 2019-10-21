import tqdm
import logging
import argparse

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import time
import os
import matplotlib.pyplot as plt
import pickle
import json
from enn import enn, enrml, lamuda
from net import netLSTM_withbn
from data import WelllogDataset, TEST_ID, COLUMNS_TARGET
from configuration import config
import util
from util import Record, save_var, get_file_list, list_to_csv, shrink, save_txt
from collections import OrderedDict
"""
Cascading: use a list of (input_dim, output_dim) to define the cascading scheam
    e.g.
        [(2, 1), (3, 1)]
"""

parser = argparse.ArgumentParse()
parser.add_argument("--resotre_file", default=None,
                    help='Optional, name fo the file to reload before training')

CASCADING = OrderedDict()
CASCADING_MODEL = OrderedDict()

CASCADING['model_1'] = (5, 1)
CASCADING['model_2'] = (6, 1)
CASCADING['model_3'] = (7, 1)

def enn_optimizer(model, input_, target, cascading=''):
    net_enn = model
    dstb_y = lamuda.Lamuda(target, NE, ERROR_PER)
    train_losses = Record()
    losses = Record()
    lamuda_history = Record()
    std_history = Record()
    pred_history = Record()

    initial_parameters = net_enn.initial_parameters
    initial_pred = net_enn.output(input_)
    train_losses.update(criterion(initial_pred.mean(0), target).tolist())
    losses.update(criterion(initial_pred.mean(0), target).tolist())
    std_history.update(dstb_y.std(initial_pred))
    pred_history.update(initial_pred)
    lamuda_history.update(dstb_y.lamuda(initial_pred))

    for j in range(T):
        torch.cuda.empty_cache()
        params = net_enn.get_parameter()
        dstb_y.update()
        # time_ = time.strftime('%Y%m%d_%H_%M_%S')
        delta = enrml.EnRML(pred_history.get_latest(mean=False), params, initial_parameters,
                            lamuda_history.get_latest(mean=False), dstb_y.dstb, ERROR_PER)
        params_raw = net_enn.update_parameter(delta)
        torch.cuda.empty_cache()
        pred = net_enn.output(input_)
        loss_new = criterion(pred.mean(0), target).tolist()
        bigger = train_losses.check(loss_new)
        record_while = 0
        while bigger:
            record_while += 1
            lamuda_history.update(lamuda_history.get_latest(mean=False) * GAMMA)
            if lamuda_history.get_latest(mean=False) > GAMMA ** 10:
                lamuda_history.update(lamuda_history.data[0])
                # print('abandon current iteration')
                logging.info("Abandon current batch")
                net_enn.set_parameter(params)
                loss_new = train_losses.get_latest()
                dstb_y.update()
                params_raw = params
                break
            dstb_y.update()
            net_enn.set_parameter(params)
            delta = enrml.EnRML(pred_history.get_latest(mean=False), params, initial_parameters,
                                lamuda_history.get_latest(mean=False), dstb_y.dstb, ERROR_PER)
            params_raw = net_enn.update_parameter(delta)
            torch.cuda.empty_cache()
            pred = net_enn.output(input_)
            loss_new = criterion(pred.mean(0), target).tolist()
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
            lamuda_tmp = lamuda_history.get_latest(mean=False) / GAMMA
            if lamuda_tmp < 0.005:
                lamuda_tmp = 0.005
            lamuda_history.update(lamuda_tmp)
    return net_enn, params_raw, train_losses.get_latest(mean=True), pred_history.get_latest(mean=False)



def evaluate(cascaded_model, loss_fn, evaluate_dataset):
    
    # define evaluate dataset
    val_dl = [evaluate_dataset.test_dataset(i) for i in TEST_ID]

    # Evaluate for one well log validation set
    for i in TEST_ID:
        input_, target = evaluate_dataset.test_dataset(i) 

        # convert to torch variable
        input_ = torch.from_numpy(input_)
        target = torch.from_numpy(target)
        input_, target = map(torch.autograd.Variable, (input_, target))
        
        # cascading
        cascaded_pred = []
        for model_name in cascaded_model:
            model = cascaded_model[model_name]      
            # make prediction
            pred = model.output(input_)
            input_ = np.concatenate([input_, pred.mean(0)], axis=1)
            cascaded_pred.append(pred)
        # calculate loss
        cascaded_pred = np.concatenate(cascaded_pred, axis=1)
        loss = loss_fn(cascaded_pred.mean(0), target)
        
        draw_comparing_diagram(*map(evaluate_dataset.inverser_normalize,
                                    (cascaded_pred.mean(0),
                                     cascaded_pred.std(0),
                                     target)))


def train(model, optimizer, loss_fm, dataloader, params, name):
    # runing average object for loss
    loss_avg = util.RunningAverage()
    # use tqdm for pregress bar
    with tqdm(total=len(dataloader)) as t:
        for i,  (in_feature, target) in enumerate(dataloader):
            # convert to troch Variables
            in_feature, target = map(Variable, (in_feature, target))
            # move to GPU if avriable
            if params.cuda:
                in_feature = in_feature.cuda()
                target = target.cuda()

            # compute the model output and loss
            model, weights, loss, pred = optimizer(model, in_feature, target, cascading=name)

            # save model weights
            torch.save(weights, 'weights')
            # update the average loss
            loss_avg.update(np.average(loss))

            t.set_postfix(loss='{:05.3f} avg: {:05.3f}'.format(np.average(loss), loss_avg()))
            t.update()
    return model
    # logging.info('- Train ')


def draw_comparing_diagram(pred, pred_std, target, title):
    for feature in COLUMNS_TARGET:
        x = np.arange(len(target))
        plt.figure(figsize=(60, 50))
        plt.plot(target, label='target', color='black', alpah=0.4)
        plt.errorbar(x, pred[:, 0], yerr=pred_std[:, 0], color='red', alpha=0.7)
        plt.tittle()
        plt.legend()
        y_label = feature
        plt.tight_layout()
        plt.savefig('{}.png')


def train_and_evaluate(dataset, optimizer, loss_fn, params, model_dir):
    """
    Train the model and evaluate every epoch,

    Args:
        model: (torch.nn.Module) the neural network
        dataset: (DataLoader) a torch.utils.data.DataLoader object that fetches training and testing data
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch 
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
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
                                num_workers=4, drop_last=params.drop_last)
            
            # define the model and optimizer
            net = netLSTM_withbn(params)
            model = enn.ENN(net, params.ensemble_size)

            # train the model
            model = train(model, optimizer, loss_fn, train_dl, params, name=model_name)

            # save model
            CASCADING_MODEL[model_name] = model



if __name__ == '__main__':
    
    # Load the parameters from json file
    args = parser.parse_args()
    json_path = 'params.json'
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = util.Params(json_path)

    # use GPU if available 
    params.cuda = torch.cuda.is_avaiable()

    # set the random seed for reproducible experiments
    torch.manual_seed(666)
    if params.cuda: torch.cuda.manual_seed(666)

    # set the logger
    util.set_logger(os.path.join(args.model_dir, 'train.log'))

    # create the input data
    logging.info("Loading the datasets...")
    
    # fetch dataloaders
    dataset = WelllogDataset(*CASCADING['model_1'])
    logging.info("- done.")
    
    # define optimizer
    optimizer = enn_optimizer
    
    # fetch the loss function
    loss_fn = torch.nn.MSELoss()
    