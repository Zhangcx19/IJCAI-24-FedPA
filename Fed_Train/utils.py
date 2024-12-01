# -*- coding: utf-8 -*-

import pandas as pd
import random
import copy
from numpy import linspace
import numpy as np
import matplotlib.pyplot as plt
import torch
import logging



# Checkpoints
def save_checkpoint(model, model_dir):
    torch.save(model.state_dict(), model_dir)

def resume_checkpoint(model, model_dir, device_id):
    state_dict = torch.load(model_dir,
                            map_location=lambda storage, loc: storage.cuda(device=device_id))  # ensure all storage are on gpu
    model.load_state_dict(state_dict)

# Hyper params
def use_cuda(enabled, device_id=0):
    if enabled:
        assert torch.cuda.is_available(), 'CUDA is not available'
        torch.cuda.set_device(device_id)
        
def use_optimizer(network, params):
    if params['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(network.parameters(),
                                    lr=params['sgd_lr'],
                                    momentum=params['sgd_momentum'],
                                    weight_decay=params['l2_regularization'])
    elif params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), 
                                     lr=params['lr'],
                                     weight_decay=params['l2_regularization'])
    elif params['optimizer'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(network.parameters(),
                                        lr=params['rmsprop_lr'],
                                        alpha=params['rmsprop_alpha'],
                                        momentum=params['rmsprop_momentum'])
    return optimizer

def initLogging(logFilename):
    """Init for logging
    """
    logging.basicConfig(
                    level    = logging.DEBUG,
                    format='%(asctime)s-%(levelname)s-%(message)s',
                    datefmt  = '%y-%m-%d %H:%M',
                    filename = logFilename,
                    filemode = 'w');
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

"""load data from the processed dataset"""
def load_data(data_dir):
    df = pd.read_csv(data_dir + "df.csv")
    user_features = pd.read_csv(data_dir + "new_user_features.csv")
    video_features = pd.read_csv(data_dir + "new_video_features.csv")
    
    return df, user_features, video_features

def read_bin_dict(file_dir):
    with open(file_dir, 'r') as f:
        content = f.readlines()
    content = [x.strip('\n') for x in content] 
    content_dict = {}
    for line in content:
        key, value = line.split(':')
        content_dict[key] = [float(item) for item in value.split(',')]
        content_dict[key][-1] = int(content_dict[key][-1])
        
    return content_dict

"""modify the float column value of dataframe by reindexing the value with bin groups"""
def preprocess_float_df(df, column_name, column_bins):
    value_bins = linspace(column_bins[0], column_bins[1], column_bins[2]).tolist()
    df[column_name] = df[column_name].apply(np.log1p)
    value_bins.insert(0, df[column_name].min()-1)
    value_bins.append(df[column_name].max()+1)
    labels = range(column_bins[2]+1)
    df[column_name] = pd.cut(x = df[column_name], 
                      bins = value_bins, 
                      labels = labels, 
                      include_lowest = True)
    
    return df

""""wrap the dataframe preprocess function for float feature values"""
def wrap_preprocess_float_df(df, column_name_value_bins_dict):
    new_df = copy.deepcopy(df)
    for column_name in column_name_value_bins_dict.keys():
        column_bins = column_name_value_bins_dict[column_name]
        new_df = preprocess_float_df(new_df, column_name, column_bins)
    
    return new_df

"""read the feature names from txt file"""
def read_feature_names(file_dir):
    feature_names = []
    with open(file_dir, 'r') as f:
        for line in f:
            feature_names.append(line.strip('\n'))
            
    return feature_names
