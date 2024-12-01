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

"""divide the users into two sets, i.e., open user set and private user set, according to specific divide style"""
def divide_users(df, divide_style, divide_ratio, divide_feature_values):
    """divide users randomly according to divide ratio"""
    if divide_style == "random":
        user_id_unique = list(df["user_id"].unique())
        open_user_num = int(len(user_id_unique) * divide_ratio)
        open_users = random.sample(user_id_unique, open_user_num)
        private_users = list(set(user_id_unique).difference(set(open_users)))
    else:
        user_id_unique = list(df["user_id"].unique())
        feature_user_ids = df.loc[df[divide_style].isin(divide_feature_values), "user_id"].tolist()
        non_feature_user_ids = list(set(user_id_unique).difference(set(feature_user_ids)))
        open_user_num_in_feature = int(len(feature_user_ids) * 0.8)
        open_users = random.sample(feature_user_ids, open_user_num_in_feature)
        open_user_num_in_non_feature = int(len(non_feature_user_ids) * 0.2)
        open_users.extend(random.sample(non_feature_user_ids, open_user_num_in_non_feature))
        private_users = list(set(user_id_unique).difference(set(open_users)))
    
    return open_users, private_users

"""split the private user data into train/val/test sets"""
def split_data(private_user_df, train_val_ratio):
    private_user_df["rank_latest"] = private_user_df.groupby(["user_id"])["timestamp"].rank(method="first", ascending=False)
    train_df = pd.DataFrame(columns=["user_id", "video_id", "timestamp", "is_click"])
    val_df = pd.DataFrame(columns=["user_id", "video_id", "timestamp", "is_click"])
    test_df = pd.DataFrame(columns=["user_id", "video_id", "timestamp", "is_click"])
    for user in private_user_df["user_id"].unique():
        user_df = private_user_df[private_user_df["user_id"]==user]
        test_num = int(len(user_df) * (1-train_val_ratio[0]-train_val_ratio[1]))
        val_num = int(len(user_df) * train_val_ratio[1])
        user_test = user_df[user_df["rank_latest"] <= test_num]
        user_val = user_df[(user_df["rank_latest"] > test_num) & (user_df["rank_latest"] <= test_num+val_num)]
        user_train = user_df[user_df["rank_latest"] > test_num+val_num]
        
        train_df = pd.concat([train_df,user_train], axis=0, ignore_index=True)
        val_df = pd.concat([val_df,user_val], axis=0, ignore_index=True)
        test_df = pd.concat([test_df,user_test], axis=0, ignore_index=True)
    
    train_df.sort_values(by=["user_id", "timestamp"], ascending=[True, True], inplace=True, ignore_index=True)
    val_df.sort_values(by=["user_id", "timestamp"], ascending=[True, True], inplace=True, ignore_index=True)
    test_df.sort_values(by=["user_id", "timestamp"], ascending=[True, True], inplace=True, ignore_index=True)
    assert train_df["user_id"].nunique() == test_df["user_id"].nunique() == val_df["user_id"].nunique()
    assert len(train_df) + len(test_df) + len(val_df) == len(private_user_df)
    
    return train_df, val_df, test_df

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