import pandas as pd
import numpy as np
import datetime
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
from model import MLPEngine
from data import SampleGenerator
from utils import *


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--num_round', type=int, default=30)
parser.add_argument('--local_epoch', type=int, default=1)
parser.add_argument('--clients_sample_ratio', type=float, default=1.0)
parser.add_argument('--clients_sample_num', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--dataset', type=str, default='KuaiRand-Pure')
parser.add_argument('--user_feature_bin_flag', type=bool, default=False)
parser.add_argument('--video_feature_bin_flag', type=bool, default=True)
parser.add_argument('--divide_style', type=str, default='random')
parser.add_argument('--divide_ratio', type=float, default=0.)
parser.add_argument('--divide_feature_values', type=str, default='5,6,7,8')
parser.add_argument('--train_val_ratio', type=str, default='0.6,0.2')
parser.add_argument('--pretrain_model_dir', type=str)
parser.add_argument('--layers', type=str, default="32,8")
parser.add_argument('--latent_dim', type=int, default=8)
parser.add_argument('--lora_dropout', type=float, default=0.)
parser.add_argument('--lora_alpha', type=int, default=1)
parser.add_argument('--lora_user_features', type=str, default='follow_user_num_range,user_active_degree')
parser.add_argument('--rank', type=int, default=4)
parser.add_argument('--gate_num', type=int, default=2)
parser.add_argument('--gate_dim', type=str, default='8,4')
parser.add_argument('--l2_regularization', type=float, default=0.)
parser.add_argument('--use_cuda', type=bool, default=True)
parser.add_argument('--device_id', type=int, default=0)
args = parser.parse_args()

# Transform string argument into list.
config = vars(args)
config['train_val_ratio'] = [float(item) for item in config['train_val_ratio'].split(',')]
config['lora_user_features'] = [str(item) for item in config['lora_user_features'].split(',')]
if len(config['layers']) == 1:
    config['layers'] = [int(config['layers'])]
else:
    config['layers'] = [int(item) for item in config['layers'].split(',')]
if len(config['divide_feature_values']) == 1:
    config['divide_feature_values'] = [int(config['divide_feature_values'])]
else:
    config['divide_feature_values'] = [int(item) for item in config['divide_feature_values'].split(',')]
if len(config['gate_dim']) == 1:
    config['gate_dim'] = [int(config['gate_dim'])]
else:
    config['gate_dim'] = [int(item) for item in config['gate_dim'].split(',')]

# Logging.
path = 'log/'+config['dataset']+'/'
current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
logname = os.path.join(path, current_time+'.txt')
initLogging(logname)

# Load the data
dataset_dir = '../dataset/' + config['dataset'] + '/'
df, user_features, video_features = load_data(dataset_dir)
# Process the feature dataframe.
logging.info('Preprocess the user feature dataframe')
user_features.index = user_features['user_id'].values
if config['user_feature_bin_flag'] is True:
    user_feature_bin_dir = '../' + config['dataset'] + '_user_feature_bins_dict.txt'
    user_column_name_value_bins_dict = read_bin_dict(user_feature_bin_dir)
    new_user_features = wrap_preprocess_float_df(user_features, user_column_name_value_bins_dict)
    new_user_features.to_csv(dataset_dir+'bin_user_features.csv', index=False)
else:
    new_user_features = user_features
# new_user_features = pd.read_csv(dataset_dir+"bin_user_features.csv")

logging.info('Preprocess the video feature dataframe')
video_features.index = video_features['video_id'].values
if config['video_feature_bin_flag'] is True:
    video_feature_bin_dir = '../' + config['dataset'] + '_video_feature_bins_dict.txt'
    video_column_name_value_bins_dict = read_bin_dict(video_feature_bin_dir)
    new_video_features = wrap_preprocess_float_df(video_features, video_column_name_value_bins_dict)
    new_video_features.to_csv(dataset_dir+'bin_video_features.csv', index=False)
else:
    new_video_features = video_features
# new_video_features = pd.read_csv(dataset_dir+"bin_video_features.csv")

# Split private user's train/val/test data.
logging.info('Split the private user data')

if config['divide_style'] == 'random':
    train_data = pd.read_csv(dataset_dir+'data_divide/private_users_'+config['divide_style']+'_'+str(config['divide_ratio'])+'_'+str(config['train_val_ratio'])+'_train.csv')
    val_data = pd.read_csv(dataset_dir+'data_divide/private_users_'+config['divide_style']+'_'+str(config['divide_ratio'])+'_'+str(config['train_val_ratio'])+'_val.csv')
    test_data = pd.read_csv(dataset_dir+'data_divide/private_users_'+config['divide_style']+'_'+str(config['divide_ratio'])+'_'+str(config['train_val_ratio'])+'_test.csv')
else:
    train_data = pd.read_csv(dataset_dir+'data_divide/private_users_'+config['divide_style']+'_'+str(config['divide_ratio'])+'_'+str(config['divide_feature_values'])+'_'+str(config['train_val_ratio'])+'_train.csv')
    val_data = pd.read_csv(dataset_dir+'data_divide/private_users_'+config['divide_style']+'_'+str(config['divide_ratio'])+'_'+str(config['divide_feature_values'])+'_'+str(config['train_val_ratio'])+'_val.csv')
    test_data = pd.read_csv(dataset_dir+'data_divide/private_users_'+config['divide_style']+'_'+str(config['divide_ratio'])+'_'+str(config['divide_feature_values'])+'_'+str(config['train_val_ratio'])+'_test.csv')
    
# Model.
logging.info('Define the model')
selected_user_feature_file_dir = '../' + config['dataset'] + '_selected_user_feature_dict.txt'
selected_user_features = read_feature_names(selected_user_feature_file_dir)
selected_video_feature_file_dir = '../' + config['dataset'] + '_selected_video_feature_dict.txt'
selected_video_features = read_feature_names(selected_video_feature_file_dir)
engine = MLPEngine(config, new_user_features, new_video_features, selected_user_features, selected_video_features)

# DataLoader for training
sample_generator = SampleGenerator(ratings=train_data)
train_loader = sample_generator.store_all_train_data(train_data)
val_loader = sample_generator.validate_data(val_data)
test_loader = sample_generator.test_data(test_data)

val_auc_list = []
test_auc_list = []
val_we_result_list = []
test_we_result_list = []
best_val_auc = 0
final_test_round = 0
temp = 0
for round_idx in range(config['num_round']):
#     break
    logging.info('-' * 80)
    logging.info('Round {} starts !'.format(round_idx))

    logging.info('-' * 80)
    logging.info('Training phase!')
    engine.fed_train_a_round(train_loader, train_data['user_id'].unique().tolist(), round_idx, new_user_features)
    # break
    
    if (round_idx+1) % 1 == 0:
        logging.info('-' * 80)
        logging.info('Evaluating phase!')
        train_auc, train_we_result = engine.fed_evaluate(train_loader, new_user_features)
        # break
        logging.info('[Evaluating Epoch {}] AUC = {:.4f}'.format(round_idx, train_auc))
        logging.info('[Evaluating Epoch {}] We_Result = {}'.format(round_idx, str(train_we_result)))

        logging.info('-' * 80)
        logging.info('Validating phase!')
        val_auc, val_we_result = engine.fed_evaluate(val_loader, new_user_features)
        # break
        logging.info('[Validating Round {}] AUC = {:.4f}'.format(round_idx, val_auc))
        val_auc_list.append(val_auc)
        logging.info('[Validating Round {}] We_Result = {}'.format(round_idx, val_we_result))
        val_we_result_list.append(val_we_result)

        logging.info('-' * 80)
        logging.info('Testing phase!')
        test_auc, test_we_result = engine.fed_evaluate(test_loader, new_user_features)
        logging.info('[Testing Round {}] AUC = {:.4f}'.format(round_idx, test_auc))
        test_auc_list.append(test_auc)
        logging.info('[Testing Round {}] We_Result = {}'.format(round_idx, test_we_result))
        test_we_result_list.append(test_we_result)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            final_test_round = temp
            
        temp += 1

current_time = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
file_str = current_time + '-' + 'latent_dim: ' + str(config['latent_dim']) + '-' + 'lr: ' + str(config['lr']) + '-' + \
           'layers: ' + str(config['layers']) + '-' + 'num_round: ' + str(config['num_round']) + '-' + 'local_epoch: ' + \
           str(config['local_epoch'])+ '-' + 'dataset: ' + config['dataset'] + '-' + \
           'lora_dropout: ' + str(config['lora_dropout']) + '-' + 'lora_alpha: ' + str(config['lora_alpha']) + '-' + \
           'lora_user_features: ' + str(config['lora_user_features']) + '-' + 'rank: ' + str(config['rank']) + '-' + \
           'clients_sample_ratio: ' + str(config['clients_sample_ratio']) + '-' + 'divide_style: ' + \
           config['divide_style'] + '-' + 'divide_feature_values: ' + str(config['divide_feature_values']) + '-' + 'batch_size: ' + str(config['batch_size']) + '-' + 'divide_ratio: ' + \
           str(config['divide_ratio']) + '-' + 'clients_sample_num: ' + str(config['clients_sample_num']) + '-' + \
           'train_val_ratio: ' + str(config['train_val_ratio']) + '-' + 'optimizer: ' + config['optimizer'] + '-' + \
           'l2_regularization: ' + str(config['l2_regularization']) + '-' + 'gate_num: ' + str(config['gate_num']) + '-' + 'gate_dim: ' + str(config['gate_dim']) + '-' + 'auc: ' + \
           str(test_auc_list[final_test_round]) + '-' + 'we_result: ' + str(test_we_result_list[final_test_round]) + '-' + 'best_round: ' + str(final_test_round)
file_name = "sh_result/"+config['dataset']+".txt"
with open(file_name, 'a') as file:
    file.write(file_str + '\n')

logging.info('FedPA')
logging.info('latent_dim: {}, layers: {}, bz: {}, lr: {}, dataset: {},' 
             'divide_style: {}, divide_ratio: {}, train_val_ratio: {},'
             'clients_sample_ratio: {}, clients_sample_num: {}, local_epoch: {}'.
             format(config['latent_dim'], config['layers'], config['batch_size'], config['lr'],
                    config['dataset'], config['divide_style'], config['divide_ratio'], config['train_val_ratio'],
                    config['clients_sample_ratio'], config['clients_sample_num'], config['local_epoch']))

logging.info('test_auc_list: {}'.format(test_auc_list))
logging.info('test_we_result_list: {}'.format(test_we_result_list))
logging.info('Best test auc: {}, test we_result: {} at epoch {}'.format(test_auc_list[final_test_round], test_we_result_list[final_test_round], final_test_round))
