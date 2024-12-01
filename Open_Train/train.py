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
parser.add_argument('--num_epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--evaluation_bz', type=int, default=1000000)
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--dataset', type=str, default='KuaiRand-Pure')
parser.add_argument('--user_feature_bin_flag', type=bool, default=False)
parser.add_argument('--video_feature_bin_flag', type=bool, default=True)
parser.add_argument('--divide_button', type=bool, default=True)
parser.add_argument('--divide_style', type=str, default='user_active_degree')
parser.add_argument('--divide_ratio', type=float, default=0.)
parser.add_argument('--divide_feature_values', type=str, default='5,6,7,8')
parser.add_argument('--split_button', type=bool, default=False)
parser.add_argument('--train_val_ratio', type=str, default='0.6,0.2')
parser.add_argument('--layers', type=str, default="32,8")
parser.add_argument('--latent_dim', type=int, default=8)
parser.add_argument('--l2_regularization', type=float, default=0.)
parser.add_argument('--use_cuda', type=bool, default=True)
parser.add_argument('--device_id', type=int, default=0)
args = parser.parse_args()

# Transform string argument into list.
config = vars(args)
config['train_val_ratio'] = [float(item) for item in config['train_val_ratio'].split(',')]
if len(config['divide_feature_values']) == 1:
    config['divide_feature_values'] = [int(config['divide_feature_values'])]
else:
    config['divide_feature_values'] = [int(item) for item in config['divide_feature_values'].split(',')]
if len(config['layers']) == 0:
    config['layers'] = []
elif len(config['layers']) == 1:
    config['layers'] = [int(config['layers'])]
else:
    config['layers'] = [int(item) for item in config['layers'].split(',')]

# Logging.
path = 'log/'+config['dataset']+'/'
current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
logname = os.path.join(path, current_time+'.txt')
initLogging(logname)

# Load the data
dataset_dir = '../dataset/' + config['dataset'] + '/processed_data/'
df, user_features, video_features = load_data(dataset_dir)
# Process the feature dataframe.
logging.info('Preprocess the user feature dataframe')
user_features.index = user_features['user_id'].values
if config['user_feature_bin_flag'] is True:
    user_feature_bin_dir = '../' + config['dataset'] + '_user_feature_bins_dict.txt'
    user_column_name_value_bins_dict = read_bin_dict(user_feature_bin_dir)
    new_user_features = wrap_preprocess_float_df(user_features, user_column_name_value_bins_dict)
#     new_user_features.to_csv(dataset_dir+'bin_user_features.csv', index=False)
else:
    new_user_features = user_features
# new_user_features = pd.read_csv(dataset_dir+"bin_user_features.csv")

logging.info('Preprocess the video feature dataframe')
video_features.index = video_features['video_id'].values
if config['video_feature_bin_flag'] is True:
    video_feature_bin_dir = '../' + config['dataset'] + '_video_feature_bins_dict.txt'
    video_column_name_value_bins_dict = read_bin_dict(video_feature_bin_dir)
    new_video_features = wrap_preprocess_float_df(video_features, video_column_name_value_bins_dict)
#     new_video_features.to_csv(dataset_dir+'bin_video_features.csv', index=False)
else:
    new_video_features = video_features
# new_video_features = pd.read_csv(dataset_dir+"bin_video_features.csv")

# Load open user's data.
logging.info('Load open user data')
if config['divide_button'] is True:
    open_users, private_users = divide_users(user_features, config['divide_style'], config['divide_ratio'], config['divide_feature_values'])
    np.save(dataset_dir+'data_divide/open_users_'+config['divide_style']+'_'+str(config['divide_ratio'])+'_'+str(config['divide_feature_values'])+'.npy', open_users)
    np.save(dataset_dir+'data_divide/private_users_'+config['divide_style']+'_'+str(config['divide_ratio'])+'_'+str(config['divide_feature_values'])+'.npy', private_users)
else:
    open_users = np.load(dataset_dir+'data_divide/open_users_'+config['divide_style']+'_'+str(config['divide_ratio'])+'_'+str(config['divide_feature_values'])+'.npy')  
open_user_df = df[df["user_id"].isin(open_users)]
open_user_df.sort_values(by=["user_id", "timestamp"], ascending=[True, True], inplace=True, ignore_index=True)
logging.info('Number of open users is {}'.format(len(open_users)))

# Split private user's train/val/test data.
logging.info('Split the private user data')
if config['split_button'] is True:
    private_users = np.load(dataset_dir+'data_divide/private_users_'+config['divide_style']+'_'+str(config['divide_ratio'])+'_'+str(config['divide_feature_values'])+'.npy')
    private_user_df = df[df["user_id"].isin(private_users)]
    private_user_df.sort_values(by=["user_id", "timestamp"], ascending=[True, True], inplace=True, ignore_index=True)
    train_data, val_data, test_data = split_data(private_user_df, config['train_val_ratio'])
    train_data.to_csv(dataset_dir+'data_divide/private_users_'+config['divide_style']+'_'+str(config['divide_ratio'])+'_'+str(config['divide_feature_values'])+'_'+str(config['train_val_ratio'])+'_train.csv', index=False)
    val_data.to_csv(dataset_dir+'data_divide/private_users_'+config['divide_style']+'_'+str(config['divide_ratio'])+'_'+str(config['divide_feature_values'])+'_'+str(config['train_val_ratio'])+'_val.csv', index=False)
    test_data.to_csv(dataset_dir+'data_divide/private_users_'+config['divide_style']+'_'+str(config['divide_ratio'])+'_'+str(config['divide_feature_values'])+'_'+str(config['train_val_ratio'])+'_test.csv', index=False)
else:
#     pass
#     train_data = pd.read_csv(dataset_dir+'data_divide/private_users_'+config['divide_style']+'_'+str(config['divide_ratio'])+'_'+str(config['train_val_ratio'])+'_train.csv')
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
sample_generator = SampleGenerator(ratings=open_user_df)
train_loader = sample_generator.instance_a_train_loader(open_user_df, config['batch_size'])
eval_loader = sample_generator.evaluate_data(open_user_df, config['evaluation_bz'])
val_loader = sample_generator.validate_data(val_data, config['evaluation_bz'])
test_loader = sample_generator.test_data(test_data, config['evaluation_bz'])

val_auc_list = []
test_auc_list = []
val_logloss_list = []
test_logloss_list = []
val_mi_result_list = []
test_mi_result_list = []
val_ma_result_list = []
test_ma_result_list = []
val_we_result_list = []
test_we_result_list = []
best_val_auc = 0
final_test_epoch = 0
best_train_auc = 0
for epoch in range(config['num_epoch']):
#     break
    logging.info('-' * 80)
    logging.info('Epoch {} starts !'.format(epoch))

    logging.info('-' * 80)
    logging.info('Training phase!')
    engine.train_an_epoch(train_loader, epoch_id=epoch)
    # break
    
    logging.info('-' * 80)
    logging.info('Evaluating phase!')
    train_auc, train_logloss, train_mi_result, train_ma_result, train_we_result = engine.evaluate(eval_loader)
    # break
    logging.info('[Evaluating Epoch {}] AUC = {:.4f}'.format(epoch, train_auc))
    logging.info('[Evaluating Epoch {}] LogLoss = {:.4f}'.format(epoch, train_logloss))
    logging.info('[Evaluating Epoch {}] Mi_Result = {}'.format(epoch, str(train_mi_result)))
    logging.info('[Evaluating Epoch {}] Ma_Result = {}'.format(epoch, str(train_ma_result)))
    logging.info('[Evaluating Epoch {}] We_Result = {}'.format(epoch, str(train_we_result)))
    if train_auc > best_train_auc:
        best_train_auc = train_auc
        torch.save(engine.model.state_dict(), './pretrain_model/'+config['dataset']+'/'+current_time+config['divide_style']+'_'+str(config['divide_ratio'])+'_'+str(config['divide_feature_values'])+'_'+str(config['train_val_ratio'])+'_epoch_'+str(epoch)+'.pth')

    logging.info('-' * 80)
    logging.info('Validating phase!')
    val_auc, val_logloss, val_mi_result, val_ma_result, val_we_result = engine.evaluate(val_loader)
    # break
    logging.info('[Validating Epoch {}] AUC = {:.4f}'.format(epoch, val_auc))
    val_auc_list.append(val_auc)
    logging.info('[Validating Round {}] LogLoss = {:.4f}'.format(epoch, val_logloss))
    val_logloss_list.append(val_logloss)
    logging.info('[Validating Round {}] Mi_Result = {}'.format(epoch, val_mi_result))
    val_mi_result_list.append(val_mi_result)
    logging.info('[Validating Round {}] Ma_Result = {}'.format(epoch, val_ma_result))
    val_ma_result_list.append(val_ma_result)
    logging.info('[Validating Round {}] We_Result = {}'.format(epoch, val_we_result))
    val_we_result_list.append(val_we_result)

    logging.info('-' * 80)
    logging.info('Testing phase!')
    test_auc, test_logloss, test_mi_result, test_ma_result, test_we_result = engine.evaluate(test_loader)
    logging.info('[Testing Epoch {}] AUC = {:.4f}'.format(epoch, test_auc))
    test_auc_list.append(test_auc)
    logging.info('[Testing Round {}] LogLoss = {:.4f}'.format(epoch, test_logloss))
    test_logloss_list.append(test_logloss)
    logging.info('[Testing Round {}] Mi_Result = {}'.format(epoch, test_mi_result))
    test_mi_result_list.append(test_mi_result)
    logging.info('[Testing Round {}] Ma_Result = {}'.format(epoch, test_ma_result))
    test_ma_result_list.append(test_ma_result)
    logging.info('[Testing Round {}] We_Result = {}'.format(epoch, test_we_result))
    test_we_result_list.append(test_we_result)

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        final_test_epoch = epoch

current_time = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
file_str = current_time + '-' + 'latent_dim: ' + str(config['latent_dim']) + '-' + 'lr: ' + str(config['lr']) + '-' + \
           'layers: ' + str(config['layers']) + '-' + 'num_epoch: ' + str(config['num_epoch']) + '-' + 'dataset: ' + \
           config['dataset'] + '-' + 'divide_button: ' + str(config['divide_button']) + '-' + 'divide_style: ' + \
           config['divide_style'] + '-' + 'divide_feature_values: ' + str(config['divide_feature_values']) + '-' + 'batch_size: ' + str(config['batch_size']) + '-' + 'divide_ratio: ' + \
           str(config['divide_ratio'])  + '-' + 'split_button: ' + str(config['split_button']) + '-' + \
           'train_val_ratio: ' + str(config['train_val_ratio']) + '-' + 'optimizer: ' + config['optimizer'] + '-' + \
           'l2_regularization: ' + str(config['l2_regularization']) + '-' + 'auc: ' + \
           str(test_auc_list[final_test_epoch]) + '-' + 'logloss: ' + str(test_logloss_list[final_test_epoch]) + '-' + 'mi_result: ' + str(test_mi_result_list[final_test_epoch]) + '-' + 'ma_result: ' + str(test_ma_result_list[final_test_epoch]) + '-' + 'we_result: ' + str(test_we_result_list[final_test_epoch]) + '-' + 'best_epoch: ' + str(final_test_epoch)
file_name = "sh_result/"+config['dataset']+".txt"
with open(file_name, 'a') as file:
    file.write(file_str + '\n')

logging.info('OpenRec')
logging.info('latent_dim: {}, layers: {}, bz: {}, lr: {}, dataset: {}, ' 
             'divide_button: {}, divide_style: {}, divide_feature_values: {}, '
             'divide_ratio: {}, split_button: {}, train_val_ratio: {}'.
             format(config['latent_dim'], config['layers'], config['batch_size'], config['lr'],
                    config['dataset'], config['divide_button'], config['divide_style'], config['divide_feature_values'],
                    config['divide_ratio'], config['split_button'], config['train_val_ratio']))

logging.info('test_auc_list: {}'.format(test_auc_list))
logging.info('test_logloss_list: {}'.format(test_logloss_list))
logging.info('test_mi_result_list: {}'.format(test_mi_result_list))
logging.info('test_ma_result_list: {}'.format(test_ma_result_list))
logging.info('test_we_result_list: {}'.format(test_we_result_list))
logging.info('Best test auc: {}, test logloss: {}, test mi_result: {}, test ma_result: {}, test we_result: {} at epoch {}'.format(test_auc_list[final_test_epoch], test_logloss_list[final_test_epoch], test_mi_result_list[final_test_epoch], test_ma_result_list[final_test_epoch], test_we_result_list[final_test_epoch], final_test_epoch))
