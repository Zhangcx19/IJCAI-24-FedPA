# -*- coding: utf-8 -*-

import pandas as pd
import random
import copy
from numpy import linspace
import numpy as np


"""replace the missing values with column mean values"""
def replace_missing_features(df):
    feature_names = df.columns.values.tolist()
    for idx in range(len(feature_names)):
        feature = feature_names[idx]
        if df[feature].isnull().values.any():
            replaced_value = -100
            feature_type = df[feature].dtypes
            if feature_type == "int64":  
                df[feature][df[feature].isnull()] = int(replaced_value)
            elif feature_type == "float64":
                df[feature][df[feature].isnull()] = float(replaced_value)
            elif feature_type == "str":
                df[feature][df[feature].isnull()] = str(replaced_value)
            elif feature_type == "object":
                df[feature][df[feature].isnull()] = str(replaced_value)
            else:
                print(feature_type)
    
    return df


"""read the feature names from txt file"""
def read_feature_names(file_dir):
    feature_names = []
    with open(file_dir, 'r') as f:
        for line in f:
            feature_names.append(line.strip('\n'))
            
    return feature_names


"""modify the discrete column value of dataframe by reindexing the values"""
def preprocess_discrete_df(df, column_names):
    for feature in column_names:
        df[feature] = pd.Categorical(df[feature])
        df[feature] = df[feature].cat.codes
    
    return df


"""create features bin dict"""
def create_bin_dict(df, feature_names, bin_values, file_dir):
    p1_list = []
    p2_list = []
    for idx in range(len(feature_names)):
        feature_name = feature_names[idx]
        feature_data = df[feature_name].values
        log_feature_data = np.log(feature_data+1)
        p1 = np.percentile(log_feature_data, 1)
        p2 = np.percentile(log_feature_data, 99)
        p1_list.append(p1)
        p2_list.append(p2)
        
    file_data = []
    for idx in range(len(feature_names)):
        feature_name = feature_names[idx]
        add_str = feature_name + ': ' + str(p1_list[idx]) + ',' + str(p2_list[idx]) + ',' + str(bin_values[idx])
        file_data.append(add_str)
    for data in file_data:
        f = open(file_dir, 'a')
        f.write(data + '\n')
        f.close()


###########################################################################################
###########################################################################################
###########################################################################################
"""load the original data and check it about the ids and the useless features"""
original_data_dir = "./dataset/KuaiRand-Pure/"
df_rand = pd.read_csv(original_data_dir+"data/log_random_4_22_to_5_08_pure.csv")
df1 = pd.read_csv(original_data_dir+"data/log_standard_4_08_to_4_21_pure.csv")
df2 = pd.read_csv(original_data_dir+"data/log_standard_4_22_to_5_08_pure.csv")
df_merge = pd.concat([df1, df2], axis=0, ignore_index=True)
df_merge = pd.concat([df_merge, df_rand], axis=0, ignore_index=True)
df_merge.sort_values(by=["user_id", "time_ms"], ascending=[True, True], inplace=True, ignore_index=True)
df = df_merge[["user_id", "video_id", "time_ms", "is_click"]]
df.rename(columns={"time_ms": "timestamp"}, inplace=True)

user_features = pd.read_csv(original_data_dir+"data/user_features_pure.csv")

video_features_basic = pd.read_csv(original_data_dir+"data/video_features_basic_pure.csv")
video_features_statistics = pd.read_csv(original_data_dir+"data/video_features_statistic_pure.csv")
video_features = pd.merge(video_features_basic, video_features_statistics, on=['video_id'], how='left')


user_feature_names = user_features.columns.values.tolist()
for feature_name in user_feature_names:
    feature_values = user_features[feature_name].unique()
    feature_value_counts = user_features[feature_name].value_counts(dropna=False)
    print("------------------------------------------------------------------------")
    print(feature_name)
    print(feature_values)
    print(feature_value_counts)
    print("------------------------------------------------------------------------")

video_feature_names = video_features.columns.values.tolist()
for feature_name in video_feature_names:
    feature_values = video_features[feature_name].unique()
    feature_value_counts = video_features[feature_name].value_counts(dropna=False)
    print("------------------------------------------------------------------------")
    print(feature_name)
    print(feature_values)
    print(feature_value_counts)
    print("------------------------------------------------------------------------")


###########################################################################################
###########################################################################################
###########################################################################################
"""discretize the feature values of user and videos"""
new_data_dir = original_data_dir + "processed_data/"
"""replace the missing feature values of df with the random value"""
df.to_csv(new_data_dir+'df.csv', index=False)
user_features = replace_missing_features(user_features)
video_features = replace_missing_features(video_features)
user_discrete_feature_file_dir = original_data_dir + 'user_feature_discrete_dict.txt'
user_discrete_column_names = read_feature_names(user_discrete_feature_file_dir)
new_user_features = preprocess_discrete_df(user_features, user_discrete_column_names)
new_user_features.to_csv(new_data_dir+'new_user_features.csv',index=False)
for user_feature in new_user_features.columns.values.tolist():
    print(user_feature)
    print(new_user_features[user_feature].values.min())
    print(new_user_features[user_feature].values.max())
    print(new_user_features[user_feature].nunique())
video_discrete_feature_file_dir = original_data_dir + 'video_feature_discrete_dict.txt'
video_discrete_column_names = read_feature_names(video_discrete_feature_file_dir)
new_video_features = preprocess_discrete_df(video_features, video_discrete_column_names)
new_video_features.to_csv(new_data_dir+'new_video_features.csv',index=False)
for video_feature in new_video_features.columns.values.tolist():
    print(video_feature)
    print(new_video_features[video_feature].values.min())
    print(new_video_features[video_feature].values.max())
    print(new_video_features[video_feature].nunique())


###########################################################################################
###########################################################################################
###########################################################################################
"""load the processed date after discretization operation"""
original_data_dir = "./dataset/KuaiRand-Pure/"
new_data_dir = original_data_dir + "processed_data/"
df, user_features, video_features = load_data(new_data_dir)


###########################################################################################
###########################################################################################
###########################################################################################
"""create the bin dict"""
feature_names = ["show_cnt","show_user_num","play_cnt","play_user_num","play_duration","complete_play_cnt","complete_play_user_num","valid_play_cnt","valid_play_user_num","long_time_play_cnt","long_time_play_user_num","short_time_play_cnt","short_time_play_user_num","play_progress","comment_stay_duration","like_cnt","like_user_num","click_like_cnt","double_click_cnt","cancel_like_cnt","cancel_like_user_num","comment_cnt","comment_user_num","direct_comment_cnt","reply_comment_cnt","delete_comment_cnt","delete_comment_user_num","comment_like_cnt","comment_like_user_num","follow_cnt","follow_user_num","cancel_follow_cnt","cancel_follow_user_num","share_cnt","share_user_num","download_cnt","download_user_num","report_cnt","report_user_num","reduce_similar_cnt","reduce_similar_user_num","collect_cnt","collect_user_num","cancel_collect_cnt","cancel_collect_user_num","direct_comment_user_num","reply_comment_user_num","share_all_cnt","share_all_user_num","outsite_share_all_cnt"]
bin_values = [999 for i in range(20)]
bin_values += [99 for i in range(30)]
file_dir = 'KuaiRand-Pure_video_feature_bins_dict.txt'
create_bin_dict(video_features, feature_names, bin_values, file_dir)