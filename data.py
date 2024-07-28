import torch
import random
import pandas as pd
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset

random.seed(0)


class UserItemRatingDataset(Dataset):
    """Wrapper, convert <user, item, rating> Tensor into Pytorch Dataset"""
    def __init__(self, user_tensor, item_tensor, target_tensor):
        """
        args:

            target_tensor: torch.Tensor, the corresponding rating for <user, item> pair
        """
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)


class SampleGenerator(object):
    """Construct dataset"""

    def __init__(self, ratings):
        """
        args:
            ratings: pd.DataFrame, which contains 4 columns = ['user_id', 'video_id', 'timestamp', 'is_click']
        """
        assert 'user_id' in ratings.columns
        assert 'video_id' in ratings.columns
        assert 'is_click' in ratings.columns
    
    def store_all_train_data(self, train_data):
        """store all the train data as a list including users, items and ratings. each dict consists of all users' information"""
        users, videos, ratings = {}, {}, {}
        # split train_ratings into groups according to userId.
        grouped_train_ratings = train_data.groupby('user_id')
        for userId, user_train_ratings in grouped_train_ratings:
            single_user = []
            user_video = []
            user_rating = []
            for row in user_train_ratings.itertuples():
                single_user.append(int(row.user_id))
                user_video.append(int(row.video_id))
                user_rating.append(float(row.is_click))
            users[single_user[-1]] = single_user
            videos[single_user[-1]] = user_video
            ratings[single_user[-1]] = user_rating
        assert len(users) == len(videos) == len(ratings)
        return [users, videos, ratings]
    
    def validate_data(self, val_data):
        """create validation data"""
        val_users, val_videos, val_ratings = {}, {}, {}
        grouped_val_ratings = val_data.groupby('user_id')
        for userId, user_val_ratings in grouped_val_ratings:
            single_user = []
            user_video = []
            user_rating = []
            for row in user_val_ratings.itertuples():
                single_user.append(int(row.user_id))
                user_video.append(int(row.video_id))
                user_rating.append(float(row.is_click))
            val_users[single_user[-1]] = single_user
            val_videos[single_user[-1]] = user_video
            val_ratings[single_user[-1]] = user_rating
        assert len(val_users) == len(val_videos) == len(val_ratings)
        return [val_users, val_videos, val_ratings]

    def test_data(self, test_data):
        """create test data"""
        test_users, test_videos, test_ratings = {}, {}, {}
        grouped_test_ratings = test_data.groupby('user_id')
        for userId, user_val_ratings in grouped_test_ratings:
            single_user = []
            user_video = []
            user_rating = []
            for row in user_val_ratings.itertuples():
                single_user.append(int(row.user_id))
                user_video.append(int(row.video_id))
                user_rating.append(float(row.is_click))
            test_users[single_user[-1]] = single_user
            test_videos[single_user[-1]] = user_video
            test_ratings[single_user[-1]] = user_rating
        assert len(test_users) == len(test_videos) == len(test_ratings)
        return [test_users, test_videos, test_ratings]