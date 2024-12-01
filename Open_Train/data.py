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

    def instance_a_train_loader(self, train_data, batch_size):
        """instance train loader for the training epoch"""
        users, videos, ratings = [], [], []
        for row in train_data.itertuples():
            users.append(int(row.user_id))
            videos.append(int(row.video_id))
            ratings.append(float(row.is_click))
        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(users),
                                        item_tensor=torch.LongTensor(videos),
                                        target_tensor=torch.FloatTensor(ratings))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    def evaluate_data(self, train_data, batch_size):
        """create evaluation data"""
        eval_users, eval_videos, eval_ratings = [], [], []
        for row in train_data.itertuples():
            eval_users.append(int(row.user_id))
            eval_videos.append(int(row.video_id))
            eval_ratings.append(float(row.is_click))
        assert len(eval_users) == len(eval_videos) == len(eval_ratings)
        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(eval_users),
                                        item_tensor=torch.LongTensor(eval_videos),
                                        target_tensor=torch.FloatTensor(eval_ratings))
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def validate_data(self, val_data, batch_size):
        """create validation data"""
        val_users, val_videos, val_ratings = [], [], []
        for row in val_data.itertuples():
            val_users.append(int(row.user_id))
            val_videos.append(int(row.video_id))
            val_ratings.append(float(row.is_click))
        assert len(val_users) == len(val_videos) == len(val_ratings)
        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(val_users),
                                        item_tensor=torch.LongTensor(val_videos),
                                        target_tensor=torch.FloatTensor(val_ratings))
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def test_data(self, test_data, batch_size):
        """create test data"""
        test_users, test_videos, test_ratings = [], [], []
        for row in test_data.itertuples():
            test_users.append(int(row.user_id))
            test_videos.append(int(row.video_id))
            test_ratings.append(float(row.is_click))
        assert len(test_users) == len(test_videos) == len(test_ratings)
        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(test_users),
                                        item_tensor=torch.LongTensor(test_videos),
                                        target_tensor=torch.FloatTensor(test_ratings))
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)