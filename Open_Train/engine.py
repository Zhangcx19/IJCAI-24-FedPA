import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from utils import save_checkpoint, use_optimizer
from sklearn.metrics import roc_auc_score, log_loss, precision_recall_fscore_support
import numpy as np


class Engine(object):
    """Meta Engine for training & evaluating NCF model

    Note: Subclass should implement self.model !
    """

    def __init__(self, config):
        self.config = config  # model configuration
        # self._writer = SummaryWriter(log_dir='runs/{}'.format(config['alias']))  # tensorboard writer
        # self._writer.add_text('config', str(config), 0)
        self.opt = use_optimizer(self.model, config)
        # implicit feedback
        self.crit = torch.nn.BCELoss()

    def train_single_batch(self, users, videos, ratings):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        if self.config['use_cuda'] is True:
            ratings = ratings.cuda()
        self.opt.zero_grad()
        ratings_pred = self.model(users, videos)
        loss = self.crit(ratings_pred.view(-1), ratings)
        loss.backward()
        self.opt.step()

    def train_an_epoch(self, train_loader, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.train()
        for batch_id, batch in enumerate(train_loader):
            assert isinstance(batch[0], torch.LongTensor)
            user, video, rating = batch[0], batch[1], batch[2]
            rating = rating.float()
            self.train_single_batch(user, video, rating)
        # self._writer.add_scalar('model/loss', total_loss, epoch_id)

    def evaluate(self, evaluate_loader):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.eval()
        with torch.no_grad():
            for batch_id, batch in enumerate(evaluate_loader):
                test_users, test_videos, test_ratings = batch[0], batch[1], batch[2]
                test_ratings = test_ratings.float()
                test_pred = self.model(test_users, test_videos)
                if self.config['use_cuda'] is True:
                    test_pred = test_pred.cpu()
                    
                if batch_id == 0:
                    test_preds = test_pred
                    total_test_ratings = test_ratings
                else:
                    test_preds = torch.cat((test_preds, test_pred))
                    total_test_ratings = torch.cat((total_test_ratings, test_ratings))
            
        auc = roc_auc_score(total_test_ratings.numpy(), test_preds.numpy())
        logloss = log_loss(total_test_ratings.numpy(), test_preds.numpy().astype("float64"))
        mi_result = precision_recall_fscore_support(total_test_ratings.numpy().astype(int), np.round(test_preds.numpy()).astype(int), average='micro')
        ma_result = precision_recall_fscore_support(total_test_ratings.numpy().astype(int), np.round(test_preds.numpy()).astype(int), average='macro')
        we_result = precision_recall_fscore_support(total_test_ratings.numpy().astype(int), np.round(test_preds.numpy()).astype(int), average='weighted')
        # self._writer.add_scalar('performance/HR', hit_ratio, epoch_id)
        # self._writer.add_scalar('performance/NDCG', ndcg, epoch_id)
        return auc, logloss, mi_result, ma_result, we_result

    def save(self, alias, epoch_id, hit_ratio, ndcg):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        model_dir = self.config['model_dir'].format(alias, epoch_id, hit_ratio, ndcg)
        save_checkpoint(self.model, model_dir)