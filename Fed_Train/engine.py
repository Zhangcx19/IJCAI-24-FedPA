import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from utils import *
import random
import copy
import numpy as np
from data import UserItemRatingDataset
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support


class Engine(object):

    def __init__(self, config):
        self.config = config  # model configuration
        # self._writer = SummaryWriter(log_dir='runs/{}'.format(config['alias']))  # tensorboard writer
        # self._writer.add_text('config', str(config), 0)
        self.server_model_param = {}
        self.server_lora_param = {}
        self.client_model_params = {}
        # implicit feedback
        self.crit = torch.nn.BCELoss()
    
    def instance_user_train_loader(self, user_train_data):
        """instance a user's train loader."""
        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(user_train_data[0]),
                                        item_tensor=torch.LongTensor(user_train_data[1]),
                                        target_tensor=torch.FloatTensor(user_train_data[2]))
        return DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True)
    
    def fed_train_single_batch(self, model_client, batch_data, optimizer, lora_user_feature_values):
        """train a batch and return an updated model."""
        # load batch data.
        users, videos, ratings = batch_data[0], batch_data[1], batch_data[2]
        ratings = ratings.float()

        if self.config['use_cuda'] is True:
            ratings = ratings.cuda()

        optimizer.zero_grad()
        ratings_pred = model_client(users, videos, lora_user_feature_values)
        loss = self.crit(ratings_pred.view(-1), ratings)
        loss.backward()
        optimizer.step()

        return model_client
    
    def aggregate_clients_params(self, round_user_params, round_lora_params, lora_user_feature_value_dict):
        """receive client models' parameters in a round, aggregate them and store the aggregated result for server."""
        # aggregate item embedding and score function via averaged aggregation.
        t = 0
        for user in round_user_params.keys():
            # load a user's parameters.
            user_params = round_user_params[user]
            if t == 0:
                self.server_model_param = copy.deepcopy(user_params)
            else:
                for key in user_params.keys():
                    self.server_model_param[key].data += user_params[key].data
            t += 1
        for key in self.server_model_param.keys():
            self.server_model_param[key].data = self.server_model_param[key].data / len(round_user_params)
                    
        for idx in range(len(self.model.fc_layers)):
            for lora_user_feature in self.config['lora_user_features']:
                lora_user_feature_values = lora_user_feature_value_dict[lora_user_feature]
                idx_lora_a_name = lora_user_feature + str(idx)+'_lora_a_'
                idx_lora_b_name = lora_user_feature + str(idx)+'_lora_b_'
                for lora_user_feature_value in lora_user_feature_values:
                    round_lora_a_params = round_lora_params[idx_lora_a_name]
                    round_lora_b_params = round_lora_params[idx_lora_b_name]
                    for lora_user_feature_value in round_lora_a_params.keys():
                        self.server_lora_param[idx_lora_a_name+str(lora_user_feature_value)] = torch.tensor(np.mean(round_lora_a_params[lora_user_feature_value], axis=0))
                        self.server_lora_param[idx_lora_b_name+str(lora_user_feature_value)] = torch.tensor(np.mean(round_lora_b_params[lora_user_feature_value], axis=0))

    
    def fed_train_a_round(self, all_train_data, user_ids, round_id, user_features):
        """train a round."""
        # sample users participating in single round.
        if self.config['clients_sample_ratio'] <= 1:
            num_participants = int(len(user_ids) * self.config['clients_sample_ratio'])
            participants = random.sample(user_ids, num_participants)
        else:
            participants = random.sample(user_ids, self.config['clients_sample_num'])
                
        # load the pretrain parameter.
        if round_id == 0:
            self.model.load_state_dict(torch.load(self.config['pretrain_model_dir']), strict=False)
#             self.model.load_state_dict(torch.load('./pretrain_model/2023-10-17 21:25:15.pth'), strict=False)
        
        # store users' model parameters of current round.
        round_participant_params = {}
        round_lora_params = {}
        for idx in range(len(self.model.fc_layers)):
            for lora_user_feature in self.config['lora_user_features']:
                idx_lora_a_name = lora_user_feature + str(idx)+'_lora_a_'
                idx_lora_b_name = lora_user_feature + str(idx)+'_lora_b_'
                round_lora_params[idx_lora_a_name] = {}
                round_lora_params[idx_lora_b_name] = {}
        
        lora_user_feature_value_dict = {}
        for lora_user_feature in self.config['lora_user_features']:
            lora_user_feature_value_dict[lora_user_feature] = list(user_features[lora_user_feature].unique()) 
        
        # perform model update for each participated user.
        for user in participants:
            lora_user_feature_values = {}
            for lora_user_feature in self.config['lora_user_features']:
                lora_user_feature_values[lora_user_feature] = user_features.loc[user_features['user_id'] == user, lora_user_feature].iloc[0]
                
            model_client = copy.deepcopy(self.model)
            if round_id != 0:
                user_param_dict = copy.deepcopy(self.model.state_dict())
                if user in self.client_model_params.keys():
                    for key in self.client_model_params[user].keys():
                        user_param_dict[key] = copy.deepcopy(self.client_model_params[user][key].data).cuda()
                for key in self.server_model_param.keys():
                    user_param_dict[key] = copy.deepcopy(self.server_model_param[key].data).cuda()
                for idx in range(len(model_client.fc_layers)):
                    for lora_user_feature in lora_user_feature_values.keys():
                        lora_user_feature_value = lora_user_feature_values[lora_user_feature]
                        idx_lora_a_name = lora_user_feature + str(idx)+'_lora_a_'
                        idx_lora_b_name = lora_user_feature + str(idx)+'_lora_b_'
                        user_param_dict['loras.'+idx_lora_a_name+str(lora_user_feature_value)] = copy.deepcopy(self.server_lora_param[idx_lora_a_name+str(lora_user_feature_value)].data).cuda()
                        user_param_dict['loras.'+idx_lora_b_name+str(lora_user_feature_value)] = copy.deepcopy(self.server_lora_param[idx_lora_b_name+str(lora_user_feature_value)].data).cuda()
                model_client.load_state_dict(user_param_dict)
            model_client.freeze()
            # Defining optimizers
            optimizer = use_optimizer(model_client, self.config)
            user_train_data = [all_train_data[0][user], all_train_data[1][user], all_train_data[2][user]]
            user_dataloader = self.instance_user_train_loader(user_train_data)
            model_client.train()
            # update client model.
            for epoch in range(self.config['local_epoch']):
                for batch_id, batch in enumerate(user_dataloader):
                    assert isinstance(batch[0], torch.LongTensor)
                    model_client = self.fed_train_single_batch(model_client, batch, optimizer, lora_user_feature_values)
            # obtain client model parameters.
            client_param = model_client.state_dict()
            # store client models' local parameters for personalization.
            self.client_model_params[user] = {}
            for client_param_name in model_client.selected_user_features:
                self.client_model_params[user]['embedding_user_features.'+client_param_name+'.weight'] = copy.deepcopy(client_param['embedding_user_features.'+client_param_name+'.weight'].data.cpu())
            for idx in range(len(model_client.fc_layers)):
                lora_a_name = 'loras.'+str(idx)+'_lora_a'
                self.client_model_params[user][lora_a_name] = copy.deepcopy(client_param[lora_a_name]).data.cpu()
                lora_b_name = 'loras.'+str(idx)+'_lora_b'
                self.client_model_params[user][lora_b_name] = copy.deepcopy(client_param[lora_b_name]).data.cpu()
            
                
            round_participant_params[user] = {}
            for idx in range(len(model_client.fc_layers)):
                gate_weight_1_name = 'gate_layers.'+str(idx*2)+'.weight'
                gate_bias_1_name = 'gate_layers.'+str(idx*2)+'.bias'
                gate_weight_2_name = 'gate_layers.'+str(idx*2+1)+'.weight'
                gate_bias_2_name = 'gate_layers.'+str(idx*2+1)+'.bias'
                round_participant_params[user][gate_weight_1_name] = copy.deepcopy(client_param[gate_weight_1_name].data.cpu())
                round_participant_params[user][gate_bias_1_name] = copy.deepcopy(client_param[gate_bias_1_name].data.cpu())
                round_participant_params[user][gate_weight_2_name] = copy.deepcopy(client_param[gate_weight_2_name].data.cpu())
                round_participant_params[user][gate_bias_2_name] = copy.deepcopy(client_param[gate_bias_2_name].data.cpu())

            round_participant_params[user]['affine_output.weight'] = copy.deepcopy(
                client_param['affine_output.weight'].data.cpu())
            round_participant_params[user]['affine_output.bias'] = copy.deepcopy(
                client_param['affine_output.bias'].data.cpu())
            
            for idx in range(len(self.model.fc_layers)):
                for lora_user_feature in lora_user_feature_values.keys():
                    lora_user_feature_value = lora_user_feature_values[lora_user_feature]
                    idx_lora_a_name = lora_user_feature + str(idx)+'_lora_a_'
                    idx_lora_b_name = lora_user_feature + str(idx)+'_lora_b_'
                    if lora_user_feature_value not in round_lora_params[idx_lora_a_name].keys():
                        round_lora_params[idx_lora_a_name][lora_user_feature_value] = []
                        round_lora_params[idx_lora_b_name][lora_user_feature_value] = []
                    round_lora_params[idx_lora_a_name][lora_user_feature_value].append(copy.deepcopy(client_param['loras.'+idx_lora_a_name+str(lora_user_feature_value)].data.cpu().numpy()))
                    round_lora_params[idx_lora_b_name][lora_user_feature_value].append(copy.deepcopy(client_param['loras.'+idx_lora_b_name+str(lora_user_feature_value)].data.cpu().numpy()))
                
        # aggregate client models in server side.
        self.aggregate_clients_params(round_participant_params, round_lora_params, lora_user_feature_value_dict)

    def fed_evaluate(self, evaluate_data, user_features):
        users, videos, ratings = copy.deepcopy(evaluate_data[0]), copy.deepcopy(evaluate_data[1]), copy.deepcopy(evaluate_data[2])
        a = [len(ratings[item]) for item in ratings]
        temp = 0
        for user in users.keys():
            test_user, test_video, test_rating = users[user], videos[user], ratings[user]
            lora_user_feature_values = {}
            for lora_user_feature in self.config['lora_user_features']:
                lora_user_feature_values[lora_user_feature] = user_features.loc[user_features['user_id'] == user, lora_user_feature].iloc[0]
            user_model = copy.deepcopy(self.model)
            user_param_dict = copy.deepcopy(self.model.state_dict())
            if user in self.client_model_params.keys():
                for key in self.client_model_params[user].keys():
                    user_param_dict[key] = copy.deepcopy(self.client_model_params[user][key].data).cuda()
            for key in self.server_model_param.keys():
                user_param_dict[key] = copy.deepcopy(self.server_model_param[key].data).cuda()
            for idx in range(len(user_model.fc_layers)):
                for lora_user_feature in lora_user_feature_values.keys():
                    lora_user_feature_value = lora_user_feature_values[lora_user_feature]
                    idx_lora_a_name = lora_user_feature + str(idx)+'_lora_a_'
                    idx_lora_b_name = lora_user_feature + str(idx)+'_lora_b_'
                    user_param_dict['loras.'+idx_lora_a_name+str(lora_user_feature_value)] = copy.deepcopy(self.server_lora_param[idx_lora_a_name+str(lora_user_feature_value)].data).cuda()
                    user_param_dict['loras.'+idx_lora_b_name+str(lora_user_feature_value)] = copy.deepcopy(self.server_lora_param[idx_lora_b_name+str(lora_user_feature_value)].data).cuda()
            user_model.load_state_dict(user_param_dict)
            user_model.eval()
            with torch.no_grad():
                test_pred = user_model(test_user, test_video, lora_user_feature_values)
                if temp == 0:
                    test_preds = test_pred
                    test_ratings = test_rating
                else:
                    test_preds = torch.cat((test_preds, test_pred))
                    test_ratings.extend(test_rating) 
            temp += 1
                   
        if self.config['use_cuda'] is True:
            test_preds = test_preds.cpu()
                    
        auc = roc_auc_score(np.array(test_ratings), test_preds.numpy())
        we_result = precision_recall_fscore_support(np.array(test_ratings).astype(int), np.round(test_preds.numpy()).astype(int), average='weighted')
        # self._writer.add_scalar('performance/HR', hit_ratio, epoch_id)
        # self._writer.add_scalar('performance/NDCG', ndcg, epoch_id)
        return auc, we_result

    def save(self, alias, epoch_id, hit_ratio, ndcg):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        model_dir = self.config['model_dir'].format(alias, epoch_id, hit_ratio, ndcg)
        save_checkpoint(self.model, model_dir)
