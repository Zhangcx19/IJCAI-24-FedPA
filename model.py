import torch
from engine import Engine
from utils import use_cuda, resume_checkpoint
import copy


class MLP(torch.nn.Module):
    def __init__(self, config, user_features, video_features, selected_user_features, selected_video_features):
        super(MLP, self).__init__()
        self.config = config
        self.latent_dim = config['latent_dim']
        self.user_features = user_features
        self.video_features = video_features
        self.selected_user_features = selected_user_features
        self.selected_video_features = selected_video_features
        self.layers = config['layers']
        self.lora_user_feature_value_dict = {}
        for lora_user_feature in config['lora_user_features']:
            self.lora_user_feature_value_dict[lora_user_feature] = list(user_features[lora_user_feature].unique())
        
        
        self.embedding_user_features = torch.nn.ModuleDict()
        for feature_name in self.selected_user_features:
            feature_unique_value_count = user_features[feature_name].max() + 1
            self.embedding_user_features[feature_name] = torch.nn.Embedding(num_embeddings=feature_unique_value_count, embedding_dim=self.latent_dim)
            
        self.embedding_video_features = torch.nn.ModuleDict()
        for feature_name in self.selected_video_features:
            feature_unique_value_count = video_features[feature_name].max() + 1
            self.embedding_video_features[feature_name] = torch.nn.Embedding(num_embeddings=feature_unique_value_count, embedding_dim=self.latent_dim)  

        self.layers.insert(0, self.latent_dim*(len(self.selected_user_features)+len(self.selected_video_features)))
        self.fc_layers = torch.nn.ModuleList()
        self.gate_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))
            self.gate_layers.append(torch.nn.Linear(in_size, config['gate_dim'][idx]))
            self.gate_layers.append(torch.nn.Linear(config['gate_dim'][idx], config['gate_num']))
        
        self.loras = torch.nn.ParameterDict()
        for idx, (in_size, out_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            """user lora"""
            self.loras[str(idx)+'_lora_a'] = torch.nn.Parameter(torch.zeros(in_size, config['rank']))
            self.loras[str(idx)+'_lora_b'] = torch.nn.Parameter(torch.zeros(config['rank'], out_size))
            """group lora"""
            for lora_user_feature in config['lora_user_features']:
                lora_user_feature_values = self.lora_user_feature_value_dict[lora_user_feature]
                idx_lora_a_name = lora_user_feature + str(idx)+'_lora_a_'
                idx_lora_b_name = lora_user_feature + str(idx)+'_lora_b_'
                for lora_user_feature_value in lora_user_feature_values:
                    self.loras[idx_lora_a_name+str(lora_user_feature_value)] = torch.nn.Parameter(torch.zeros(in_size, config['rank']))
                    self.loras[idx_lora_b_name+str(lora_user_feature_value)] = torch.nn.Parameter(torch.zeros(config['rank'], out_size)) 
        if config['lora_dropout'] > 0.:
            self.lora_dropout = torch.nn.Dropout(p=config['lora_dropout'])
        else:
            self.lora_dropout = lambda x: x
        self.lora_scaling = config['lora_alpha'] / config['rank']


        self.affine_output = torch.nn.Linear(in_features=self.layers[-1], out_features=1)
        self.logistic = torch.nn.Sigmoid()
        
    
    def embedding_consrtuction(self, embedding_flag, indices):   
        if embedding_flag == 'user':   
            for feature_idx in range(len(self.selected_user_features)):
                feature_name = self.selected_user_features[feature_idx]
                user_feature_values = self.user_features.loc[indices][feature_name].values
                user_feature_embedding = self.embedding_user_features[feature_name](torch.LongTensor(user_feature_values).cuda())
                if feature_idx == 0:
                    user_embedding = user_feature_embedding
                else:
                    user_embedding = torch.cat([user_embedding, user_feature_embedding], dim=-1)
            return user_embedding
        elif embedding_flag == 'video':
            for feature_idx in range(len(self.selected_video_features)):
                feature_name = self.selected_video_features[feature_idx]
                video_feature_values = self.video_features.loc[indices][feature_name].values
                video_feature_embedding = self.embedding_video_features[feature_name](torch.LongTensor(video_feature_values).cuda())
                if feature_idx == 0:
                    video_embedding = video_feature_embedding
                else:
                    video_embedding = torch.cat([video_embedding, video_feature_embedding], dim=-1)
            return video_embedding
        else:
            pass

    def forward(self, user_indices, item_indices, lora_user_feature_values):
        user_embedding = self.embedding_consrtuction('user', user_indices)
        item_embedding = self.embedding_consrtuction('video', item_indices)
        vector = torch.cat([user_embedding, item_embedding], dim=-1)  # the concat latent vector
        for idx, _ in enumerate(range(len(self.fc_layers))):
            gate_vector = self.gate_layers[idx*2](vector)
            gate_vector = torch.nn.ReLU()(gate_vector)
            gate_vector = self.gate_layers[idx*2+1](gate_vector)
            gate_vector = torch.nn.Softmax()(gate_vector)
            user_lora_vector = (self.lora_dropout(vector) @ self.loras[str(idx)+'_lora_a'] @ self.loras[str(idx)+'_lora_b']) * self.lora_scaling
            lora_vector_list = [user_lora_vector]
            for lora_user_feature in lora_user_feature_values.keys():
                lora_user_feature_value = lora_user_feature_values[lora_user_feature]
                lora_vector_list.append((self.lora_dropout(vector) @ self.loras[lora_user_feature+str(idx)+'_lora_a_'+str(lora_user_feature_value)] @ self.loras[lora_user_feature+str(idx)+'_lora_b_'+str(lora_user_feature_value)]) * self.lora_scaling)
            vector = self.fc_layers[idx](vector)
            vector = torch.nn.ReLU()(vector)
            # vector = torch.nn.BatchNorm1d()(vector)
            # vector = torch.nn.Dropout(p=0.5)(vector)
            vector = vector * torch.unsqueeze(gate_vector[:, 0], dim=1)
            for ind in range(len(lora_vector_list)):
                vector += lora_vector_list[ind] * torch.unsqueeze(gate_vector[:, ind+1], dim=1)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating
    
    def freeze(self):
        for feature_name in self.selected_video_features:
            self.embedding_video_features[feature_name].weight.requires_grad = False
        for param in self.fc_layers.parameters():
            param.requires_grad = False

    def init_weight(self):
        pass


class MLPEngine(Engine):
    def __init__(self, config, user_features, video_features, selected_user_features, selected_video_features):
        self.model = MLP(config, user_features, video_features, selected_user_features, selected_video_features)
        if config['use_cuda'] is True:
#             use_cuda(True, config['device_id'])
            self.model.cuda()
        super(MLPEngine, self).__init__(config)
        print(self.model)
#         print(self.model.state_dict().keys())