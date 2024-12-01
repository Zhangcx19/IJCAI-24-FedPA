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
        for idx, (in_size, out_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

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

    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_consrtuction('user', user_indices)
        item_embedding = self.embedding_consrtuction('video', item_indices)
        vector = torch.cat([user_embedding, item_embedding], dim=-1)  # the concat latent vector
        for idx, _ in enumerate(range(len(self.fc_layers))):
            vector = self.fc_layers[idx](vector)
            vector = torch.nn.ReLU()(vector)
            # vector = torch.nn.BatchNorm1d()(vector)
            # vector = torch.nn.Dropout(p=0.5)(vector)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating

    def init_weight(self):
        pass


class MLPEngine(Engine):
    """Engine for training & evaluating GMF model"""
    def __init__(self, config, user_features, video_features, selected_user_features, selected_video_features):
        self.model = MLP(config, user_features, video_features, selected_user_features, selected_video_features)
        if config['use_cuda'] is True:
#             use_cuda(True, config['device_id'])
            self.model.cuda()
        super(MLPEngine, self).__init__(config)
        print(self.model)
#         print(self.model.state_dict())