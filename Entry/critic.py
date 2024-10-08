from transformers import RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification

from Models.StagedModel_long_seq_p import Model
import torch.nn as nn
import torch
from copy import deepcopy


class critic(nn.Module):
    '''
    critic用来评估模型的好坏
    '''

    def __init__(self):
        super(critic, self).__init__()
        config = RobertaConfig.from_pretrained("./resources/codebert-base")
        config.num_labels = 1
        config.num_attention_heads = 12
        config.num_hidden_layers = 6
        tokenizer = RobertaTokenizer.from_pretrained("./resources/codebert-base")
        model_fine = RobertaForSequenceClassification(config=config)
        model_coarse = RobertaForSequenceClassification(config=config)
        self.target_pred = Model(model_fine, model_coarse, config, tokenizer)
        self.target_pred.load_state_dict(torch.load('./resources/SavedModels/model_line_only_2048_mean_real.bin', map_location='cuda'), strict=False)
        self.active_pred = Model(model_fine, model_coarse, config, tokenizer)
        self.active_pred.load_state_dict(torch.load('./resources/SavedModels/model_line_only_2048_mean_real.bin', map_location='cuda'), strict=False)

    def forward(self, x, scope, labels_c=None, labels_f=None):
        if scope == "target":
            out = self.target_pred(x, labels_c, labels_f)
        if scope == "active":
            out = self.active_pred(x, labels_c, labels_f)
        return out

    def assign_target_network(self):
        params = []
        for name, x in self.active_pred.named_parameters():
            params.append(x)
        i = 0
        for name, x in self.target_pred.named_parameters():
            x.data = deepcopy(params[i].data)
            i += 1

    def update_target_network(self):
        params = []
        for name, x in self.active_pred.named_parameters():
            params.append(x)
        i = 0
        for name, x in self.target_pred.named_parameters():
            x.data = deepcopy(params[i].data)
            i += 1

    def assign_active_network(self):
        params = []
        for name, x in self.target_pred.named_parameters():
            params.append(x)
        i = 0
        for name, x in self.active_pred.named_parameters():
            x.data = deepcopy(params[i].data)
            i += 1

    def assign_active_network_gradients(self):
        params = []
        for name, x in self.target_pred.named_parameters():
            params.append(x)
        i = 0
        for name, x in self.active_pred.named_parameters():
            x.grad = deepcopy(params[i].grad)
            i += 1
        for name, x in self.target_pred.named_parameters():
            x.grad = None

    def forward_lstm(self, h, x_vector, scope):
        if scope == "target":
            out, state = self.target_pred.getNextHiddenState(h, x_vector)
        if scope == "active":
            out, state = self.active_pred.getNextHiddenState(h, x_vector)
        return out, state

    def proglines_encode(self, x):
        return self.target_pred.proglines_encode(x)
