import numpy as np
import torch.nn as nn
import torch
class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(768, 2)

    def forward(self, features):
        x = features  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class policyNet(nn.Module):
    def __init__(self, config):
        super(policyNet, self).__init__()
        self.classifier = RobertaClassificationHead()

    def forward(self, h, x, labels_c, labels_f):
        actions = []

        logits_f = self.classifier(x)
        prob_f = torch.softmax(logits_f, dim=-1)
        c_prob_f = torch.clamp(prob_f, min=1e-10, max=1-1e-10)

        if labels_f is not None and labels_c is not None and labels_c.sum() > 0:
            labels_f = labels_f[labels_c.bool()][:, 1:x.size(1) + 1]

        return prob_f, c_prob_f, labels_f, actions
