import torch.nn as nn
from Models.policyNet import policyNet
import torch
from copy import deepcopy

class actor(nn.Module):

    def __init__(self, config):
        super(actor, self).__init__()
        self.config = config
        self.target_policy = policyNet(config)
        self.active_policy = policyNet(config)

    def get_target_logOutput(self, h, x):
        out = self.target_policy(h, x)
        logOut = torch.log(out)
        return logOut

    # predict the Q value
    def forward(self, h, x, scope, labels_c=None, labels_f=None):
        if scope == "target":
            out = self.target_policy(h, x, labels_c, labels_f)
        if scope == "active":
            out = self.active_policy(h, x, labels_c, labels_f)
        return out

    def get_gradient(self, h, x, reward, label_f, pred, scope):
        if scope == "target":
            out = pred
            logout = torch.log(out).view(-1)
            index = reward.index(0)
            index = (index + 1) % 2

            grad = torch.autograd.grad(logout[index].view(-1), self.target_policy.parameters(), retain_graph=True, allow_unused=True) # torch.cuda.FloatTensor(reward[index])

            grads = []
            for grad_ in grad:
                if grad_ is None:
                    grads.append(None)
                else:
                    grads.append((-1) * grad_ * reward[index])
            return grads

        if scope == "active":
            out = self.active_policy(h, x)
            return out
    def assign_active_network_gradients(self, params):
        i=0
        for name, x in self.active_policy.named_parameters():
            if params[i] is None:
                x.grad = None
            else:
                x.grad = params[i].clone()
            i+=1

    def target_gradients_to_active(self):
        params = []
        for name, x in self.target_policy.named_parameters():
            params.append(x.grad.clone())
        i = 0
        for name, x in self.active_policy.named_parameters():
            x.grad = params[i].clone()
            i += 1
        for name, x in self.target_policy.named_parameters():
            x.grad = None

    def update_target_network(self):
        params = []
        for name, x in self.active_policy.named_parameters():
            params.append(x.data.clone())
        i=0

        for name, x in self.target_policy.named_parameters():
            x.data = params[i].clone()
            i+=1
    def assign_active_network(self):
        params = []
        for name, x in self.target_policy.named_parameters():
            params.append(x.data.clone())
        i=0
        for name, x in self.active_policy.named_parameters():
            x.data = params[i].clone()
            i+=1
