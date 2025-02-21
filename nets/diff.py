"""
Code Reference: https://github.com/yixinliu233/D2PT
"""

import torch.nn as nn
import torch.nn.functional as F
from utils import edge_index_to_sparse_mx, process_adj, feature_propagation


class Diff(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, is_prop, T, alpha):
        super(Diff, self).__init__()
        self.linear1 = nn.Linear(nfeat, nhid)
        self.linear2 = nn.Linear(nhid, nclass)
        self.dropout = dropout
        self.is_prop = is_prop
        self.T = T
        self.alpha = alpha

        self.reg_params = self.linear1.parameters()
        self.non_reg_params = self.linear2.parameters()

    def get_prop_feature(self, x, edge_index):
        adj = edge_index_to_sparse_mx(edge_index.cpu(), x.shape[0])
        adj = process_adj(adj)
        x_prop = feature_propagation(adj, x.cpu(), self.T, self.alpha)
        return x_prop

    def forward(self, x, edge_index=None):
        if self.is_prop:
            x = self.get_prop_feature(x, edge_index).to(x.device)
        x = F.relu(self.linear1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.linear2(x)


def create_diff(nfeat, nhid, nclass, dropout, is_prop=False, T=10, alpha=0.01):
    model = Diff(nfeat, nhid, nclass, dropout, is_prop, T, alpha)
    return model