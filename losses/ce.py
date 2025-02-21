import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropy(nn.Module):
    def __init__(self, class_weight=None):
        super(CrossEntropy, self).__init__()
        self.class_weight = class_weight

    def forward(self, input, target, sample_per_class=None, weight=None, reduction='mean'):
        return F.cross_entropy(input, target, weight=self.class_weight, reduction=reduction)