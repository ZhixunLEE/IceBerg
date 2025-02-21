"""
Code Reference: https://github.com/jiawei-ren/BalancedMetaSoftmax-Classification/blob/main/loss/BalancedSoftmaxLoss.py
"""

"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
Portions of the source code are from the OLTR project which
notice below and in LICENSE in the root directory of
this source tree.
Copyright (c) 2019, Zhongqi Miao
All rights reserved.
"""


import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import json


class RobustBalancedSoftmax(_Loss):
    """
    Balanced Softmax Loss
    """
    def __init__(self, num_cls, beta):
        super(RobustBalancedSoftmax, self).__init__()
        self.num_cls = num_cls
        self.beta = beta

    def forward(self, input, label, sample_per_class, reduction='mean', weight=None):
        return robust_balanced_softmax_loss(label, input, sample_per_class, reduction, self.num_cls, self.beta, weight)


def robust_balanced_softmax_loss(labels, logits, sample_per_class, reduction, num_cls, beta, weight=None):
    """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      sample_per_class: A int tensor of size [no of classes].
      reduction: string. One of "none", "mean", "sum"
    Returns:
      loss: A float tensor. Balanced Softmax Loss.
    """
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + spc.log()
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    if weight is not None:
        loss = (weight * loss).mean()

    pred = F.softmax(logits, dim=1)
    pred = torch.clamp(pred, min=1e-7, max=1.0)
    label_one_hot = F.one_hot(labels, num_cls).float().to(labels.device)
    label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
    rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))
    loss += beta * rce.mean()
    
    return loss