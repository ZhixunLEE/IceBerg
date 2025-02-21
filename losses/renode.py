import torch
import math
import torch.nn.functional as F
import numpy as np


def index2dense(edge_index,nnode=2708):

    indx = edge_index.cpu().numpy()
    adj = np.zeros((nnode,nnode),dtype = 'int8')
    adj[(indx[0],indx[1])]=1
    new_adj = torch.from_numpy(adj).float()
    return new_adj

def focal_loss(labels, logits, alpha, gamma):

    BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels,reduction = "none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + 
            torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss,dim=1)

    return focal_loss

def get_renode_weight(Pi, gpr, y, data_train_mask, base_weight, max_weight):

    ##hyperparams##
    rn_base_weight = base_weight
    rn_scale_weight = max_weight - base_weight
    assert rn_scale_weight in [0.5 , 0.75, 1.0, 1.25, 1.5]

    ppr_matrix = Pi  #personlized pagerank
    gpr_matrix = torch.tensor(gpr).float() #class-accumulated personlized pagerank

    base_w  = rn_base_weight
    scale_w = rn_scale_weight
    nnode = ppr_matrix.size(0)
    unlabel_mask = data_train_mask.int().ne(1)#unlabled node

    #computing the Totoro values for labeled nodes
    gpr_sum = torch.sum(gpr_matrix,dim=1)
    gpr_rn  = gpr_sum.unsqueeze(1) - gpr_matrix

    label_matrix = F.one_hot(y, gpr_matrix.size(1)).float()
    label_matrix[unlabel_mask] = 0
    rn_matrix = torch.mm(ppr_matrix,gpr_rn).to(label_matrix.device)
    rn_matrix = torch.sum(rn_matrix * label_matrix,dim=1)
    rn_matrix[unlabel_mask] = rn_matrix.max() + 99 #exclude the influence of unlabeled node

    #computing the ReNode Weight
    train_size    = torch.sum(data_train_mask.int()).item()
    totoro_list   = rn_matrix.tolist()
    id2totoro     = {i:totoro_list[i] for i in range(len(totoro_list))}
    sorted_totoro = sorted(id2totoro.items(),key=lambda x:x[1],reverse=False)
    id2rank       = {sorted_totoro[i][0]:i for i in range(nnode)}
    totoro_rank   = [id2rank[i] for i in range(nnode)]

    rn_weight = [(base_w + 0.5 * scale_w * (1 + math.cos(x*1.0*math.pi/(train_size-1)))) for x in totoro_rank]
    rn_weight = torch.from_numpy(np.array(rn_weight)).type(torch.FloatTensor)
    rn_weight = rn_weight.to(data_train_mask.device)
    rn_weight = rn_weight * data_train_mask.float()

    return rn_weight

class IMB_LOSS:
    def __init__(self, loss_name, num_classes, idx_info, factor_focal, factor_cb, device):
        self.loss_name = loss_name
        self.device    = device
        self.cls_num   = num_classes

        #train_size = [len(x) for x in data.train_node]
        train_size = [len(x) for x in idx_info]
        train_size_arr = np.array(train_size)
        train_size_mean = np.mean(train_size_arr)
        train_size_factor = train_size_mean / train_size_arr

        #alpha in re-weight
        self.factor_train = torch.from_numpy(train_size_factor).type(torch.FloatTensor)

        #gamma in focal
        self.factor_focal = factor_focal

        #beta in CB
        weights = torch.from_numpy(np.array([1.0 for _ in range(self.cls_num)])).float()

        if self.loss_name == 'focal':
            weights = self.factor_train

        if self.loss_name == 'cb-softmax':
            beta = factor_cb
            effective_num = 1.0 - np.power(beta, train_size_arr)
            weights = (1.0 - beta) / np.array(effective_num)
            weights = weights / np.sum(weights) * self.cls_num
            weights = torch.tensor(weights).float()

        self.weights = weights.unsqueeze(0).to(device)

    def compute(self,pred,target):

        if self.loss_name == 'ce':
            return F.cross_entropy(pred,target,weight=None,reduction='none')

        elif self.loss_name == 're-weight':
            return F.cross_entropy(pred,target,weight=self.factor_train.to(self.device),reduction='none')

        elif self.loss_name == 'focal':
            labels_one_hot = F.one_hot(target, self.cls_num).type(torch.FloatTensor).to(self.device)
            weights = self.weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
            weights = weights.sum(1)
            weights = weights.unsqueeze(1)
            weights = weights.repeat(1,self.cls_num)

            return focal_loss(labels_one_hot,pred,weights,self.factor_focal)

        elif self.loss_name == 'cb-softmax':
            labels_one_hot = F.one_hot(target, self.cls_num).type(torch.FloatTensor).to(self.device)
            weights = self.weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
            weights = weights.sum(1)
            weights = weights.unsqueeze(1)
            weights = weights.repeat(1,self.cls_num)

            pred = pred.softmax(dim = 1)
            temp_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights,reduction='none') 
            return torch.mean(temp_loss,dim=1)

        else:
            raise Exception("No Implentation Loss")



class ReNode(torch.nn.Module):
    def __init__(self, args, device):
        super(ReNode, self).__init__()
        self.args = args
        self.device = device

    def calculate(self, y, edge_index, idx_info, data_train_mask):
        pagerank_prob = 0.85
        num_nodes = y.shape[0]
        num_cls = y.max().item() + 1

        # calculating the Personalized PageRank Matrix
        pr_prob = 1 - pagerank_prob
        A = index2dense(edge_index, num_nodes)
        A_hat   = A.to(self.device) + torch.eye(A.size(0)).to(self.device) # add self-loop
        D       = torch.diag(torch.sum(A_hat,1))
        D       = D.inverse().sqrt()
        A_hat   = torch.mm(torch.mm(D, A_hat), D)
        Pi = pr_prob * ((torch.eye(A.size(0)).to(self.device) - (1 - pr_prob) * A_hat).inverse())

        # calculating the ReNode Weight
        gpr_matrix = [] # the class-level influence distribution
        for iter_c in range(num_cls):
            #iter_Pi = data.Pi[torch.tensor(target_data.train_node[iter_c]).long()]
            iter_Pi = Pi[idx_info[iter_c].long()] # check! is it same with above line?
            iter_gpr = torch.mean(iter_Pi,dim=0).squeeze()
            gpr_matrix.append(iter_gpr)

        temp_gpr = torch.stack(gpr_matrix,dim=0)
        temp_gpr = temp_gpr.transpose(0,1)
        gpr = temp_gpr
        self.rn_weight =  get_renode_weight(Pi, gpr, y, data_train_mask, self.args.rn_base, self.args.rn_max)[data_train_mask] #ReNode Weight
        self.renode_loss = IMB_LOSS(self.args.loss_name, num_cls, idx_info, self.args.factor_focal, self.args.factor_cb, self.device)

    def forward(self, logits, y, class_num_list=None):

        cls_loss= self.renode_loss.compute(logits, y)
        cls_loss = torch.sum(cls_loss * self.rn_weight.to(self.device)) / cls_loss.size(0)

        return cls_loss
