import torch
import numpy as np
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, to_dense_batch
from torch_scatter import scatter_add
from scipy.linalg import expm


@torch.no_grad()
def duplicate_neighbor(total_node, edge_index, sampling_src_idx):
    """
    Duplicate edges of source nodes for sampled nodes.
    Input:
        total_node:         # of nodes; scalar
        edge_index:         Edge index; [2, # of edges]
        sampling_src_idx:   Source node index for augmented nodes; [# of augmented nodes]
    Output:
        new_edge_index:     original_edge_index + duplicated_edge_index
    """
    device = edge_index.device

    # Assign node index for augmented nodes
    row, col = edge_index[0], edge_index[1]
    row, sort_idx = torch.sort(row)
    col = col[sort_idx]
    degree = scatter_add(torch.ones_like(col), col)
    new_row =(torch.arange(len(sampling_src_idx)).to(device)+ total_node).repeat_interleave(degree[sampling_src_idx])
    temp = scatter_add(torch.ones_like(sampling_src_idx), sampling_src_idx).to(device)

    # Duplicate the edges of source nodes
    node_mask = torch.zeros(total_node, dtype=torch.bool).to(device)
    unique_src = torch.unique(sampling_src_idx)
    node_mask[unique_src] = True
    row_mask = node_mask[row]
    edge_mask = col[row_mask]
    b_idx = torch.arange(len(unique_src)).to(device).repeat_interleave(degree[unique_src])
    edge_dense, _ = to_dense_batch(edge_mask, b_idx, fill_value=-1)
    if len(temp[temp!=0]) != edge_dense.shape[0]:
        cut_num =len(temp[temp!=0]) - edge_dense.shape[0]
        cut_temp = temp[temp!=0][:-cut_num]
    else:
        cut_temp = temp[temp!=0]
    edge_dense  = edge_dense.repeat_interleave(cut_temp, dim=0)
    new_col = edge_dense[edge_dense!= -1]
    inv_edge_index = torch.stack([new_col, new_row], dim=0)
    new_edge_index = torch.cat([edge_index, inv_edge_index], dim=1)

    return new_edge_index


def saliency_mixup(x, sampling_src_idx, sampling_dst_idx, lam, saliency=None,
                   dist_kl = None, keep_prob = 0.3):
    """
    Saliency-based node mixing - Mix node features
    Input:
        x:                  Node features; [# of nodes, input feature dimension]
        sampling_src_idx:   Source node index for augmented nodes; [# of augmented nodes]
        sampling_dst_idx:   Target node index for augmented nodes; [# of augmented nodes]
        lam:                Sampled mixing ratio; [# of augmented nodes, 1]
        saliency:           Saliency map of input feature; [# of nodes, input feature dimension]
        dist_kl:             KLD between source node and target node predictions; [# of augmented nodes, 1]
        keep_prob:          Ratio of keeping source node feature; scalar
    Output:
        new_x:              [# of original nodes + # of augmented nodes, feature dimension]
    """
    ## Mixup ##
    new_src = x[sampling_src_idx.to(x.device), :].clone()
    new_dst = x[sampling_dst_idx.to(x.device), :].clone()
    lam = lam.to(x.device)

    # Saliency Mixup
    if saliency != None:
        node_dim = saliency.shape[1]
        saliency_dst = saliency[sampling_dst_idx].abs()
        saliency_dst += 1e-10
        saliency_dst /= torch.sum(saliency_dst, dim=1).unsqueeze(1)

        K = int(node_dim * keep_prob)
        mask_idx = torch.multinomial(saliency_dst, K)
        lam = lam.expand(-1,node_dim).clone()
        if dist_kl != None: # Adaptive
            kl_mask = (torch.sigmoid(dist_kl/3.) * K).squeeze().long()
            idx_matrix = (torch.arange(K).unsqueeze(dim=0).to(kl_mask.device) >= kl_mask.unsqueeze(dim=1))
            zero_repeat_idx = mask_idx[:,0:1].repeat(1,mask_idx.size(1))
            mask_idx[idx_matrix] = zero_repeat_idx[idx_matrix]

        lam[torch.arange(lam.shape[0]).unsqueeze(1), mask_idx] = 1.
    mixed_node = lam * new_src + (1-lam) * new_dst
    new_x = torch.cat([x, mixed_node], dim =0)
    return new_x


@torch.no_grad()
def get_ins_neighbor_dist(num_nodes, edge_index, train_mask, device):
    """
    Compute adjacent node distribution.
    """
    ## Utilize GPU ##
    train_mask = train_mask.clone().to(device)
    edge_index = edge_index.clone().to(device)
    row, col = edge_index[0], edge_index[1]

    # Compute neighbor distribution
    neighbor_dist_list = []
    for j in range(num_nodes):
        neighbor_dist = torch.zeros(num_nodes, dtype=torch.float32).to(device)

        idx = row[(col==j)]
        neighbor_dist[idx] = neighbor_dist[idx] + 1
        neighbor_dist_list.append(neighbor_dist)

    neighbor_dist_list = torch.stack(neighbor_dist_list,dim=0)
    neighbor_dist_list = F.normalize(neighbor_dist_list,dim=1,p=1)

    return neighbor_dist_list

import pdb
@torch.no_grad()
def sampling_idx_individual_dst(class_num_list, idx_info, device):
    """
    Samples source and target nodes
    """
    # Selecting src & dst nodes
    max_num, n_cls = max(class_num_list), len(class_num_list)
    sampling_list = max_num * torch.ones(n_cls) - torch.tensor(class_num_list)
    # pdb.set_trace()
    new_class_num_list = torch.tensor(class_num_list).to(device)

    # Compute # of source nodes
    sampling_src_idx =[cls_idx[torch.randint(len(cls_idx),(int(samp_num.item()),))]
                        for cls_idx, samp_num in zip(idx_info, sampling_list)]
    sampling_src_idx = torch.cat(sampling_src_idx)

    # Generate corresponding destination nodes
    prob = torch.log(new_class_num_list.float())/ new_class_num_list.float()
    prob = prob.repeat_interleave(new_class_num_list.long())
    temp_idx_info = torch.cat(idx_info)
    dst_idx = torch.multinomial(prob, sampling_src_idx.shape[0], True)
    sampling_dst_idx = temp_idx_info[dst_idx]

    # Sorting src idx with corresponding dst idx
    sampling_src_idx, sorted_idx = torch.sort(sampling_src_idx)
    sampling_dst_idx = sampling_dst_idx[sorted_idx]

    return sampling_src_idx, sampling_dst_idx

class MeanAggregation(MessagePassing):
    def __init__(self):
        super(MeanAggregation, self).__init__(aggr='mean')

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 4-5: Start propagating messages.
        return self.propagate(edge_index, x=x)
    

def get_dist_kl(prev_out, sampling_src_idx, sampling_dst_idx):
    """
    Compute KL divergence
    """
    device = prev_out.device
    dist_kl = F.kl_div(torch.log(prev_out[sampling_dst_idx.to(device)]), prev_out[sampling_src_idx.to(device)], \
                    reduction='none').sum(dim=1,keepdim=True)
    dist_kl[dist_kl<0] = 0
    return dist_kl
    

@torch.no_grad()
def neighbor_sampling_ens(total_node, edge_index, sampling_src_idx, sampling_dst_idx,
        neighbor_dist_list, prev_out, train_node_mask=None):
    """
    Neighbor Sampling - Mix adjacent node distribution and samples neighbors from it
    Input:
        total_node:         # of nodes; scalar
        edge_index:         Edge index; [2, # of edges]
        sampling_src_idx:   Source node index for augmented nodes; [# of augmented nodes]
        sampling_dst_idx:   Target node index for augmented nodes; [# of augmented nodes]
        neighbor_dist_list: Adjacent node distribution of whole nodes; [# of nodes, # of nodes]
        prev_out:           Model prediction of the previous step; [# of nodes, n_cls]
        train_node_mask:    Mask for not removed nodes; [# of nodes]
    Output:
        new_edge_index:     original edge index + sampled edge index
        dist_kl:            kl divergence of target nodes from source nodes; [# of sampling nodes, 1]
    """
    ## Exception Handling ##
    device = edge_index.device
    n_candidate = 1
    sampling_src_idx = sampling_src_idx.clone().to(device)

    # Find the nearest nodes and mix target pool
    if prev_out is not None:
        sampling_dst_idx = sampling_dst_idx.clone().to(device)
        dist_kl = get_dist_kl(prev_out, sampling_src_idx, sampling_dst_idx)
        ratio = F.softmax(torch.cat([dist_kl.new_zeros(dist_kl.size(0),1), -dist_kl], dim=1), dim=1)
        mixed_neighbor_dist = ratio[:,:1] * neighbor_dist_list[sampling_src_idx]
        for i in range(n_candidate):
            mixed_neighbor_dist += ratio[:,i+1:i+2] * neighbor_dist_list[sampling_dst_idx.unsqueeze(dim=1)[:,i]]
    else:
        mixed_neighbor_dist = neighbor_dist_list[sampling_src_idx]

    # Compute degree
    col = edge_index[1]
    degree = scatter_add(torch.ones_like(col), col)
    if len(degree) < total_node:
        degree = torch.cat([degree, degree.new_zeros(total_node-len(degree))],dim=0)
    if train_node_mask is None:
        train_node_mask = torch.ones_like(degree,dtype=torch.bool)
    degree_dist = scatter_add(torch.ones_like(degree[train_node_mask]), degree[train_node_mask]).to(device).type(torch.float32)

    # Sample degree for augmented nodes
    prob = degree_dist.unsqueeze(dim=0).repeat(len(sampling_src_idx),1)
    aug_degree = torch.multinomial(prob, 1).to(device).squeeze(dim=1) # (m)
    max_degree = degree.max().item() + 1
    aug_degree = torch.min(aug_degree, degree[sampling_src_idx])

    # Sample neighbors
    new_tgt = torch.multinomial(mixed_neighbor_dist + 1e-12, max_degree)
    tgt_index = torch.arange(max_degree).unsqueeze(dim=0).to(device)
    new_col = new_tgt[(tgt_index - aug_degree.unsqueeze(dim=1) < 0)]
    new_row = (torch.arange(len(sampling_src_idx)).to(device)+ total_node)
    new_row = new_row.repeat_interleave(aug_degree)
    inv_edge_index = torch.stack([new_col, new_row], dim=0)
    new_edge_index = torch.cat([edge_index, inv_edge_index], dim=1)

    return new_edge_index, dist_kl

@torch.no_grad()
def neighbor_sampling_sha(total_node, edge_index, sampling_src_idx,
        neighbor_dist_list, train_node_mask=None):
    """
    Neighbor Sampling - Mix adjacent node distribution and samples neighbors from it
    Input:
        total_node:         # of nodes; scalar
        edge_index:         Edge index; [2, # of edges]
        sampling_src_idx:   Source node index for augmented nodes; [# of augmented nodes]
        sampling_dst_idx:   Target node index for augmented nodes; [# of augmented nodes]
        neighbor_dist_list: Adjacent node distribution of whole nodes; [# of nodes, # of nodes]
        prev_out:           Model prediction of the previous step; [# of nodes, n_cls]
        train_node_mask:    Mask for not removed nodes; [# of nodes]
    Output:
        new_edge_index:     original edge index + sampled edge index
        dist_kl:            kl divergence of target nodes from source nodes; [# of sampling nodes, 1]
    """
    ## Exception Handling ##
    device = edge_index.device
    sampling_src_idx = sampling_src_idx.clone().to(device)
    
    # Find the nearest nodes and mix target pool
    mixed_neighbor_dist = neighbor_dist_list[sampling_src_idx]

    # Compute degree
    col = edge_index[1]
    degree = scatter_add(torch.ones_like(col), col)
    if len(degree) < total_node:
        degree = torch.cat([degree, degree.new_zeros(total_node-len(degree))],dim=0)
    if train_node_mask is None:
        train_node_mask = torch.ones_like(degree,dtype=torch.bool)
    degree_dist = scatter_add(torch.ones_like(degree[train_node_mask]), degree[train_node_mask]).to(device).type(torch.float32)

    # Sample degree for augmented nodes
    prob = degree_dist.unsqueeze(dim=0).repeat(len(sampling_src_idx),1)
    aug_degree = torch.multinomial(prob, 1).to(device).squeeze(dim=1) # (m)
    max_degree = degree.max().item() + 1
    aug_degree = torch.min(aug_degree, degree[sampling_src_idx])

    # Sample neighbors
    new_tgt = torch.multinomial(mixed_neighbor_dist + 1e-12, max_degree)
    tgt_index = torch.arange(max_degree).unsqueeze(dim=0).to(device)
    new_col = new_tgt[(tgt_index - aug_degree.unsqueeze(dim=1) < 0)]
    new_row = (torch.arange(len(sampling_src_idx)).to(device)+ total_node)
    new_row = new_row.repeat_interleave(aug_degree)
    inv_edge_index = torch.stack([new_col, new_row], dim=0)
    new_edge_index = torch.cat([edge_index, inv_edge_index], dim=1)

    return new_edge_index

@torch.no_grad()
def sampling_node_source(class_num_list, prev_out_local, idx_info_local, train_idx, tau=2, max_flag=False, no_mask=False):
    max_num, n_cls = max(class_num_list), len(class_num_list) 
    if not max_flag: # mean
        max_num = sum(class_num_list) / n_cls
    sampling_list = max_num * torch.ones(n_cls) - torch.tensor(class_num_list)

    prev_out_local = F.softmax(prev_out_local/tau, dim=1)
    prev_out_local = prev_out_local.cpu() 

    src_idx_all = []
    dst_idx_all = []
    for cls_idx, num in enumerate(sampling_list):
        num = int(num.item())
        if num <= 0: 
            continue

        # first sampling
        prob = 1 - prev_out_local[idx_info_local[cls_idx]][:,cls_idx].squeeze() 
        src_idx_local = torch.multinomial(prob + 1e-12, num, replacement=True) 
        src_idx = train_idx[idx_info_local[cls_idx][src_idx_local]] 

        # second sampling
        conf_src = prev_out_local[idx_info_local[cls_idx][src_idx_local]] 
        if not no_mask:
            conf_src[:,cls_idx] = 0
        neighbor_cls = torch.multinomial(conf_src + 1e-12, 1).squeeze().tolist() 

        # third sampling
        neighbor = [prev_out_local[idx_info_local[cls]][:,cls_idx] for cls in neighbor_cls] 
        dst_idx = []
        for i, item in enumerate(neighbor):
            dst_idx_local = torch.multinomial(item + 1e-12, 1)[0] 
            dst_idx.append(train_idx[idx_info_local[neighbor_cls[i]][dst_idx_local]])
        dst_idx = torch.tensor(dst_idx).to(src_idx.device)

        src_idx_all.append(src_idx)
        dst_idx_all.append(dst_idx)
    
    src_idx_all = torch.cat(src_idx_all)
    dst_idx_all = torch.cat(dst_idx_all)
    
    return src_idx_all, dst_idx_all
    
# ==== For GraphSHA gdc ======
def get_adj_matrix(x, edge_index) -> np.ndarray:
    num_nodes = x.shape[0]
    adj_matrix = np.zeros(shape=(num_nodes, num_nodes))
    for i, j in zip(edge_index[0], edge_index[1]):
        adj_matrix[i, j] = 1.
    return adj_matrix

def get_heat_matrix(
        adj_matrix: np.ndarray,
        t: float = 5.0) -> np.ndarray:
    num_nodes = adj_matrix.shape[0]
    A_tilde = adj_matrix + np.eye(num_nodes)
    D_tilde = np.diag(1/np.sqrt(A_tilde.sum(axis=1)))
    H = D_tilde @ A_tilde @ D_tilde
    return expm(-t * (np.eye(num_nodes) - H))

def get_top_k_matrix(A: np.ndarray, k: int = 128) -> np.ndarray:
    num_nodes = A.shape[0]
    row_idx = np.arange(num_nodes)
    A[A.argsort(axis=0)[:num_nodes - k], row_idx] = 0.
    norm = A.sum(axis=0)
    norm[norm <= 0] = 1 # avoid dividing by zero
    return A/norm

def get_ppr_matrix(
        adj_matrix: np.ndarray,
        alpha: float = 0.05) -> np.ndarray:
    num_nodes = adj_matrix.shape[0]
    A_tilde = adj_matrix + np.eye(num_nodes)
    D_tilde = np.diag(1/np.sqrt(A_tilde.sum(axis=1)))
    H = D_tilde @ A_tilde @ D_tilde
    return alpha * np.linalg.inv(np.eye(num_nodes) - (1 - alpha) * H) # numpy, [N, N]

def get_clipped_matrix(A: np.ndarray, eps: float = 0.01) -> np.ndarray:
    num_nodes = A.shape[0]
    A[A < eps] = 0.
    norm = A.sum(axis=0)
    norm[norm <= 0] = 1 # avoid dividing by zero
    return A/norm

def get_PPR_adj(x, edge_index, alpha=0.05, k=None, eps=None):
    assert ((k==None and eps!=None) or (k!=None and eps==None))

    adj_matrix = get_adj_matrix(x, edge_index)
    ppr_matrix = get_ppr_matrix(adj_matrix, alpha=alpha)

    if k!=None:
        ppr_matrix = get_top_k_matrix(ppr_matrix, k=k)
    elif eps!=None:
        ppr_matrix = get_clipped_matrix(ppr_matrix, eps=eps)
    else:
        raise ValueError
    
    return torch.tensor(ppr_matrix).float().to(x.device)

def get_heat_adj(x, edge_index, t=5.0, k=None, eps=None):
    assert ((k==None and eps!=None) or (k!=None and eps==None))
    adj_matrix = get_adj_matrix(x, edge_index)
    heat_matrix = get_heat_matrix(adj_matrix, t=t)

    if k!=None:
        heat_matrix = get_top_k_matrix(heat_matrix, k=k)
    elif eps!=None:
        heat_matrix = get_clipped_matrix(heat_matrix, eps=eps)
    else:
        raise ValueError
    
    return torch.tensor(heat_matrix).float().to(x.device)
