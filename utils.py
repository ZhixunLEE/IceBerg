import torch
import random
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csc_matrix
from torch_geometric.utils import add_remaining_self_loops, add_self_loops
from torch_scatter import scatter_add


def seed_everything(seed):
    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


def get_confidence(output, with_softmax=False):
    if not with_softmax:
        output = torch.softmax(output, dim=1)

    confidence, pred_label = torch.max(output, dim=1)
    return confidence, pred_label
    

def edge_index_to_sparse_mx(edge_index, num_nodes):
    edge_weight = np.array([1] * len(edge_index[0]))
    adj = csc_matrix((edge_weight, (edge_index[0], edge_index[1])),
                     shape=(num_nodes, num_nodes)).tolil()
    return adj


def process_adj(adj):
    adj.setdiag(1)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize_adj(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj


def normalize_adj(mx):
    """Row-column-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1/2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def feature_propagation(adj, features, K, alpha):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features = features.to(device)
    adj = adj.to(device)
    features_prop = features.clone()
    for _ in range(1, K + 1):
        features_prop = torch.sparse.mm(adj, features_prop)
        features_prop = (1 - alpha) * features_prop + alpha * features
    return features_prop.cpu()


def get_weight(class_num_list, is_reweight=True):
    if is_reweight:
        min_number = np.min(class_num_list)
        class_weight_list = [float(min_number)/float(num) for num in class_num_list]
    else:
        class_weight_list = [1. for _ in class_num_list]
    class_weight = torch.tensor(class_weight_list).type(torch.float32)

    return class_weight


def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

    fill_value = 2. if improved else 1.
    num_nodes = int(edge_index.max()) + 1 if num_nodes is None else num_nodes
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)

    if add_self_loops:
        edge_index, tmp_edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)
        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


