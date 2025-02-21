import torch
from torch_geometric.utils import coalesce
from utils import *
from data_utils import *
from nets import *
from losses import *
from baselines import *
from sklearn.metrics import balanced_accuracy_score, f1_score
import torch_geometric.utils as pygutils
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore")

class Trainer():
    def __init__(self, args, data, device):

        self.data = data.to(device)
        self.dataset = args.dataset
        self.args = args
        self.num_features = data.x.shape[1]
        self.num_nodes = data.x.shape[0]
        self.num_cls = data.y.max().item() + 1
        self.num_edges = data.edge_index.shape[1]
        self.seed = 100
        self.device = device

        ## Criterion Selection ##
        if args.loss_type in ['ce', 'rw']:
            self.criterion = CrossEntropy()
        elif args.loss_type == 'bs':
            self.criterion = BalancedSoftmax()
        elif args.loss_type == 'rn':
            self.criterion = ReNode(args, device)
        else:
            raise NotImplementedError("Not Implemented Loss!")
        
        self.criterion_u = RobustBalancedSoftmax(self.num_cls, self.args.beta)
        # self.criterion_u = CrossEntropy()

        if args.ens or args.tam:
            self.aggregator = MeanAggregation()
            self.saliency, self.prev_out = None, None

        if args.sha:
            self.prev_out = None

    def init_model(self):
        if self.args.net == 'GCN':
            model = create_gcn(nfeat=self.num_features, nhid=self.args.feat_dim,
                            nclass=self.num_cls, dropout=self.args.dropout, nlayer=self.args.n_layer)
        elif self.args.net == 'GAT':
            model = create_gat(nfeat=self.num_features, nhid=self.args.feat_dim,
                            nclass=self.num_cls, dropout=self.args.dropout, nlayer=self.args.n_layer)
        elif self.args.net == "SAGE":
            model = create_sage(nfeat=self.num_features, nhid=self.args.feat_dim,
                            nclass=self.num_cls, dropout=self.args.dropout, nlayer=self.args.n_layer)
        elif self.args.net == 'Diff':
            model = create_diff(nfeat=self.num_features, nhid=self.args.feat_dim, nclass=self.num_cls,
                            dropout=self.args.dropout, is_prop=self.args.is_prop, T=self.args.T, alpha=self.args.alpha)
        else:
            raise NotImplementedError("Not Implemented Architecture!")
        
        optimizer = torch.optim.Adam([
            dict(params=model.reg_params, weight_decay=self.args.weight_decay),
            dict(params=model.non_reg_params, weight_decay=0),], lr=self.args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                            factor = 0.5,
                                                            patience = 100,
                                                           verbose=False)
        
        return model, optimizer, scheduler
    
    def init_data_imb(self):
        data_train_mask = self.data_train_mask
        stats = self.data.y[data_train_mask]
        n_data = []
        for i in range(self.num_cls):
            data_num = (stats == i).sum()
            n_data.append(int(data_num.item()))
        idx_info = get_idx_info(self.data.y, self.num_cls, data_train_mask)
        class_num_list = n_data

        imb_class_num = self.num_cls // 2
        new_class_num_list = []
        max_num = np.max(class_num_list[:self.num_cls-imb_class_num])
        for i in range(self.num_cls):
            if self.args.imb_ratio > 1 and i > self.num_cls-1-imb_class_num: #only imbalance the last classes
                new_class_num_list.append(min(int(max_num*(1./self.args.imb_ratio)), class_num_list[i]))
            elif self.args.imb_ratio > 1:
                new_class_num_list.append(class_num_list[i])
            else:
                new_class_num_list.append(self.args.label_per_class)
        class_num_list = new_class_num_list

        if self.args.imb_ratio >= 1:
            self.data_train_mask, self.idx_info = split_semi_dataset(len(self.data.x), n_data, self.num_cls, class_num_list, idx_info, self.device)
        # self.data_unlabel_mask = ~(self.data_train_mask | self.data_val_mask | self.data_test_mask)
        self.data_unlabel_mask = ~self.data_train_mask

        min_number = np.min(class_num_list)
        self.class_weight_list = np.array([float(min_number)/float(num) for num in class_num_list]) * self.args.imb_ratio
        self.class_num_list = torch.tensor(class_num_list)
        self.class_num_list_u = torch.zeros_like(self.class_num_list)
        if self.args.loss_type == 'rw':
            class_weight = get_weight(class_num_list).to(self.device)
            self.criterion.class_weight = class_weight
        elif self.args.loss_type == 'rn':
            self.criterion.calculate(self.data.y, self.data.edge_index, self.idx_info, self.data_train_mask)

    def init_random(self, train_ratio=0.1, val_ratio=0.4):
        y_len = self.data.y.size(0)
        indices = torch.randperm(y_len)

        train_index = indices[:int(y_len * train_ratio)]
        val_index = indices[int(y_len * train_ratio):int(y_len * (train_ratio + val_ratio))]
        test_index = indices[int(y_len * (train_ratio + val_ratio)):]

        self.data_train_mask = pygutils.index_to_mask(train_index, size=self.num_nodes).to(self.device)
        self.data_val_mask = pygutils.index_to_mask(val_index, size=self.num_nodes).to(self.device)
        self.data_test_mask = pygutils.index_to_mask(test_index, size=self.num_nodes).to(self.device)
        self.data_unlabel_mask = ~self.data_train_mask

        self.idx_info = get_idx_info(self.data.y, self.num_cls, self.data_train_mask)
        self.class_num_list = torch.tensor([(self.data.y[self.data_train_mask] == i).sum().item() for i in range(self.num_cls)])

    def init_data_few(self, val_per_class=30):
        indices = []
        y = self.data.y
        
        for i in range(self.num_cls):
            index = (y == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

        train_index = torch.cat([i[:self.args.label_per_class] for i in indices], dim=0)
        val_index = torch.cat([i[self.args.label_per_class:self.args.label_per_class+val_per_class] for i in indices], dim=0)
        test_index = torch.cat([i[self.args.label_per_class+val_per_class:] for i in indices], dim=0)

        self.data_train_mask = pygutils.index_to_mask(train_index, size=self.num_nodes)
        self.data_val_mask = pygutils.index_to_mask(val_index, size=self.num_nodes)
        self.data_test_mask = pygutils.index_to_mask(test_index, size=self.num_nodes)
        self.data_unlabel_mask = ~self.data_train_mask

        self.class_num_list = torch.tensor([self.args.label_per_class for _ in range(self.num_cls)])

    def test_epoch(self):
        model = self.model
        y = self.data.y
        if self.args.net == 'Diff':
            x = self.x_prop
        else:
            x = self.data.x
        edge_index = self.data.edge_index
        
        with torch.no_grad():
            model.eval()
            logits = model(x, edge_index)

        accs, baccs, f1s = [], [], []
        for _, mask in enumerate([self.data_train_mask, self.data_val_mask, self.data_test_mask]):
            pred = logits[mask].max(1)[1]
            y_pred = pred.cpu().numpy()
            y_true = y[mask].cpu().numpy()
            acc = pred.eq(y[mask]).sum().item() / mask.sum().item()
            bacc = balanced_accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='macro')

            accs.append(acc)
            baccs.append(bacc)
            f1s.append(f1)
        
        return accs, baccs, f1s
    
    def get_pred_label(self):
        model = self.model
        if self.args.net == 'Diff':
            x = self.x_prop
        else:
            x = self.data.x
        edge_index = self.data.edge_index

        with torch.no_grad():
            model.eval()
            logits = model(x, edge_index)

        confidence, self.pred_label = get_confidence(logits, with_softmax=False)
        self.pred_label[self.data_train_mask] = self.data.y[self.data_train_mask]
        threshold = confidence[self.data_unlabel_mask].mean().item()
        self.pseudo_mask = confidence.ge(threshold) & self.data_unlabel_mask

        self.utilize_list.append(self.pseudo_mask.sum().item() / self.data_unlabel_mask.sum().item())
        if self.pseudo_mask.sum().item() == 0:
            self.pseudo_acc.append(0)
        else:
            self.pseudo_acc.append((self.pred_label[self.pseudo_mask] == self.data.y[self.pseudo_mask]).sum().item() / self.pseudo_mask.sum().item())

    def get_prop_feature(self):
        edge_index = self.data.edge_index
        x = self.data.x

        adj = edge_index_to_sparse_mx(edge_index.cpu(), self.num_nodes)
        adj = process_adj(adj)
        x_prop = feature_propagation(adj, x, self.args.T, self.args.alpha)
        self.x_prop = x_prop.to(self.device)

    # For GraphENS
    def backward_hook(self, module, grad_input, grad_output):
        self.saliency = grad_input[0].data

    def gens_sampling(self, epoch, y, data_train_mask, x, edge_index):
        model = self.model
        criterion = self.criterion
        num_nodes = x.shape[0]

        # Hook saliency map of input features
        model.conv1.lin.register_backward_hook(self.backward_hook)
        # model.linear1.register_backward_hook(self.backward_hook)

        # Sampling source and destination nodes
        sampling_src_idx, sampling_dst_idx = sampling_idx_individual_dst(self.class_num_list, self.idx_info, self.device)
        beta = torch.distributions.beta.Beta(2, 2)
        lam = beta.sample((len(sampling_src_idx),) ).unsqueeze(1)
        # self.saliency = None
        ori_saliency = self.saliency[:num_nodes] if (self.saliency != None) else None

        # Augment nodes
        if epoch > self.args.warmup:
            with torch.no_grad():
                self.prev_out = self.aggregator(self.prev_out, edge_index)
                self.prev_out = F.softmax(self.prev_out / self.args.pred_temp, dim=1).detach().clone()
            new_edge_index, dist_kl = neighbor_sampling_ens(num_nodes, edge_index, sampling_src_idx, sampling_dst_idx,
                                    self.neighbor_dist_list, self.prev_out)
            new_x = saliency_mixup(x, sampling_src_idx, sampling_dst_idx, lam, ori_saliency, dist_kl = dist_kl, keep_prob=self.args.keep_prob)
        else:
            new_edge_index = duplicate_neighbor(num_nodes, edge_index, sampling_src_idx)
            dist_kl, ori_saliency = None, None
            new_x = saliency_mixup(x, sampling_src_idx, sampling_dst_idx, lam, ori_saliency, dist_kl=dist_kl)
        new_x.requires_grad = True

        # Get predictions
        output = model(new_x, new_edge_index)
        self.prev_out = (output[:num_nodes]).detach().clone() # logit propagation

        ## Train_mask modification ##
        add_num = output.shape[0] - num_nodes
        new_train_mask = torch.ones(add_num, dtype=torch.bool, device=self.device)
        new_train_mask = torch.cat((data_train_mask, new_train_mask), dim=0)

        ## Label modification ##
        _new_y = y[sampling_src_idx].clone()
        new_y = torch.cat((y[data_train_mask], _new_y), dim =0)

        if self.args.tam:
            output = adjust_output(self.args, output, new_edge_index, torch.cat((y,_new_y),dim=0), new_train_mask, \
                                    self.aggregator, self.class_num_list, epoch)

        loss = criterion(output[new_train_mask], new_y)

        return loss, output[:num_nodes]
    
    def graphsha_sampling(self, epoch, y, data_train_mask, x, edge_index):
        model = self.model
        criterion = self.criterion
        num_nodes = x.shape[0]

        if epoch > self.args.wamrup:
            # identifying source samples
            train_idx = data_train_mask.nonzero().squeeze()
            prev_out_local = self.prev_out[train_idx]
            train_idx_list = train_idx.cpu().tolist()
            local2global = {i:train_idx_list[i] for i in range(len(train_idx_list))}
            global2local = dict([val, key] for key, val in local2global.items())
            idx_info_list = [item.cpu().tolist() for item in self.idx_info]
            idx_info_local = [torch.tensor(list(map(global2local.get, cls_idx))) for cls_idx in idx_info_list]

            sampling_src_idx, sampling_dst_idx = sampling_node_source(self.class_num_list, prev_out_local, idx_info_local, 
                                                                      train_idx, self.args.tau, self.args.max, self.args.no_mask) 
            # semimxup
            new_edge_index = neighbor_sampling_sha(num_nodes, edge_index, sampling_src_idx, 
                                               self.neighbor_dist_list)
            beta = torch.distributions.beta.Beta(1, 100)
            lam = beta.sample((len(sampling_src_idx),) ).unsqueeze(1)
            new_x = saliency_mixup(x, sampling_src_idx, sampling_dst_idx, lam)

        else:
            sampling_src_idx, sampling_dst_idx = sampling_idx_individual_dst(self.class_num_list, self.idx_info, self.device)
            beta = torch.distributions.beta.Beta(2, 2)
            lam = beta.sample((len(sampling_src_idx),) ).unsqueeze(1)
            new_edge_index = duplicate_neighbor(num_nodes, edge_index, sampling_src_idx)
            new_x = saliency_mixup(x, sampling_src_idx, sampling_dst_idx, lam)

        output = model(new_x, new_edge_index)
        self.prev_out = (output[:num_nodes]).detach().clone()
        add_num = output.shape[0] - data_train_mask.shape[0]
        new_train_mask = torch.ones(add_num, dtype=torch.bool, device= x.device)
        new_train_mask = torch.cat((data_train_mask, new_train_mask), dim =0)
        _new_y = y[sampling_src_idx].clone()
        new_y = torch.cat((y[data_train_mask], _new_y),dim =0)

        if self.args.tam:
            output = adjust_output(self.args, output, new_edge_index, torch.cat((y,_new_y),dim=0), new_train_mask, \
                                    self.aggregator, self.class_num_list, epoch)

        loss = criterion(output[new_train_mask], new_y)

        return loss, output[:self.num_nodes]
    
    def mixup(self, epoch, y, data_train_mask, x, edge_index):
        model = self.model
        criterion = self.criterion
        num_nodes = x.shape[0]

        sampling_src_idx, sampling_dst_idx = sampling_idx_individual_dst(self.class_num_list, self.idx_info, self.device)
        beta = torch.distributions.beta.Beta(2, 2)
        lam = beta.sample((len(sampling_src_idx),) ).unsqueeze(1)
        new_edge_index = duplicate_neighbor(num_nodes, edge_index, sampling_src_idx)
        new_x = saliency_mixup(x, sampling_src_idx, sampling_dst_idx, lam)

        output = model(new_x, new_edge_index)
        self.prev_out = (output[:num_nodes]).detach().clone()
        add_num = output.shape[0] - data_train_mask.shape[0]
        new_train_mask = torch.ones(add_num, dtype=torch.bool, device= x.device)
        new_train_mask = torch.cat((data_train_mask, new_train_mask), dim =0)
        _new_y = y[sampling_src_idx].clone()
        new_y = torch.cat((y[data_train_mask], _new_y),dim =0)

        if self.args.tam:
            output = adjust_output(self.args, output, new_edge_index, torch.cat((y,_new_y),dim=0), new_train_mask, \
                                    self.aggregator, self.class_num_list, epoch)

        loss = criterion(output[new_train_mask], new_y)

        return loss, output[:self.num_nodes]


    def train_epoch(self, epoch):
        model = self.model
        optimizer = self.optimizer
        y = self.data.y
        criterion = self.criterion
        criterion_u = self.criterion_u
        data_train_mask = self.data_train_mask
        if self.args.net == 'Diff' and (not (self.args.ens or self.args.sha or self.args.mix)):
            x = self.x_prop
        else:
            x = self.data.x
        edge_index = self.data.edge_index
        loss = 0

        if self.args.bat:
            x, edge_index, _ = self.augmenter.augment(model, x, edge_index)
            y, data_train_mask = self.augmenter.adapt_labels_and_train_mask(y, data_train_mask)
            idx_info = get_idx_info(y, self.num_cls, data_train_mask)
            if self.args.loss_type == 'rn':
                self.criterion.calculate(y, edge_index, idx_info, data_train_mask)

        model.train()
        optimizer.zero_grad()

        if self.args.ens:
            loss_l, output = self.gens_sampling(epoch, y, data_train_mask, x, edge_index)
            loss += loss_l
        elif self.args.sha:
            loss_l, output = self.graphsha_sampling(epoch, y, data_train_mask, x, edge_index)
            loss += loss_l
        elif self.args.mix:
            loss_l, output = self.mixup(epoch, y, data_train_mask, x, edge_index)
            loss += loss_l
        else:
            output = model(x, edge_index)
            if self.args.tam:
                output = adjust_output(self.args, output, edge_index, y, data_train_mask, \
                                        self.aggregator, self.class_num_list, epoch)
            
            loss += criterion(output[data_train_mask], y[data_train_mask], self.class_num_list)

        # Double Balancing
        if not self.args.no_pseudo and epoch >= self.args.warmup:
            self.class_num_list_u = torch.tensor([(self.pred_label[self.pseudo_mask] == i).sum().item() for i in range(self.num_cls)])
            loss += criterion_u(output[self.pseudo_mask], self.pred_label[self.pseudo_mask], self.class_num_list_u) * self.args.lamda
        
        loss.backward()
        optimizer.step()

    def train(self, r):

        ## Fix seed ##
        seed_everything(self.seed + r)   

        self.utilize_list = []  
        self.pseudo_acc = [] 
 

        if self.dataset == 'arxiv':
            self.data.y = self.data.y.reshape(1, -1)[0]

        if self.dataset in ['cs', 'physics', 'arxiv']:
            self.init_data_few()
        elif self.dataset == 'corafull':
            self.init_random()
        elif self.dataset in ['roman-empire', 'penn94']:
            self.data_train_mask = self.data.train_mask[:,r%10].clone()
            self.data_val_mask = self.data.val_mask[:,r%10].clone()
            self.data_test_mask = self.data.test_mask[:,r%10].clone()
        else:
            self.data_train_mask = self.data.train_mask.clone()
            self.data_val_mask = self.data.val_mask.clone()
            self.data_test_mask = self.data.test_mask.clone()

        if self.dataset == 'corafull':
            pass
        elif self.args.imb_ratio > 1:
            self.init_data_imb()
        else:
            self.init_data_few()

        if self.args.bat:
            augmenter = BatAugmenter(mode='bat1', random_state=self.seed+r)
            self.augmenter = augmenter.init_with_data(self.data, self.data_train_mask)

        if self.args.net == 'Diff':
            self.get_prop_feature()

        if self.args.ens:
            self.neighbor_dist_list = get_ins_neighbor_dist(self.data.y.size(0), self.data.edge_index, self.data_train_mask, self.device)
        if self.args.sha:
            assert not (self.args.ens and self.args.sha)
            if self.args.gdc=='ppr':
                self.neighbor_dist_list = get_PPR_adj(self.data.x, self.data.edge_index, alpha=0.05, k=128, eps=None)
            elif self.args.gdc=='hk':
                self.neighbor_dist_list = get_heat_adj(self.data.x, self.data.edge_index, t=5.0, k=None, eps=0.0001)
            elif self.args.gdc=='none':
                self.neighbor_dist_list = get_ins_neighbor_dist(self.data.y.size(0), self.data.edge_index, self.data_train_mask, self.device)

        self.model, self.optimizer, self.scheduler = self.init_model()

        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)

        best_val_acc_f1, test_bacc, test_acc, p, best_epoch = 0, 0, 0, 0, 0
        for epoch in range(1, self.args.epochs+1):
            if not self.args.no_pseudo:
                self.get_pred_label()
            self.train_epoch(epoch)

            accs, baccs, f1s = self.test_epoch()
            if epoch % 10 == 0:
                print("Iter: {} | Epoch: {} | Test Acc: {:.4f} | Test BAcc: {:.4f} | Test F1: {:.4f} | Best Acc: {:.4f} | Best BAcc: {:.4f} | Best Epoch: {}".format(
                    r, epoch, accs[2], baccs[2], f1s[2], test_acc, test_bacc, best_epoch
                ))

            val_acc, val_f1 = accs[1], f1s[1]
            val_acc_f1 = (val_acc + val_f1) / 2.
            if val_acc_f1 > best_val_acc_f1:
                best_val_acc_f1 = val_acc_f1
                test_acc = accs[2]
                test_bacc = baccs[2]
                test_f1 = f1s[2]
                p = 0
                best_epoch = epoch
            else:
                p += 1
                if p == self.args.patience: break
        
        return best_val_acc_f1, test_acc, test_bacc, test_f1