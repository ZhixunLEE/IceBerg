import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu_id', type=int, default=0,
                        help='the number of gpu')
    parser.add_argument('--runs', type=int, default=10,
                        help='the number of repeatition')
    parser.add_argument('--no_pseudo', action='store_true',
                        help='prohibit to use pseudo labeling')
    # Learning setting
    parser.add_argument('--warmup', type=int, default=0,
                        help='warmup for double balancing')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='weight decay')
    parser.add_argument('--epochs',type=int, default=5000,
                        help='training epochs')
    parser.add_argument('--patience', type=int, default=2000,
                        help='patience for training')
    parser.add_argument('--lamda', type=float, default=1)
    parser.add_argument('--mix', action='store_true')
    parser.add_argument('--beta',type=float, default=0)
    
    # For BAT
    parser.add_argument('--bat', action='store_true',
                        help='use BAT')
    
    # For TAM
    parser.add_argument('--tam', action='store_true',
                        help='use TAM')
    parser.add_argument('--tam_alpha', type=float, default=2.5,
                        help='coefficient of ACM')
    parser.add_argument('--tam_beta', type=float, default=0.5,
                        help='coefficient of ADM')
    parser.add_argument('--temp_phi', type=float, default=1.2,
                        help='classwise temperature')
    
    # Hyperparameters for GraphENS
    parser.add_argument('--ens', action='store_true',
                        help='use GraphENS')
    parser.add_argument('--keep_prob', type=float, default=0.01,
                        help='Keeping Probability')
    parser.add_argument('--pred_temp', type=float, default=2,
                        help='Prediction temperature')
    
    # Hyperparameters for GraphSHA
    parser.add_argument('--sha', action='store_true',
                        help='use GraphSHA')
    parser.add_argument('--tau', type=int, default=2,
                        help='Temperature in the softmax function when calculating confidence-based node hardness')
    parser.add_argument('--max', action="store_true", 
                        help='synthesizing to max or mean num of training set. default is mean')
    parser.add_argument('--no_mask', action="store_true", 
                        help='whether to mask the self class in sampling neighbor classes. default is mask')
    parser.add_argument('--gdc', type=str, choices=['ppr', 'hk', 'none'], default='ppr', 
                        help='how to convert to weighted graph')

    # Hyperparameters for Diffusion
    parser.add_argument('--T', type=int, default=10,
                        help='Propogation step')
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='restart probability')
    parser.add_argument('--is_prop', action='store_true',
                        help='propogate at each epoch')
    
    # ReNode
    parser.add_argument('--loss_name', default="re-weight", type=str, help="the training loss") #ce focal re-weight cb-softmax
    parser.add_argument('--factor_focal', default=2.0,    type=float, help="alpha in Focal Loss")
    parser.add_argument('--factor_cb',    default=0.9999, type=float, help="beta  in CB Loss")
    parser.add_argument('--rn_base',    default=0.5, type=float, help="Lower bound of RN")
    parser.add_argument('--rn_max',    default=1.5, type=float, help="Upper bound of RN")

    # Dataset
    parser.add_argument('--dataset', type=str, default='CiteSeer',
                        help='Dataset Name')
    parser.add_argument('--imb_ratio', type=float, default=10,
                        help='Imbalance Ratio')
    parser.add_argument('--label_per_class', type=int, default=20,
                        help='the number of labels per class') # set imb_ratio to 1
    parser.add_argument('--no_feat_norm', action='store_true')

    # Architecture
    parser.add_argument('--net', type=str, default='GCN',
                        help='Architecture name')
    parser.add_argument('--n_layer', type=int, default=2,
                        help='the number of layers')
    parser.add_argument('--feat_dim', type=int, default=256,
                        help='Feature dimension')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    
    # Imbalance Loss
    parser.add_argument('--loss_type', type=str, default='ce',
                        help='Loss type', choices=['ce', 'rw', 'bs', 'rn'])
    args = parser.parse_args()

    return args