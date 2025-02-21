import os.path as osp
import torch
import os
from nets import *
from data_utils import *
from args import parse_args
from trainer import Trainer
import statistics


if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')

    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', args.dataset)
    dataset = get_dataset(args.dataset, path, split_type='public', no_feat_norm=args.no_feat_norm)
    data = dataset[0]

    if args.dataset == 'arxiv':
        data.y = data.y.reshape(1, -1)[0]

    setting_log = "IR: {}, net: {}, n_layer: {}, feat_dim: {}, lr: {}, weight_decay: {}, dropout: {} \n\
            TAM: {}, BAT: {}, ENS: {}, SHA: {}, loss_type: {}, no_pseudo: {}, label_per_class: {}, T: {}, alpha: {}".format(
            str(args.imb_ratio), args.net, str(args.n_layer), str(args.feat_dim), str(args.lr),
            str(args.weight_decay), str(args.dropout), str(args.tam), str(args.bat), str(args.ens), str(args.sha), str(args.loss_type),
            str(args.no_pseudo), str(args.label_per_class), str(args.T), str(args.alpha))

    avg_val_acc_f1, avg_test_acc, avg_test_bacc, avg_test_f1 = [], [], [], []

    trainer = Trainer(args, data, device)

    for r in range(args.runs):
        best_val_acc_f1, test_acc, test_bacc, test_f1 = trainer.train(r)
        
        avg_val_acc_f1.append(best_val_acc_f1)
        avg_test_acc.append(test_acc)
        avg_test_bacc.append(test_bacc)
        avg_test_f1.append(test_f1)

    ## Calculate statistics ##
    acc_CI =  (statistics.stdev(avg_test_acc) / (args.runs ** (1/2)))
    bacc_CI =  (statistics.stdev(avg_test_bacc) / (args.runs ** (1/2)))
    f1_CI =  (statistics.stdev(avg_test_f1) / (args.runs ** (1/2)))
    avg_acc = statistics.mean(avg_test_acc)
    avg_bacc = statistics.mean(avg_test_bacc)
    avg_f1 = statistics.mean(avg_test_f1)
    avg_val_acc_f1 = statistics.mean(avg_val_acc_f1)

    avg_log = 'Test Acc: {:.4f} +- {:.4f}, BAcc: {:.4f} +- {:.4f}, F1: {:.4f} +- {:.4f}, Val Acc F1: {:.4f}'
    avg_log = avg_log.format(avg_acc ,acc_CI ,avg_bacc, bacc_CI, avg_f1, f1_CI, avg_val_acc_f1)
    log = "{}\n{}".format(setting_log, avg_log)
    print(log)

    out_folder = 'result'
    if not osp.exists(out_folder):
        os.mkdir(out_folder)

    file_name = args.net + '_' + args.dataset + '_' + str(not args.no_pseudo) + '.txt'
    out_path = osp.join(out_folder, file_name)
    with open(out_path, 'a+') as f:
        f.write(log)
        f.write('\n\n')