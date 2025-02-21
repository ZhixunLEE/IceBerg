# You can select three classic backbones: GCN, GAT, and SAGE.
# And multiple mainstream benchmark datasets: Cora, CiteSeer, PubMed, cs, physics, arxiv, corafull, roman-empire, penn94
# Feel free to change the net and dataset in the following commands

# ERM; Imbalance Ratio: 10; Loss Type: CrossEntropy; Resampling: None
python main.py --net GCN --imb_ratio 10 --dataset Cora --patience 500 --loss_type ce --no_pseudo

# RW; Imbalance Ratio: 10; Loss Type: Reweighting; Resampling: None
python main.py --net GCN --imb_ratio 10 --dataset Cora --patience 500 --loss_type rw --no_pseudo

# BS; Imbalance Ratio: 10; Loss Type: BalancedSoftmax; Resampling: None
python main.py --net GCN --imb_ratio 10 --dataset Cora --patience 500 --loss_type bs --no_pseudo

# RN; Imbalance Ratio: 10; Loss Type: ReNode; Resampling: None
python main.py --net GCN --imb_ratio 10 --dataset Cora --patience 500 --loss_type rn --no_pseudo

# MIX; Imbalance Ratio: 10; Loss Type: CrossEntropy; Resampling: Mixup
python main.py --net GCN --imb_ratio 10 --dataset Cora --patience 500 --loss_type ce --no_pseudo --mix

# ENS; Imbalance Ratio: 10; Loss Type: CrossEntropy; Resampling: GraphENS
python main.py --net GCN --imb_ratio 10 --dataset Cora --patience 500 --loss_type ce --no_pseudo --ens --warmup 10

# SHA; Imbalance Ratio: 10; Loss Type: CrossEntropy; Resampling: GraphSHA
python main.py --net GCN --imb_ratio 10 --dataset Cora --patience 500 --loss_type ce --no_pseudo --sha --warmup 10

# For TAM and BAT, you can easily conbine them with the aforementioned BASE balancing methods. For example:
python main.py --net GCN --imb_ratio 10 --dataset Cora --patience 500 --loss_type ce --no_pseudo --tam
python main.py --net GCN --imb_ratio 10 --dataset Cora --patience 500 --loss_type ce --no_pseudo --bat