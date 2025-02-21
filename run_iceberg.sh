# BASE balancing method: BalancedSoftmax; Imbalance Ratio: 10
python main.py --net Diff --dataset Cora --patience 2000 --loss_type bs --T 20 --alpha 0.1 --lr 0.1 --feat_dim 128 --no_feat_norm --weight_decay 0.005

python main.py --net Diff --dataset CiteSeer --patience 2000 --loss_type bs --T 20 --alpha 0.1 --lr 0.1 --feat_dim 128 --no_feat_norm --weight_decay 0.005

python main.py --net Diff --dataset PubMed --patience 2000 --loss_type bs --T 20 --alpha 0.1 --lr 0.05 --feat_dim 128 --no_feat_norm --weight_decay 0.005

python main.py --net Diff --dataset cs --patience 2000 --loss_type bs --T 5 --alpha 0.1 --lr 0.01 --feat_dim 256 --no_feat_norm --weight_decay 0.005
