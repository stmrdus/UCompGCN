##### with DistMult Score Function
# CompGCN (Composition: Subtraction)
CUDA_VISIBLE_DEVICES=0 python run_UCompGCN.py -model UCompGCN -score_func distmult -opn sub -gcn_dim 150 -gcn_layer 2 -data WN18RR

# CompGCN (Composition: Multiplication)
CUDA_VISIBLE_DEVICES=0 python run_UCompGCN.py -model UCompGCN -score_func distmult -opn mult -gcn_dim 150 -gcn_layer 2 -data WN18RR

# CompGCN (Composition: Circular Correlation)
CUDA_VISIBLE_DEVICES=0 python run_UCompGCN.py -model UCompGCN -score_func distmult -opn corr -gcn_dim 150 -gcn_layer 2 -data WN18RR