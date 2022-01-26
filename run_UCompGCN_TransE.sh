##### with TransE Score Function
# CompGCN (Composition: Subtraction)
CUDA_VISIBLE_DEVICES=0 python run_UCompGCN.py -model UCompGCN -score_func transe -opn sub -gamma 9 -hid_drop 0.1 -init_dim 200 -data WN18RR

# CompGCN (Composition: Multiplication)
CUDA_VISIBLE_DEVICES=0 python run_UCompGCN.py -model UCompGCN -score_func transe -opn mult -gamma 9 -hid_drop 0.2 -init_dim 200 -data WN18RR

# CompGCN (Composition: Circular Correlation)
CUDA_VISIBLE_DEVICES=0 python run_UCompGCN.py -model UCompGCN -score_func transe -opn corr -gamma 40 -hid_drop 0.1 -init_dim 200 -data WN18RR


CUDA_VISIBLE_DEVICES=0 python run_UCompGCN.py -model UCompGCN -score_func transe -opn sub -gamma 9 -hid_drop 0.1 -init_dim 200 -data kinship -depth 1 -sumres False