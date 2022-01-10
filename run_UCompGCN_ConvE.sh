##### with ConvE Score Function
# CompGCN (Composition: Subtraction)
CUDA_VISIBLE_DEVICES=0 python run_UCompGCN.py -model UCompGCN -score_func conve -opn sub -ker_sz 5 -data WN18RR

# CompGCN (Composition: Multiplication)
CUDA_VISIBLE_DEVICES=0 python run_UCompGCN.py -model UCompGCN -score_func conve -opn mult -data WN18RR

# CompGCN (Composition: Circular Correlation)
CUDA_VISIBLE_DEVICES=0 python run_UCompGCN.py -model UCompGCN -score_func conve -opn corr -data WN18RR