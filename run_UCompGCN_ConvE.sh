##### with ConvE Score Function
# CompGCN (Composition: Subtraction)
CUDA_VISIBLE_DEVICES=0 python run_UCompGCN.py -score_func conve -opn sub -ker_sz 5 -data kinship

# CompGCN (Composition: Multiplication)
CUDA_VISIBLE_DEVICES=0 python run_UCompGCN.py -score_func conve -opn mult -data kinship

# CompGCN (Composition: Circular Correlation)
CUDA_VISIBLE_DEVICES=0 python run_UCompGCN.py -score_func conve -opn corr -data kinship