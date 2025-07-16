
# CUDA_VISIBLE_DEVICES=0 ~/.conda/envs/pathology/bin/python /ailab/user/liumianxin/CLAM/eval.py \
# --drop_out \
# --k 10 \
# --split test \
# --models_exp_code /ailab/user/liumianxin/CLAM/results/task_tumor_H3K27M_pos_chief_AIFFPE_vit_final_s1 \
# --save_exp_code pos_H3K27M \
# --model_type vit \
# --results_dir /ailab/user/liumianxin/CLAM/results/ \
# --data_root_dir /ailab/user/liumianxin/transfer/Cryo_path/CLAM_DATA2/ \
# --task task_3_tumor_H3K27M

# CUDA_VISIBLE_DEVICES=0 ~/.conda/envs/pathology/bin/python /ailab/user/liumianxin/CLAM/eval.py \
# --drop_out \
# --k 10 \
# --split test \
# --models_exp_code /ailab/user/liumianxin/CLAM/results/all/task_tumor_ATRX_all_chief_AIFFPE_vit_final_s1 \
# --save_exp_code all_ATRX \
# --model_type vit \
# --results_dir /ailab/user/liumianxin/CLAM/results/ \
# --data_root_dir /ailab/user/liumianxin/transfer/Cryo_path/CLAM_DATA2/ \
# --task task_3_tumor_ATRX_all

# CUDA_VISIBLE_DEVICES=0 ~/.conda/envs/pathology/bin/python /ailab/user/liumianxin/CLAM/eval.py \
# --drop_out \
# --k 1 \
# --split all \
# --models_exp_code /ailab/user/liumianxin/CLAM/results/all_ext/task_tumor_H3K27M_all_ext_chief_AIFFPE_vit_3_s1 \
# --save_exp_code all_H3K27M_rec_train \
# --model_type vit \
# --results_dir /ailab/user/liumianxin/CLAM/results/ \
# --data_root_dir /ailab/user/liumianxin/transfer/Cryo_path/Validation_Rec_patch/ \
# --task task_3_tumor_H3K27M_rec_all


# CUDA_VISIBLE_DEVICES=0 ~/.conda/envs/pathology/bin/python /ailab/user/liumianxin/CLAM/eval.py \
# --drop_out \
# --k 10 \
# --split all \
# --models_exp_code /ailab/user/liumianxin/CLAM/results/task_tumor_ATRX_all_chief_AIFFPE_vit_s1 \
# --save_exp_code all_ATRX_rec \
# --model_type vit \
# --results_dir /ailab/user/liumianxin/CLAM/results/ \
# --data_root_dir /ailab/user/liumianxin/transfer/Cryo_path/Validation_Rec_patch/ \
# --task task_3_tumor_ATRX_rec_all

# CUDA_VISIBLE_DEVICES=0 ~/.conda/envs/pathology/bin/python /ailab/user/liumianxin/CLAM/eval.py \
# --drop_out \
# --k 10 \
# --split all \
# --models_exp_code /ailab/user/liumianxin/CLAM/results/task_tumor_H3K27M_pos_chief_AIFFPE_vit_final_s1 \
# --save_exp_code pos_H3K27M_rec \
# --model_type vit \
# --results_dir /ailab/user/liumianxin/CLAM/results/ \
# --data_root_dir /ailab/user/liumianxin/transfer/Cryo_path/Validation_Rec_patch/ \
# --task task_3_tumor_H3K27M_rec

# CUDA_VISIBLE_DEVICES=0 ~/.conda/envs/pathology/bin/python /ailab/user/liumianxin/CLAM/eval.py \
# --drop_out \
# --k 10 \
# --split all \
# --models_exp_code /ailab/user/liumianxin/CLAM/results/task_tumor_H3K27M_pos_chief_AIFFPE_vit_ft_s1 \
# --save_exp_code pos_H3K27M_fujian \
# --model_type vit \
# --results_dir /ailab/user/liumianxin/CLAM/results/ \
# --data_root_dir /ailab/user/liumianxin/transfer/Cryo_path/Validation_fujian_patch/ \
# --task task_3_tumor_H3K27M_fujian

# CUDA_VISIBLE_DEVICES=0 ~/.conda/envs/pathology/bin/python /ailab/user/liumianxin/CLAM/eval.py \
# --drop_out \
# --k 10 \
# --split all \
# --models_exp_code /ailab/user/liumianxin/CLAM/results/task_tumor_H3K27M_pos_chief_AIFFPE_vit_final_s1 \
# --save_exp_code pos_H3K27M_north \
# --model_type vit \
# --results_dir /ailab/user/liumianxin/CLAM/results/ \
# --data_root_dir /ailab/user/liumianxin/transfer/Cryo_path/Validation_north_patch/ \
# --task task_3_tumor_H3K27M_north


# CUDA_VISIBLE_DEVICES=0 ~/.conda/envs/pathology/bin/python /ailab/user/liumianxin/CLAM/eval.py \
# --drop_out \
# --k 1 \
# --split all \
# --models_exp_code /ailab/user/liumianxin/CLAM/results/pos_ext/task_tumor_P53_pos_rec_chief_AIFFPE_vit_s1 \
# --save_exp_code pos_P53_rec_train \
# --model_type vit \
# --results_dir /ailab/user/liumianxin/CLAM/results/ \
# --data_root_dir /ailab/user/liumianxin/transfer/Cryo_path/Validation_Rec_patch/ \
# --task task_3_tumor_P53_rec

CUDA_VISIBLE_DEVICES=0 ~/.conda/envs/pathology/bin/python /ailab/user/liumianxin/CLAM/eval.py \
--drop_out \
--k 1 \
--split all \
--models_exp_code /ailab/user/liumianxin/CLAM/results/task_tumor_ATRX_pos_rec_chief_AIFFPE_vit_s1 \
--save_exp_code pos_ATRX_rec_train \
--model_type vit \
--results_dir /ailab/user/liumianxin/CLAM/results/ \
--data_root_dir /ailab/user/liumianxin/transfer/Cryo_path/Validation_Rec_patch/ \
--task task_3_tumor_ATRX_rec

# CUDA_VISIBLE_DEVICES=0 ~/.conda/envs/pathology/bin/python /ailab/user/liumianxin/CLAM/eval.py \
# --drop_out \
# --k 1 \
# --split all \
# --models_exp_code /ailab/user/liumianxin/CLAM/results/pos_ext/task_tumor_P53_pos_north_chief_AIFFPE_vit_s1 \
# --save_exp_code pos_P53_north_train \
# --model_type vit \
# --results_dir /ailab/user/liumianxin/CLAM/results/ \
# --data_root_dir /ailab/user/liumianxin/transfer/Cryo_path/Validation_north_patch/ \
# --task task_3_tumor_P53_north

# CUDA_VISIBLE_DEVICES=0 ~/.conda/envs/pathology/bin/python /ailab/user/liumianxin/CLAM/eval.py \
# --drop_out \
# --k 10 \
# --split all \
# --models_exp_code /ailab/user/liumianxin/CLAM/results/task_tumor_P53_pos_chief_AIFFPE_vit_final_s1 \
# --save_exp_code task_3_tumor_P53_pos2neg \
# --model_type vit \
# --results_dir /ailab/user/liumianxin/CLAM/results/ \
# --data_root_dir /ailab/user/liumianxin/transfer/Cryo_path/CLAM_DATA2/ \
# --task task_3_tumor_P53_neg


# CUDA_VISIBLE_DEVICES=0 ~/.conda/envs/pathology/bin/python /ailab/user/liumianxin/CLAM/eval.py \
# --drop_out \
# --k 10 \
# --split test \
# --models_exp_code /ailab/user/liumianxin/CLAM/results/neg2neg/task_tumor_P53_neg_chief_AIFFPE_vit_final_s1 \
# --save_exp_code P53_neg2neg \
# --model_type vit \
# --results_dir /ailab/user/liumianxin/CLAM/results/ \
# --data_root_dir /ailab/user/liumianxin/transfer/Cryo_path/CLAM_DATA2/ \
# --task task_3_tumor_P53_neg