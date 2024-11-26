# CUDA_VISIBLE_DEVICES=0 python /ailab/user/liumianxin/CLAM/eval.py \
# --drop_out \
# --k 1 \
# --split test \
# --models_exp_code /ailab/user/liumianxin/CLAM/results/task_tumor_ATRX_ALL_ext_pathoduet_p2_AIFFPE_vit_s1 \
# --save_exp_code task_3_tumor_ATRX_rec_all \
# --model_type vit \
# --results_dir /ailab/user/liumianxin/CLAM/results \
# --data_root_dir /ailab/group/pjlab-smarthealth03/transfers_cpfs_test/liumianxin/CLAM_DATA2/ \
# --task task_3_tumor_ATRX_rec_all

CUDA_VISIBLE_DEVICES=0 python /ailab/user/liumianxin/CLAM/eval.py \
--drop_out \
--k 10 \
--split all \
--models_exp_code /ailab/user/liumianxin/CLAM/results/task_tumor_ATRX_pathoduet_p2_AIFFPE_vit_test_s1 \
--save_exp_code task_3_tumor_ATRX_rec2 \
--model_type vit \
--results_dir /ailab/user/liumianxin/CLAM/results/ \
--data_root_dir /ailab/group/pjlab-smarthealth03/transfers/liumianxin/Validation_Rec_patch/ \
--task task_3_tumor_ATRX_rec

CUDA_VISIBLE_DEVICES=0 python /ailab/user/liumianxin/CLAM/eval.py \
--drop_out \
--k 10 \
--split all \
--models_exp_code /ailab/user/liumianxin/CLAM/results/task_tumor_ATRX_pathoduet_p2_AIFFPE_vit_test_s1 \
--save_exp_code task_3_tumor_ATRX_fujian2 \
--model_type vit \
--results_dir /ailab/user/liumianxin/CLAM/results/ \
--data_root_dir /ailab/group/pjlab-smarthealth03/transfers/liumianxin/Validation_fujian_patch/ \
--task task_3_tumor_ATRX_fujian

CUDA_VISIBLE_DEVICES=0 python /ailab/user/liumianxin/CLAM/eval.py \
--drop_out \
--k 10 \
--split all \
--models_exp_code /ailab/user/liumianxin/CLAM/results/task_tumor_ATRX_pathoduet_p2_AIFFPE_vit_test_s1 \
--save_exp_code task_3_tumor_ATRX_north2 \
--model_type vit \
--results_dir /ailab/user/liumianxin/CLAM/results/ \
--data_root_dir /ailab/group/pjlab-smarthealth03/transfers/liumianxin/Validation_north_patch/ \
--task task_3_tumor_ATRX_north