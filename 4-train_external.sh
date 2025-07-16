CUDA_VISIBLE_DEVICES=0 ~/.conda/envs/pathology/bin/python \
/ailab/user/liumianxin/CLAM/main_external.py \
--drop_out \
--early_stopping \
--lr 1e-3 \
--reg 1e-3 \
--k 1 \
--k_start 0 \
--k_end 1 \
--label_frac 1 \
--exp_code task_tumor_ATRX_pos_rec_chief_AIFFPE_vit \
--weighted_sample \
--bag_loss ce \
--inst_loss ce \
--task task_tumor_ATRX_rec \
--model_type vit \
--log_data \
--data_root_dir /ailab/user/liumianxin/transfer/Cryo_path/CLAM_DATA2/ \