CUDA_VISIBLE_DEVICES=0 ~/.conda/envs/pathology/bin/python \
/ailab/user/liumianxin/CLAM/main.py \
--drop_out \
--early_stopping \
--k 10 \
--feature_path features_chief \
--enco_dim 768 \
--label_frac 1 \
--exp_code task_tumor_P53_pos_chief_AIFFPE_vit \
--weighted_sample \
--bag_loss ce \
--inst_loss ce \
--task task_tumor_P53 \
--model_type vit \
--log_data \
--data_root_dir /ailab/user/liumianxin/transfer/Cryo_path/CLAM_DATA2/ \