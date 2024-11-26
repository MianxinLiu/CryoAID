CUDA_VISIBLE_DEVICES=0 python \
/ailab/user/liumianxin/CLAM/main.py \
--drop_out \
--early_stopping \
--lr 1e-4 \
--k 10 \
--label_frac 1 \
--exp_code task_tumor_ATRX_neg_pathoduet_p2_AIFFPE_vit \
--weighted_sample \
--bag_loss ce \
--inst_loss ce \
--task task_tumor_ATRX_neg \
--model_type vit \
--log_data \
--data_root_dir /ailab/group/pjlab-smarthealth03/transfers_cpfs_test/liumianxin/CLAM_DATA2/ \

