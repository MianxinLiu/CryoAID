CUDA_VISIBLE_DEVICES=0 python \
/ailab/user/liumianxin/CLAM/main_external.py \
--drop_out \
--early_stopping \
--lr 1e-4 \
--k 10 \
--k_start 0 \
--k_end 1 \
--label_frac 1 \
--exp_code task_tumor_ATRX_ALL_ext_pathoduet_p2_AIFFPE_vit \
--weighted_sample \
--bag_loss ce \
--inst_loss ce \
--task task_tumor_ATRX_all2 \
--model_type vit \
--log_data \
--data_root_dir /ailab/group/pjlab-smarthealth03/transfers_cpfs_test/liumianxin/CLAM_DATA2/ \

