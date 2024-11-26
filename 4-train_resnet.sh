CUDA_VISIBLE_DEVICES=0 python \
/ailab/user/liumianxin/CLAM/main_resnet.py \
--drop_out \
--early_stopping \
--lr 1e-4 \
--k 10 \
--label_frac 1 \
--exp_code task_tumor_ATRX_resnet_pathoduet_p2_AIFFPE_vit \
--weighted_sample \
--bag_loss ce \
--inst_loss ce \
--task task_tumor_ATRX \
--model_type vit_large \
--log_data \
--data_root_dir /ailab/group/pjlab-smarthealth03/transfers_cpfs_test/liumianxin/CLAM_DATA2/ \

