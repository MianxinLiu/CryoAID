CUDA_VISIBLE_DEVICES=0 python create_patches_fp.py \
--patch_level 1 \
--patch_size 224 \
--step_size 224 \
--preset bwh_biopsy.csv --seg --patch --stitch