export OPENCV_IO_MAX_IMAGE_PIXELS=10995116277760

item="Lauren"
hospital="XJ"
save_dir="/mnt/petrelfs/yanfang/workspace/hw/yanfang/FM_code/test/${hospital}/Patches/"$item
source_dir="/mnt/petrelfs/yanfang/workspace/hw/yanfang/FM_code/dataset_csv/${item}_slide.csv"
wsi_format="png"
patch_size=512


/mnt/petrelfs/yanfang/anaconda3/envs/allslide/bin/python /mnt/petrelfs/yanfang/workspace/hw/yanfang/FM_code/Patching/create_patches_fp.py \
        --source $source_dir \
        --save_dir $save_dir\
        --preset /mnt/petrelfs/yanfang/workspace/hw/yanfang/FM_code/Patching/presets/tcga.csv \
        --patch_level 0 \
        --patch_size $patch_size \
        --step_size $patch_size \
        --wsi_format $wsi_format \
        --seg \
        --patch \


# srun -p smart_health_00 --time=05:00:00 bash Patching/get_coor_scripts/Lauren.sh