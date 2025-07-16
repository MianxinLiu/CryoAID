
# save log
prefix="/mnt/petrelfs/yanfang/workspace/hw/yanfang/FM_GPFM/RUIJIN"
skip_partial="no" # yes to skip partial file

models="chief resnet50 uni conch ctranspath GPFM phikon plip virchow2 gigapath"


for model in $models
do
        DIR_TO_COORDS=$prefix"/rawdata/Intestine_patch"
        CSV_FILE_NAME="dataset_csv/Intestine_patch.csv"
        FEATURES_DIRECTORY=$prefix"/Patches/Intestine_patch"

        echo $model, "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
        
        if [[ "$model" =~ ^(chief|ctranspath|GPFM|resnet50)$ ]]; then
            envs=/mnt/petrelfs/yanfang/anaconda3/envs/allslide/bin/python
        elif [[ "$model" =~ ^(conch|gigapath|phikon|plip|uni|virchow2)$ ]]; then
            envs=/mnt/petrelfs/yanfang/anaconda3/envs/dinov2/bin/python 
        fi

        # nohup 
        $envs extract_features_fp_from_patch_usethis.py \
                --csv_path $CSV_FILE_NAME \
                --feat_dir $FEATURES_DIRECTORY \
                --batch_size 128 \
                --model $model \
                --skip_partial $skip_partial #> "extract_scripts/logs/PANDA_log_$model.log" 2>&1 &
done


# srun -p smart_health_00 -w SH-IDC1-10-140-1-167 --gres=gpu:1 --cpus-per-task=16 --time=60:00:00 bash extract_scripts/7-Intestine.sh