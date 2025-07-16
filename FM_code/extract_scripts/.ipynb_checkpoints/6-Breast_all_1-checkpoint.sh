root_dir="extract_scripts/logs/"
use_cache="no"

# models="chief resnet50 uni conch ctranspath GPFM phikon plip gigapath virchow2"
# models="GPFM phikon plip gigapath virchow2"
models="gigapath"

for model in $models
do
        DIR_TO_COORDS=/mnt/petrelfs/yanfang/workspace/hw/yanfang/FM_GPFM/RUIJIN/Patches/Breast
        DATA_DIRECTORY=/
        CSV_FILE_NAME=/mnt/petrelfs/yanfang/workspace/hw/yanfang/FM_GPFM/GPFM-master/dataset_csv/Breast.csv
        FEATURES_DIRECTORY=/mnt/petrelfs/yanfang/workspace/hw/yanfang/FM_GPFM/RUIJIN/Patches/Breast
        ext=".svs"
        save_storage="yes"
        datatype="auto" # extra path process for TCGA dataset, direct mode do not care use extra path

        # echo $model", GPU is:"${gpus[$model]}
        # export CUDA_VISIBLE_DEVICES=${gpus[$model]}
        echo $model, "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
        
        if [[ "$model" =~ ^(chief|ctranspath|GPFM|resnet50)$ ]]; then
            envs=/mnt/petrelfs/yanfang/anaconda3/envs/allslide/bin/python
        elif [[ "$model" =~ ^(conch|gigapath|phikon|plip|uni|virchow2)$ ]]; then
            envs=/mnt/petrelfs/yanfang/anaconda3/envs/dinov2/bin/python 
        fi

        #nohup 
        $envs /mnt/petrelfs/yanfang/workspace/hw/yanfang/FM_GPFM/GPFM-master/extract_features_fp_fast.py \
                --data_h5_dir $DIR_TO_COORDS \
                --data_slide_dir $DATA_DIRECTORY \
                --csv_path $CSV_FILE_NAME \
                --feat_dir $FEATURES_DIRECTORY \
                --batch_size 2048 \
                --use_cache $use_cache \
                --model $model \
                --datatype $datatype \
                --slide_ext $ext \
                --save_storage $save_storage #> $root_dir"Breast_log_$model.log" 2>&1 &
done




















# srun -p smart_health_00 --gres=gpu:1 --cpus-per-task=16 --time=4-60:00:00 bash extract_scripts/6-Breast_all_1.sh