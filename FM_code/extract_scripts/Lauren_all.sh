root_dir="extract_scripts/logs/"
use_cache="no"

# models="uni uni2 chief gigapath virchow2 ctranspath resnet50"
models="uni2"
hospital="XJ"
task="Lauren"

for model in $models
do
        DIR_TO_COORDS=/mnt/petrelfs/yanfang/workspace/hw/yanfang/FM_code/test/$hospital/Patches/$task
        DATA_DIRECTORY=/
        CSV_FILE_NAME=/mnt/petrelfs/yanfang/workspace/hw/yanfang/FM_code/dataset_csv/${task}_slide.csv
        FEATURES_DIRECTORY=/mnt/petrelfs/yanfang/workspace/hw/yanfang/FM_code/test/$hospital/Patches/$task
        ext=".svs"
        save_storage="yes"
        datatype="auto"

        # echo $model", GPU is:"${gpus[$model]}
        # export CUDA_VISIBLE_DEVICES=${gpus[$model]}
        echo $model, "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
        
        if [[ "$model" =~ ^(chief|ctranspath|resnet50)$ ]]; then
            envs=/mnt/petrelfs/yanfang/anaconda3/envs/allslide/bin/python
        elif [[ "$model" =~ ^(gigapath|uni|uni2|virchow2)$ ]]; then
            envs=/mnt/petrelfs/yanfang/anaconda3/envs/dinov2/bin/python 
        fi

        $envs /mnt/petrelfs/yanfang/workspace/hw/yanfang/FM_code/extract_features_fp_fast.py \
                --data_h5_dir $DIR_TO_COORDS \
                --data_slide_dir $DATA_DIRECTORY \
                --csv_path $CSV_FILE_NAME \
                --feat_dir $FEATURES_DIRECTORY \
                --batch_size 1024 \
                --use_cache $use_cache \
                --model $model \
                --datatype $datatype \
                --slide_ext $ext \
                --save_storage $save_storage 
done



# srun -p smart_health_00 --gres=gpu:1 --cpus-per-task=16 --time=60:00:00 bash extract_scripts/Lauren_all.sh















# srun -p smart_health_00 --gres=gpu:1 --cpus-per-task=16 --time=4-60:00:00 bash extract_scripts/6-Breast_all.sh