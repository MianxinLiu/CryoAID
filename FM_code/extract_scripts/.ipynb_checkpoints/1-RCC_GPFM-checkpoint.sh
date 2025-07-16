
root_dir="extract_scripts/logs/"
use_cache="no"

models="GPFM"
# models="resnet50 uni conch ctranspath GPFM phikon plip"

declare -A gpus
gpus["uni"]=0
gpus["conch"]=0
gpus["ctranspath"]=0
gpus["GPFM"]=0
gpus["plip"]=0
gpus["resnet50"]=0


for model in $models
do
        DIR_TO_COORDS=/mnt/petrelfs/yanfang/workspace/hw/yanfang/FM_GPFM/RUIJIN/Patches/RCC
        DATA_DIRECTORY=/
        CSV_FILE_NAME=/mnt/petrelfs/yanfang/workspace/hw/yanfang/FM_GPFM/GPFM-master/dataset_csv/RCC.csv
        FEATURES_DIRECTORY=/mnt/petrelfs/yanfang/workspace/hw/yanfang/FM_GPFM/RUIJIN/Patches/RCC
        ext=".svs"
        save_storage="yes"
        datatype="auto" # extra path process for TCGA dataset, direct mode do not care use extra path

        echo $model", GPU is:"${gpus[$model]}
        export CUDA_VISIBLE_DEVICES=${gpus[$model]}

        #nohup 
        /mnt/petrelfs/yanfang/anaconda3/envs/allslide/bin/python /mnt/petrelfs/yanfang/workspace/hw/yanfang/FM_GPFM/GPFM-master/extract_features_fp_fast.py \
                --data_h5_dir $DIR_TO_COORDS \
                --data_slide_dir $DATA_DIRECTORY \
                --csv_path $CSV_FILE_NAME \
                --feat_dir $FEATURES_DIRECTORY \
                --batch_size 512 \
                --use_cache $use_cache \
                --model $model \
                --datatype $datatype \
                --slide_ext $ext \
                --save_storage $save_storage #> $root_dir"Nanfang_log_$model.log" 2>&1 &
done

# srun -p smart_health_00 -n1 -w SH-IDC1-10-140-1-166 --pty bash

# srun -p smart_health_00 -w SH-IDC1-10-140-1-166 --gres=gpu:1 --cpus-per-task=16 --time=60:00:00 bash extract_scripts/1-RCC_1.sh

# for model in $models
# do
#         DIR_TO_COORDS=/hdd/RUIJIN/Patches/RCC
#         DATA_DIRECTORY=/
#         CSV_FILE_NAME=/home/yf/workspace/project/GPFM/code/GPFM-master/dataset_csv/RCC.csv
#         FEATURES_DIRECTORY=/hdd/RUIJIN/Patches/RCC
#         ext=".svs"
#         save_storage="yes"
#         datatype="auto" # extra path process for TCGA dataset, direct mode do not care use extra path

#         echo $model", GPU is:"${gpus[$model]}
#         export CUDA_VISIBLE_DEVICES=${gpus[$model]}

#         #nohup 
#         /home/yf/miniconda3/envs/py39/bin/python /home/yf/workspace/project/GPFM/code/GPFM-master/extract_features_fp.py \
#                 --data_h5_dir $DIR_TO_COORDS \
#                 --data_slide_dir $DATA_DIRECTORY \
#                 --csv_path $CSV_FILE_NAME \
#                 --feat_dir $FEATURES_DIRECTORY \
#                 --batch_size 32 \
#                 --model $model \
#                 --datatype $datatype \
#                 --slide_ext $ext \
#                 --save_storage $save_storage #> $root_dir"Nanfang_log_$model.log" 2>&1 &
# done
