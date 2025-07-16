prefix="/hdd/RUIJIN"
dataset="RCC"
skip_partial="no" # yes to skip partial file
root="/home/yf/workspace/project/GPFM/code/GPFM-master/"

# models="resnet50 uni conch ctranspath gpfm phikon plip"
models="resnet50"

declare -A gpus
gpus["uni"]=0
gpus["conch"]=0
gpus["ctranspath"]=0
gpus["gpfm"]=0
gpus["plip"]=0
gpus["resnet50"]=0


for model in $models
do
        DIR_TO_COORDS=$prefix"/Patches/"$dataset
        CSV_FILE_NAME=$root"dataset_csv/"$dataset".csv"
        FEATURES_DIRECTORY=$prefix"/Patches/"$dataset

        echo $model", GPU is:"${gpus[$model]}
        export CUDA_VISIBLE_DEVICES=${gpus[$model]}

        #nohup 
        /home/yf/miniconda3/envs/py39/bin/python $root/extract_features_fp_from_patch.py \
                --patch_img_dir /mnt/mount/RUIJIN/Kidney
                --data_h5_dir $DIR_TO_COORDS \
                --csv_path $CSV_FILE_NAME \
                --feat_dir $FEATURES_DIRECTORY \
                --batch_size 128 \
                --model $model \
                --skip_partial $skip_partial #> $root"extract_scripts/logs/RCC_log_$model.log" 2>&1 &
done

