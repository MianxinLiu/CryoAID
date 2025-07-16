# export http_proxy=http://yanfang:XfAOA6iyl73G4GsMPkipuNE0tEwxCqrm6Jwqtb3SP4OCxjk2TOzWEsa5s8SY@10.1.20.50:23128/ ; 
# export https_proxy=http://yanfang:XfAOA6iyl73G4GsMPkipuNE0tEwxCqrm6Jwqtb3SP4OCxjk2TOzWEsa5s8SY@10.1.20.50:23128/ ; 
# export HTTP_PROXY=http://yanfang:XfAOA6iyl73G4GsMPkipuNE0tEwxCqrm6Jwqtb3SP4OCxjk2TOzWEsa5s8SY@10.1.20.50:23128/ ; 
# export HTTPS_PROXY=http://yanfang:XfAOA6iyl73G4GsMPkipuNE0tEwxCqrm6Jwqtb3SP4OCxjk2TOzWEsa5s8SY@10.1.10.50:23128/

root_dir="extract_scripts/logs/"
use_cache="no"

models="virchow2"
# models="resnet50 uni conch ctranspath gpfm phikon plip chief gigapath virchow2"

declare -A gpus
gpus["uni"]=0
gpus["conch"]=2
gpus["ctranspath"]=3
gpus["GPFM"]=1
gpus["plip"]=4
gpus["resnet50"]=5
gpus["phikon"]=6
gpus["virchow2"]=3


for model in $models
do
        DIR_TO_COORDS=/mnt/petrelfs/yanfang/workspace/hw/yanfang/FM_GPFM/RUIJIN/Patches/Ovary
        DATA_DIRECTORY=/
        CSV_FILE_NAME=/mnt/petrelfs/yanfang/workspace/hw/yanfang/FM_GPFM/GPFM-master/dataset_csv/Ovary.csv
        FEATURES_DIRECTORY=/mnt/petrelfs/yanfang/workspace/hw/yanfang/FM_GPFM/RUIJIN/Patches/Ovary
        ext=".svs"
        save_storage="yes"
        datatype="auto" # extra path process for TCGA dataset, direct mode do not care use extra path

        echo $model", GPU is:"${gpus[$model]}
        export CUDA_VISIBLE_DEVICES=${gpus[$model]}

        #nohup 
        /mnt/petrelfs/yanfang/anaconda3/envs/dinov2/bin/python /mnt/petrelfs/yanfang/workspace/hw/yanfang/FM_GPFM/GPFM-master/extract_features_fp_fast.py \
                --data_h5_dir $DIR_TO_COORDS \
                --data_slide_dir $DATA_DIRECTORY \
                --csv_path $CSV_FILE_NAME \
                --feat_dir $FEATURES_DIRECTORY \
                --batch_size 2048 \
                --use_cache $use_cache \
                --model $model \
                --datatype $datatype \
                --slide_ext $ext \
                --save_storage $save_storage #> $root_dir"Ovary_log_$model.log" 2>&1 &
done


# srun -p smart_health_00 -w SH-IDC1-10-140-1-156 --gres=gpu:1 --cpus-per-task=16 --time=60:00:00 bash extract_scripts/4-Ovary_virchow2.sh