# CryoAID
The codes for CryoAID

## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN
- The model for AI-FFPE is available at https://github.com/DeepMIALab/AI-FFPE (Brain/FrozGanModels/wAtt_wLoss/Case-I).
- CHIEF model is available at https://github.com/hms-dbmi/CHIEF.
- UNI model is available at https://github.com/mahmoodlab/UNI.
- Gigapath model is available at https://github.com/prov-gigapath/prov-gigapath.
- Virchow2 model is available at https://huggingface.co/paige-ai/Virchow2. 
- Pathoduet model is available at https://github.com/openmedlab/PathoDuet (checkpoint p2/checkpoint HE).
- The code for ABMIL is available at https://github.com/AMLab-Amsterdam/AttentionDeepMIL. The code for CLAM and the ResNet model is available at https://github.com/mahmoodlab/CLAM. The code for TransMIL is available at https://github.com/szc19990412/TransMIL.
- Other settings are refer to CLAM at https://github.com/mahmoodlab/CLAM (such as the preprocessing part).

### Install
- Clone this repo:
```bash
git clone https://github.com/MianxinLiu/CryoAID
cd CryoAID
```

- For pip users, please type the command `pip install -r requirements.txt`. Note that for running CHIEF, one should create a copied environment, download "timm-0.5.4.tar" (offered here or their page) and install the tar. 

- For Conda users,  you can create a new Conda environment using `conda env create -f environment.yml`.

### Configuration
- AI-FFPE: The code for AI-FFPE should be downloaded at https://github.com/DeepMIALab/AI-FFPE). In the paper we downloaded and used AI-FFPE pre-trained model parameter (Brain/FrozGanModels/wAtt_wLoss/Case-I). In addition, modify the --name and --checkpoints_dir argument in options/base_options.py. If you put the model under path as "PATH/wAtt_wLoss/Case-I", then --name=wAtt_wLoss/Case-I and --checkpoints_dir=PATH. Remember to add the path when running feature extraction (search "sys.path.append('/ailab/user/liumianxin/transfer/AI_FFPE_main/')" in "extract_features_fp_FM_AIFFPE.py" and set to your path).
- FMs: For most of the FMs, one can refer to their origianl Github pages. For CHIEF model, note that they used a modified swimtransformer. For running CHIEF, one should create a copied environment, download "timm-0.5.4.tar" (offered here or their page) and install the tar (details refer to ctranspath or CHIEF model). 

### Training and Test
Run the bash files in order.
- segment and patching: 1-create_patches.sh (can refer to CLAM at https://github.com/mahmoodlab/CLAM). It will generate tissue segmentation and patches for input slides.
- extract features from pathches: 2-extract_features.sh. The base commend will run "extract_features_fp_FM_AIFFPE.py" to apply AI-FFPE and different pathology foundation model to extract feature from patches. The --no_gen argument can be set to skip AI-FFPE generation. --FM can select the foundation model after downloading the FM parameters and configuring the path. It is recommended to set the output path as "features_xxx" where "xxx" is the FM or setting (e.g. features_chief, features_resnet). This will benefit to the model training with different configurations.
- train and test splits: 3-create_splits.sh. Generate cross-valiation splits from the excels in Dataset_csv folder. Notet that this bash generate splits with overlaps. Run "create_splits_seq_kfold.py" to obtain splits without overlaps. In addition, "create_splits_seq_traval.py" combines internal and external datasets to generate a split for external testing with a training on internal data.
- train model: 4-train.sh. Modifing the feature_path parameter for ablation studies. Using other 4-train_external.sh for external evaluations. --model_type can specify the classifier model.
- evaluate: 5-eval.sh. Evaluate trained model on desirde datasets.
- visualize the heatmaps: 6-heatmap.sh (can refer to CLAM at https://github.com/mahmoodlab/CLAM).
