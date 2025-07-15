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

- For pip users, please type the command `pip install -r requirements.txt`.

- For Conda users,  you can create a new Conda environment using `conda env create -f environment.yml`.

### Training and Test
Run the bash files in order.
- segment and patching: 1-create_patches.sh
- extract features from pathches: 2-extract_features.sh
- train and test splits: 3-create_splits.sh
- train model: 4-train.sh. Using other 4-train-xxx.sh for ablation models
- evaluate: 5-eval.sh
- visualize the heatmaps: 6-heatmap.sh
