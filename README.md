# CryoAIMD
The codes for CryoAIMD

## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN
- The model for AI-FFPE is available at https://github.com/DeepMIALab/AI-FFPE (Brain/FrozGanModels/wAtt_wLoss/Case-II). 
- Pathoduet model is available at https://github.com/openmedlab/PathoDuet (checkpoint p2).

### Install
- Clone this repo:
```bash
git clone https://github.com/MianxinLiu/CryoAIMD
cd CryoAIMD
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
