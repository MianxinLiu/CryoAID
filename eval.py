from __future__ import print_function

import numpy as np

import argparse
import torch
import torch.nn as nn
import pdb
import os
import pandas as pd
from utils.utils import *
from math import floor
import matplotlib.pyplot as plt
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
import h5py
from utils.eval_utils import *

# Training settings
parser = argparse.ArgumentParser(description='CLAM Evaluation Script')
parser.add_argument('--data_root_dir', type=str, default=None,
                    help='data directory')
parser.add_argument('--results_dir', type=str, default='./results',
                    help='relative path to results folder, i.e. '+
                    'the directory containing models_exp_code relative to project root (default: ./results)')
parser.add_argument('--save_exp_code', type=str, default=None,
                    help='experiment code to save eval results')
parser.add_argument('--models_exp_code', type=str, default=None,
                    help='experiment code to load trained models (directory under results_dir containing model checkpoints')
parser.add_argument('--splits_dir', type=str, default=None,
                    help='splits directory, if using custom splits other than what matches the task (default: None)')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', 
                    help='size of model (default: small)')
parser.add_argument('--enco_dim', type=int, default=768,
                    help='encoding dimension (default: enco_dim=768)') 
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil', 'vit', 'abmil','transmil'], default='vit', 
                    help='type of model (default: clam_sb, clam w/ single attention branch)')
parser.add_argument('--drop_out', action='store_true', default=False, 
                    help='whether model uses dropout')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--fold', type=int, default=-1, help='single fold to evaluate')
parser.add_argument('--micro_average', action='store_true', default=False, 
                    help='use micro_average instead of macro_avearge for multiclass AUC')
parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'], default='all')
parser.add_argument('--task', type=str, choices=['task_3_tumor_H3K27M_fujian', 'task_3_tumor_ATRX_fujian', \
    'task_3_tumor_P53_fujian','task_3_tumor_H3K27M_north', 'task_3_tumor_ATRX_north','task_3_tumor_P53_north', \
        'task_3_tumor_H3K27M', 'task_3_tumor_ATRX','task_3_tumor_P53', \
            'task_3_tumor_H3K27M_nogen', 'task_3_tumor_ATRX_nogen','task_3_tumor_P53_nogen', \
                'task_3_tumor_H3K27M_resnet', 'task_3_tumor_ATRX_resnet','task_3_tumor_P53_resnet', \
    'task_3_tumor_ATRX_neg', 'task_3_tumor_H3K27M_neg', 'task_3_tumor_P53_neg', \
         'task_3_tumor_H3K27M_all', 'task_3_tumor_ATRX_all','task_3_tumor_P53_all', \
            'task_3_tumor_H3K27M_rec', 'task_3_tumor_ATRX_rec','task_3_tumor_P53_rec', \
                'task_3_tumor_H3K27M_rec_all', 'task_3_tumor_ATRX_rec_all','task_3_tumor_P53_rec_all'])
args = parser.parse_args()

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

args.save_dir = os.path.join('/ailab/user/liumianxin/CLAM/eval_results', 'EVAL_' + str(args.save_exp_code))
args.models_dir = os.path.join(args.results_dir, str(args.models_exp_code))

os.makedirs(args.save_dir, exist_ok=True)

if args.splits_dir is None:
    args.splits_dir = args.models_dir

assert os.path.isdir(args.models_dir)
assert os.path.isdir(args.splits_dir)

settings = {'task': args.task,
            'split': args.split,
            'save_dir': args.save_dir, 
            'models_dir': args.models_dir,
            'model_type': args.model_type,
            'drop_out': args.drop_out,
            'model_size': args.model_size}

with open(args.save_dir + '/eval_experiment_{}.txt'.format(args.save_exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print(settings)
if args.task == 'task_3_tumor_H3K27M_fujian':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '/ailab/user/liumianxin/CLAM/dataset_csv/tumor_H3K27M_dummy_fujian.csv',
                            data_dir= os.path.join(args.data_root_dir, 'features_chief'),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'neg':0, 'pos':1},
                            patient_strat= False,
                            ignore=[])
elif args.task == 'task_3_tumor_ATRX_fujian':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '/ailab/user/liumianxin/CLAM/dataset_csv/tumor_ATRX_dummy_fujian.csv',
                            data_dir= os.path.join(args.data_root_dir, 'features_chief'),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'neg':1, 'pos':0},
                            patient_strat= False,
                            ignore=[])
elif args.task == 'task_3_tumor_P53_fujian':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '/ailab/user/liumianxin/CLAM/dataset_csv/tumor_P53_dummy_fujian.csv',
                            data_dir= os.path.join(args.data_root_dir, 'features_chief'),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'neg':0, 'pos':1},
                            patient_strat= False,
                            ignore=[])
elif args.task == 'task_3_tumor_H3K27M':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '/ailab/user/liumianxin/CLAM/dataset_csv/tumor_H3K27M_dummy_clean.csv',
                            data_dir= os.path.join(args.data_root_dir, 'features_chief'),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'neg':0, 'pos':1},
                            patient_strat= False,
                            ignore=[])
elif args.task == 'task_3_tumor_ATRX':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '/ailab/user/liumianxin/CLAM/dataset_csv/tumor_ATRX_dummy_clean.csv',
                            data_dir= os.path.join(args.data_root_dir, 'features_chief'),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'neg':1, 'pos':0},
                            patient_strat= False,
                            ignore=[])
elif args.task == 'task_3_tumor_P53':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '/ailab/user/liumianxin/CLAM/dataset_csv/tumor_P53_dummy_clean.csv',
                            data_dir= os.path.join(args.data_root_dir, 'features_chief'),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'neg':0, 'pos':1},
                            patient_strat= False,
                            ignore=[])
elif args.task == 'task_3_tumor_H3K27M_nogen':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '/ailab/user/liumianxin/CLAM/dataset_csv/tumor_H3K27M_dummy_clean.csv',
                            data_dir= os.path.join(args.data_root_dir, 'features_chief_nogen'),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'neg':0, 'pos':1},
                            patient_strat= False,
                            ignore=[])
elif args.task == 'task_3_tumor_ATRX_nogen':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '/ailab/user/liumianxin/CLAM/dataset_csv/tumor_ATRX_dummy_clean.csv',
                            data_dir= os.path.join(args.data_root_dir, 'features_chief_nogen'),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'neg':1, 'pos':0},
                            patient_strat= False,
                            ignore=[])
elif args.task == 'task_3_tumor_P53_nogen':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '/ailab/user/liumianxin/CLAM/dataset_csv/tumor_P53_dummy_clean.csv',
                            data_dir= os.path.join(args.data_root_dir, 'features_chief_nogen'),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'neg':0, 'pos':1},
                            patient_strat= False,
                            ignore=[])
elif args.task == 'task_3_tumor_H3K27M_resnet':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '/ailab/user/liumianxin/CLAM/dataset_csv/tumor_H3K27M_dummy_clean.csv',
                            data_dir= os.path.join(args.data_root_dir, 'features_resnet'),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'neg':0, 'pos':1},
                            patient_strat= False,
                            ignore=[])
elif args.task == 'task_3_tumor_ATRX_resnet':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '/ailab/user/liumianxin/CLAM/dataset_csv/tumor_ATRX_dummy_clean.csv',
                            data_dir= os.path.join(args.data_root_dir, 'features_resnet'),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'neg':1, 'pos':0},
                            patient_strat= False,
                            ignore=[])
elif args.task == 'task_3_tumor_P53_resnet':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '/ailab/user/liumianxin/CLAM/dataset_csv/tumor_P53_dummy_clean.csv',
                            data_dir= os.path.join(args.data_root_dir, 'features_resnet'),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'neg':0, 'pos':1},
                            patient_strat= False,
                            ignore=[])
elif args.task == 'task_3_tumor_H3K27M_north':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '/ailab/user/liumianxin/CLAM/dataset_csv/tumor_H3K27M_dummy_north.csv',
                            data_dir= os.path.join(args.data_root_dir, 'features_chief'),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'neg':0, 'pos':1},
                            patient_strat= False,
                            ignore=[])
elif args.task == 'task_3_tumor_ATRX_north':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '/ailab/user/liumianxin/CLAM/dataset_csv/tumor_ATRX_dummy_north.csv',
                            data_dir= os.path.join(args.data_root_dir, 'features_chief'),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'neg':1, 'pos':0},
                            patient_strat= False,
                            ignore=[])
elif args.task == 'task_3_tumor_P53_north':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '/ailab/user/liumianxin/CLAM/dataset_csv/tumor_P53_dummy_north.csv',
                            data_dir= os.path.join(args.data_root_dir, 'features_chief'),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'neg':0, 'pos':1},
                            patient_strat= False,
                            ignore=[])
elif args.task == 'task_3_tumor_ATRX_neg':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '/ailab/user/liumianxin/CLAM/dataset_csv/tumor_ATRX_dummy_neg.csv',
                            data_dir= os.path.join(args.data_root_dir, 'features_chief'),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'neg':1, 'pos':0},
                            patient_strat= False,
                            ignore=[])
elif args.task == 'task_3_tumor_H3K27M_neg':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '/ailab/user/liumianxin/CLAM/dataset_csv/tumor_H3K27M_dummy_neg.csv',
                            data_dir= os.path.join(args.data_root_dir, 'features_chief'),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'neg':0, 'pos':1},
                            patient_strat= False,
                            ignore=[])
elif args.task == 'task_3_tumor_P53_neg':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '/ailab/user/liumianxin/CLAM/dataset_csv/tumor_P53_dummy_neg.csv',
                            data_dir= os.path.join(args.data_root_dir, 'features_chief'),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'neg':0, 'pos':1},
                            patient_strat= False,
                            ignore=[])
elif args.task == 'task_3_tumor_H3K27M_all':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '/ailab/user/liumianxin/CLAM/dataset_csv/tumor_H3K27M_dummy_all.csv',
                            data_dir= os.path.join(args.data_root_dir, 'features_chief'),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'neg':0, 'pos':1},
                            patient_strat= False,
                            ignore=[])
elif args.task == 'task_3_tumor_ATRX_all':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '/ailab/user/liumianxin/CLAM/dataset_csv/tumor_ATRX_dummy_all.csv',
                            data_dir= os.path.join(args.data_root_dir, 'features_chief'),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'neg':1, 'pos':0},
                            patient_strat= False,
                            ignore=[])
elif args.task == 'task_3_tumor_P53_all':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '/ailab/user/liumianxin/CLAM/dataset_csv/tumor_P53_dummy_all.csv',
                            data_dir= os.path.join(args.data_root_dir, 'features_chief'),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'neg':0, 'pos':1},
                            patient_strat= False,
                            ignore=[])
elif args.task == 'task_3_tumor_H3K27M_rec':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '/ailab/user/liumianxin/CLAM/dataset_csv/tumor_H3K27M_dummy_Rec.csv',
                            data_dir= os.path.join(args.data_root_dir, 'features_chief'),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'neg':0, 'pos':1},
                            patient_strat= False,
                            ignore=[])
elif args.task == 'task_3_tumor_ATRX_rec':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '/ailab/user/liumianxin/CLAM/dataset_csv/tumor_ATRX_dummy_Rec.csv',
                            data_dir= os.path.join(args.data_root_dir, 'features_chief'),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'neg':1, 'pos':0},
                            patient_strat= False,
                            ignore=[])
elif args.task == 'task_3_tumor_P53_rec':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '/ailab/user/liumianxin/CLAM/dataset_csv/tumor_P53_dummy_Rec.csv',
                            data_dir= os.path.join(args.data_root_dir, 'features_chief'),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'neg':0, 'pos':1},
                            patient_strat= False,
                            ignore=[])
elif args.task == 'task_3_tumor_H3K27M_rec_all':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '/ailab/user/liumianxin/CLAM/dataset_csv/tumor_H3K27M_dummy_RecAll.csv',
                            data_dir= os.path.join(args.data_root_dir, 'features_chief'),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'neg':0, 'pos':1},
                            patient_strat= False,
                            ignore=[])
elif args.task == 'task_3_tumor_ATRX_rec_all':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '/ailab/user/liumianxin/CLAM/dataset_csv/tumor_ATRX_dummy_RecAll.csv',
                            data_dir= os.path.join(args.data_root_dir, 'features_chief'),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'neg':1, 'pos':0},
                            patient_strat= False,
                            ignore=[])
elif args.task == 'task_3_tumor_P53_rec_all':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '/ailab/user/liumianxin/CLAM/dataset_csv/tumor_P53_dummy_RecAll.csv',
                            data_dir= os.path.join(args.data_root_dir, 'features_chief'),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'neg':0, 'pos':1},
                            patient_strat= False,
                            ignore=[])
else:
    raise NotImplementedError

if args.k_start == -1:
    start = 0
else:
    start = args.k_start
if args.k_end == -1:
    end = args.k
else:
    end = args.k_end

if args.fold == -1:
    folds = range(start, end)
else:
    folds = range(args.fold, args.fold+1)
ckpt_paths = [os.path.join(args.models_dir, 's_{}_checkpoint.pt'.format(fold)) for fold in folds]
datasets_id = {'train': 0, 'val': 1, 'test': 2, 'all': -1}

if __name__ == "__main__":
    all_results = []
    all_auc = []
    all_acc = []
    for ckpt_idx in range(len(ckpt_paths)):
        if datasets_id[args.split] < 0:
            split_dataset = dataset
        else:
            csv_path = '{}/splits_{}.csv'.format(args.splits_dir, folds[ckpt_idx])
            datasets = dataset.return_splits(from_id=False, csv_path=csv_path)
            split_dataset = datasets[datasets_id[args.split]]
        model, patient_results, test_error, auc, df  = eval(split_dataset, args, ckpt_paths[ckpt_idx])
        all_results.append(all_results)
        all_auc.append(auc)
        all_acc.append(1-test_error)
        df.to_csv(os.path.join(args.save_dir, 'fold_{}.csv'.format(folds[ckpt_idx])), index=False)

    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_auc, 'test_acc': all_acc})
    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(folds[0], folds[-1])
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.save_dir, save_name))
