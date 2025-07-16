from .dataset_generic import Generic_MIL_Dataset
import os


def get_survival_dataset(task, seed=119, data_root_dir = None):
    study = '_'.join(task.split('_')[:2])
    if study == 'tcga_kirc' or study == 'tcga_kirp':
        combined_study = 'tcga_kidney'
    elif study == 'tcga_luad' or study == 'tcga_lusc':
        combined_study = 'tcga_lung'
    else:
        combined_study = study
    # combined_study = combined_study.split('_')[1]
    csv_path = 'dataset_csv/survival_by_case/{}_Splits.csv'.format(combined_study)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    
    # dataset = Generic_MIL_Survival_Dataset(csv_path = 'dataset_csv/%s_processed.csv' % combined_study,
    print(csv_path)
    dataset = Generic_MIL_Survival_Dataset(csv_path = csv_path,
                                            data_dir= data_root_dir,
                                            shuffle = False, 
                                            seed = seed, 
                                            print_info = True,
                                            patient_strat= False,
                                            n_bins=4,
                                            label_col = 'survival_months',
                                            ignore=[])
    return dataset


def get_subtying_dataset(task, seed=119, data_dir=None):
    if task == 'LUAD_LUSC':
        dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/LUAD_LUSC.csv',
                                data_dir= data_dir,
                                shuffle = False, 
                                seed = seed, 
                                print_info = True,
                                label_dict = {'LUAD':0, 'LUSC':1},
                                patient_strat=False,
                                ignore=[])

    elif task == 'camelyon':
        dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/camelyon.csv',
                                data_dir= None,
                                shuffle = False, 
                                seed = seed, 
                                print_info = True,
                                label_dict = {'normal':0, 'tumor':1},
                                patient_strat= False,
                                ignore=[])
        
    elif task == 'RCC':
        dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/RCC.csv',
                                data_dir= None,
                                shuffle = False, 
                                seed = seed, 
                                print_info = True,
                                label_dict = {'KICH':0, 'KIRP':1, 'KIRC':2},
                                patient_strat= False,
                                ignore=[])
 
    else:
        raise NotImplementedError
    return dataset
        