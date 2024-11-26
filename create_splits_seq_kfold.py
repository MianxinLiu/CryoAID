import os
import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import KFold

gene='ATRX'
csvpath ='/ailab/user/liumianxin/CLAM/dataset_csv/'
outpath ='/ailab/user/liumianxin/CLAM/splits/'

datatable = pd.read_csv(csvpath+'tumor_'+gene+'_dummy_all.csv')

kf = KFold(n_splits=10)
datalist = datatable['case_id'].values
datalist = np.unique(np.array(datalist))
slide_id = datatable['slide_id'].values


for i, (train_index, test_index) in enumerate(kf.split(datalist)):
    patch_train = []
    patch_test = []

    train_pos = 0
    val_pos = 0
    test_pos = 0
    train_neg = 0
    val_neg = 0
    test_neg = 0

    for j in range(len(test_index)):
        tmp = datatable[datatable['case_id']==datalist[test_index][j]]['slide_id'].values
        for k in range(len(tmp)):
            patch_test.append(tmp[k])
        if datatable[datatable['case_id']==datalist[test_index][j]]['label'].values[0] == 'pos':
            test_pos+=len(tmp)
        else:
            test_neg+=len(tmp)

    for j in range(len(train_index)):
        tmp = datatable[datatable['case_id']==datalist[train_index][j]]['slide_id'].values
        for k in range(len(tmp)):
            patch_train.append(tmp[k])
        if datatable[datatable['case_id']==datalist[train_index][j]]['label'].values[0] == 'pos':
            train_pos+=len(tmp)
        else:
            train_neg+=len(tmp)

    assert len(np.intersect1d(patch_train, patch_test)) == 0

    patch_all = patch_train+patch_test
    patch_all_cache = [patch_train, patch_test]

    index = patch_all
    one_hot = np.eye(2).astype(bool)
    bool_array = np.repeat(one_hot, [len(dset) for dset in patch_all_cache], axis=0)
    print([len(dset) for dset in patch_all_cache])
    print(len(bool_array))
    df = pd.DataFrame(bool_array, index=index, columns = ['train', 'test'])
    df.to_csv(outpath+'task_tumor_'+gene+'_all_100/splits_'+str(i)+'_bool.csv')

    record = len(patch_test)
    for j in range(record,len(patch_train)):
        patch_test.append('')

    d = {'train':patch_train, 'test': patch_test}
    df = pd.DataFrame(d)
    df.to_csv(outpath+'task_tumor_'+gene+'_all_100/splits_'+str(i)+'.csv')

    d = {'train':[train_pos, train_neg],'val':[val_pos, val_neg],'test':[test_pos, test_neg]}
    df = pd.DataFrame(d, index=['pos','neg'])
    df.to_csv(outpath+'task_tumor_'+gene+'_all_100/splits_'+str(i)+'_descriptor.csv')
        




