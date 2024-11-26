import os
import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import KFold

gene='ATRX'
csvpath ='/ailab/user/liumianxin/CLAM/dataset_csv/'
outpath ='/ailab/user/liumianxin/CLAM/splits/'
i=0
train_pos = 0
val_pos = 0
test_pos = 0
train_neg = 0
val_neg = 0
test_neg = 0

datatable = pd.read_csv(csvpath+'tumor_'+gene+'_dummy_all.csv')
train_index = datatable['slide_id'].values

datalist = datatable['case_id'].values
datalist = np.unique(np.array(datalist))
for j in range(len(datalist)):
    tmp = datatable[datatable['case_id']==datalist[j]]['slide_id'].values
    if datatable[datatable['case_id']==datalist[j]]['label'].values[0] == 'pos':
        train_pos+=len(tmp)
    else:
        train_neg+=len(tmp)

datatable2 = pd.read_csv(csvpath+'tumor_'+gene+'_dummy_RecAll.csv')
test_index = datatable2['slide_id'].values

datalist = datatable2['case_id'].values
datalist = np.unique(np.array(datalist))
for j in range(len(datalist)):
    tmp = datatable2[datatable2['case_id']==datalist[j]]['slide_id'].values
    if datatable2[datatable2['case_id']==datalist[j]]['label'].values[0] == 'pos':
        test_pos+=len(tmp)
    else:
        test_neg+=len(tmp)

# datatable = pd.concat((datatable,datatable2))
# datatable.reset_index(inplace=True, drop=True) 
# datatable.to_csv(csvpath+'tumor_'+gene+'_dummy_all2.csv')


assert len(np.intersect1d(train_index, test_index)) == 0

train_index = train_index.tolist()
test_index = test_index.tolist()
patch_all = train_index+test_index
patch_all_cache = [train_index, test_index]

index = patch_all
one_hot = np.eye(2).astype(bool)
bool_array = np.repeat(one_hot, [len(dset) for dset in patch_all_cache], axis=0)
print([len(dset) for dset in patch_all_cache])
print(len(bool_array))
df = pd.DataFrame(bool_array, index=index, columns = ['train', 'test'])
df.to_csv(outpath+'task_tumor_'+gene+'_all2_100/splits_'+str(i)+'_bool.csv')

record = len(test_index)
for j in range(record,len(train_index)):
    test_index.append('')

d = {'train':train_index, 'test': test_index}
df = pd.DataFrame(d)
df.to_csv(outpath+'task_tumor_'+gene+'_all2_100/splits_'+str(i)+'.csv')

d = {'train':[train_pos, train_neg],'val':[val_pos, val_neg],'test':[test_pos, test_neg]}
df = pd.DataFrame(d, index=['pos','neg'])
df.to_csv(outpath+'task_tumor_'+gene+'_all2_100/splits_'+str(i)+'_descriptor.csv')
    




