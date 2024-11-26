import pdb
import os
import pandas as pd
import numpy as np
import json
import glob

datapath = '/ailab/group/pjlab-smarthealth03/transfers_cpfs_test/liumianxin/CLAM_DATA2/features_p2_nm/h5_files/'
csvpath ='/ailab/user/liumianxin/CLAM/dataset_csv/'
ave_datapath = '/ailab/group/pjlab-smarthealth03/transfers_cpfs_test/liumianxin/CLAM_DATA2/features_p2_nm/'

datalist = os.listdir(datapath)

for i in range(len(datalist)):
    if len(datalist[i].split('-'))==2:
        datalist[i] = datalist[i].split('.')[0] + '-1.pt'

print(sorted(datalist))
datalist=sorted(datalist)
tabledata=[]

csvpath ='/ailab/user/liumianxin/CLAM/dataset_csv/'
datatable = pd.read_csv(csvpath+'Training.csv')

for i in range(len(datalist)):
    case_id=datalist[i].split(' ')[0]
    slide_id=datalist[i].split('.')[0]
    hop_id=datalist[i].split(' ')[1]
    hop_id=hop_id.split('.')[0]
    hop_id=hop_id.split('-')[0]+'-'+hop_id.split('-')[1]
    if i!=(len(datalist)-1):
        next_id=datalist[i+1].split(' ')[0]
    if case_id==next_id:
        pos = datatable.loc[datatable['ID']==hop_id]
        if not pos.empty:
            if pos["ATRX"].values[0]==1:
                label = 'pos'
            elif pos["ATRX"].values[0]==0:
                label = 'neg'
            else:
                label = ' '
            if slide_id.split('-')[2]=='1':
                slide_id = slide_id.split('-')[0]+'-'+slide_id.split('-')[1]
            if label!=' ':
                tabledata.append([case_id, slide_id, label])

df = pd.DataFrame(tabledata, columns=['case_id','slide_id','label'])
csvname = csvpath+'tumor_ATRX_dummy_neg.csv'
print(len(df))
# df=df.drop_duplicates(subset=['case_id'])
# print(len(df))
df.to_csv(csvname, index=False)
