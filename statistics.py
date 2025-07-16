import os
import pandas as pd
import numpy as np
import glob
from sklearn import metrics


def Find_Optimal_Cutoff(target, predicted):
    fpr, tpr, threshold = metrics.roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold']) 


# AUC at times
gene = 'ATRX'
datapath = '/ailab/user/liumianxin/CLAM/eval_results/EVAL_all_'+gene+'/'

df = pd.read_csv(datapath+'fold_all.csv',index_col=None)
# for i in range(1,10):
#     df_tmp = pd.read_csv(datapath+'fold_'+str(i)+'.csv',index_col=None)
#     df = pd.concat((df,df_tmp))

datalist = df['slide_id'].values
Y = df['Y'].values
Y_hat = df['Y_hat'].values
p_1 = df['p_1'].values

Y_one = []
Y_hat_one = []
p_one = []
for i in range(len(datalist)):
    if len(datalist[i].split('-'))==2:
        Y_one.append(Y[i])
        Y_hat_one.append(Y_hat[i])
        p_one.append(p_1[i])

Y_two = []
Y_hat_two = []
p_two = []
for i in range(1,len(datalist)):
    if len(datalist[i].split('-'))==3 and datalist[i].split('-')[2] == '2' and datalist[i].split('-')[0] == datalist[i-1].split('-')[0]:
        # print(datalist[i],datalist[i-1])
        p = (p_1[i]+p_1[i-1])/2
        if p>0.5:
            y=1
        else:
            y=0
        # print(Y_hat[i])
        # print(p)
        Y_two.append(Y[i])
        Y_hat_two.append(y)
        p_two.append(p)

from sklearn import metrics
print(len(Y_one))

threshold = Find_Optimal_Cutoff(Y_one,p_one)
print(threshold)
Y_hat_cut = []
for i in range(len(Y_one)):
    if p_one[i]>threshold:
        Y_hat_cut.append(1)
    else:
        Y_hat_cut.append(0)

correct = (np.array(Y_hat_cut) == np.array(Y_one)).sum()
accuracy = float(correct) / float(len(Y_one))
sens = metrics.recall_score(Y_one, Y_hat_cut, pos_label=1)
spec = metrics.recall_score(Y_one, Y_hat_cut, pos_label=0)
auc = metrics.roc_auc_score(Y_one, p_one)
print('|test accuracy:', accuracy,
        '|test sen:', sens,
        '|test spe:', spec,
        '|test auc:', auc,
        )

d = {'Y_one':Y_one, 'p_one': p_one, 'Y_hat_cut': Y_hat_cut}
df = pd.DataFrame(d)
# df.to_csv('/ailab/user/liumianxin/CLAM/statistics/first_auc'+gene+'.csv')

from sklearn import metrics
print(len(Y_two))

threshold = Find_Optimal_Cutoff(Y_two, p_two)
print(threshold)
Y_hat_cut = []
for i in range(len(Y_two)):
    if p_two[i]>threshold:
        Y_hat_cut.append(1)
    else:
        Y_hat_cut.append(0)

correct = (np.array(Y_hat_cut) == np.array(Y_two)).sum()
accuracy = float(correct) / float(len(Y_two))
sens = metrics.recall_score(Y_two, Y_hat_cut, pos_label=1)
spec = metrics.recall_score(Y_two, Y_hat_cut, pos_label=0)
auc = metrics.roc_auc_score(Y_two, p_two)

print('|test accuracy:', accuracy,
        '|test sen:', sens,
        '|test spe:', spec,
        '|test auc:', auc,
        )

d = {'Y_two':Y_two, 'p_two': p_two, 'Y_hat_cut': Y_hat_cut}
df = pd.DataFrame(d)
# df.to_csv('/ailab/user/liumianxin/CLAM/statistics/second_auc'+gene+'.csv')


## Times reduced
datapath = '/ailab/user/liumianxin/CLAM/eval_results/EVAL_all_H3K27M/'
df = pd.read_csv(datapath+'fold_'+str(0)+'.csv',index_col=None)
datalist = df['slide_id'].values
Y = df['Y'].values
Y_hat = df['Y_hat'].values
p_1 = df['p_1'].values
threshold = Find_Optimal_Cutoff(Y, df['p_1'].values)
print(threshold)
Y_hat_cut = []
for i in range(len(Y)):
    if df['p_1'].values[i]>threshold:
        Y_hat_cut.append(1)
    else:
        Y_hat_cut.append(0)
df.iloc[:,2]=Y_hat_cut
df1=df

for i in range(1,10):
    df = pd.read_csv(datapath+'fold_'+str(i)+'.csv',index_col=None)
    datalist = df['slide_id'].values
    Y = df['Y'].values
    Y_hat = df['Y_hat'].values
    p_1 = df['p_1'].values
    threshold = Find_Optimal_Cutoff(Y, df['p_1'].values)
    print(threshold)
    Y_hat_cut = []
    for i in range(len(Y)):
        if df['p_1'].values[i]>threshold:
            Y_hat_cut.append(1)
        else:
            Y_hat_cut.append(0)
    df.iloc[:,2]=Y_hat_cut

    df1 = pd.concat((df1,df))

datapath = '/ailab/user/liumianxin/CLAM/eval_results/EVAL_all_ATRX/'

df = pd.read_csv(datapath+'fold_'+str(0)+'.csv',index_col=None)
datalist = df['slide_id'].values
Y = df['Y'].values
Y_hat = df['Y_hat'].values
p_1 = df['p_1'].values
threshold = Find_Optimal_Cutoff(Y, df['p_1'].values)
print(threshold)
Y_hat_cut = []
for i in range(len(Y)):
    if df['p_1'].values[i]>threshold:
        Y_hat_cut.append(1)
    else:
        Y_hat_cut.append(0)
df.iloc[:,2]=Y_hat_cut
df2=df

for i in range(1,10):
    df = pd.read_csv(datapath+'fold_'+str(i)+'.csv',index_col=None)
    datalist = df['slide_id'].values
    Y = df['Y'].values
    Y_hat = df['Y_hat'].values
    p_1 = df['p_1'].values
    threshold = Find_Optimal_Cutoff(Y, df['p_1'].values)
    print(threshold)
    Y_hat_cut = []
    for i in range(len(Y)):
        if df['p_1'].values[i]>threshold:
            Y_hat_cut.append(1)
        else:
            Y_hat_cut.append(0)
    df.iloc[:,2]=Y_hat_cut
    df2 = pd.concat((df2,df))

df2.columns = ['slide_id','Y2','Y_hat2','p_02','p_12']

datapath = '/ailab/user/liumianxin/CLAM/eval_results/EVAL_all_P53/'
df = pd.read_csv(datapath+'fold_'+str(0)+'.csv',index_col=None)
datalist = df['slide_id'].values
Y = df['Y'].values
Y_hat = df['Y_hat'].values
p_1 = df['p_1'].values
threshold = Find_Optimal_Cutoff(Y, df['p_1'].values)
print(threshold)
Y_hat_cut = []
for i in range(len(Y)):
    if df['p_1'].values[i]>threshold:
        Y_hat_cut.append(1)
    else:
        Y_hat_cut.append(0)
df.iloc[:,2]=Y_hat_cut
df3=df

for i in range(1,10):
    df = pd.read_csv(datapath+'fold_'+str(i)+'.csv',index_col=None)
    datalist = df['slide_id'].values
    Y = df['Y'].values
    Y_hat = df['Y_hat'].values
    p_1 = df['p_1'].values
    threshold = Find_Optimal_Cutoff(Y, df['p_1'].values)
    print(threshold)
    Y_hat_cut = []
    for i in range(len(Y)):
        if df['p_1'].values[i]>threshold:
            Y_hat_cut.append(1)
        else:
            Y_hat_cut.append(0)
    df.iloc[:,2]=Y_hat_cut

    df3 = pd.concat((df3,df))

df3.columns = ['slide_id','Y3','Y_hat3','p_03','p_13']

df1 = df1.merge(df2, on='slide_id')
df1 = df1.merge(df3, on='slide_id')
df = df1

datalist = df['slide_id'].values
Y1 = df['Y'].values
Y_hat1 = df['Y_hat'].values
Y2 = df['Y2'].values
Y_hat2 = df['Y_hat2'].values
Y3 = df['Y3'].values
Y_hat3 = df['Y_hat3'].values

datalist=np.append(datalist, 'abc abc')

time = []
sub = []
count = 1
for i in range(len(datalist)-1):
    if Y1[i].size>0:
        if datalist[i].split(' ')[0] == datalist[i+1].split(' ')[0]:
            count+=1
        else:
            time.append(count)
            count = 1
            sub.append(datalist[i].split(' ')[1])
print(time)
print(len(time))
print(np.mean(time))
print(np.sum(time))
time_human = time

time = []
count = 1
model_flag = 0
for i in range(len(datalist)-1):
    if Y1[i].size>0:
        if datalist[i].split(' ')[0] == datalist[i+1].split(' ')[0]:
            if Y_hat1[i]==1 and Y1[i]==1:
                model_flag = 1
            if Y_hat2[i]==1 and Y2[i]==1:
                model_flag = 1
            if Y_hat3[i]==1 and Y3[i]==1:
                model_flag = 1
            if model_flag==0:
                count+=1
        else:
            time.append(count)
            count = 1
            model_flag = 0
print(time)
print(np.mean(time))
temp = np.array(time_human)-np.array(time)
print(temp)
print(temp.sum())

d = {'sub_id':sub, 'human': time_human, 'AI': time}
df = pd.DataFrame(d)
df.to_csv('/ailab/user/liumianxin/CLAM/statistics/times2.csv')

## Times reduced external
datapath = '/ailab/user/liumianxin/CLAM/eval_results/EVAL_all_H3K27M_rec_train/'

df = pd.read_csv(datapath+'fold_0.csv',index_col=None)

df1=df
print(len(df1))
df1=df1.drop_duplicates(subset=['slide_id'])
print(len(df1))
datalist = df['slide_id'].values
Y = df['Y'].values
Y_hat = df['Y_hat'].values
p_1 = df['p_1'].values


threshold = Find_Optimal_Cutoff(Y, df1['p_1'].values)
print(threshold)
Y_hat_cut = []
for i in range(len(Y)):
    if df1['p_1'].values[i]>threshold:
        Y_hat_cut.append(1)
    else:
        Y_hat_cut.append(0)
df1.iloc[:,2]=Y_hat_cut

datapath = '/ailab/user/liumianxin/CLAM/eval_results/EVAL_all_ATRX_rec_train/'

df = pd.read_csv(datapath+'fold_0.csv',index_col=None)

df2=df
print(len(df2))
df2=df2.drop_duplicates(subset=['slide_id'])
print(len(df2))
datalist = df['slide_id'].values
Y = df['Y'].values
Y_hat = df['Y_hat'].values
p_2 = df['p_1'].values

threshold = Find_Optimal_Cutoff(Y, df2['p_1'].values)
print(threshold)
Y_hat_cut = []
for i in range(len(Y)):
    if df2['p_1'].values[i]>threshold:
        Y_hat_cut.append(1)
    else:
        Y_hat_cut.append(0)
df2.iloc[:,2]=Y_hat_cut
df2.columns = ['slide_id','Y2','Y_hat2','p_02','p_12']

datapath = '/ailab/user/liumianxin/CLAM/eval_results/EVAL_all_P53_rec_train/'

df = pd.read_csv(datapath+'fold_0.csv',index_col=False)

df3=df
df3=df3.drop_duplicates(subset=['slide_id'])
datalist = df['slide_id'].values
Y = df['Y'].values
Y_hat = df['Y_hat'].values
p_1 = df['p_1'].values

threshold = Find_Optimal_Cutoff(Y, df3['p_1'].values)
print(threshold)
Y_hat_cut = []
for i in range(len(Y)):
    if df3['p_1'].values[i]>threshold:
        Y_hat_cut.append(1)
    else:
        Y_hat_cut.append(0)
df3.iloc[:,2]=Y_hat_cut
df3.columns = ['slide_id','Y3','Y_hat3','p_03','p_13']

df1 = df1.merge(df2, on='slide_id')
df1 = df1.merge(df3, on='slide_id')
df = df1
# df.to_csv('/ailab/user/liumianxin/CLAM/statistics/pred_ext.csv')

datalist = df['slide_id'].values
Y1 = df['Y'].values
Y_hat1 = df['Y_hat'].values

Y2 = df['Y'].values
Y_hat2 = df['Y_hat2'].values

Y3 = df['Y3'].values
Y_hat3 = df['Y_hat3'].values

datalist=np.append(datalist, 'abc-abc')
time = []
count = 1
for i in range(len(datalist)-1):
    if datalist[i].split('-')[1] == datalist[i+1].split('-')[1]:
        count+=1
    else:
        time.append(count)
        count = 1

print(time)
print(len(time))
print(np.mean(time))
print(np.sum(time))
time_human = time

time = []
count = 1
model_flag = 0
for i in range(len(datalist)-1):
    if datalist[i].split('-')[1] == datalist[i+1].split('-')[1]:
        if Y_hat1[i]==1 and Y1[i]==1:
            model_flag = 1
        if Y_hat2[i]==1 and Y2[i]==1:
            model_flag = 1
        if Y_hat3[i]==1 and Y3[i]==1:
            model_flag = 1
        if model_flag==0:
            count+=1
    else:
        time.append(count)
        count = 1
        model_flag = 0

print(time)
print(np.mean(time))
temp = np.array(time_human)-np.array(time)
# print(temp)
print(temp.sum())

d = {'human': time_human, 'AI': time}
df = pd.DataFrame(d)
df.to_csv('/ailab/user/liumianxin/CLAM/statistics/times_ext.csv')