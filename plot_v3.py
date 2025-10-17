import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from random import shuffle
import numpy as np
from sklearn import metrics
from compare_auc_delong_xu import delong_roc_variance
import scipy.stats as stats

def compute_permu_aucP(pred,y_test):
    roc_auc = metrics.roc_auc_score(y_test, pred)
    count=0
    for i in range(1000):
        permu_index=y_test.copy()
        shuffle(permu_index)

        auc_temp = metrics.roc_auc_score(permu_index, pred)
        if auc_temp>roc_auc:
            count=count+1
    return (count / 1000)

def compute_permu_P(pred,y_test):
    correct = (np.array(pred) == np.array(y_test)).sum()
    acc = float(correct) / float(len(y_test))
    sen = metrics.recall_score(y_test, pred, pos_label=1)
    spe = metrics.recall_score(y_test, pred, pos_label=0)

    count_acc = 0
    count_sen = 0
    count_spe = 0
    for i in range(1000):
        permu_index=y_test.copy()
        shuffle(permu_index)

        correct = (np.array(pred) == np.array(permu_index)).sum()
        acc_temp = float(correct) / float(len(permu_index))
        sen_temp = metrics.recall_score(permu_index, pred, pos_label=1)
        spe_temp = metrics.recall_score(permu_index, pred, pos_label=0)

        if acc_temp>acc:
            count_acc=count_acc+1
        if sen_temp>sen:
            count_sen=count_sen+1
        if spe_temp>spe:
            count_spe=count_spe+1
    return (count_acc / 1000), (count_sen / 1000),(count_spe / 1000)

def compute_permu_P_F1(pred,y_test):
    f1 = metrics.f1_score(y_test, pred, pos_label=1)

    count_f1 = 0
    for i in range(1000):
        permu_index=y_test.copy()
        shuffle(permu_index)

        f1_temp = metrics.f1_score(permu_index, pred, pos_label=1)

        if f1_temp>f1:
            count_f1=count_f1+1
    return (count_f1 / 1000)

def AUC_CI(y_true, y_pred):
    auc, auc_cov = delong_roc_variance(
        y_true,
        y_pred)

    auc_std = np.sqrt(auc_cov)
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - 0.95) / 2)

    ci = stats.norm.ppf(
        lower_upper_q,
        loc=auc,
        scale=auc_std)

    ci[ci > 1] = 1
    return ci

def Find_Optimal_Cutoff(target, predicted):
    fpr, tpr, threshold = metrics.roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold'])

def find_best_f1_threshold(y_true, y_pred_prob, step=0.01):
    thresholds = np.arange(0.0, 1.0, step)
    f1_scores = [metrics.f1_score(y_true, (y_pred_prob >= t).astype(int)) for t in thresholds]

    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    return best_threshold

savepath = '/home/PJLAB/liumianxin/Desktop/AI_FFPE/figures/'

# Internal boxplot
data = pd.read_csv('./DataFrames/Comp_FM.csv')
genelist= ['ATRX', 'H3K27M', 'P53']
y = data[(data['Method']=='CryoAIMD') & (data['Gene']== genelist[2])]['AUC'].values
print(np.mean(y))
print(np.std(y))
stats.t.interval(confidence=0.95, df=len(y)-1, loc=np.mean(y, axis=0), scale=stats.sem(y, axis=0))


data = pd.read_csv('./DataFrames/Comp_FM.csv')
plt.figure(figsize=(10, 10))
sns.set_theme(style='white', font_scale=2.0)
palette = sns.color_palette("mako")
test=sns.boxplot(x='Gene', y='AUC', hue = 'Method', data = data, palette=palette)
shift = [-0.335, -0.2, -0.065, 0.07, 0.205]
methodlist = ['ResNet', 'Pathoduet', 'UNI', 'Virchow2', 'Gigapath']
genelist= ['ATRX', 'H3K27M', 'TP53']
for i in range(3):
    for j in range(5):
        x = data[(data['Method']==methodlist[j]) & (data['Gene']== genelist[i])]['AUC'].values
        y = data[(data['Method']=='CryoAID') & (data['Gene']== genelist[i])]['AUC'].values
        p = stats.ttest_ind(x, y, alternative='less')
        print(p)
        if p[1]<0.05:
            pval = '*'
            plt.text(x=i+shift[j]-0.02, y=1, s=pval)
        if p[1]<0.01:
            pval = '*'
            plt.text(x=i+shift[j]-0.02, y=1.05, s=pval)
        if p[1]<0.001:
            pval = '*'
            plt.text(x=i+shift[j]-0.02, y=1.1, s=pval)
plt.ylim([0.3, 1.15])
plt.title('Comparison on FMs')
plt.legend(loc='lower right')
plt.savefig(savepath + 'Fig1A.tif')

# Internal boxplot
data = pd.read_csv('./DataFrames/Comp_CLS.csv')
plt.figure(figsize=(8, 10))
sns.set_theme(style='white', font_scale=2.0)
palette = sns.color_palette("rocket")
test=sns.boxplot(x='Gene', y='AUC', hue = 'Method', data = data, palette=palette)

shift = [-0.31, -0.12, 0.08]
methodlist = ['TransMIL', 'ABMIL', 'CLAM', 'CryoAID']
genelist= ['ATRX', 'H3K27M', 'TP53']
for i in range(3):
    for j in range(3):
        x = data[(data['Method']==methodlist[j]) & (data['Gene']== genelist[i])]['AUC'].values
        y = data[(data['Method']=='CryoAID') & (data['Gene']== genelist[i])]['AUC'].values
        p = stats.ttest_ind(x, y, alternative='less')
        print(p)
        if p[1]<0.05:
            pval = '*'
            plt.text(x=i+shift[j]-0.02, y=1, s=pval)
        if p[1]<0.01:
            pval = '*'
            plt.text(x=i+shift[j]-0.02, y=1.05, s=pval)
        if p[1]<0.001:
            pval = '*'
            plt.text(x=i+shift[j]-0.02, y=1.1, s=pval)
plt.ylim([0.3, 1.15])
plt.legend(loc='lower right')
plt.title('Comparison on Classifier')
plt.savefig(savepath + 'Fig1B.tif')

data = pd.read_csv('./DataFrames/Comp_gen.csv')
sns.set_theme(style='white', font_scale=2.0)
plt.figure(figsize=(7.5, 10))
palette = sns.color_palette("Paired")
test=sns.boxplot(x='Gene', y='AUC', hue = 'Method', data = data, palette=palette)

shift = [-0.22, 0.1]
methodlist = ['w/o AI-FFPE', 'CryoAID']
genelist= ['ATRX', 'H3K27M', 'TP53']
for i in range(3):
    for j in range(2):
        x = data[(data['Method']==methodlist[j]) & (data['Gene']== genelist[i])]['AUC'].values
        y = data[(data['Method']=='CryoAID') & (data['Gene']== genelist[i])]['AUC'].values
        p = stats.ttest_ind(x, y, alternative='less')
        print(p)
        if p[1]<0.05:
            pval = '*'
            plt.text(x=i+shift[j]-0.02, y=1, s=pval)
        if p[1]<0.01:
            pval = '*'
            plt.text(x=i+shift[j]-0.02, y=1.05, s=pval)
        if p[1]<0.001:
            pval = '*'
            plt.text(x=i+shift[j]-0.02, y=1.1, s=pval)
plt.ylim([0.3, 1.15])
plt.legend(loc='lower right')
plt.title('Effect of AI-FFPE')
plt.savefig(savepath + 'Fig1C.tif')

# Internal AUC
sns.set_theme(style='white', font_scale=2.5)
plt.figure(figsize=(10, 10))
cmap = plt.get_cmap('mako')
colors = [cmap(i) for i in np.linspace(0, 0.5, 3)]
index = 0
for gene in ['ATRX','H3K27M','TP53']:
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 326)

    for i in range(10):
        data = pd.read_csv('./eval_results/EVAL_pos_' + gene + '/fold_'+str(i)+'.csv')
        pred = data['p_1'].values
        y_test = data['Y'].values
        fpr, tpr, threshold = metrics.roc_curve(y_test, pred)
        roc_auc = metrics.auc(fpr, tpr)
        aucs.append(roc_auc)

        # 插值，使每一折的TPR对应统一的FPR（才能平均）
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)

    # 平均与标准差
    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)
    mean_auc = metrics.auc(mean_fpr, mean_tpr)

    data = pd.read_csv('./eval_results/EVAL_pos_'+gene+'/fold_all.csv')
    pred = data['p_1'].values
    y_test = data['Y'].values
    fpr, tpr, threshold = metrics.roc_curve(y_test, pred)
    P = compute_permu_aucP(pred, y_test)
    plt.title('Internal ROC')
    plt.plot(mean_fpr, mean_tpr, label = gene+' AUC = %0.3f P=%0.3f' % (mean_auc, P), linewidth=3.0, color=colors[index])
    index+=1

plt.legend(loc = 'lower right', fontsize=25)
plt.plot([0, 1], [0, 1],'k--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
plt.savefig(savepath + 'Fig1D.tif')

# Internal metrics
accuracy_p, f1_p, sens_p, spec_p = [], [], [], []
for gene in ['ATRX','H3K27M','TP53']:
    datapath = './eval_results/EVAL_pos_'+gene+'/'

    accuracy =[]
    f1 = []
    sens =[]
    spec =[]
    pre_all = []
    pre_all2 = []
    Y_all = []
    for i in range(0,10):
        df = pd.read_csv(datapath+'fold_'+str(i)+'.csv',index_col=None)
        datalist = df['slide_id'].values
        Y = df['Y'].values
        Y_hat = df['Y_hat'].values
        p_1 = df['p_1'].values

        threshold = Find_Optimal_Cutoff(Y,p_1)
        Y_hat_cut = []
        for i in range(len(p_1)):
            if p_1[i]>threshold:
                Y_hat_cut.append(1)
            else:
                Y_hat_cut.append(0)

        threshold = find_best_f1_threshold(Y,p_1)
        Y_hat_cut2 = []
        for i in range(len(p_1)):
            if p_1[i]>threshold:
                Y_hat_cut2.append(1)
            else:
                Y_hat_cut2.append(0)

        correct = (np.array(Y_hat_cut) == np.array(Y)).sum()
        accuracy.append(float(correct) / float(len(Y)))
        f1.append(metrics.f1_score(Y, Y_hat_cut2))
        sens.append(metrics.recall_score(Y, Y_hat_cut, pos_label=1))
        spec.append(metrics.recall_score(Y, Y_hat_cut, pos_label=0))
        pre_all += Y_hat_cut
        pre_all2 += Y_hat_cut2
        Y_all += Y.tolist()

    p1, p2, p3 = compute_permu_P(pre_all, Y_all)
    accuracy_p.append(p1)
    sens_p.append(p2)
    spec_p.append(p3)
    p4 = compute_permu_P_F1(pre_all2, Y_all)
    f1_p.append(p4)

    d = {'accuracy':accuracy, 'f1':f1, 'sens': sens, 'spec': spec}
    df = pd.DataFrame(d)
    df.to_csv('./DataFrames/Inter_metrics_'+gene+'.csv')

# data = pd.read_csv('./DataFrames/Inter_metrics.csv')
sns.set_theme(style='white', font_scale=1.7)
# test=sns.boxplot(x='Gene', y='Value', hue = 'Metrics', data = data)
data = pd.read_csv('./DataFrames/Inter_metrics.csv')
plt.figure(figsize=(8, 8))
palette = sns.color_palette("viridis")
ax = sns.barplot(x="Gene", y="Value", hue='Metrics', data = data, capsize=.1, errorbar="sd", palette=palette)
positions = [patch.get_x() + patch.get_width() / 2 for patch in ax.patches]
heights = [patch.get_height() for patch in ax.patches]
p = accuracy_p+f1_p+sens_p+spec_p

for i in range(len(positions[:12])):
    if p[i] < 0.05:
        pval = '*'
        plt.text(x=positions[i]-0.04, y=0.95, s=pval)
    if p[i] < 0.01:
        pval = '*'
        plt.text(x=positions[i]-0.04, y=0.96, s=pval)
    if p[i] < 0.001:
        pval = '*'
        plt.text(x=positions[i]-0.04, y=0.97, s=pval)

sns.swarmplot(x="Gene", y="Value", hue='Metrics', data = data, alpha=.35, color="0", dodge=True, legend=False)
plt.ylim([0.4, 1])
plt.show()
plt.title('Internal Metrics')
plt.legend(loc = 'lower right', fontsize=25)
plt.savefig(savepath + 'Fig1G.tif')

data = pd.read_csv('./DataFrames/statistics_times2.csv')
genelist= ['ATRX', 'H3K27M', 'TP53']
y = data[(data['Metrics']=='AUC') & (data['Gene']== genelist[2])]['Value'].values
print(np.mean(y))
print(np.std(y))
stats.t.interval(confidence=0.95, df=len(y)-1, loc=np.mean(y, axis=0), scale=stats.sem(y, axis=0))


# External AUC
sns.set_theme(style='white', font_scale=2.5)
plt.figure(figsize=(10, 10))
cmap = plt.get_cmap('mako')
colors = [cmap(i) for i in np.linspace(0, 0.5, 3)]
index = 0
for gene in ['ATRX','H3K27M','TP53']:
    data = pd.read_csv('./eval_results/EVAL_pos_'+gene+'_rec_train/fold_0.csv')
    pred = data['p_1'].values
    y_test = data['Y'].values
    fpr, tpr, threshold = metrics.roc_curve(y_test, pred)
    roc_auc = metrics.auc(fpr, tpr)
    CI = AUC_CI(y_test, pred)
    P = compute_permu_aucP(pred, y_test)
    print(CI)
    plt.title('External ROC-Consecutive')
    plt.plot(fpr, tpr, label = gene+' AUC = %0.3f P=%0.3f' % (roc_auc, P), linewidth=3.0, color=colors[index])
    index+=1

plt.legend(loc = 'lower right', fontsize=25)
plt.plot([0, 1], [0, 1],'k--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
plt.savefig(savepath + 'Fig1E.tif')

sns.set_theme(style='white', font_scale=2.5)
plt.figure(figsize=(10, 10))
cmap = plt.get_cmap('mako')
colors = [cmap(i) for i in np.linspace(0, 0.5, 3)]
index = 0
for gene in ['ATRX','H3K27M','TP53']:
    data = pd.read_csv('./eval_results/EVAL_pos_'+gene+'_fujian_train/fold_0.csv')
    pred = data['p_1'].values
    y_test = data['Y'].values
    data = pd.read_csv('./eval_results/EVAL_pos_' + gene + '_north_train/fold_0.csv')
    pred2 = data['p_1'].values
    y_test2 = data['Y'].values
    pred = np.concatenate([pred,pred2])
    y_test = np.concatenate([y_test, y_test2])
    fpr, tpr, threshold = metrics.roc_curve(y_test, pred)
    roc_auc = metrics.auc(fpr, tpr)
    CI = AUC_CI(y_test, pred)
    P = compute_permu_aucP(pred, y_test)
    print(CI)
    plt.title('External ROC-Multisite')
    plt.plot(fpr, tpr, label = gene+' AUC = %0.3f P=%0.3f' % (roc_auc, P), linewidth=3.0, color=colors[index])
    index+=1

plt.legend(loc = 'lower right', fontsize=25)
plt.plot([0, 1], [0, 1],'k--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
plt.savefig(savepath + 'Fig1F.tif')

sns.set_theme(style='white', font_scale=2.5)
plt.figure(figsize=(10, 10))
cmap = plt.get_cmap('mako')
colors = [cmap(i) for i in np.linspace(0, 0.5, 3)]
index = 0
for gene in ['ATRX','H3K27M','TP53']:
    data = pd.read_csv('./eval_results/EVAL_pos_'+gene+'_north_train/fold_0.csv')
    pred = data['p_1'].values
    y_test = data['Y'].values
    fpr, tpr, threshold = metrics.roc_curve(y_test, pred)
    roc_auc = metrics.auc(fpr, tpr)
    CI = AUC_CI(y_test, pred)
    P = compute_permu_aucP(pred, y_test)
    print(CI)
    plt.title('External ROC-North')
    plt.plot(fpr, tpr, label = gene+' AUC = %0.3f P=%0.3f' % (roc_auc, P), linewidth=3.0, color=colors[index])
    index+=1

plt.legend(loc = 'lower right', fontsize=25)
plt.plot([0, 1], [0, 1],'k--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
plt.savefig(savepath + 'FigS2A.tif')

sns.set_theme(style='white', font_scale=2.5)
plt.figure(figsize=(10, 10))
cmap = plt.get_cmap('mako')
colors = [cmap(i) for i in np.linspace(0, 0.9, 5)]


tprs = []
aucs = []
pre_all = []
Y_all = []
YP_all = []
mean_fpr = np.linspace(0, 1, 301)

for fold in range(5):
    data = pd.read_csv('./eval_results/EVAL_task_3_tumor_ASTER/fold_'+str(fold)+'.csv')
    pred = data['p_1'].values
    y_test = data['Y'].values
    fpr, tpr, threshold = metrics.roc_curve(y_test, pred)
    roc_auc = metrics.auc(fpr, tpr)
    aucs.append(roc_auc)

    # 插值，使每一折的TPR对应统一的FPR（才能平均）
    tpr_interp = np.interp(mean_fpr, fpr, tpr)
    tpr_interp[0] = 0.0
    tprs.append(tpr_interp)

    threshold = find_best_f1_threshold(y_test, pred, step=0.0001)
    print(threshold)
    Y_hat_cut = []
    for i in range(len(y_test)):
        if pred[i] > threshold:
            Y_hat_cut.append(1)
        else:
            Y_hat_cut.append(0)
    pre_all += Y_hat_cut
    Y_all += y_test.tolist()
    YP_all += pred.tolist()
    print(metrics.f1_score(y_test, Y_hat_cut))

# 平均与标准差
mean_tpr = np.mean(tprs, axis=0)
std_tpr = np.std(tprs, axis=0)
mean_auc = metrics.auc(mean_fpr, mean_tpr)
P = compute_permu_aucP(YP_all, Y_all)

plt.figure(figsize=(10, 10))
plt.title('Internal ROC-Pilocytic Astrocytoma')
plt.plot(mean_fpr, mean_tpr, label = 'AUC = %0.3f P = %0.3f' % (mean_auc, P), linewidth=3.0)
plt.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, color='lightsteelblue', alpha=0.2)
plt.legend(loc = 'lower right', fontsize=25)
plt.plot([0, 1], [0, 1],'k--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
plt.savefig(savepath + 'FigS3A.tif')


p1, p3, p4 = compute_permu_P(pre_all, Y_all)
p2 = compute_permu_P_F1(pre_all, Y_all)
data = pd.read_csv('./DataFrames/statistics_aster.csv')
plt.figure(figsize=(10,8))
palette = sns.color_palette("viridis")
ax = sns.barplot(x="Metrics", y="Value", data = data, capsize=.1, ci="sd", palette=palette)

positions = [patch.get_x() + patch.get_width() / 2 for patch in ax.patches]
heights = [patch.get_height() for patch in ax.patches]
p = [p1, p2, p3, p4]

for i in range(len(positions)):
    if p[i] < 0.05:
        pval = '*'
        plt.text(x=positions[i]-0.03, y=0.95, s=pval)
    if p[i] < 0.01:
        pval = '*'
        plt.text(x=positions[i]-0.03, y=0.97, s=pval)
    if p[i] < 0.001:
        pval = '*'
        plt.text(x=positions[i]-0.03, y=0.99, s=pval)

sns.swarmplot(x="Metrics", y="Value", data = data, alpha=.5, color="0", dodge=True, legend=False)
plt.show()
plt.ylim(0,1.1)
plt.title('Metrics-Pilocytic Astrocytoma')
plt.savefig(savepath + 'FigS3B.tif')

# external metrics
accuracy=[]
f1=[]
sens=[]
spec=[]
accuracy_p=[]
f1_p=[]
sens_p=[]
spec_p=[]
for gene in ['ATRX','H3K27M','TP53']:
    data = pd.read_csv('./eval_results/EVAL_pos_'+gene+'_rec_train/fold_0.csv')
    pred = data['p_1'].values
    y_test = data['Y'].values

    # data = pd.read_csv('./eval_results/EVAL_pos_' + gene + '_fujian_train/fold_0.csv')
    # pred = data['p_1'].values
    # y_test = data['Y'].values
    # data = pd.read_csv('./eval_results/EVAL_pos_' + gene + '_north_train/fold_0.csv')
    # pred2 = data['p_1'].values
    # y_test2 = data['Y'].values
    # pred = np.concatenate([pred, pred2])
    # y_test = np.concatenate([y_test, y_test2])

    threshold = Find_Optimal_Cutoff(y_test, pred)
    # print(threshold)
    Y_hat_cut = []
    for i in range(len(y_test)):
        if pred[i] > threshold:
            Y_hat_cut.append(1)
        else:
            Y_hat_cut.append(0)
    threshold = find_best_f1_threshold(y_test, pred, step=0.0001)
    # print(threshold)
    Y_hat_cut2 = []
    for i in range(len(y_test)):
        if pred[i] > threshold:
            Y_hat_cut2.append(1)
        else:
            Y_hat_cut2.append(0)

    correct = (np.array(Y_hat_cut) == np.array(y_test)).sum()
    accuracy.append(float(correct) / float(len(y_test)))
    f1.append(metrics.f1_score(y_test, Y_hat_cut2, pos_label=1))
    sens.append(metrics.recall_score(y_test, Y_hat_cut, pos_label=1))
    spec.append(metrics.recall_score(y_test, Y_hat_cut, pos_label=0))
    p1, p3, p4 = compute_permu_P(Y_hat_cut, y_test)
    p2 = compute_permu_P_F1(Y_hat_cut2, y_test)
    accuracy_p.append(p1)
    f1_p.append(p2)
    sens_p.append(p3)
    spec_p.append(p4)


plt.figure(figsize=(8, 8))
data = pd.read_csv('./DataFrames/statistics_consecutive.csv')
sns.set_theme(style='white', font_scale=1.7)
palette = sns.color_palette("viridis")
ax=sns.barplot(x='Gene', y='Value', hue = 'Metrics', data = data, palette=palette, legend=False)
positions = [patch.get_x() + patch.get_width() / 2 for patch in ax.patches]
heights = [patch.get_height() for patch in ax.patches]
p = accuracy_p+f1_p+sens_p+spec_p

for i in range(len(positions)):
    if p[i] < 0.05:
        pval = '*'
        plt.text(x=positions[i]-0.05, y=0.77, s=pval)
    if p[i] < 0.01:
        pval = '*'
        plt.text(x=positions[i]-0.05, y=0.78, s=pval)
    if p[i] < 0.001:
        pval = '*'
        plt.text(x=positions[i]-0.05, y=0.79, s=pval)
plt.ylim([0.4, 1])
# sns.move_legend(test, "lower right")
plt.title('Consecutive Metrics')
plt.savefig(savepath + 'Fig1H.tif')

plt.figure(figsize=(8, 8))
data = pd.read_csv('./DataFrames/statistics_multisite.csv')
sns.set_theme(style='white', font_scale=1.7)
palette = sns.color_palette("viridis")
plt.ylim([0.4, 1])
test=sns.barplot(x='Gene', y='Value', hue = 'Metrics', data = data, palette=palette, legend=False)
positions = [patch.get_x() + patch.get_width() / 2 for patch in ax.patches]
heights = [patch.get_height() for patch in ax.patches]
p = accuracy_p+f1_p+sens_p+spec_p

for i in range(len(positions[:12])):
    if p[i] < 0.05:
        pval = '*'
        plt.text(x=positions[i]-0.04, y=0.77, s=pval)
    if p[i] < 0.01:
        pval = '*'
        plt.text(x=positions[i]-0.04, y=0.78, s=pval)
    if p[i] < 0.001:
        pval = '*'
        plt.text(x=positions[i]-0.04, y=0.79, s=pval)

# sns.move_legend(test, "lower right")
plt.title('Multisite Metrics')
plt.savefig(savepath + 'Fig1I.tif')

# bar for re-check results
patch_ratio = [715/1035, 108/1035, 212/1035]
slide_ratio = [60/69, 9/69]
name = pd.read_csv('./DataFrames/patch_name2.csv')
name = name['Name'].values
tmp = []
for i in range(len(name)):
    tmp.append(name[i].split('_')[1])
len(np.unique(np.array(tmp)))

final = []
tmp2= np.unique(np.array(tmp))
for i in range(len(tmp2)):
    prefix = tmp2[i].split('-')[0]
    prefix = prefix.split(' ')[-1]
    posfix = tmp2[i].split('-')[1]
    final.append(prefix + '-' + posfix)
len(np.unique(np.array(final)))

plt.figure(figsize=(8,8))
d = {'Ratio': [715/1035, 212/1035, 108/1035], 'Group': ['Malignant', 'Uncertain', 'Benign']}
df = pd.DataFrame(d)
sns.set_theme(style='white', font_scale=2)
palette = sns.color_palette("Spectral")
sns.barplot(df, x='Group', y='Ratio', hue='Group', legend=False, palette=palette)
plt.title('Patch-level counts')
plt.savefig(savepath + 'Fig4E.tif')

plt.figure(figsize=(8,8))
d = {'Ratio': [9/69, 60/69], 'Group': ['Not Identified','Identified tumour']}
df = pd.DataFrame(d)
palette = sns.color_palette("Paired")
sns.set_theme(style='white', font_scale=2)
sns.barplot(df, x='Group', y='Ratio', hue='Group', legend=False, palette=palette)
plt.title('Slide-level counts')
plt.savefig(savepath + 'Fig4F.tif')

# Pos to neg AUC
gene = 'ATRX'
sns.set_theme(style='white', font_scale=2.5)
data = pd.read_csv('./DataFrames/'+gene+'/pos2neg.csv')
pred = data['p_1'].values
y_test = data['Y'].values
fpr, tpr, threshold = metrics.roc_curve(y_test, pred)
roc_auc = metrics.auc(fpr, tpr)
CI = AUC_CI(y_test, pred)
print(CI)
plt.figure(figsize=(10,10))
P = compute_permu_aucP(pred, y_test)
plt.title('Pass to No pass-'+gene)
plt.plot(fpr, tpr, label = 'AUC = %0.3f P=%0.3f' % (roc_auc, P), linewidth=3.0)
# plt.fill_between(fpr, tpr - (roc_auc-CI[0]), tpr + (CI[1]-roc_auc), color='lightsteelblue', alpha=0.2)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'k--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
plt.savefig(savepath + 'Fig4C.tif')

d = {'Ratio': [1082/1470, 247/1470, 141/1470], 'Group': ['Malignant', 'Uncertain', 'Benign']}
df = pd.DataFrame(d)
sns.set_theme(style='white', font_scale=1.5)
sns.barplot(df, x='Group', y='Ratio', hue='Group', legend=False)

# neg to neg AUC
gene = 'TP53'
sns.set_theme(style='white', font_scale=2.5)
plt.figure(figsize=(10, 10))
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 238)

for i in range(10):
    data = pd.read_csv('./eval_results/EVAL_' + gene + '_neg2neg/fold_' + str(i) + '.csv')
    pred = data['p_1'].values
    y_test = data['Y'].values
    fpr, tpr, threshold = metrics.roc_curve(y_test, pred)
    roc_auc = metrics.auc(fpr, tpr)
    aucs.append(roc_auc)

    # 插值，使每一折的TPR对应统一的FPR（才能平均）
    tpr_interp = np.interp(mean_fpr, fpr, tpr)
    tpr_interp[0] = 0.0
    tprs.append(tpr_interp)

# 平均与标准差
mean_tpr = np.mean(tprs, axis=0)
std_tpr = np.std(tprs, axis=0)
mean_auc = metrics.auc(mean_fpr, mean_tpr)

data = pd.read_csv('./DataFrames/'+gene+'/neg2neg.csv')
pred = data['p_1'].values
y_test = data['Y'].values
# fpr, tpr, threshold = metrics.roc_curve(y_test, pred)
# roc_auc = metrics.auc(fpr, tpr)
# CI = AUC_CI(y_test, pred)
P = compute_permu_aucP(pred, y_test)
plt.title('No Pass to No Pass-'+gene)
plt.plot(mean_fpr, mean_tpr, label = 'AUC = %0.3f P=%0.3f' % (mean_auc, P), linewidth=3.0)
plt.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, color='lightsteelblue', alpha=0.2)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'k--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
plt.savefig(savepath + 'Fig5C.tif')

# all AUC

sns.set_theme(style='white', font_scale=2.5)
plt.figure(figsize=(10, 10))
cmap = plt.get_cmap('mako')
colors = [cmap(i) for i in np.linspace(0, 0.5, 3)]
index = 0
for gene in ['ATRX','H3K27M','TP53']:
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 564)

    for i in range(10):
        data = pd.read_csv('./eval_results/EVAL_all_' + gene + '/fold_'+str(i)+'.csv')
        pred = data['p_1'].values
        y_test = data['Y'].values
        fpr, tpr, threshold = metrics.roc_curve(y_test, pred)
        roc_auc = metrics.auc(fpr, tpr)
        aucs.append(roc_auc)

        # 插值，使每一折的TPR对应统一的FPR（才能平均）
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)

    # 平均与标准差
    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)
    mean_auc = metrics.auc(mean_fpr, mean_tpr)

    data = pd.read_csv('./eval_results/EVAL_all_'+gene+'/fold_all.csv')
    pred = data['p_1'].values
    y_test = data['Y'].values
    fpr, tpr, threshold = metrics.roc_curve(y_test, pred)
    P = compute_permu_aucP(pred, y_test)
    plt.title('Internal ROC')
    plt.plot(mean_fpr, mean_tpr, label = gene+' AUC = %0.3f P=%0.3f' % (mean_auc, P), linewidth=3.0, color=colors[index])
    index+=1

plt.legend(loc = 'lower right', fontsize=25)
plt.plot([0, 1], [0, 1],'k--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
plt.savefig(savepath + 'Fig6B.tif')

# gene = 'ATRX'
# sns.set_theme(style='white', font_scale=2.5)
# data = pd.read_csv('./DataFrames/'+gene+'/All.csv')
# pred = data['p_1'].values
# y_test = data['Y'].values
# fpr, tpr, threshold = metrics.roc_curve(y_test, pred)
# roc_auc = metrics.auc(fpr, tpr)
# CI = AUC_CI(y_test, pred)
# plt.figure(figsize=(10,10))
# P = compute_permu_aucP(pred, y_test)
# plt.title('Internal All-'+gene)
# plt.plot(fpr, tpr, label = 'AUC = %0.3f P=%0.3f' % (roc_auc, P))
# plt.fill_between(fpr, tpr - (roc_auc-CI[0]), tpr + (CI[1]-roc_auc), color='lightsteelblue', alpha=0.2)
# plt.legend(loc = 'lower right')
# plt.plot([0, 1], [0, 1],'k--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.show()
# plt.savefig(savepath + 'Fig6A.tif')

# all AUC ext
sns.set_theme(style='white', font_scale=2.5)
plt.figure(figsize=(10, 10))
cmap = plt.get_cmap('mako')
colors = [cmap(i) for i in np.linspace(0, 0.5, 3)]
index = 0
for gene in ['ATRX','H3K27M','TP53']:
    data = pd.read_csv('./DataFrames/'+gene+'/Rec_all.csv')
    pred = data['p_1'].values
    y_test = data['Y'].values
    fpr, tpr, threshold = metrics.roc_curve(y_test, pred)
    roc_auc = metrics.auc(fpr, tpr)
    CI = AUC_CI(y_test, pred)
    P = compute_permu_aucP(pred, y_test)
    print(CI)
    plt.title('Consecutive All')
    plt.plot(fpr, tpr, label = gene+' AUC = %0.3f P=%0.3f' % (roc_auc, P), linewidth=3.0, color=colors[index])
    index+=1

plt.legend(loc = 'lower right', fontsize=25)
plt.plot([0, 1], [0, 1],'k--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
plt.savefig(savepath + 'Fig6C.tif')
# gene = 'P53'
# sns.set_theme(style='white', font_scale=2.5)
# data = pd.read_csv('./DataFrames/'+gene+'/Rec_all.csv')
# pred = data['p_1'].values
# y_test = data['Y'].values
# fpr, tpr, threshold = metrics.roc_curve(y_test, pred)
# roc_auc = metrics.auc(fpr, tpr)
# CI = AUC_CI(y_test, pred)
# print(CI)
# plt.figure(figsize=(10,10))
# P = compute_permu_aucP(pred, y_test)
# plt.title('Consecutive All-'+gene)
# plt.plot(fpr, tpr, label = 'AUC = %0.3f P=%0.3f' % (roc_auc, P))
# plt.fill_between(fpr, tpr - (roc_auc-CI[0]), tpr + (CI[1]-roc_auc), color='lightsteelblue', alpha=0.2)
# plt.legend(loc = 'lower right')
# plt.plot([0, 1], [0, 1],'k--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.show()
# plt.savefig(savepath + 'Fig6D.tif')


# all - metrics at first and second biopsy


accuracy_p, f1_p, sens_p, spec_p, auc_p = [], [], [], [], []

for gene in ['ATRX','H3K27M','TP53']:
    datapath = './eval_results/EVAL_all_'+gene+'/'
    # df = pd.read_csv(datapath+'fold_all.csv',index_col=None)
    accuracy = []
    f1 = []
    sens = []
    spec = []
    auc = []
    accuracy2 = []
    f12 = []
    sens2 = []
    spec2 = []
    auc2 = []
    pre_all = []
    pre_all2 = []
    Y_all = []
    YP_all = []
    for i in range(0,10):
        df = pd.read_csv(datapath+'fold_'+str(i)+'.csv',index_col=None)
        # df = pd.concat((df,df_tmp))

        datalist = df['slide_id'].values
        Y = df['Y'].values
        Y_hat = df['Y_hat'].values
        p_1 = df['p_1'].values

        Y_one = []
        p_one = []
        for i in range(len(datalist)):
            if len(datalist[i].split('-'))==2:
                Y_one.append(Y[i])
                p_one.append(p_1[i])

        Y_two = []
        p_two = []
        for i in range(1,len(datalist)):
            if len(datalist[i].split('-'))==3 and datalist[i].split('-')[2] == '2' and datalist[i].split('-')[0] == datalist[i-1].split('-')[0]:
                # print(datalist[i],datalist[i-1])
                p = (p_1[i]+p_1[i-1])/2
                # print(Y_hat[i])
                # print(p)
                Y_two.append(Y[i])
                p_two.append(p)

        # from sklearn import metrics
        # print(len(Y_one))

        threshold = Find_Optimal_Cutoff(Y_one,p_one)
        # print(threshold)
        Y_hat_cut = []
        for i in range(len(Y_one)):
            if p_one[i]>threshold:
                Y_hat_cut.append(1)
            else:
                Y_hat_cut.append(0)
        threshold = find_best_f1_threshold(Y_one,p_one,step=0.01)
        # print(threshold)
        Y_hat_cut2 = []
        for i in range(len(Y_one)):
            if p_one[i]>threshold:
                Y_hat_cut2.append(1)
            else:
                Y_hat_cut2.append(0)

        correct = (np.array(Y_hat_cut) == np.array(Y_one)).sum()
        accuracy.append(float(correct) / float(len(Y_one)))
        f1.append(metrics.f1_score(Y_one, Y_hat_cut2))
        sens.append(metrics.recall_score(Y_one, Y_hat_cut, pos_label=1))
        spec.append(metrics.recall_score(Y_one, Y_hat_cut, pos_label=0))
        auc.append(metrics.roc_auc_score(Y_one, p_one))

        # pre_all += Y_hat_cut
        # pre_all2 += Y_hat_cut2
        # Y_all += Y_one
        # YP_all += p_one

        threshold = Find_Optimal_Cutoff(Y_two, p_two)
        # print(threshold)
        Y_hat_cut = []
        for i in range(len(Y_two)):
            if p_two[i]>threshold:
                Y_hat_cut.append(1)
            else:
                Y_hat_cut.append(0)
        threshold = find_best_f1_threshold(Y_two,p_two,step=0.01)
        # print(threshold)
        Y_hat_cut2 = []
        for i in range(len(Y_two)):
            if p_two[i]>threshold:
                Y_hat_cut2.append(1)
            else:
                Y_hat_cut2.append(0)

        correct = (np.array(Y_hat_cut) == np.array(Y_two)).sum()
        accuracy2.append(float(correct) / float(len(Y_two)))
        f12.append(metrics.f1_score(Y_two, Y_hat_cut2))
        sens2.append(metrics.recall_score(Y_two, Y_hat_cut, pos_label=1))
        spec2.append(metrics.recall_score(Y_two, Y_hat_cut, pos_label=0))
        auc2.append(metrics.roc_auc_score(Y_two, p_two))

        pre_all += Y_hat_cut
        pre_all2 += Y_hat_cut2
        Y_all += Y_two
        YP_all += p_two

    p1, p2, p3 = compute_permu_P(pre_all, Y_all)
    accuracy_p.append(p1)
    sens_p.append(p2)
    spec_p.append(p3)
    p4 = compute_permu_aucP(YP_all, Y_all)
    auc_p.append(p4)
    p5 = compute_permu_P_F1(pre_all2, Y_all)
    f1_p.append(p5)

    print('|test accuracy:', np.mean(accuracy),np.std(accuracy),
            '|test f1:', np.mean(f1),np.std(f1),
            '|test sen:', np.mean(sens),np.std(sens),
            '|test spe:', np.mean(spec),np.std(spec),
            '|test auc:', np.mean(auc),np.std(auc),
            )
    print('|test accuracy:', np.mean(accuracy2),np.std(accuracy2),
            '|test f1:', np.mean(f12),np.std(f12),
            '|test sen:', np.mean(sens2),np.std(sens2),
            '|test spe:', np.mean(spec2),np.std(spec2),
            '|test auc', np.mean(auc2),np.std(auc2),
            )

    d = {'accuracy':accuracy, 'f1':f1, 'sens': sens, 'spec': spec, 'auc': auc}
    df = pd.DataFrame(d)
    df.to_csv('./DataFrames/biospy/first_metrics'+gene+'.csv')

    d = {'accuracy':accuracy2, 'f1':f12, 'sens': sens2, 'spec': spec2, 'auc': auc2}
    df = pd.DataFrame(d)
    df.to_csv('./DataFrames/biospy/second_metrics'+gene+'.csv')

data = pd.read_csv('./DataFrames/statistics_times.csv')
plt.figure(figsize=(10,8))
palette = sns.color_palette("viridis")
ax = sns.barplot(x="Gene", y="Value", hue='Metrics', data = data, capsize=.1, ci="sd", palette=palette)
positions = [patch.get_x() + patch.get_width() / 2 for patch in ax.patches]
heights = [patch.get_height() for patch in ax.patches]
p = accuracy_p+f1_p+sens_p+spec_p+auc_p

for i in range(len(positions[:15])):
    if p[i] < 0.05:
        pval = '*'
        plt.text(x=positions[i]-0.02, y=1.0, s=pval)
    if p[i] < 0.01:
        pval = '*'
        plt.text(x=positions[i]-0.02, y=1.015, s=pval)
    if p[i] < 0.001:
        pval = '*'
        plt.text(x=positions[i]-0.02, y=1.03, s=pval)

sns.swarmplot(x="Gene", y="Value", hue='Metrics', data = data, alpha=.35, color="0", dodge=True, legend=False)
plt.show()
plt.ylim([0,1.1])
plt.title('Metrics at first sampling')
plt.savefig(savepath + 'Fig6D.tif')

data = pd.read_csv('./DataFrames/statistics_times2.csv')
plt.figure(figsize=(10,8))
palette = sns.color_palette("viridis")
ax = sns.barplot(x="Gene", y="Value", hue='Metrics', data = data, capsize=.1, ci="sd", palette=palette, legend=False)
positions = [patch.get_x() + patch.get_width() / 2 for patch in ax.patches]
heights = [patch.get_height() for patch in ax.patches]
p = accuracy_p+f1_p+sens_p+spec_p+auc_p

for i in range(len(positions[:15])):
    if p[i] < 0.05:
        pval = '*'
        plt.text(x=positions[i]-0.02, y=1.0, s=pval)
    if p[i] < 0.01:
        pval = '*'
        plt.text(x=positions[i]-0.02, y=1.015, s=pval)
    if p[i] < 0.001:
        pval = '*'
        plt.text(x=positions[i]-0.02, y=1.03, s=pval)

sns.swarmplot(x="Gene", y="Value", hue='Metrics', data = data, alpha=.35, color="0", dodge=True, legend=False)
plt.show()
plt.ylim([0,1.1])
plt.title('Metrics at second sampling')
plt.savefig(savepath + 'Fig6E.tif')

# data = pd.read_csv('./DataFrames/statistics_times.csv')
# plt.figure(figsize=(9,7))
# sns.set_theme(style='white', font_scale=1.5)
# sns.barplot(data, x='gene', y='value', hue='Metrics', legend=True)
# plt.title('Metrice at first sampling')
# plt.ylim(0.5,0.8)
# plt.savefig(savepath + 'Fig6G.tif')
#
# data = pd.read_csv('./DataFrames/statistics_times2.csv')
# plt.figure(figsize=(9,7))
# sns.set_theme(style='white', font_scale=1.5)
# ax = sns.barplot(data, x='gene', y='value', hue='Metrics', legend=True)
# plt.title('Metrice at second sampling')
# ax.legend_.remove()
# plt.ylim(0.5,0.8)
# plt.savefig(savepath + 'Fig6H.tif')



from sankeyflow import Sankey

nodes = [
    [('H1', 147), ('H2', 135),('H3', 32),('H4', 10),('H5', 1),('H6', 1,  {'label_pos':'bottom'})],
    [('A1', 255), ('A2', 59),('A3', 8),('A4', 3),('A6', 1)],
]
flows = [
    ('H6', 'A6', 1),
    ('H5', 'A1', 1),
    ('H4', 'A4', 3),
    ('H4', 'A2', 1),
    ('H4', 'A1', 6),
    ('H3', 'A3', 8),
    ('H3', 'A2', 6),
    ('H3', 'A1', 18),
    ('H2', 'A2', 52),
    ('H2', 'A1', 83),
    ('H1', 'A1', 147),
]

plt.figure(figsize=(12, 10), dpi=150)
sns.set_theme(style='white', font_scale=2.0)
s = Sankey(flows=flows, nodes=nodes, node_opts=dict(label_format='{label} {value:.0f}'),cmap=plt.cm.Pastel1)
plt.title('Biospy change: Internal dataset')
s.draw()
plt.show()
plt.savefig(savepath + 'Fig6F.tif')


nodes = [
    [('H1', 6), ('H2', 50),('H3', 11),('H4', 1)],
    [('A1', 37), ('A2', 26),('A3', 4),('A4', 1)],
]
flows = [
    ('H4', 'A4', 1),
    ('H3', 'A3', 4),
    ('H3', 'A2', 1),
    ('H3', 'A1', 6),
    ('H2', 'A2', 25),
    ('H2', 'A1', 25),
    ('H1', 'A1', 6),
]

plt.figure(figsize=(12, 10), dpi=150)
sns.set_theme(style='white', font_scale=2.0)
s = Sankey(flows=flows, nodes=nodes, node_opts=dict(label_format='{label} {value:.0f}'),cmap=plt.cm.Pastel1)
plt.title('Biospy change: Consecutive dataset')
s.draw()
plt.show()
plt.savefig(savepath + 'Fig6G.tif')

data = pd.read_csv('./DataFrames/times_ext.csv')
sns.set_theme(style='white', font_scale=1.7)
x = data['human'].values
y = data['AI'].values
p = stats.ttest_rel(x, y, alternative='greater')
print(p)

# data = pd.read_csv('./DataFrames/times2.csv')
# sns.set_theme(style='white', font_scale=1.7)
# ax = sns.violinplot(x='group', y='times', hue = 'group', data = data, fill=False)
# sns.stripplot(x='group', y='times', hue = 'group', data = data, ax=ax)
# plt.title('Internal dataset')
# for i in range(324):
#     plt.plot([data['group'].values[i], data['group'].values[324+i]],[data['times'].values[i],data['times'].values[i+324]], color='0.8')
#
# data = pd.read_csv('./DataFrames/times_ext.csv')
# sns.set_theme(style='white', font_scale=1.7)
# ax = sns.violinplot(x='group', y='times', hue = 'group', data = data, fill=False)
# sns.stripplot(x='group', y='times', hue = 'group', data = data, ax=ax)
# plt.title('Prospective dataset')
# for i in range(66):
#     plt.plot([data['group'].values[i], data['group'].values[66+i]],[data['times'].values[i],data['times'].values[i+66]], color='0.8')

import matplotlib.pyplot as plt
import matplotlib as mpl

# 创建画布和坐标轴
fig = plt.figure(figsize=(0.5, 5.0))
ax = fig.add_axes([0.3, 0.05, 0.3, 0.9])  # [left, bottom, width, height]

# 白色背景
fig.patch.set_facecolor('white')

# 颜色映射与归一化
cmap = plt.cm.viridis       # 可改为 'plasma', 'coolwarm', 'inferno' 等
norm = mpl.colors.Normalize(vmin=0, vmax=1)

# 创建纵向 colorbar
cb = mpl.colorbar.ColorbarBase(
    ax,
    cmap=cmap,
    norm=norm,
    orientation='vertical'
)

# 设置刻度
cb.set_ticks([0, 0.5, 1])
cb.set_ticklabels(['0', '0.5', '1'])

# 刻度字体大小
cb.ax.tick_params(labelsize=20)

# 标签（可选）
# cb.set_label('Attention weights', fontsize=20)

plt.show()
plt.savefig(savepath + 'colorbar.tif')
