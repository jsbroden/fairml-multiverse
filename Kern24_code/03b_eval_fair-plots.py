"""
Fair Algorithmic Profiling
Evaluate Fairness
"""

# Setup

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
from matplotlib.lines import Line2D
plt.style.use('seaborn')

from sklearn.metrics import precision_score, recall_score, make_scorer, roc_curve, auc, precision_recall_curve, classification_report, confusion_matrix, accuracy_score, log_loss

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric

from utils import aif_test, aif_plot, aif_plot2

X_train_f = pd.read_csv("./output/X_train_f.csv")
X_train_s = pd.read_csv("./output/X_train_s.csv")

X_test_f = pd.read_csv("./output/X_test_f.csv")
X_test_s = pd.read_csv("./output/X_test_s.csv")
y_test = pd.read_csv("./output/y_test.csv")

preds_test = pd.read_csv("./output/preds_test.csv")

# 01 Plot risk scores

sns.histplot(data = comb_test, x = 'glm2_p', kde = True, stat = 'density', common_norm = False, element = 'step')
sns.histplot(data = comb_test, x = 'glm2b_p', kde = True, stat = 'density', common_norm = False, element = 'step', color = 'g')

sns.histplot(data = comb_test, x = 'net2_p', kde = True, stat = 'density', common_norm = False, element = 'step')
sns.histplot(data = comb_test, x = 'net2b_p', kde = True, stat = 'density', common_norm = False, element = 'step', color = 'g')

sns.histplot(data = comb_test, x = 'rf2_p', kde = True, stat = 'density', common_norm = False, element = 'step')
sns.histplot(data = comb_test, x = 'rf2b_p', kde = True, stat = 'density', common_norm = False, element = 'step', color = 'g')

sns.histplot(data = comb_test, x = 'gbm2_p', kde = True, stat = 'density', common_norm = False, element = 'step')
sns.histplot(data = comb_test, x = 'gbm2b_p', kde = True, stat = 'density', common_norm = False, element = 'step', color = 'g')

sns.set(font_scale = 1.1)

threshold25 = np.sort(comb_test.glm2_p)[::-1][int(0.25*len(comb_test.glm2_p))]
threshold10 = np.sort(comb_test.glm2_p)[::-1][int(0.10*len(comb_test.glm2_p))]

sns_plot = sns.histplot(data = comb_test, x = 'glm2_p', hue = 'frau1', kde = True, stat = 'density', common_norm = False, element = 'step')
sns_plot.set(xlabel = 'Risk Score')
sns_plot.legend(title = '', labels = ['Female', 'Male'])
sns_plot.axvline(threshold25, color='k', linestyle='dashed')
sns_plot.axvline(threshold10, color='k', linestyle='dotted')
sns_plot.text(threshold10 - 0.02, 5.7, 'P1a')
sns_plot.text(threshold25 - 0.02, 5.7, 'P1b')
sns_plot.figure.savefig('glm2_p_sex.png', dpi = 300)

sns_plot = sns.histplot(data = comb_test, x = 'glm2_p', hue = 'nongerman', kde = True, stat = 'density', common_norm = False, element = 'step')
sns_plot.set(xlabel = 'Risk Score')
sns_plot.legend(title = '', labels = ['Non-German', 'German'])
sns_plot.axvline(threshold25, color='k', linestyle='dashed')
sns_plot.axvline(threshold10, color='k', linestyle='dotted')
sns_plot.text(threshold10 - 0.02, 6.2, 'P1a')
sns_plot.text(threshold25 - 0.02, 6.2, 'P1b')
sns_plot.figure.savefig('glm2_p_ger.png', dpi = 300)

threshold25 = np.sort(comb_test.net2_p)[::-1][int(0.25*len(comb_test.net2_p))]
threshold10 = np.sort(comb_test.net2_p)[::-1][int(0.10*len(comb_test.net2_p))]

sns_plot = sns.histplot(data = comb_test, x = 'net2_p', hue = 'frau1', kde = True, stat = 'density', common_norm = False, element = 'step')
sns_plot.set(xlabel = 'Risk Score')
sns_plot.legend(title = '', labels = ['Female', 'Male'])
sns_plot.axvline(threshold25, color='k', linestyle='dashed')
sns_plot.axvline(threshold10, color='k', linestyle='dotted')
sns_plot.text(threshold10 - 0.02, 7.6, 'P1a')
sns_plot.text(threshold25 - 0.02, 7.6, 'P1b')
sns_plot.figure.savefig('net2_p_sex.png', dpi = 300)

sns_plot = sns.histplot(data = comb_test, x = 'net2_p', hue = 'nongerman', kde = True, stat = 'density', common_norm = False, element = 'step')
sns_plot.set(xlabel = 'Risk Score')
sns_plot.legend(title = '', labels = ['Non-German', 'German'])
sns_plot.axvline(threshold25, color='k', linestyle='dashed')
sns_plot.axvline(threshold10, color='k', linestyle='dotted')
sns_plot.text(threshold10 - 0.02, 8.7, 'P1a')
sns_plot.text(threshold25 - 0.02, 8.7, 'P1b')
sns_plot.figure.savefig('net2_p_ger.png', dpi = 300)

threshold25 = np.sort(comb_test.rf2_p)[::-1][int(0.25*len(comb_test.rf2_p))]
threshold10 = np.sort(comb_test.rf2_p)[::-1][int(0.10*len(comb_test.rf2_p))]

sns_plot = sns.histplot(data = comb_test, x = 'rf2_p', hue = 'frau1', kde = True, stat = 'density', common_norm = False, element = 'step')
sns_plot.set(xlabel = 'Risk Score')
sns_plot.legend(title = '', labels = ['Female', 'Male'])
sns_plot.axvline(threshold25, color='k', linestyle='dashed')
sns_plot.axvline(threshold10, color='k', linestyle='dotted')
sns_plot.text(threshold10 - 0.02, 5.85, 'P1a')
sns_plot.text(threshold25 - 0.02, 5.85, 'P1b')
sns_plot.figure.savefig('rf2_p_sex.png', dpi = 300)

sns_plot = sns.histplot(data = comb_test, x = 'rf2_p', hue = 'nongerman', kde = True, stat = 'density', common_norm = False, element = 'step')
sns_plot.set(xlabel = 'Risk Score')
sns_plot.legend(title = '', labels = ['Non-German', 'German'])
sns_plot.axvline(threshold25, color='k', linestyle='dashed')
sns_plot.axvline(threshold10, color='k', linestyle='dotted')
sns_plot.text(threshold10 - 0.02, 9.25, 'P1a')
sns_plot.text(threshold25 - 0.02, 9.25, 'P1b')
sns_plot.figure.savefig('rf2_p_ger.png', dpi = 300)

threshold25 = np.sort(comb_test.gbm2_p)[::-1][int(0.25*len(comb_test.gbm2_p))]
threshold10 = np.sort(comb_test.gbm2_p)[::-1][int(0.10*len(comb_test.gbm2_p))]

sns_plot = sns.histplot(data = comb_test, x = 'gbm2_p', hue = 'frau1', kde = True, stat = 'density', common_norm = False, element = 'step')
sns_plot.set(xlabel = 'Risk Score')
sns_plot.legend(title = '', labels = ['Female', 'Male'])
sns_plot.axvline(threshold25, color='k', linestyle='dashed')
sns_plot.axvline(threshold10, color='k', linestyle='dotted')
sns_plot.text(threshold10 - 0.02, 8.55, 'P1a')
sns_plot.text(threshold25 - 0.02, 8.55, 'P1b')
sns_plot.figure.savefig('gbm2_p_sex.png', dpi = 300)

sns_plot = sns.histplot(data = comb_test, x = 'gbm2_p', hue = 'nongerman', kde = True, stat = 'density', common_norm = False, element = 'step')
sns_plot.set(xlabel = 'Risk Score')
sns_plot.legend(title = '', labels = ['Non-German', 'German'])
sns_plot.axvline(threshold25, color='k', linestyle='dashed')
sns_plot.axvline(threshold10, color='k', linestyle='dotted')
sns_plot.text(threshold10 - 0.02, 10.15, 'P1a')
sns_plot.text(threshold25 - 0.02, 10.15, 'P1b')
sns_plot.figure.savefig('gbm2_p_ger.png', dpi = 300)

# 02 Performance and Fairness vs. Threshold Plots
# https://nbviewer.jupyter.org/github/IBM/AIF360/blob/master/examples/tutorial_medical_expenditure.ipynb

label_test = pd.concat([y_test, X_test_f], axis = 1)

label_test.loc[label_test['maxdeutsch.Missing.'] == 1, 'maxdeutsch1'] = np.nan
preds_test.loc[label_test['maxdeutsch.Missing.'] == 1, 'y_test'] = np.nan

label_test = label_test.dropna()
preds_test = preds_test.dropna()

# Loop over models (w/o protected attributes) and create plots

for column in preds_test[['glm2_p', 'net2_p', 'rf2_p', 'gbm2_p']]:
    
    scores = preds_test[column]

    protected_attribute = ['frau1']
    unprivileged_group = [{'frau1': 1}]
    privileged_group = [{'frau1': 0}]

    test_label = BinaryLabelDataset(df = label_test,
                                    label_names = ['ltue'], 
                                    protected_attribute_names = protected_attribute)

    threshold_max = np.sort(scores)[::-1][int(0.001*len(scores))]
    threshold25 = np.sort(scores)[::-1][int(0.25*len(scores))]
    threshold10 = np.sort(scores)[::-1][int(0.10*len(scores))]
    thresh_arr = np.linspace(0.01, threshold_max, 50)

    val_metrics = aif_test(dataset = test_label,
                           scores = scores,
                           thresh_arr = thresh_arr,
                           unpriv_group = unprivileged_group,
                           priv_group = privileged_group)

    aif_plot(thresh_arr, 'Classification Threshold',
             val_metrics['rec'], 'Recall',
             val_metrics['prec'], 'Precision',
             cutoff1 = threshold25, cutoff2 = threshold10)

    plt.savefig('./output/' + column + '_pr_rec', dpi = 300) # Recall and Precision vs. Threshold
    plt.clf()
    
    aif_plot(thresh_arr, 'Classification Threshold', 
             val_metrics['f1'], 'F1 Score',
             val_metrics['stat_par_diff'], 'Stat. Parity Difference', 
             cutoff1 = threshold25, cutoff2 = threshold10,
             ax1min = 0, ax1max = 0.5, ax2min = 0, ax2max = 0.5)

    plt.savefig('./output/' + column + '_f1_diff_sex', dpi = 300) # F1 and Parity Difference vs. Threshold (sex)
    plt.clf()
    
    protected_attribute = ['maxdeutsch1']
    unprivileged_group = [{'maxdeutsch1': 0}]
    privileged_group = [{'maxdeutsch1': 1}]

    test_label = BinaryLabelDataset(df = label_test,
                                    label_names = ['ltue'], 
                                    protected_attribute_names = protected_attribute)

    val_metrics = aif_test(dataset = test_label,
                           scores = scores,
                           thresh_arr = thresh_arr,
                           unpriv_group = unprivileged_group,
                           priv_group = privileged_group)

    aif_plot(thresh_arr, 'Classification Threshold', 
             val_metrics['f1'], 'F1 Score',
             val_metrics['stat_par_diff'], 'Stat. Parity Difference', 
             cutoff1 = threshold25, cutoff2 = threshold10,
             ax1min = -0.2, ax1max = 0.5, ax2min = -0.2, ax2max = 0.5)

    plt.savefig('./output/' + column + '_f1_diff_ger', dpi = 300) # F1 and Parity Difference vs. Threshold (german)
    plt.clf()

# Combined plots for models (w/o protected attributes, trained w. 2010 - 2015 vs. 2015)
        
protected_attribute = ['maxdeutsch1']
unprivileged_group = [{'maxdeutsch1': 0}]
privileged_group = [{'maxdeutsch1': 1}]    

test_label = BinaryLabelDataset(df = label_test,
                                label_names = ['ltue'], 
                                protected_attribute_names = protected_attribute)

scores1 = preds_test['glm2_p']
scores2 = preds_test['glm2b_p']

threshold_max = np.sort(scores1)[::-1][int(0.001*len(scores1))]
threshold25 = np.sort(scores1)[::-1][int(0.25*len(scores1))]
threshold10 = np.sort(scores1)[::-1][int(0.10*len(scores1))]
threshold25_2 = np.sort(scores2)[::-1][int(0.25*len(scores2))]
threshold10_2 = np.sort(scores2)[::-1][int(0.10*len(scores2))]
thresh_arr = np.linspace(0.01, threshold_max, 50)

val_metrics1 = aif_test(dataset = test_label,
                       scores = scores1,
                       thresh_arr = thresh_arr,
                       unpriv_group = unprivileged_group,
                       priv_group = privileged_group)

val_metrics2 = aif_test(dataset = test_label,
                       scores = scores2,
                       thresh_arr = thresh_arr,
                       unpriv_group = unprivileged_group,
                       priv_group = privileged_group)

legend_elements = [Line2D([0], [0], color='k', lw = 2, label='2010-2015 (l)'),
                   Line2D([0], [0], color='k', lw = 2, linestyle='dashdot', label='2015 (s)')]

aif_plot2(thresh_arr, 'Classification Threshold', 
         val_metrics1['rec'], val_metrics2['rec'], 'Recall',
         val_metrics1['prec'], val_metrics2['prec'], 'Precision',          
         cutoff1 = threshold25, cutoff2 = threshold10,
         cutoff12 = threshold25_2, cutoff22 = threshold10_2)

plt.savefig('./output/glm2plus3_p_pr_rec', dpi = 300) # Recall and Precision vs. Threshold
plt.clf()

aif_plot2(thresh_arr, 'Classification Threshold', 
         val_metrics1['f1'], val_metrics2['f1'], 'F1 Score',
         val_metrics1['stat_par_diff'], val_metrics2['stat_par_diff'], 'Stat. Parity Difference', 
         cutoff1 = threshold25, cutoff2 = threshold10,
         cutoff12 = threshold25_2, cutoff22 = threshold10_2,
         ax1min = -0.2, ax1max = 0.5, ax2min = -0.2, ax2max = 0.5)

plt.savefig('./output/glm2plus3_p_f1_diff_ger', dpi = 300) # F1 and Parity Difference vs. Threshold (german)
plt.clf()

scores1 = preds_test['net2_p']
scores2 = preds_test['net2b_p']

threshold_max = np.sort(scores1)[::-1][int(0.001*len(scores1))]
threshold25 = np.sort(scores1)[::-1][int(0.25*len(scores1))]
threshold10 = np.sort(scores1)[::-1][int(0.10*len(scores1))]
threshold25_2 = np.sort(scores2)[::-1][int(0.25*len(scores2))]
threshold10_2 = np.sort(scores2)[::-1][int(0.10*len(scores2))]
thresh_arr = np.linspace(0.01, threshold_max, 50)

val_metrics1 = aif_test(dataset = test_label,
                       scores = scores1,
                       thresh_arr = thresh_arr,
                       unpriv_group = unprivileged_group,
                       priv_group = privileged_group)

val_metrics2 = aif_test(dataset = test_label,
                       scores = scores2,
                       thresh_arr = thresh_arr,
                       unpriv_group = unprivileged_group,
                       priv_group = privileged_group)

aif_plot2(thresh_arr, 'Classification Threshold', 
         val_metrics1['rec'], val_metrics2['rec'], 'Recall',
         val_metrics1['prec'], val_metrics2['prec'], 'Precision',          
         cutoff1 = threshold25, cutoff2 = threshold10,
         cutoff12 = threshold25_2, cutoff22 = threshold10_2)

plt.savefig('./output/net2plus3_p_pr_rec', dpi = 300) # Recall and Precision vs. Threshold
plt.clf()

aif_plot2(thresh_arr, 'Classification Threshold', 
         val_metrics1['f1'], val_metrics2['f1'], 'F1 Score',
         val_metrics1['stat_par_diff'], val_metrics2['stat_par_diff'], 'Stat. Parity Difference', 
         cutoff1 = threshold25, cutoff2 = threshold10,
         cutoff12 = threshold25_2, cutoff22 = threshold10_2,
         ax1min = -0.2, ax1max = 0.5, ax2min = -0.2, ax2max = 0.5)

plt.savefig('./output/net2plus3_p_f1_diff_ger', dpi = 300) # F1 and Parity Difference vs. Threshold (german)
plt.clf()

scores1 = preds_test['rf2_p']
scores2 = preds_test['rf2b_p']

threshold_max = np.sort(scores1)[::-1][int(0.001*len(scores1))]
threshold25 = np.sort(scores1)[::-1][int(0.25*len(scores1))]
threshold10 = np.sort(scores1)[::-1][int(0.10*len(scores1))]
threshold25_2 = np.sort(scores2)[::-1][int(0.25*len(scores2))]
threshold10_2 = np.sort(scores2)[::-1][int(0.10*len(scores2))]
thresh_arr = np.linspace(0.01, threshold_max, 50)

val_metrics1 = aif_test(dataset = test_label,
                       scores = scores1,
                       thresh_arr = thresh_arr,
                       unpriv_group = unprivileged_group,
                       priv_group = privileged_group)

val_metrics2 = aif_test(dataset = test_label,
                       scores = scores2,
                       thresh_arr = thresh_arr,
                       unpriv_group = unprivileged_group,
                       priv_group = privileged_group)

aif_plot2(thresh_arr, 'Classification Threshold', 
         val_metrics1['rec'], val_metrics2['rec'], 'Recall',
         val_metrics1['prec'], val_metrics2['prec'], 'Precision',          
         cutoff1 = threshold25, cutoff2 = threshold10,
         cutoff12 = threshold25_2, cutoff22 = threshold10_2)

plt.savefig('./output/rf2plus3_p_pr_rec', dpi = 300) # Recall and Precision vs. Threshold
plt.clf()

aif_plot2(thresh_arr, 'Classification Threshold', 
         val_metrics1['f1'], val_metrics2['f1'], 'F1 Score',
         val_metrics1['stat_par_diff'], val_metrics2['stat_par_diff'], 'Stat. Parity Difference', 
         cutoff1 = threshold25, cutoff2 = threshold10,
         cutoff12 = threshold25_2, cutoff22 = threshold10_2,
         ax1min = -0.2, ax1max = 0.5, ax2min = -0.2, ax2max = 0.5)

plt.savefig('./output/rf2plus3_p_f1_diff_ger', dpi = 300) # F1 and Parity Difference vs. Threshold (german)
plt.clf()

scores1 = preds_test['gbm2_p']
scores2 = preds_test['gbm2b_p']

threshold_max = np.sort(scores1)[::-1][int(0.001*len(scores1))]
threshold25 = np.sort(scores1)[::-1][int(0.25*len(scores1))]
threshold10 = np.sort(scores1)[::-1][int(0.10*len(scores1))]
threshold25_2 = np.sort(scores2)[::-1][int(0.25*len(scores2))]
threshold10_2 = np.sort(scores2)[::-1][int(0.10*len(scores2))]
thresh_arr = np.linspace(0.01, threshold_max, 50)

val_metrics1 = aif_test(dataset = test_label,
                       scores = scores1,
                       thresh_arr = thresh_arr,
                       unpriv_group = unprivileged_group,
                       priv_group = privileged_group)

val_metrics2 = aif_test(dataset = test_label,
                       scores = scores2,
                       thresh_arr = thresh_arr,
                       unpriv_group = unprivileged_group,
                       priv_group = privileged_group)

aif_plot2(thresh_arr, 'Classification Threshold', 
         val_metrics1['rec'], val_metrics2['rec'], 'Recall',
         val_metrics1['prec'], val_metrics2['prec'], 'Precision',          
         cutoff1 = threshold25, cutoff2 = threshold10,
         cutoff12 = threshold25_2, cutoff22 = threshold10_2)

plt.savefig('./output/gbm2plus3_p_pr_rec', dpi = 300) # Recall and Precision vs. Threshold
plt.clf()

aif_plot2(thresh_arr, 'Classification Threshold', 
         val_metrics1['f1'], val_metrics2['f1'], 'F1 Score',
         val_metrics1['stat_par_diff'], val_metrics2['stat_par_diff'], 'Stat. Parity Difference', 
         cutoff1 = threshold25, cutoff2 = threshold10,
         cutoff12 = threshold25_2, cutoff22 = threshold10_2,
         ax1min = -0.2, ax1max = 0.5, ax2min = -0.2, ax2max = 0.5)

plt.savefig('./output/gbm2plus3_p_f1_diff_ger', dpi = 300) # F1 and Parity Difference vs. Threshold (german)
plt.clf()

# 03 Fairness vs Accuracy Plots

f1_group = pd.read_csv("./output/f1_group.csv")
f1_group = f1_group.loc[(f1_group['pop'] != 'Overall')]
acc_group = pd.read_csv("./output/acc_group.csv")
acc_group = acc_group.loc[(acc_group['pop'] != 'Overall')]

fairness1 = pd.read_csv("./output/test_fairness1.csv")
fairness1 = fairness1.iloc[1: , :]
cond_fair1 = pd.read_csv("./output/test_cond_fairness1.csv")
cond_fair1 = cond_fair1.iloc[1: , :]
fairness1b = pd.read_csv("./output/test_fairness1b.csv")
fairness1b = fairness1b.iloc[1: , :]
cond_fair1b = pd.read_csv("./output/test_cond_fairness1b.csv")
cond_fair1b = cond_fair1b.iloc[1: , :]
fairness2 = pd.read_csv("./output/test_fairness2.csv")
fairness2 = fairness2.iloc[1: , :]
cond_fair2 = pd.read_csv("./output/test_cond_fairness2.csv")
cond_fair2 = cond_fair2.iloc[1: , :]
fairness2b = pd.read_csv("./output/test_fairness2b.csv")
fairness2b = fairness2b.iloc[1: , :]
cond_fair2b = pd.read_csv("./output/test_cond_fairness2b.csv")
cond_fair2b = cond_fair2b.iloc[1: , :]

fairness1[['method', 'Type', 'Cutoff']] = fairness1['Model'].str.split(pat='(\d)', n=1, expand=True) # Split up cols
fairness1b[['method', 'Type', 'Cutoff']] = fairness1b['Model'].str.split(pat='(\d[a-zA-Z])', n=1, expand=True)
fairness2[['method', 'Type', 'Cutoff']] = fairness2['Model'].str.split(pat='(\d)', n=1, expand=True)
fairness2b[['method', 'Type', 'Cutoff']] =  fairness2b['Model'].str.split(pat='(\d[a-zA-Z])', n=1, expand=True)

fairness = fairness1.append([fairness1b, fairness2, fairness2b]) # Append
fairness = fairness.reset_index(drop = True)

fairness['Cutoff'] = fairness['Cutoff'].str.replace(r'_', '') # Clean up
fairness = fairness.loc[(fairness['Cutoff'] == 'c1') | (fairness['Cutoff'] == 'c2')]
fairness['method'] = fairness['method'].astype('category')
fairness['method'] = fairness['method'].cat.rename_categories({'glm': 'LR', 
                                                               'net': 'PLR', 
                                                               'rf': 'RF', 
                                                               'gbm': 'GBM'})
fairness = fairness.drop(columns = ['Model'])

fairness = fairness.melt(id_vars=['method', 'Type', 'Cutoff'], var_name = 'pop') # Long format
fairness['pop'] = fairness['pop'].astype('category')
fairness['pop'] = fairness['pop'].cat.rename_categories({'Parity Diff. (Female)': 'Female', 
                                                         'Parity Diff. (Non-German)': 'Non-German', 
                                                         'Parity Diff. (Non-German-Male)': 'Non-Ger. M', 
                                                         'Parity Diff. (Non-German-Female)': 'Non-Ger. F'})

fairness = fairness.rename(columns={'value': 'Parity Diff.'})

fair_f1 = pd.merge(fairness, f1_group, on=['pop', 'method', 'Type', 'Cutoff']) # Merge
fair_acc = pd.merge(fairness, acc_group, on=['pop', 'method', 'Type', 'Cutoff'])

# Plots
sns.set(font_scale = 1.2)

sns.scatterplot(x = "Parity Diff.", y = "value", hue = 'method', data = fair_f1)
sns.scatterplot(x = "Parity Diff.", y = "value", hue = 'pop', data = fair_f1)
sns.scatterplot(x = "Parity Diff.", y = "value", hue = 'Cutoff', data = fair_f1)

fig, ax = plt.subplots(figsize = (8.5, 7))
ax = sns.scatterplot(data = fair_f1, x = "Parity Diff.", y = "value", hue = 'Cutoff', style = 'pop', palette = "muted", s = 90, alpha = 0.85)
ax.set_xlabel("Parity Diff.")
ax.set_ylabel("F1 Score")
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles, ['', 'Policy 1a', 'Policy 1b', '', 'Female', 'Non-German', 'Non-Ger. M', 'Non-Ger. F'], bbox_to_anchor = (1, 0.8), loc = 2)
plt.tight_layout()
plt.savefig('./output/fair_f1', dpi = 300)

g = sns.JointGrid(data = fair_f1, x = "Parity Diff.", y = "value", hue = 'Cutoff', palette = "muted", xlim = [-0.225, 0.1], ylim = [0.1, 0.44], height = 7, ratio = 5)
g.plot_joint(sns.scatterplot, style = 'pop', data = fair_f1, s = 75, alpha = 0.8)
g.plot_marginals(sns.kdeplot, fill = True, alpha = 0.15, bw_adjust = .9, linewidth = 1)
g.set_axis_labels('Parity Diff.', 'F1 Score')
g.ax_joint.legend_._visible = False
handles, labels = g.ax_joint.get_legend_handles_labels()
g.fig.legend(handles = handles, labels = ['', 'Policy 1a', 'Policy 1b', '', 'Female', 'Non-German', 'Non-Ger. M', 'Non-Ger. F'], bbox_to_anchor = (0.935, 0.675), loc = 2)
g.savefig('./output/fair_f1_joint', dpi = 300)

sns.scatterplot(x = "Parity Diff.", y = "value", hue = 'method', data = fair_acc)
sns.scatterplot(x = "Parity Diff.", y = "value", hue = 'pop', data = fair_acc)
sns.scatterplot(x = "Parity Diff.", y = "value", hue = 'Cutoff', data = fair_acc)

fig, ax = plt.subplots(figsize = (8.5, 7))
ax = sns.scatterplot(data = fair_acc, x = "Parity Diff.", y = "value", hue = 'Cutoff', style = 'pop', palette = "muted", s = 90, alpha = 0.85)
ax.set_xlabel("Parity Diff.")
ax.set_ylabel("Bal. Accuracy")
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles, ['', 'Policy 1a', 'Policy 1b', '', 'Female', 'Non-German', 'Non-Ger. M', 'Non-Ger. F'], bbox_to_anchor = (1, 0.8), loc = 2)
plt.tight_layout()
plt.savefig('./output/fair_acc', dpi = 300)

g = sns.JointGrid(data = fair_acc, x = "Parity Diff.", y = "value", hue = 'Cutoff', palette = "muted", xlim = [-0.225, 0.1], ylim = [0.485, 0.715], height = 7, ratio = 5)
g.plot_joint(sns.scatterplot, style = 'pop', data = fair_f1, s = 75, alpha = 0.8)
g.plot_marginals(sns.kdeplot, fill = True, alpha = 0.15, bw_adjust = .9, linewidth = 1)
g.set_axis_labels('Parity Diff.', 'Bal. Accuracy')
g.ax_joint.legend_._visible = False
handles, labels = g.ax_joint.get_legend_handles_labels()
g.fig.legend(handles = handles, labels = ['', 'Policy 1a', 'Policy 1b', '', 'Female', 'Non-German', 'Non-Ger. M', 'Non-Ger. F'], bbox_to_anchor = (0.935, 0.675), loc = 2)
g.savefig('./output/fair_acc_joint', dpi = 300)

# to do: scatterplot of partity difference (accuracy) vs. time frame of training data 
