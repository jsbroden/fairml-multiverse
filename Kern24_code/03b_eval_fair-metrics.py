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

## Descriptive

comb_test = pd.concat([preds_test, X_test_f], axis = 1)

comb_test['nongerman'] = np.where(comb_test['maxdeutsch1'] == 0, 1, 0)
comb_test.loc[comb_test['maxdeutsch.Missing.'] == 1, 'nongerman'] = np.nan
comb_test['nongerman_male'] = np.where((comb_test['nongerman'] == 1) & (comb_test['frau1'] == 0), 1, 0)
comb_test['nongerman_female'] = np.where((comb_test['nongerman'] == 1) & (comb_test['frau1'] == 1), 1, 0)

comb_test = comb_test.dropna()

comb_test[['y_test', 'nongerman']].groupby(['nongerman']).mean() # Baseline
comb_test[['rf2_c1', 'nongerman']].groupby(['nongerman']).mean() # High risk (w/o protected attributes)
comb_test[['rf2_c2', 'nongerman']].groupby(['nongerman']).mean() # High risk (w/o protected attributes)
comb_test[['rf2_c3', 'nongerman']].groupby(['nongerman']).mean() # Middle risk (w/o protected attributes)

comb_test[['y_test', 'frau1', 'nongerman', 'nongerman_male', 'nongerman_female']].groupby(['y_test']).mean() # Baseline
comb_test[['rf2_c1', 'frau1', 'nongerman', 'nongerman_male', 'nongerman_female']].groupby(['rf2_c1']).mean() # High risk (w/o protected attributes)
comb_test[['rf2_c2', 'frau1', 'nongerman', 'nongerman_male', 'nongerman_female']].groupby(['rf2_c2']).mean() # High risk (w/o protected attributes)
comb_test[['rf2_c3', 'frau1', 'nongerman', 'nongerman_male', 'nongerman_female']].groupby(['rf2_c3']).mean() # Middle risk (w/o protected attributes)

# 01 Fairness Metrics

label_test_s = pd.concat([y_test, X_test_s], axis = 1) # w/o protected attributes
preds_test_s = preds_test

label_test = pd.concat([y_test, X_test_f], axis = 1) # with protected attributes

label_test.loc[label_test['maxdeutsch.Missing.'] == 1, 'maxdeutsch1'] = np.nan
preds_test.loc[label_test['maxdeutsch.Missing.'] == 1, 'y_test'] = np.nan

label_test['nongerman'] = np.where(label_test['maxdeutsch1'] == 0, 1, 0)
label_test['nongerman_male'] = np.where((label_test['nongerman'] == 1) & (label_test['frau1'] == 0), 1, 0)
label_test['nongerman_female'] = np.where((label_test['nongerman'] == 1) & (label_test['frau1'] == 1), 1, 0)

label_test = label_test.dropna().reset_index(drop = True)
preds_test = preds_test.dropna().reset_index(drop = True)

# 01a Consistency

# Baseline Fairness for observed label

protected_attribute = ['employed_before'] # Set dummy 'protected attribute'
unprivileged_group = [{'employed_before': 0}]
privileged_group = [{'employed_before': 1}]

test_label = BinaryLabelDataset(df = label_test_s,
                                label_names = ['ltue'], 
                                protected_attribute_names = protected_attribute)

metric_test_label = BinaryLabelDatasetMetric(test_label, 
                                             unprivileged_groups = unprivileged_group,
                                             privileged_groups = privileged_group)

# pd.DataFrame(metric_test_pred.dataset.features) # AIF360 uses all columns to compute Consistency
base_consist = metric_test_label.consistency() # Consistency

# Loop over models (w protected attributes) and cutoffs to calculate metrics

consist1 = []

for column in preds_test_s[['glm1_c1', 'glm1_c2', 'glm1_c3',
                            'net1_c1', 'net1_c2', 'net1_c3',
                            'rf1_c1', 'rf1_c2', 'rf1_c3',
                            'gbm1_c1', 'gbm1_c2', 'gbm1_c3']]:
    
    pred = preds_test_s[column]
    test_pred = test_label.copy()
    test_pred.labels = pred
    
    metric_test_pred = BinaryLabelDatasetMetric(test_pred, 
                                                unprivileged_groups = unprivileged_group,
                                                privileged_groups = privileged_group)
    
    pred_consist = metric_test_pred.consistency() # Consistency

    consist1.append([column,
                    pred_consist])
    
    print(column)

consist1 = pd.DataFrame(consist1)

consist1.loc[-1] = ['label', base_consist[0]]
consist1 = consist1.sort_index()

consist1 = consist1.rename(columns={0: "Model", 1: "Consistency"})

consist1.to_latex('./output/test_consistency1.tex', index = False)
consist1.to_csv('./output/test_consistency1.csv', index = False)

# Loop over models (w protected attributes) and cutoffs to calculate metrics (train w. 2015)

consist1b = []

for column in preds_test_s[['glm1b_c1', 'glm1b_c2', 'glm1b_c3',
                            'net1b_c1', 'net1b_c2', 'net1b_c3',
                            'rf1b_c1', 'rf1b_c2', 'rf1b_c3',
                            'gbm1b_c1', 'gbm1b_c2', 'gbm1b_c3']]:
    
    pred = preds_test_s[column]
    test_pred = test_label.copy()
    test_pred.labels = pred
    
    metric_test_pred = BinaryLabelDatasetMetric(test_pred, 
                                                unprivileged_groups = unprivileged_group,
                                                privileged_groups = privileged_group)
    
    pred_consist = metric_test_pred.consistency() # Consistency

    consist1b.append([column,
                    pred_consist])
    
    print(column)

consist1b = pd.DataFrame(consist1b)

consist1b.loc[-1] = ['label', base_consist[0]]
consist1b = consist1b.sort_index()

consist1b = consist1b.rename(columns={0: "Model", 1: "Consistency"})

consist1b.to_latex('./output/test_consistency1b.tex', index = False)
consist1b.to_csv('./output/test_consistency1b.csv', index = False)

# Loop over models (w/o protected attributes) and cutoffs to calculate metrics

consist2 = []

for column in preds_test_s[['glm2_c1', 'glm2_c2', 'glm2_c3',
                            'net2_c1', 'net2_c2', 'net2_c3',
                            'rf2_c1', 'rf2_c2', 'rf2_c3',
                            'gbm2_c1', 'gbm2_c2', 'gbm2_c3']]:
    
    pred = preds_test_s[column]
    test_pred = test_label.copy()
    test_pred.labels = pred
    
    metric_test_pred = BinaryLabelDatasetMetric(test_pred, 
                                                unprivileged_groups = unprivileged_group,
                                                privileged_groups = privileged_group)
    
    pred_consist = metric_test_pred.consistency() # Consistency

    consist2.append([column,
                    pred_consist])
    
    print(column)

consist2 = pd.DataFrame(consist2)

consist2.loc[-1] = ['label', base_consist[0]]
consist2 = consist2.sort_index()

consist2 = consist2.rename(columns={0: "Model", 1: "Consistency"})

consist2.to_latex('./output/test_consistency2.tex', index = False)
consist2.to_csv('./output/test_consistency2.csv', index = False)

# Loop over models (w/o protected attributes) and cutoffs to calculate metrics (train w. 2015)

consist2b = []

for column in preds_test_s[['glm2b_c1', 'glm2b_c2', 'glm2b_c3',
                            'net2b_c1', 'net2b_c2', 'net2b_c3',
                            'rf2b_c1', 'rf2b_c2', 'rf2b_c3',
                            'gbm2b_c1', 'gbm2b_c2', 'gbm2b_c3']]:
    
    pred = preds_test_s[column]
    test_pred = test_label.copy()
    test_pred.labels = pred
    
    metric_test_pred = BinaryLabelDatasetMetric(test_pred, 
                                                unprivileged_groups = unprivileged_group,
                                                privileged_groups = privileged_group)
    
    pred_consist = metric_test_pred.consistency() # Consistency

    consist2b.append([column,
                    pred_consist])
    
    print(column)

consist2b = pd.DataFrame(consist2b)

consist2b.loc[-1] = ['label', base_consist[0]]
consist2b = consist2b.sort_index()

consist2b = consist2b.rename(columns={0: "Model", 1: "Consistency"})

consist2b.to_latex('./output/test_consistency2b.tex', index = False)
consist2b.to_csv('./output/test_consistency2b.csv', index = False)

# 01b Stat. Parity Difference

# Baseline Fairness for observed label

protected_attribute = ['frau1']
unprivileged_group = [{'frau1': 1}]
privileged_group = [{'frau1': 0}]

test_label = BinaryLabelDataset(df = label_test,
                                label_names = ['ltue'], 
                                protected_attribute_names = protected_attribute)

metric_test_label = BinaryLabelDatasetMetric(test_label, 
                                             unprivileged_groups = unprivileged_group,
                                             privileged_groups = privileged_group)

base_par_sex = metric_test_label.statistical_parity_difference() # Label diff female

protected_attribute = ['maxdeutsch1']
unprivileged_group = [{'maxdeutsch1': 0}]
privileged_group = [{'maxdeutsch1': 1}]

test_label = BinaryLabelDataset(df = label_test,
                                label_names = ['ltue'], 
                                protected_attribute_names = protected_attribute)

metric_test_label = BinaryLabelDatasetMetric(test_label, 
                                             unprivileged_groups = unprivileged_group,
                                             privileged_groups = privileged_group)

base_par_ger = metric_test_label.statistical_parity_difference() # Label diff nongerman

protected_attribute = ['nongerman_male']
unprivileged_group = [{'nongerman_male': 1}]
privileged_group = [{'nongerman_male': 0}]

test_label = BinaryLabelDataset(df = label_test,
                                label_names = ['ltue'], 
                                protected_attribute_names = protected_attribute)

metric_test_label = BinaryLabelDatasetMetric(test_label, 
                                             unprivileged_groups = unprivileged_group,
                                             privileged_groups = privileged_group)

base_par_ger_male = metric_test_label.statistical_parity_difference() # Label diff nongerman male

protected_attribute = ['nongerman_female']
unprivileged_group = [{'nongerman_female': 1}]
privileged_group = [{'nongerman_female': 0}]

test_label = BinaryLabelDataset(df = label_test,
                                label_names = ['ltue'], 
                                protected_attribute_names = protected_attribute)

metric_test_label = BinaryLabelDatasetMetric(test_label, 
                                             unprivileged_groups = unprivileged_group,
                                             privileged_groups = privileged_group)

base_par_ger_female = metric_test_label.statistical_parity_difference() # Label diff nongerman female

# Loop over models (w protected attributes) and cutoffs to calculate metrics

fairness1 = []

for column in preds_test[['glm1_c1', 'glm1_c2', 'glm1_c3',
                          'net1_c1', 'net1_c2', 'net1_c3',
                          'rf1_c1', 'rf1_c2', 'rf1_c3',
                          'gbm1_c1', 'gbm1_c2', 'gbm1_c3']]:

    protected_attribute = ['frau1']
    unprivileged_group = [{'frau1': 1}]
    privileged_group = [{'frau1': 0}]
    
    test_label = BinaryLabelDataset(df = label_test,
                                    label_names = ['ltue'], 
                                    protected_attribute_names = protected_attribute)

    pred = preds_test[column]
    test_pred = test_label.copy()
    test_pred.labels = pred
    
    metric_test_pred = BinaryLabelDatasetMetric(test_pred, 
                                                unprivileged_groups = unprivileged_group,
                                                privileged_groups = privileged_group)
    
    par_sex = metric_test_pred.statistical_parity_difference() # Parity difference for female
    
    protected_attribute = ['maxdeutsch1']
    unprivileged_group = [{'maxdeutsch1': 0}]
    privileged_group = [{'maxdeutsch1': 1}]
    
    test_label = BinaryLabelDataset(df = label_test,
                                    label_names = ['ltue'], 
                                    protected_attribute_names = protected_attribute)

    test_pred = test_label.copy()
    test_pred.labels = pred
    
    metric_test_pred = BinaryLabelDatasetMetric(test_pred, 
                                                unprivileged_groups = unprivileged_group,
                                                privileged_groups = privileged_group)
    
    par_ger = metric_test_pred.statistical_parity_difference() # Parity difference for nongerman
    
    protected_attribute = ['nongerman_male']
    unprivileged_group = [{'nongerman_male': 1}]
    privileged_group = [{'nongerman_male': 0}]
    
    test_label = BinaryLabelDataset(df = label_test,
                                    label_names = ['ltue'], 
                                    protected_attribute_names = protected_attribute)

    test_pred = test_label.copy()
    test_pred.labels = pred
    
    metric_test_pred = BinaryLabelDatasetMetric(test_pred, 
                                                unprivileged_groups = unprivileged_group,
                                                privileged_groups = privileged_group)
    
    par_ger_male = metric_test_pred.statistical_parity_difference() # Parity difference for nongerman male
    
    protected_attribute = ['nongerman_female']
    unprivileged_group = [{'nongerman_female': 1}]
    privileged_group = [{'nongerman_female': 0}]
    
    test_label = BinaryLabelDataset(df = label_test,
                                    label_names = ['ltue'], 
                                    protected_attribute_names = protected_attribute)

    test_pred = test_label.copy()
    test_pred.labels = pred
    
    metric_test_pred = BinaryLabelDatasetMetric(test_pred, 
                                                unprivileged_groups = unprivileged_group,
                                                privileged_groups = privileged_group)
    
    par_ger_female = metric_test_pred.statistical_parity_difference() # Parity difference for nongerman female
    
    fairness1.append([column,
                     par_sex,
                     par_ger,
                     par_ger_male,
                     par_ger_female])

fairness1 = pd.DataFrame(fairness1)

fairness1.loc[-1] = ['label', base_par_sex, base_par_ger, base_par_ger_male, base_par_ger_female]
fairness1 = fairness1.sort_index()

fairness1 = fairness1.rename(columns={0: "Model", 1: "Parity Diff. (Female)", 2: "Parity Diff. (Non-German)", 3: "Parity Diff. (Non-German-Male)", 4: "Parity Diff. (Non-German-Female)"})

fairness1.to_latex('./output/test_fairness1.tex', index = False, float_format = "%.3f")
fairness1.to_csv('./output/test_fairness1.csv', index = False)

# Loop over models (w protected attributes) and cutoffs to calculate metrics (train w. 2015)

fairness1b = []

for column in preds_test[['glm1b_c1', 'glm1b_c2', 'glm1b_c3',
                          'net1b_c1', 'net1b_c2', 'net1b_c3',
                          'rf1b_c1', 'rf1b_c2', 'rf1b_c3',
                          'gbm1b_c1', 'gbm1b_c2', 'gbm1b_c3']]:

    protected_attribute = ['frau1']
    unprivileged_group = [{'frau1': 1}]
    privileged_group = [{'frau1': 0}]
    
    test_label = BinaryLabelDataset(df = label_test,
                                    label_names = ['ltue'], 
                                    protected_attribute_names = protected_attribute)

    pred = preds_test[column]
    test_pred = test_label.copy()
    test_pred.labels = pred
    
    metric_test_pred = BinaryLabelDatasetMetric(test_pred, 
                                                unprivileged_groups = unprivileged_group,
                                                privileged_groups = privileged_group)
    
    par_sex = metric_test_pred.statistical_parity_difference() # Parity difference for female
    
    protected_attribute = ['maxdeutsch1']
    unprivileged_group = [{'maxdeutsch1': 0}]
    privileged_group = [{'maxdeutsch1': 1}]
    
    test_label = BinaryLabelDataset(df = label_test,
                                    label_names = ['ltue'], 
                                    protected_attribute_names = protected_attribute)

    test_pred = test_label.copy()
    test_pred.labels = pred
    
    metric_test_pred = BinaryLabelDatasetMetric(test_pred, 
                                                unprivileged_groups = unprivileged_group,
                                                privileged_groups = privileged_group)
    
    par_ger = metric_test_pred.statistical_parity_difference() # Parity difference for nongerman
    
    protected_attribute = ['nongerman_male']
    unprivileged_group = [{'nongerman_male': 1}]
    privileged_group = [{'nongerman_male': 0}]
    
    test_label = BinaryLabelDataset(df = label_test,
                                    label_names = ['ltue'], 
                                    protected_attribute_names = protected_attribute)

    test_pred = test_label.copy()
    test_pred.labels = pred
    
    metric_test_pred = BinaryLabelDatasetMetric(test_pred, 
                                                unprivileged_groups = unprivileged_group,
                                                privileged_groups = privileged_group)
    
    par_ger_male = metric_test_pred.statistical_parity_difference() # Parity difference for nongerman male
    
    protected_attribute = ['nongerman_female']
    unprivileged_group = [{'nongerman_female': 1}]
    privileged_group = [{'nongerman_female': 0}]
    
    test_label = BinaryLabelDataset(df = label_test,
                                    label_names = ['ltue'], 
                                    protected_attribute_names = protected_attribute)

    test_pred = test_label.copy()
    test_pred.labels = pred
    
    metric_test_pred = BinaryLabelDatasetMetric(test_pred, 
                                                unprivileged_groups = unprivileged_group,
                                                privileged_groups = privileged_group)
    
    par_ger_female = metric_test_pred.statistical_parity_difference() # Parity difference for nongerman female
    
    fairness1b.append([column,
                     par_sex,
                     par_ger,
                     par_ger_male,
                     par_ger_female])

fairness1b = pd.DataFrame(fairness1b)

fairness1b.loc[-1] = ['label', base_par_sex, base_par_ger, base_par_ger_male, base_par_ger_female]
fairness1b = fairness1b.sort_index()

fairness1b = fairness1b.rename(columns={0: "Model", 1: "Parity Diff. (Female)", 2: "Parity Diff. (Non-German)", 3: "Parity Diff. (Non-German-Male)", 4: "Parity Diff. (Non-German-Female)"})

fairness1b.to_latex('./output/test_fairness1b.tex', index = False, float_format = "%.3f")
fairness1b.to_csv('./output/test_fairness1b.csv', index = False)

# Loop over models (w/o protected attributes) and cutoffs to calculate metrics

fairness2 = []

for column in preds_test[['glm2_c1', 'glm2_c2', 'glm2_c3',
                          'net2_c1', 'net2_c2', 'net2_c3',
                          'rf2_c1', 'rf2_c2', 'rf2_c3',
                          'gbm2_c1', 'gbm2_c2', 'gbm2_c3']]:

    protected_attribute = ['frau1']
    unprivileged_group = [{'frau1': 1}]
    privileged_group = [{'frau1': 0}]
    
    test_label = BinaryLabelDataset(df = label_test,
                                    label_names = ['ltue'], 
                                    protected_attribute_names = protected_attribute)

    pred = preds_test[column]
    test_pred = test_label.copy()
    test_pred.labels = pred
    
    metric_test_pred = BinaryLabelDatasetMetric(test_pred, 
                                                unprivileged_groups = unprivileged_group,
                                                privileged_groups = privileged_group)
    
    par_sex = metric_test_pred.statistical_parity_difference() # Parity difference for female
    
    protected_attribute = ['maxdeutsch1']
    unprivileged_group = [{'maxdeutsch1': 0}]
    privileged_group = [{'maxdeutsch1': 1}]
    
    test_label = BinaryLabelDataset(df = label_test,
                                    label_names = ['ltue'], 
                                    protected_attribute_names = protected_attribute)

    test_pred = test_label.copy()
    test_pred.labels = pred
    
    metric_test_pred = BinaryLabelDatasetMetric(test_pred, 
                                                unprivileged_groups = unprivileged_group,
                                                privileged_groups = privileged_group)
    
    par_ger = metric_test_pred.statistical_parity_difference() # Parity difference for nongerman
    
    protected_attribute = ['nongerman_male']
    unprivileged_group = [{'nongerman_male': 1}]
    privileged_group = [{'nongerman_male': 0}]
    
    test_label = BinaryLabelDataset(df = label_test,
                                    label_names = ['ltue'], 
                                    protected_attribute_names = protected_attribute)

    test_pred = test_label.copy()
    test_pred.labels = pred
    
    metric_test_pred = BinaryLabelDatasetMetric(test_pred, 
                                                unprivileged_groups = unprivileged_group,
                                                privileged_groups = privileged_group)
    
    par_ger_male = metric_test_pred.statistical_parity_difference() # Parity difference for nongerman male
    
    protected_attribute = ['nongerman_female']
    unprivileged_group = [{'nongerman_female': 1}]
    privileged_group = [{'nongerman_female': 0}]
    
    test_label = BinaryLabelDataset(df = label_test,
                                    label_names = ['ltue'], 
                                    protected_attribute_names = protected_attribute)

    test_pred = test_label.copy()
    test_pred.labels = pred
    
    metric_test_pred = BinaryLabelDatasetMetric(test_pred, 
                                                unprivileged_groups = unprivileged_group,
                                                privileged_groups = privileged_group)
    
    par_ger_female = metric_test_pred.statistical_parity_difference() # Parity difference for nongerman female
    
    fairness2.append([column,
                     par_sex,
                     par_ger,
                     par_ger_male,
                     par_ger_female])

fairness2 = pd.DataFrame(fairness2)

fairness2.loc[-1] = ['label', base_par_sex, base_par_ger, base_par_ger_male, base_par_ger_female]
fairness2 = fairness2.sort_index()

fairness2 = fairness2.rename(columns={0: "Model", 1: "Parity Diff. (Female)", 2: "Parity Diff. (Non-German)", 3: "Parity Diff. (Non-German-Male)", 4: "Parity Diff. (Non-German-Female)"})

fairness2.to_latex('./output/test_fairness2.tex', index = False, float_format = "%.3f")
fairness2.to_csv('./output/test_fairness2.csv', index = False)

# Loop over models (w/o protected attributes) and cutoffs to calculate metrics (train w. 2015)

fairness2b = []

for column in preds_test[['glm2b_c1', 'glm2b_c2', 'glm2b_c3',
                          'net2b_c1', 'net2b_c2', 'net2b_c3',
                          'rf2b_c1', 'rf2b_c2', 'rf2b_c3',
                          'gbm2b_c1', 'gbm2b_c2', 'gbm2b_c3']]:

    protected_attribute = ['frau1']
    unprivileged_group = [{'frau1': 1}]
    privileged_group = [{'frau1': 0}]
    
    test_label = BinaryLabelDataset(df = label_test,
                                    label_names = ['ltue'], 
                                    protected_attribute_names = protected_attribute)

    pred = preds_test[column]
    test_pred = test_label.copy()
    test_pred.labels = pred
    
    metric_test_pred = BinaryLabelDatasetMetric(test_pred, 
                                                unprivileged_groups = unprivileged_group,
                                                privileged_groups = privileged_group)
    
    par_sex = metric_test_pred.statistical_parity_difference() # Parity difference for female
    
    protected_attribute = ['maxdeutsch1']
    unprivileged_group = [{'maxdeutsch1': 0}]
    privileged_group = [{'maxdeutsch1': 1}]
    
    test_label = BinaryLabelDataset(df = label_test,
                                    label_names = ['ltue'], 
                                    protected_attribute_names = protected_attribute)

    test_pred = test_label.copy()
    test_pred.labels = pred
    
    metric_test_pred = BinaryLabelDatasetMetric(test_pred, 
                                                unprivileged_groups = unprivileged_group,
                                                privileged_groups = privileged_group)
    
    par_ger = metric_test_pred.statistical_parity_difference() # Parity difference for nongerman
    
    protected_attribute = ['nongerman_male']
    unprivileged_group = [{'nongerman_male': 1}]
    privileged_group = [{'nongerman_male': 0}]
    
    test_label = BinaryLabelDataset(df = label_test,
                                    label_names = ['ltue'], 
                                    protected_attribute_names = protected_attribute)

    test_pred = test_label.copy()
    test_pred.labels = pred
    
    metric_test_pred = BinaryLabelDatasetMetric(test_pred, 
                                                unprivileged_groups = unprivileged_group,
                                                privileged_groups = privileged_group)
    
    par_ger_male = metric_test_pred.statistical_parity_difference() # Parity difference for nongerman male
    
    protected_attribute = ['nongerman_female']
    unprivileged_group = [{'nongerman_female': 1}]
    privileged_group = [{'nongerman_female': 0}]
    
    test_label = BinaryLabelDataset(df = label_test,
                                    label_names = ['ltue'], 
                                    protected_attribute_names = protected_attribute)

    test_pred = test_label.copy()
    test_pred.labels = pred
    
    metric_test_pred = BinaryLabelDatasetMetric(test_pred, 
                                                unprivileged_groups = unprivileged_group,
                                                privileged_groups = privileged_group)
    
    par_ger_female = metric_test_pred.statistical_parity_difference() # Parity difference for nongerman female
    
    fairness2b.append([column,
                     par_sex,
                     par_ger,
                     par_ger_male,
                     par_ger_female])

fairness2b = pd.DataFrame(fairness2b)

fairness2b.loc[-1] = ['label', base_par_sex, base_par_ger, base_par_ger_male, base_par_ger_female]
fairness2b = fairness2b.sort_index()

fairness2b = fairness2b.rename(columns={0: "Model", 1: "Parity Diff. (Female)", 2: "Parity Diff. (Non-German)", 3: "Parity Diff. (Non-German-Male)", 4: "Parity Diff. (Non-German-Female)"})

fairness2b.to_latex('./output/test_fairness2b.tex', index = False, float_format = "%.3f")
fairness2b.to_csv('./output/test_fairness2b.csv', index = False)

# 01c: Cond. Stat. Parity Difference (Edu = Abitur) 

# Baseline Fairness for observed label

protected_attribute = ['frau1', 'maxschule9']
unprivileged_group = [{'frau1': 1, 'maxschule9': 1}]
privileged_group = [{'frau1': 0, 'maxschule9': 1}]

test_label = BinaryLabelDataset(df = label_test,
                                label_names = ['ltue'], 
                                protected_attribute_names = protected_attribute)

metric_test_label = BinaryLabelDatasetMetric(test_label, 
                                             unprivileged_groups = unprivileged_group,
                                             privileged_groups = privileged_group)

base_cpar_sex = metric_test_label.statistical_parity_difference() # Label diff female (edu = abi)

protected_attribute = ['maxdeutsch1', 'maxschule9']
unprivileged_group = [{'maxdeutsch1': 0, 'maxschule9': 1}]
privileged_group = [{'maxdeutsch1': 1, 'maxschule9': 1}]

test_label = BinaryLabelDataset(df = label_test,
                                label_names = ['ltue'], 
                                protected_attribute_names = protected_attribute)

metric_test_label = BinaryLabelDatasetMetric(test_label, 
                                             unprivileged_groups = unprivileged_group,
                                             privileged_groups = privileged_group)

base_cpar_ger = metric_test_label.statistical_parity_difference() # Label diff nongerman (edu = abi)

protected_attribute = ['nongerman_male', 'maxschule9']
unprivileged_group = [{'nongerman_male': 1, 'maxschule9': 1}]
privileged_group = [{'nongerman_male': 0, 'maxschule9': 1}]

test_label = BinaryLabelDataset(df = label_test,
                                label_names = ['ltue'], 
                                protected_attribute_names = protected_attribute)

metric_test_label = BinaryLabelDatasetMetric(test_label, 
                                             unprivileged_groups = unprivileged_group,
                                             privileged_groups = privileged_group)

base_cpar_ger_male = metric_test_label.statistical_parity_difference() # Label diff nongerman male (edu = abi)

protected_attribute = ['nongerman_female', 'maxschule9']
unprivileged_group = [{'nongerman_female': 1, 'maxschule9': 1}]
privileged_group = [{'nongerman_female': 0, 'maxschule9': 1}]

test_label = BinaryLabelDataset(df = label_test,
                                label_names = ['ltue'], 
                                protected_attribute_names = protected_attribute)

metric_test_label = BinaryLabelDatasetMetric(test_label, 
                                             unprivileged_groups = unprivileged_group,
                                             privileged_groups = privileged_group)

base_cpar_ger_female = metric_test_label.statistical_parity_difference() # Label diff nongerman female (edu = abi)

# Loop over models (w protected attributes) and cutoffs to calculate metrics

cond_fair1 = []

for column in preds_test[['glm1_c1', 'glm1_c2', 'glm1_c3',
                          'net1_c1', 'net1_c2', 'net1_c3',
                          'rf1_c1', 'rf1_c2', 'rf1_c3',
                          'gbm1_c1', 'gbm1_c2', 'gbm1_c3']]:

    protected_attribute = ['frau1', 'maxschule9']
    unprivileged_group = [{'frau1': 1, 'maxschule9': 1}]
    privileged_group = [{'frau1': 0, 'maxschule9': 1}]
    
    test_label = BinaryLabelDataset(df = label_test,
                                    label_names = ['ltue'], 
                                    protected_attribute_names = protected_attribute)

    pred = preds_test[column]
    test_pred = test_label.copy()
    test_pred.labels = pred
    
    metric_test_pred = BinaryLabelDatasetMetric(test_pred, 
                                                unprivileged_groups = unprivileged_group,
                                                privileged_groups = privileged_group)

    par_sex = metric_test_pred.statistical_parity_difference() # Parity difference for female (edu = abi)
    
    protected_attribute = ['maxdeutsch1', 'maxschule9']
    unprivileged_group = [{'maxdeutsch1': 0, 'maxschule9': 1}]
    privileged_group = [{'maxdeutsch1': 1, 'maxschule9': 1}]
    
    test_label = BinaryLabelDataset(df = label_test,
                                    label_names = ['ltue'], 
                                    protected_attribute_names = protected_attribute)

    test_pred = test_label.copy()
    test_pred.labels = pred
    
    metric_test_pred = BinaryLabelDatasetMetric(test_pred, 
                                                unprivileged_groups = unprivileged_group,
                                                privileged_groups = privileged_group)
    
    par_ger = metric_test_pred.statistical_parity_difference() # Parity difference for nongerman (edu = abi)
    
    protected_attribute = ['nongerman_male', 'maxschule9']
    unprivileged_group = [{'nongerman_male': 1, 'maxschule9': 1}]
    privileged_group = [{'nongerman_male': 0, 'maxschule9': 1}]
    
    test_label = BinaryLabelDataset(df = label_test,
                                    label_names = ['ltue'], 
                                    protected_attribute_names = protected_attribute)

    test_pred = test_label.copy()
    test_pred.labels = pred
    
    metric_test_pred = BinaryLabelDatasetMetric(test_pred, 
                                                unprivileged_groups = unprivileged_group,
                                                privileged_groups = privileged_group)
    
    par_ger_male = metric_test_pred.statistical_parity_difference() # Parity difference for nongerman male (edu = abi)
    
    protected_attribute = ['nongerman_female', 'maxschule9']
    unprivileged_group = [{'nongerman_female': 1, 'maxschule9': 1}]
    privileged_group = [{'nongerman_female': 0, 'maxschule9': 1}]
    
    test_label = BinaryLabelDataset(df = label_test,
                                    label_names = ['ltue'], 
                                    protected_attribute_names = protected_attribute)

    test_pred = test_label.copy()
    test_pred.labels = pred
    
    metric_test_pred = BinaryLabelDatasetMetric(test_pred, 
                                                unprivileged_groups = unprivileged_group,
                                                privileged_groups = privileged_group)
    
    par_ger_female = metric_test_pred.statistical_parity_difference() # Parity difference for nongerman female (edu = abi)
    
    cond_fair1.append([column,
                      par_sex,
                      par_ger,
                      par_ger_male,
                      par_ger_female])

cond_fair1 = pd.DataFrame(cond_fair1)

cond_fair1.loc[-1] = ['label', base_cpar_sex, base_cpar_ger, base_cpar_ger_male, base_cpar_ger_female]
cond_fair1 = cond_fair1.sort_index()

cond_fair1 = cond_fair1.rename(columns={0: "Model", 1: "Cond. Parity Diff. (Female)", 2: "Cond. Parity Diff. (Non-German)", 3: "Cond. Parity Diff. (Non-German-Male)", 4: "Cond. Parity Diff. (Non-German-Female)"})

cond_fair1.to_latex('./output/test_cond_fairness1.tex', index = False, float_format = "%.3f")
cond_fair1.to_csv('./output/test_cond_fairness1.csv', index = False)

# Loop over models (w protected attributes) and cutoffs to calculate metrics (train w. 2015)

cond_fair1b = []

for column in preds_test[['glm1b_c1', 'glm1b_c2', 'glm1b_c3',
                          'net1b_c1', 'net1b_c2', 'net1b_c3',
                          'rf1b_c1', 'rf1b_c2', 'rf1b_c3',
                          'gbm1b_c1', 'gbm1b_c2', 'gbm1b_c3']]:

    protected_attribute = ['frau1', 'maxschule9']
    unprivileged_group = [{'frau1': 1, 'maxschule9': 1}]
    privileged_group = [{'frau1': 0, 'maxschule9': 1}]
    
    test_label = BinaryLabelDataset(df = label_test,
                                    label_names = ['ltue'], 
                                    protected_attribute_names = protected_attribute)

    pred = preds_test[column]
    test_pred = test_label.copy()
    test_pred.labels = pred
    
    metric_test_pred = BinaryLabelDatasetMetric(test_pred, 
                                                unprivileged_groups = unprivileged_group,
                                                privileged_groups = privileged_group)

    par_sex = metric_test_pred.statistical_parity_difference() # Parity difference for female (edu = abi)
    
    protected_attribute = ['maxdeutsch1', 'maxschule9']
    unprivileged_group = [{'maxdeutsch1': 0, 'maxschule9': 1}]
    privileged_group = [{'maxdeutsch1': 1, 'maxschule9': 1}]
    
    test_label = BinaryLabelDataset(df = label_test,
                                    label_names = ['ltue'], 
                                    protected_attribute_names = protected_attribute)

    test_pred = test_label.copy()
    test_pred.labels = pred
    
    metric_test_pred = BinaryLabelDatasetMetric(test_pred, 
                                                unprivileged_groups = unprivileged_group,
                                                privileged_groups = privileged_group)
    
    par_ger = metric_test_pred.statistical_parity_difference() # Parity difference for nongerman (edu = abi)
    
    protected_attribute = ['nongerman_male', 'maxschule9']
    unprivileged_group = [{'nongerman_male': 1, 'maxschule9': 1}]
    privileged_group = [{'nongerman_male': 0, 'maxschule9': 1}]
    
    test_label = BinaryLabelDataset(df = label_test,
                                    label_names = ['ltue'], 
                                    protected_attribute_names = protected_attribute)

    test_pred = test_label.copy()
    test_pred.labels = pred
    
    metric_test_pred = BinaryLabelDatasetMetric(test_pred, 
                                                unprivileged_groups = unprivileged_group,
                                                privileged_groups = privileged_group)
    
    par_ger_male = metric_test_pred.statistical_parity_difference() # Parity difference for nongerman male (edu = abi)
    
    protected_attribute = ['nongerman_female', 'maxschule9']
    unprivileged_group = [{'nongerman_female': 1, 'maxschule9': 1}]
    privileged_group = [{'nongerman_female': 0, 'maxschule9': 1}]
    
    test_label = BinaryLabelDataset(df = label_test,
                                    label_names = ['ltue'], 
                                    protected_attribute_names = protected_attribute)

    test_pred = test_label.copy()
    test_pred.labels = pred
    
    metric_test_pred = BinaryLabelDatasetMetric(test_pred, 
                                                unprivileged_groups = unprivileged_group,
                                                privileged_groups = privileged_group)
    
    par_ger_female = metric_test_pred.statistical_parity_difference() # Parity difference for nongerman female (edu = abi)
    
    cond_fair1b.append([column,
                      par_sex,
                      par_ger,
                      par_ger_male,
                      par_ger_female])

cond_fair1b = pd.DataFrame(cond_fair1b)

cond_fair1b.loc[-1] = ['label', base_cpar_sex, base_cpar_ger, base_cpar_ger_male, base_cpar_ger_female]
cond_fair1b = cond_fair1b.sort_index()

cond_fair1b = cond_fair1b.rename(columns={0: "Model", 1: "Cond. Parity Diff. (Female)", 2: "Cond. Parity Diff. (Non-German)", 3: "Cond. Parity Diff. (Non-German-Male)", 4: "Cond. Parity Diff. (Non-German-Female)"})

cond_fair1b.to_latex('./output/test_cond_fairness1b.tex', index = False, float_format = "%.3f")
cond_fair1b.to_csv('./output/test_cond_fairness1b.csv', index = False)

# Loop over models (w/o protected attributes) and cutoffs to calculate metrics

cond_fair2 = []

for column in preds_test[['glm2_c1', 'glm2_c2', 'glm2_c3',
                          'net2_c1', 'net2_c2', 'net2_c3',
                          'rf2_c1', 'rf2_c2', 'rf2_c3',
                          'gbm2_c1', 'gbm2_c2', 'gbm2_c3']]:

    protected_attribute = ['frau1', 'maxschule9']
    unprivileged_group = [{'frau1': 1, 'maxschule9': 1}]
    privileged_group = [{'frau1': 0, 'maxschule9': 1}]
    
    test_label = BinaryLabelDataset(df = label_test,
                                    label_names = ['ltue'], 
                                    protected_attribute_names = protected_attribute)

    pred = preds_test[column]
    test_pred = test_label.copy()
    test_pred.labels = pred
    
    metric_test_pred = BinaryLabelDatasetMetric(test_pred, 
                                                unprivileged_groups = unprivileged_group,
                                                privileged_groups = privileged_group)

    par_sex = metric_test_pred.statistical_parity_difference() # Parity difference for female (edu = abi)
    
    protected_attribute = ['maxdeutsch1', 'maxschule9']
    unprivileged_group = [{'maxdeutsch1': 0, 'maxschule9': 1}]
    privileged_group = [{'maxdeutsch1': 1, 'maxschule9': 1}]
    
    test_label = BinaryLabelDataset(df = label_test,
                                    label_names = ['ltue'], 
                                    protected_attribute_names = protected_attribute)

    test_pred = test_label.copy()
    test_pred.labels = pred
    
    metric_test_pred = BinaryLabelDatasetMetric(test_pred, 
                                                unprivileged_groups = unprivileged_group,
                                                privileged_groups = privileged_group)
    
    par_ger = metric_test_pred.statistical_parity_difference() # Parity difference for nongerman (edu = abi)
    
    protected_attribute = ['nongerman_male', 'maxschule9']
    unprivileged_group = [{'nongerman_male': 1, 'maxschule9': 1}]
    privileged_group = [{'nongerman_male': 0, 'maxschule9': 1}]
    
    test_label = BinaryLabelDataset(df = label_test,
                                    label_names = ['ltue'], 
                                    protected_attribute_names = protected_attribute)

    test_pred = test_label.copy()
    test_pred.labels = pred
    
    metric_test_pred = BinaryLabelDatasetMetric(test_pred, 
                                                unprivileged_groups = unprivileged_group,
                                                privileged_groups = privileged_group)
    
    par_ger_male = metric_test_pred.statistical_parity_difference() # Parity difference for nongerman male (edu = abi)
    
    protected_attribute = ['nongerman_female', 'maxschule9']
    unprivileged_group = [{'nongerman_female': 1, 'maxschule9': 1}]
    privileged_group = [{'nongerman_female': 0, 'maxschule9': 1}]
    
    test_label = BinaryLabelDataset(df = label_test,
                                    label_names = ['ltue'], 
                                    protected_attribute_names = protected_attribute)

    test_pred = test_label.copy()
    test_pred.labels = pred
    
    metric_test_pred = BinaryLabelDatasetMetric(test_pred, 
                                                unprivileged_groups = unprivileged_group,
                                                privileged_groups = privileged_group)
    
    par_ger_female = metric_test_pred.statistical_parity_difference() # Parity difference for nongerman female (edu = abi)
    
    cond_fair2.append([column,
                      par_sex,
                      par_ger,
                      par_ger_male,
                      par_ger_female])

cond_fair2 = pd.DataFrame(cond_fair2)

cond_fair2.loc[-1] = ['label', base_cpar_sex, base_cpar_ger, base_cpar_ger_male, base_cpar_ger_female]
cond_fair2 = cond_fair2.sort_index()

cond_fair2 = cond_fair2.rename(columns={0: "Model", 1: "Cond. Parity Diff. (Female)", 2: "Cond. Parity Diff. (Non-German)", 3: "Cond. Parity Diff. (Non-German-Male)", 4: "Cond. Parity Diff. (Non-German-Female)"})

cond_fair2.to_latex('./output/test_cond_fairness2.tex', index = False, float_format = "%.3f")
cond_fair2.to_csv('./output/test_cond_fairness2.csv', index = False)

# Loop over models (w/o protected attributes) and cutoffs to calculate metrics (train w. 2015)

cond_fair2b = []

for column in preds_test[['glm2b_c1', 'glm2b_c2', 'glm2b_c3',
                          'net2b_c1', 'net2b_c2', 'net2b_c3',
                          'rf2b_c1', 'rf2b_c2', 'rf2b_c3',
                          'gbm2b_c1', 'gbm2b_c2', 'gbm2b_c3']]:

    protected_attribute = ['frau1', 'maxschule9']
    unprivileged_group = [{'frau1': 1, 'maxschule9': 1}]
    privileged_group = [{'frau1': 0, 'maxschule9': 1}]
    
    test_label = BinaryLabelDataset(df = label_test,
                                    label_names = ['ltue'], 
                                    protected_attribute_names = protected_attribute)

    pred = preds_test[column]
    test_pred = test_label.copy()
    test_pred.labels = pred
    
    metric_test_pred = BinaryLabelDatasetMetric(test_pred, 
                                                unprivileged_groups = unprivileged_group,
                                                privileged_groups = privileged_group)

    par_sex = metric_test_pred.statistical_parity_difference() # Parity difference for female (edu = abi)
    
    protected_attribute = ['maxdeutsch1', 'maxschule9']
    unprivileged_group = [{'maxdeutsch1': 0, 'maxschule9': 1}]
    privileged_group = [{'maxdeutsch1': 1, 'maxschule9': 1}]
    
    test_label = BinaryLabelDataset(df = label_test,
                                    label_names = ['ltue'], 
                                    protected_attribute_names = protected_attribute)

    test_pred = test_label.copy()
    test_pred.labels = pred
    
    metric_test_pred = BinaryLabelDatasetMetric(test_pred, 
                                                unprivileged_groups = unprivileged_group,
                                                privileged_groups = privileged_group)
    
    par_ger = metric_test_pred.statistical_parity_difference() # Parity difference for nongerman (edu = abi)
    
    protected_attribute = ['nongerman_male', 'maxschule9']
    unprivileged_group = [{'nongerman_male': 1, 'maxschule9': 1}]
    privileged_group = [{'nongerman_male': 0, 'maxschule9': 1}]
    
    test_label = BinaryLabelDataset(df = label_test,
                                    label_names = ['ltue'], 
                                    protected_attribute_names = protected_attribute)

    test_pred = test_label.copy()
    test_pred.labels = pred
    
    metric_test_pred = BinaryLabelDatasetMetric(test_pred, 
                                                unprivileged_groups = unprivileged_group,
                                                privileged_groups = privileged_group)
    
    par_ger_male = metric_test_pred.statistical_parity_difference() # Parity difference for nongerman male (edu = abi)
    
    protected_attribute = ['nongerman_female', 'maxschule9']
    unprivileged_group = [{'nongerman_female': 1, 'maxschule9': 1}]
    privileged_group = [{'nongerman_female': 0, 'maxschule9': 1}]
    
    test_label = BinaryLabelDataset(df = label_test,
                                    label_names = ['ltue'], 
                                    protected_attribute_names = protected_attribute)

    test_pred = test_label.copy()
    test_pred.labels = pred
    
    metric_test_pred = BinaryLabelDatasetMetric(test_pred, 
                                                unprivileged_groups = unprivileged_group,
                                                privileged_groups = privileged_group)
    
    par_ger_female = metric_test_pred.statistical_parity_difference() # Parity difference for nongerman female (edu = abi)
    
    cond_fair2b.append([column,
                      par_sex,
                      par_ger,
                      par_ger_male,
                      par_ger_female])

cond_fair2b = pd.DataFrame(cond_fair2b)

cond_fair2b.loc[-1] = ['label', base_cpar_sex, base_cpar_ger, base_cpar_ger_male, base_cpar_ger_female]
cond_fair2b = cond_fair2b.sort_index()

cond_fair2b = cond_fair2b.rename(columns={0: "Model", 1: "Cond. Parity Diff. (Female)", 2: "Cond. Parity Diff. (Non-German)", 3: "Cond. Parity Diff. (Non-German-Male)", 4: "Cond. Parity Diff. (Non-German-Female)"})

cond_fair2b.to_latex('./output/test_cond_fairness2b.tex', index = False, float_format = "%.3f")
cond_fair2b.to_csv('./output/test_cond_fairness2b.csv', index = False)

# Combine all metrics

consist1 = pd.read_csv("./output/test_consistency1.csv")
fairness1 = pd.read_csv("./output/test_fairness1.csv")
cond_fair1 = pd.read_csv("./output/test_cond_fairness1.csv")

cond_fair1 = cond_fair1.drop(columns={'Model'})
consist1 = consist1.drop(columns={'Model'})

test_full_fair1 = pd.concat([fairness1,
                             cond_fair1,
                             consist1],
                            axis = 1)

test_full_fair1.to_latex('./output/test_full_fairness1.tex', index = False, float_format = "%.2f")

consist1b = pd.read_csv("./output/test_consistency1b.csv")
fairness1b = pd.read_csv("./output/test_fairness1b.csv")
cond_fair1b = pd.read_csv("./output/test_cond_fairness1b.csv")

cond_fair1b = cond_fair1b.drop(columns={'Model'})
consist1b = consist1b.drop(columns={'Model'})

test_full_fair1b = pd.concat([fairness1b,
                              cond_fair1b,
                              consist1b],
                             axis = 1)

test_full_fair1b.to_latex('./output/test_full_fairness1b.tex', index = False, float_format = "%.2f")

consist2 = pd.read_csv("./output/test_consistency2.csv")
fairness2 = pd.read_csv("./output/test_fairness2.csv")
cond_fair2 = pd.read_csv("./output/test_cond_fairness2.csv")

cond_fair2 = cond_fair2.drop(columns={'Model'})
consist2 = consist2.drop(columns={'Model'})

test_full_fair2 = pd.concat([fairness2,
                             cond_fair2,
                             consist2],
                            axis = 1)

test_full_fair2.to_latex('./output/test_full_fairness2.tex', index = False, float_format = "%.2f")

consist2b = pd.read_csv("./output/test_consistency2b.csv")
fairness2b = pd.read_csv("./output/test_fairness2b.csv")
cond_fair2b = pd.read_csv("./output/test_cond_fairness2b.csv")

cond_fair2b = cond_fair2b.drop(columns={'Model'})
consist2b = consist2b.drop(columns={'Model'})

test_full_fair2b = pd.concat([fairness2b,
                              cond_fair2b,
                              consist2b],
                             axis = 1)

test_full_fair2b.to_latex('./output/test_full_fairness2b.tex', index = False, float_format = "%.2f")
