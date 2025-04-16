"""
Fair Algorithmic Profiling
Evaluate Performance
"""

# Setup

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
plt.style.use('seaborn')

from sklearn.metrics import precision_score, recall_score, make_scorer, roc_curve, auc, precision_recall_curve, classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score, log_loss, f1_score

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric

from utils import aif_test, aif_plot

X_train_f = pd.read_csv("./output/X_train_f.csv")
X_train_s = pd.read_csv("./output/X_train_s.csv")

X_test_f = pd.read_csv("./output/X_test_f.csv")
X_test_s = pd.read_csv("./output/X_test_s.csv")
y_test = pd.read_csv("./output/y_test.csv")

preds_test = pd.read_csv("./output/preds_test.csv")

# 00 Add rule-based predictions

preds_test['skill_rule'] = np.where((X_test_s['maxausbildung_imp2'] == 0) & 
                                    (X_test_s['maxausbildung_imp3'] == 0) & 
                                    (X_test_s['maxausbildung_imp4'] == 0) & 
                                    (X_test_s['maxausbildung_imp5'] == 0) & 
                                    (X_test_s['maxausbildung_imp6'] == 0) &
                                    (X_test_s['maxausbildung_imp.Missing.'] == 0), 1, 0)
preds_test['skill_rule'].describe()

preds_test['time_rule'] = np.where((X_test_s['seeking1_tot_dur'] > 730), 1, 0)
preds_test['time_rule'].describe()

# 01a Overall Performance (w. protected attributes)

fpr1, tpr1, thresholds1 = roc_curve(preds_test['y_test'], preds_test['glm1_p'])
rocauc_glm1 = auc(fpr1, tpr1)
prec1, rec1, thresholds1 = precision_recall_curve(preds_test['y_test'], preds_test['glm1_p'])
prauc_glm1 = auc(rec1, prec1)

fpr2, tpr2, thresholds2 = roc_curve(preds_test['y_test'], preds_test['net1_p'])
rocauc_net1 = auc(fpr2, tpr2)
prec2, rec2, thresholds2 = precision_recall_curve(preds_test['y_test'], preds_test['net1_p'])
prauc_net1 = auc(rec2, prec2)

fpr3, tpr3, thresholds3 = roc_curve(preds_test['y_test'], preds_test['rf1_p'])
rocauc_rf1 = auc(fpr3, tpr3)
prec3, rec3, thresholds3 = precision_recall_curve(preds_test['y_test'], preds_test['rf1_p'])
prauc_rf1 = auc(rec3, prec3)

fpr4, tpr4, thresholds4 = roc_curve(preds_test['y_test'], preds_test['gbm1_p'])
rocauc_gbm1 = auc(fpr4, tpr4)
prec4, rec4, thresholds4 = precision_recall_curve(preds_test['y_test'], preds_test['gbm1_p'])
prauc_gbm1 = auc(rec4, prec4)

# ROC curves
random_probs = [0 for i in range(len(preds_test['y_test']))]
p_fpr, p_tpr, _ = roc_curve(preds_test['y_test'], random_probs, pos_label = 1)

fig, ax = plt.subplots(figsize = (9, 7))
plt.plot(fpr1, tpr1, color = 'blue', label = 'Logistic Regression')
plt.plot(fpr2, tpr2, color = 'green', label = 'Elastic Net')
plt.plot(fpr3, tpr3, color = 'orange', label = 'Random Forest')
plt.plot(fpr4, tpr4, color = 'red', label = 'Gradient Boosting')
plt.plot(p_fpr, p_tpr, linestyle = '--', color = 'black')
ax.tick_params(axis = 'both', which = 'major', labelsize = 16)
plt.xlabel('False Positive Rate', fontsize = 18)
plt.ylabel('True Positive Rate', fontsize = 18)
plt.legend(loc = 'best', fontsize = 16)
plt.savefig('./output/ROC1', dpi = 300)
plt.show();

# PR curves
fig, ax = plt.subplots(figsize = (9, 7))
plt.plot(rec1, prec1, color = 'blue', label = 'Logistic Regression')
plt.plot(rec2, prec2, color = 'green', label = 'Elastic Net')
plt.plot(rec3, prec3, color = 'orange', label = 'Random Forest')
plt.plot(rec4, prec4, color = 'red', label = 'Gradient Boosting')
ax.tick_params(axis = 'both', which = 'major', labelsize = 16)
plt.xlabel('Recall', fontsize = 18)
plt.ylabel('Precision', fontsize = 18)
plt.legend(loc = 'best', fontsize = 16)
plt.savefig('./output/PR1', dpi = 300)
plt.show();

# 01b Overall Performance (w/o protected attributes)

fpr1, tpr1, thresholds1 = roc_curve(preds_test['y_test'], preds_test['glm2_p'])
rocauc_glm2 = auc(fpr1, tpr1)
prec1, rec1, thresholds1 = precision_recall_curve(preds_test['y_test'], preds_test['glm2_p'])
prauc_glm2 = auc(rec1, prec1)

fpr2, tpr2, thresholds2 = roc_curve(preds_test['y_test'], preds_test['net2_p'])
rocauc_net2 = auc(fpr2, tpr2)
prec2, rec2, thresholds2 = precision_recall_curve(preds_test['y_test'], preds_test['net2_p'])
prauc_net2 = auc(rec2, prec2)

fpr3, tpr3, thresholds3 = roc_curve(preds_test['y_test'], preds_test['rf2_p'])
rocauc_rf2 = auc(fpr3, tpr3)
prec3, rec3, thresholds3 = precision_recall_curve(preds_test['y_test'], preds_test['rf2_p'])
prauc_rf2 = auc(rec3, prec3)

fpr4, tpr4, thresholds4 = roc_curve(preds_test['y_test'], preds_test['gbm2_p'])
rocauc_gbm2 = auc(fpr4, tpr4)
prec4, rec4, thresholds4 = precision_recall_curve(preds_test['y_test'], preds_test['gbm2_p'])
prauc_gbm2 = auc(rec4, prec4)

# ROC curves
random_probs = [0 for i in range(len(preds_test['y_test']))]
p_fpr, p_tpr, _ = roc_curve(preds_test['y_test'], random_probs, pos_label = 1)

fig, ax = plt.subplots(figsize = (9, 7))
plt.plot(fpr1, tpr1, color = 'blue', label = 'Logistic Regression')
plt.plot(fpr2, tpr2, color = 'green', label = 'Elastic Net')
plt.plot(fpr3, tpr3, color = 'orange', label = 'Random Forest')
plt.plot(fpr4, tpr4, color = 'red', label = 'Gradient Boosting')
plt.plot(p_fpr, p_tpr, linestyle = '--', color = 'black')
ax.tick_params(axis = 'both', which = 'major', labelsize = 16)
plt.xlabel('False Positive Rate', fontsize = 18)
plt.ylabel('True Positive Rate', fontsize = 18)
plt.legend(loc = 'best', fontsize = 16)
plt.savefig('./output/ROC2', dpi = 300)
plt.show();

# PR curves
fig, ax = plt.subplots(figsize = (9, 7))
plt.plot(rec1, prec1, color = 'blue', label = 'Logistic Regression')
plt.plot(rec2, prec2, color = 'green', label = 'Elastic Net')
plt.plot(rec3, prec3, color = 'orange', label = 'Random Forest')
plt.plot(rec4, prec4, color = 'red', label = 'Gradient Boosting')
ax.tick_params(axis = 'both', which = 'major', labelsize = 16)
plt.xlabel('Recall', fontsize = 18)
plt.ylabel('Precision', fontsize = 18)
plt.legend(loc = 'best', fontsize = 16)
plt.savefig('./output/PR2', dpi = 300)
plt.show();

# 01c Overall Performance (w protected attributes, train w. 2015)

fpr1, tpr1, thresholds1 = roc_curve(preds_test['y_test'], preds_test['glm1b_p'])
rocauc_glm1b = auc(fpr1, tpr1)
prec1, rec1, thresholds1 = precision_recall_curve(preds_test['y_test'], preds_test['glm1b_p'])
prauc_glm1b = auc(rec1, prec1)

fpr2, tpr2, thresholds2 = roc_curve(preds_test['y_test'], preds_test['net1b_p'])
rocauc_net1b = auc(fpr2, tpr2)
prec2, rec2, thresholds2 = precision_recall_curve(preds_test['y_test'], preds_test['net1b_p'])
prauc_net1b = auc(rec2, prec2)

fpr3, tpr3, thresholds3 = roc_curve(preds_test['y_test'], preds_test['rf1b_p'])
rocauc_rf1b = auc(fpr3, tpr3)
prec3, rec3, thresholds3 = precision_recall_curve(preds_test['y_test'], preds_test['rf1b_p'])
prauc_rf1b = auc(rec3, prec3)

fpr4, tpr4, thresholds4 = roc_curve(preds_test['y_test'], preds_test['gbm1b_p'])
rocauc_gbm1b = auc(fpr4, tpr4)
prec4, rec4, thresholds4 = precision_recall_curve(preds_test['y_test'], preds_test['gbm1b_p'])
prauc_gbm1b = auc(rec4, prec4)

# 01d Overall Performance (w/o protected attributes, train w. 2015)

fpr1, tpr1, thresholds1 = roc_curve(preds_test['y_test'], preds_test['glm2b_p'])
rocauc_glm2b = auc(fpr1, tpr1)
prec1, rec1, thresholds1 = precision_recall_curve(preds_test['y_test'], preds_test['glm2b_p'])
prauc_glm2b = auc(rec1, prec1)

fpr2, tpr2, thresholds2 = roc_curve(preds_test['y_test'], preds_test['net2b_p'])
rocauc_net2b = auc(fpr2, tpr2)
prec2, rec2, thresholds2 = precision_recall_curve(preds_test['y_test'], preds_test['net2b_p'])
prauc_net2b = auc(rec2, prec2)

fpr3, tpr3, thresholds3 = roc_curve(preds_test['y_test'], preds_test['rf2b_p'])
rocauc_rf2b = auc(fpr3, tpr3)
prec3, rec3, thresholds3 = precision_recall_curve(preds_test['y_test'], preds_test['rf2b_p'])
prauc_rf2b = auc(rec3, prec3)

fpr4, tpr4, thresholds4 = roc_curve(preds_test['y_test'], preds_test['gbm2b_p'])
rocauc_gbm2b = auc(fpr4, tpr4)
prec4, rec4, thresholds4 = precision_recall_curve(preds_test['y_test'], preds_test['gbm2b_p'])
prauc_gbm2b = auc(rec4, prec4)

# 02 Classification Performance (rule-based predictions)

srule_rep = classification_report(preds_test['y_test'], preds_test['skill_rule'], output_dict = True)
srule_perf = pd.DataFrame(np.array([srule_rep['accuracy'], srule_rep['1']['f1-score'], srule_rep['1']['precision'], srule_rep['1']['recall']]), columns = ['skill_rule'])

trule_rep = classification_report(preds_test['y_test'], preds_test['time_rule'], output_dict = True)
trule_perf = pd.DataFrame(np.array([trule_rep['accuracy'], trule_rep['1']['f1-score'], trule_rep['1']['precision'], trule_rep['1']['recall']]), columns = ['time_rule'])

perf0 = pd.concat([srule_perf,
                   trule_perf], 
                  axis = 1).transpose()

perf0 = perf0.rename(columns={0: "Accuracy", 1: "F1 Score", 2: "Precision", 3: "Recall"})

perf0.to_latex('./output/test_perf0.tex', index = False, float_format = "%.3f")

# 02a Classification Performance (w. protected attributes)

glm1_c1_rep = classification_report(preds_test['y_test'], preds_test['glm1_c1'], output_dict = True)
glm1_c1_acc = balanced_accuracy_score(preds_test['y_test'], preds_test['glm1_c1'])
glm1_c1_perf = pd.DataFrame(np.array([rocauc_glm1, prauc_glm1, glm1_c1_rep['accuracy'], glm1_c1_acc, glm1_c1_rep['1']['f1-score'], glm1_c1_rep['1']['precision'], glm1_c1_rep['1']['recall']]), columns = ['glm1_c1'])

glm1_c2_rep = classification_report(preds_test['y_test'], preds_test['glm1_c2'], output_dict = True)
glm1_c2_acc = balanced_accuracy_score(preds_test['y_test'], preds_test['glm1_c2'])
glm1_c2_perf = pd.DataFrame(np.array([rocauc_glm1, prauc_glm1, glm1_c2_rep['accuracy'], glm1_c2_acc, glm1_c2_rep['1']['f1-score'], glm1_c2_rep['1']['precision'], glm1_c2_rep['1']['recall']]), columns = ['glm1_c2'])

net1_c1_rep = classification_report(preds_test['y_test'], preds_test['net1_c1'], output_dict = True)
net1_c1_acc = balanced_accuracy_score(preds_test['y_test'], preds_test['net1_c1'])
net1_c1_perf = pd.DataFrame(np.array([rocauc_net1, prauc_net1, net1_c1_rep['accuracy'], net1_c1_acc, net1_c1_rep['1']['f1-score'], net1_c1_rep['1']['precision'], net1_c1_rep['1']['recall']]), columns = ['net1_c1'])

net1_c2_rep = classification_report(preds_test['y_test'], preds_test['net1_c2'], output_dict = True)
net1_c2_acc = balanced_accuracy_score(preds_test['y_test'], preds_test['net1_c2'])
net1_c2_perf = pd.DataFrame(np.array([rocauc_net1, prauc_net1, net1_c2_rep['accuracy'], net1_c2_acc, net1_c2_rep['1']['f1-score'], net1_c2_rep['1']['precision'], net1_c2_rep['1']['recall']]), columns = ['net1_c2'])

rf1_c1_rep = classification_report(preds_test['y_test'], preds_test['rf1_c1'], output_dict = True)
rf1_c1_acc = balanced_accuracy_score(preds_test['y_test'], preds_test['rf1_c1'])
rf1_c1_perf = pd.DataFrame(np.array([rocauc_rf1, prauc_rf1, rf1_c1_rep['accuracy'], rf1_c1_acc, rf1_c1_rep['1']['f1-score'], rf1_c1_rep['1']['precision'], rf1_c1_rep['1']['recall']]), columns = ['rf1_c1'])

rf1_c2_rep = classification_report(preds_test['y_test'], preds_test['rf1_c2'], output_dict = True)
rf1_c2_acc = balanced_accuracy_score(preds_test['y_test'], preds_test['rf1_c2'])
rf1_c2_perf = pd.DataFrame(np.array([rocauc_rf1, prauc_rf1, rf1_c2_rep['accuracy'], rf1_c2_acc, rf1_c2_rep['1']['f1-score'], rf1_c2_rep['1']['precision'], rf1_c2_rep['1']['recall']]), columns = ['rf1_c2'])

gbm1_c1_rep = classification_report(preds_test['y_test'], preds_test['gbm1_c1'], output_dict = True)
gbm1_c1_acc = balanced_accuracy_score(preds_test['y_test'], preds_test['gbm1_c1'])
gbm1_c1_perf = pd.DataFrame(np.array([rocauc_gbm1, prauc_gbm1, gbm1_c1_rep['accuracy'], gbm1_c1_acc, gbm1_c1_rep['1']['f1-score'], gbm1_c1_rep['1']['precision'], gbm1_c1_rep['1']['recall']]), columns = ['gbm1_c1'])

gbm1_c2_rep = classification_report(preds_test['y_test'], preds_test['gbm1_c2'], output_dict = True)
gbm1_c2_acc = balanced_accuracy_score(preds_test['y_test'], preds_test['gbm1_c2'])
gbm1_c2_perf = pd.DataFrame(np.array([rocauc_gbm1, prauc_gbm1, gbm1_c2_rep['accuracy'], gbm1_c2_acc, gbm1_c2_rep['1']['f1-score'], gbm1_c2_rep['1']['precision'], gbm1_c2_rep['1']['recall']]), columns = ['gbm1_c2'])

perf1_1 = pd.concat([glm1_c1_perf,
                     net1_c1_perf,
                     rf1_c1_perf,
                     gbm1_c1_perf], 
                    axis = 1).transpose()

perf1_1 = perf1_1.rename(columns={0: "ROC-AUC", 1: "PR-AUC", 2: "Accuracy", 3: "Balanced Accuracy", 4: "F1 Score", 5: "Precision", 6: "Recall"})

perf1_1.to_latex('./output/test_perf1_c1.tex', index = False, float_format = "%.3f")

perf1_2 = pd.concat([glm1_c2_perf,
                     net1_c2_perf,
                     rf1_c2_perf,
                     gbm1_c2_perf], 
                    axis = 1).transpose()

perf1_2 = perf1_2.rename(columns={0: "ROC-AUC", 1: "PR-AUC", 2: "Accuracy", 3: "Balanced Accuracy", 4: "F1 Score", 5: "Precision", 6: "Recall"})

perf1_2.to_latex('./output/test_perf1_c2.tex', index = False, float_format = "%.3f")

# 02b Classification Performance (w/o protected attributes)

glm2_c1_rep = classification_report(preds_test['y_test'], preds_test['glm2_c1'], output_dict = True)
glm2_c1_acc = balanced_accuracy_score(preds_test['y_test'], preds_test['glm2_c1'])
glm2_c1_perf = pd.DataFrame(np.array([rocauc_glm2, prauc_glm2, glm2_c1_rep['accuracy'], glm2_c1_acc, glm2_c1_rep['1']['f1-score'], glm2_c1_rep['1']['precision'], glm2_c1_rep['1']['recall']]), columns = ['glm2_c1'])

glm2_c2_rep = classification_report(preds_test['y_test'], preds_test['glm2_c2'], output_dict = True)
glm2_c2_acc = balanced_accuracy_score(preds_test['y_test'], preds_test['glm2_c2'])
glm2_c2_perf = pd.DataFrame(np.array([rocauc_glm2, prauc_glm2, glm2_c2_rep['accuracy'], glm2_c2_acc, glm2_c2_rep['1']['f1-score'], glm2_c2_rep['1']['precision'], glm2_c2_rep['1']['recall']]), columns = ['glm2_c2'])

net2_c1_rep = classification_report(preds_test['y_test'], preds_test['net2_c1'], output_dict = True)
net2_c1_acc = balanced_accuracy_score(preds_test['y_test'], preds_test['net2_c1'])
net2_c1_perf = pd.DataFrame(np.array([rocauc_net2, prauc_net2, net2_c1_rep['accuracy'], net2_c1_acc, net2_c1_rep['1']['f1-score'], net2_c1_rep['1']['precision'], net2_c1_rep['1']['recall']]), columns = ['net2_c1'])

net2_c2_rep = classification_report(preds_test['y_test'], preds_test['net2_c2'], output_dict = True)
net2_c2_acc = balanced_accuracy_score(preds_test['y_test'], preds_test['net2_c2'])
net2_c2_perf = pd.DataFrame(np.array([rocauc_net2, prauc_net2, net2_c2_rep['accuracy'], net2_c2_acc, net2_c2_rep['1']['f1-score'], net2_c2_rep['1']['precision'], net2_c2_rep['1']['recall']]), columns = ['net2_c2'])

rf2_c1_rep = classification_report(preds_test['y_test'], preds_test['rf2_c1'], output_dict = True)
rf2_c1_acc = balanced_accuracy_score(preds_test['y_test'], preds_test['rf2_c1'])
rf2_c1_perf = pd.DataFrame(np.array([rocauc_rf2, prauc_rf2, rf2_c1_rep['accuracy'], rf2_c1_acc, rf2_c1_rep['1']['f1-score'], rf2_c1_rep['1']['precision'], rf2_c1_rep['1']['recall']]), columns = ['rf2_c1'])

rf2_c2_rep = classification_report(preds_test['y_test'], preds_test['rf2_c2'], output_dict = True)
rf2_c2_acc = balanced_accuracy_score(preds_test['y_test'], preds_test['rf2_c2'])
rf2_c2_perf = pd.DataFrame(np.array([rocauc_rf2, prauc_rf2, rf2_c2_rep['accuracy'], rf2_c2_acc, rf2_c2_rep['1']['f1-score'], rf2_c2_rep['1']['precision'], rf2_c2_rep['1']['recall']]), columns = ['rf2_c2'])

gbm2_c1_rep = classification_report(preds_test['y_test'], preds_test['gbm2_c1'], output_dict = True)
gbm2_c1_acc = balanced_accuracy_score(preds_test['y_test'], preds_test['gbm2_c1'])
gbm2_c1_perf = pd.DataFrame(np.array([rocauc_gbm2, prauc_gbm2, gbm2_c1_rep['accuracy'], gbm2_c1_acc, gbm2_c1_rep['1']['f1-score'], gbm2_c1_rep['1']['precision'], gbm2_c1_rep['1']['recall']]), columns = ['gbm2_c1'])

gbm2_c2_rep = classification_report(preds_test['y_test'], preds_test['gbm2_c2'], output_dict = True)
gbm2_c2_acc = balanced_accuracy_score(preds_test['y_test'], preds_test['gbm2_c2'])
gbm2_c2_perf = pd.DataFrame(np.array([rocauc_gbm2, prauc_gbm2, gbm2_c2_rep['accuracy'], gbm2_c2_acc, gbm2_c2_rep['1']['f1-score'], gbm2_c2_rep['1']['precision'], gbm2_c2_rep['1']['recall']]), columns = ['gbm2_c2'])

perf2_1 = pd.concat([glm2_c1_perf,
                     net2_c1_perf,
                     rf2_c1_perf,
                     gbm2_c1_perf], 
                    axis = 1).transpose()

perf2_1 = perf2_1.rename(columns={0: "ROC-AUC", 1: "PR-AUC", 2: "Accuracy", 3: "Balanced Accuracy", 4: "F1 Score", 5: "Precision", 6: "Recall"})

perf2_1.to_latex('./output/test_perf2_c1.tex', index = False, float_format = "%.3f")

perf2_2 = pd.concat([glm2_c2_perf,
                     net2_c2_perf,
                     rf2_c2_perf,
                     gbm2_c2_perf], 
                    axis = 1).transpose()

perf2_2 = perf2_2.rename(columns={0: "ROC-AUC", 1: "PR-AUC", 2: "Accuracy", 3: "Balanced Accuracy", 4: "F1 Score", 5: "Precision", 6: "Recall"})

perf2_2.to_latex('./output/test_perf2_c2.tex', index = False, float_format = "%.3f")

# 02c Classification Performance (w protected attributes, train w. 2015)

glm1b_c1_rep = classification_report(preds_test['y_test'], preds_test['glm1b_c1'], output_dict = True)
glm1b_c1_acc = balanced_accuracy_score(preds_test['y_test'], preds_test['glm1b_c1'])
glm1b_c1_perf = pd.DataFrame(np.array([rocauc_glm1b, prauc_glm1b, glm1b_c1_rep['accuracy'], glm1b_c1_acc, glm1b_c1_rep['1']['f1-score'], glm1b_c1_rep['1']['precision'], glm1b_c1_rep['1']['recall']]), columns = ['glm1b_c1'])

glm1b_c2_rep = classification_report(preds_test['y_test'], preds_test['glm1b_c2'], output_dict = True)
glm1b_c2_acc = balanced_accuracy_score(preds_test['y_test'], preds_test['glm1b_c2'])
glm1b_c2_perf = pd.DataFrame(np.array([rocauc_glm1b, prauc_glm1b, glm1b_c2_rep['accuracy'], glm1b_c2_acc, glm1b_c2_rep['1']['f1-score'], glm1b_c2_rep['1']['precision'], glm1b_c2_rep['1']['recall']]), columns = ['glm1b_c2'])

net1b_c1_rep = classification_report(preds_test['y_test'], preds_test['net1b_c1'], output_dict = True)
net1b_c1_acc = balanced_accuracy_score(preds_test['y_test'], preds_test['net1b_c1'])
net1b_c1_perf = pd.DataFrame(np.array([rocauc_net1b, prauc_net1b, net1b_c1_rep['accuracy'], net1b_c1_acc, net1b_c1_rep['1']['f1-score'], net1b_c1_rep['1']['precision'], net1b_c1_rep['1']['recall']]), columns = ['net1b_c1'])

net1b_c2_rep = classification_report(preds_test['y_test'], preds_test['net1b_c2'], output_dict = True)
net1b_c2_acc = balanced_accuracy_score(preds_test['y_test'], preds_test['net1b_c2'])
net1b_c2_perf = pd.DataFrame(np.array([rocauc_net1b, prauc_net1b, net1b_c2_rep['accuracy'], net1b_c2_acc, net1b_c2_rep['1']['f1-score'], net1b_c2_rep['1']['precision'], net1b_c2_rep['1']['recall']]), columns = ['net1b_c2'])

rf1b_c1_rep = classification_report(preds_test['y_test'], preds_test['rf1b_c1'], output_dict = True)
rf1b_c1_acc = balanced_accuracy_score(preds_test['y_test'], preds_test['rf1b_c1'])
rf1b_c1_perf = pd.DataFrame(np.array([rocauc_rf1b, prauc_rf1b, rf1b_c1_rep['accuracy'], rf1b_c1_acc, rf1b_c1_rep['1']['f1-score'], rf1b_c1_rep['1']['precision'], rf1b_c1_rep['1']['recall']]), columns = ['rf1b_c1'])

rf1b_c2_rep = classification_report(preds_test['y_test'], preds_test['rf1b_c2'], output_dict = True)
rf1b_c2_acc = balanced_accuracy_score(preds_test['y_test'], preds_test['rf1b_c2'])
rf1b_c2_perf = pd.DataFrame(np.array([rocauc_rf1b, prauc_rf1b, rf1b_c2_rep['accuracy'], rf1b_c2_acc, rf1b_c2_rep['1']['f1-score'], rf1b_c2_rep['1']['precision'], rf1b_c2_rep['1']['recall']]), columns = ['rf1b_c2'])

gbm1b_c1_rep = classification_report(preds_test['y_test'], preds_test['gbm1b_c1'], output_dict = True)
gbm1b_c1_acc = balanced_accuracy_score(preds_test['y_test'], preds_test['gbm1b_c1'])
gbm1b_c1_perf = pd.DataFrame(np.array([rocauc_gbm1b, prauc_gbm1b, gbm1b_c1_rep['accuracy'], gbm1b_c1_acc, gbm1b_c1_rep['1']['f1-score'], gbm1b_c1_rep['1']['precision'], gbm1b_c1_rep['1']['recall']]), columns = ['gbm1b_c1'])

gbm1b_c2_rep = classification_report(preds_test['y_test'], preds_test['gbm1b_c2'], output_dict = True)
gbm1b_c2_acc = balanced_accuracy_score(preds_test['y_test'], preds_test['gbm1b_c2'])
gbm1b_c2_perf = pd.DataFrame(np.array([rocauc_gbm1b, prauc_gbm1b, gbm1b_c2_rep['accuracy'], gbm1b_c2_acc, gbm1b_c2_rep['1']['f1-score'], gbm1b_c2_rep['1']['precision'], gbm1b_c2_rep['1']['recall']]), columns = ['gbm1b_c2'])

perf1b_1 = pd.concat([glm1b_c1_perf,
                     net1b_c1_perf,
                     rf1b_c1_perf,
                     gbm1b_c1_perf], 
                    axis = 1).transpose()

perf1b_1 = perf1b_1.rename(columns={0: "ROC-AUC", 1: "PR-AUC", 2: "Accuracy", 3: "Balanced Accuracy", 4: "F1 Score", 5: "Precision", 6: "Recall"})

perf1b_1.to_latex('./output/test_perf1b_c1.tex', index = False, float_format = "%.3f")

perf1b_2 = pd.concat([glm1b_c2_perf,
                     net1b_c2_perf,
                     rf1b_c2_perf,
                     gbm1b_c2_perf], 
                    axis = 1).transpose()

perf1b_2 = perf1b_2.rename(columns={0: "ROC-AUC", 1: "PR-AUC", 2: "Accuracy", 3: "Balanced Accuracy", 4: "F1 Score", 5: "Precision", 6: "Recall"})

perf1b_2.to_latex('./output/test_perf1b_c2.tex', index = False, float_format = "%.3f")

# 02d Classification Performance (w/o protected attributes, train w. 2015)

glm2b_c1_rep = classification_report(preds_test['y_test'], preds_test['glm2b_c1'], output_dict = True)
glm2b_c1_acc = balanced_accuracy_score(preds_test['y_test'], preds_test['glm2b_c1'])
glm2b_c1_perf = pd.DataFrame(np.array([rocauc_glm2b, prauc_glm2b, glm2b_c1_rep['accuracy'], glm2b_c1_acc, glm2b_c1_rep['1']['f1-score'], glm2b_c1_rep['1']['precision'], glm2b_c1_rep['1']['recall']]), columns = ['glm2b_c1'])

glm2b_c2_rep = classification_report(preds_test['y_test'], preds_test['glm2b_c2'], output_dict = True)
glm2b_c2_acc = balanced_accuracy_score(preds_test['y_test'], preds_test['glm2b_c2'])
glm2b_c2_perf = pd.DataFrame(np.array([rocauc_glm2b, prauc_glm2b, glm2b_c2_rep['accuracy'], glm2b_c2_acc, glm2b_c2_rep['1']['f1-score'], glm2b_c2_rep['1']['precision'], glm2b_c2_rep['1']['recall']]), columns = ['glm2b_c2'])

net2b_c1_rep = classification_report(preds_test['y_test'], preds_test['net2b_c1'], output_dict = True)
net2b_c1_acc = balanced_accuracy_score(preds_test['y_test'], preds_test['net2b_c1'])
net2b_c1_perf = pd.DataFrame(np.array([rocauc_net2b, prauc_net2b, net2b_c1_rep['accuracy'], net2b_c1_acc, net2b_c1_rep['1']['f1-score'], net2b_c1_rep['1']['precision'], net2b_c1_rep['1']['recall']]), columns = ['net2b_c1'])

net2b_c2_rep = classification_report(preds_test['y_test'], preds_test['net2b_c2'], output_dict = True)
net2b_c2_acc = balanced_accuracy_score(preds_test['y_test'], preds_test['net2b_c2'])
net2b_c2_perf = pd.DataFrame(np.array([rocauc_net2b, prauc_net2b, net2b_c2_rep['accuracy'], net2b_c2_acc, net2b_c2_rep['1']['f1-score'], net2b_c2_rep['1']['precision'], net2b_c2_rep['1']['recall']]), columns = ['net2b_c2'])

rf2b_c1_rep = classification_report(preds_test['y_test'], preds_test['rf2b_c1'], output_dict = True)
rf2b_c1_acc = balanced_accuracy_score(preds_test['y_test'], preds_test['rf2b_c1'])
rf2b_c1_perf = pd.DataFrame(np.array([rocauc_rf2b, prauc_rf2b, rf2b_c1_rep['accuracy'], rf2b_c1_acc, rf2b_c1_rep['1']['f1-score'], rf2b_c1_rep['1']['precision'], rf2b_c1_rep['1']['recall']]), columns = ['rf2b_c1'])

rf2b_c2_rep = classification_report(preds_test['y_test'], preds_test['rf2b_c2'], output_dict = True)
rf2b_c2_acc = balanced_accuracy_score(preds_test['y_test'], preds_test['rf2b_c2'])
rf2b_c2_perf = pd.DataFrame(np.array([rocauc_rf2b, prauc_rf2b, rf2b_c2_rep['accuracy'], rf2b_c2_acc, rf2b_c2_rep['1']['f1-score'], rf2b_c2_rep['1']['precision'], rf2b_c2_rep['1']['recall']]), columns = ['rf2b_c2'])

gbm2b_c1_rep = classification_report(preds_test['y_test'], preds_test['gbm2b_c1'], output_dict = True)
gbm2b_c1_acc = balanced_accuracy_score(preds_test['y_test'], preds_test['gbm2b_c1'])
gbm2b_c1_perf = pd.DataFrame(np.array([rocauc_gbm2b, prauc_gbm2b, gbm2b_c1_rep['accuracy'], gbm2b_c1_acc, gbm2b_c1_rep['1']['f1-score'], gbm2b_c1_rep['1']['precision'], gbm2b_c1_rep['1']['recall']]), columns = ['gbm2b_c1'])

gbm2b_c2_rep = classification_report(preds_test['y_test'], preds_test['gbm2b_c2'], output_dict = True)
gbm2b_c2_acc = balanced_accuracy_score(preds_test['y_test'], preds_test['gbm2b_c2'])
gbm2b_c2_perf = pd.DataFrame(np.array([rocauc_gbm2b, prauc_gbm2b, gbm2b_c2_rep['accuracy'], gbm2b_c2_acc, gbm2b_c2_rep['1']['f1-score'], gbm2b_c2_rep['1']['precision'], gbm2b_c2_rep['1']['recall']]), columns = ['gbm2b_c2'])

perf2b_1 = pd.concat([glm2b_c1_perf,
                     net2b_c1_perf,
                     rf2b_c1_perf,
                     gbm2b_c1_perf], 
                    axis = 1).transpose()

perf2b_1 = perf2b_1.rename(columns={0: "ROC-AUC", 1: "PR-AUC", 2: "Accuracy", 3: "Balanced Accuracy", 4: "F1 Score", 5: "Precision", 6: "Recall"})

perf2b_1.to_latex('./output/test_perf2b_c1.tex', index = False, float_format = "%.3f")

perf2b_2 = pd.concat([glm2b_c2_perf,
                     net2b_c2_perf,
                     rf2b_c2_perf,
                     gbm2b_c2_perf], 
                    axis = 1).transpose()

perf2b_2 = perf2b_2.rename(columns={0: "ROC-AUC", 1: "PR-AUC", 2: "Accuracy", 3: "Balanced Accuracy", 4: "F1 Score", 5: "Precision", 6: "Recall"})

perf2b_2.to_latex('./output/test_perf2b_c2.tex', index = False, float_format = "%.3f")

# 03 Classification Performance by Groups

comb_test = pd.concat([preds_test, X_test_f], axis = 1)

comb_test['nongerman'] = np.where(comb_test['maxdeutsch1'] == 0, 1, 0)
comb_test.loc[comb_test['maxdeutsch.Missing.'] == 1, 'nongerman'] = np.nan
comb_test['nongerman_male'] = np.where((comb_test['nongerman'] == 1) & (comb_test['frau1'] == 0), 1, 0)
comb_test['nongerman_female'] = np.where((comb_test['nongerman'] == 1) & (comb_test['frau1'] == 1), 1, 0)

comb_test = comb_test.dropna()

f1 = []
pops = ('frau1', 'nongerman', 'nongerman_male', 'nongerman_female')

f1.append(['Overall',
           f1_score(comb_test['y_test'], comb_test['glm1_c1']), 
           f1_score(comb_test['y_test'], comb_test['glm1_c2']),
           f1_score(comb_test['y_test'], comb_test['glm2_c1']),
           f1_score(comb_test['y_test'], comb_test['glm2_c2']),
           f1_score(comb_test['y_test'], comb_test['glm1b_c1']), 
           f1_score(comb_test['y_test'], comb_test['glm1b_c2']),
           f1_score(comb_test['y_test'], comb_test['glm2b_c1']), 
           f1_score(comb_test['y_test'], comb_test['glm2b_c2']),
           f1_score(comb_test['y_test'], comb_test['net1_c1']), 
           f1_score(comb_test['y_test'], comb_test['net1_c2']),
           f1_score(comb_test['y_test'], comb_test['net2_c1']), 
           f1_score(comb_test['y_test'], comb_test['net2_c2']),
           f1_score(comb_test['y_test'], comb_test['net1b_c1']), 
           f1_score(comb_test['y_test'], comb_test['net1b_c2']),
           f1_score(comb_test['y_test'], comb_test['net2b_c1']), 
           f1_score(comb_test['y_test'], comb_test['net2b_c2']),
           f1_score(comb_test['y_test'], comb_test['rf1_c1']), 
           f1_score(comb_test['y_test'], comb_test['rf1_c2']),
           f1_score(comb_test['y_test'], comb_test['rf2_c1']), 
           f1_score(comb_test['y_test'], comb_test['rf2_c2']),
           f1_score(comb_test['y_test'], comb_test['rf1b_c1']), 
           f1_score(comb_test['y_test'], comb_test['rf1b_c2']),
           f1_score(comb_test['y_test'], comb_test['rf2b_c1']), 
           f1_score(comb_test['y_test'], comb_test['rf2b_c2']),
           f1_score(comb_test['y_test'], comb_test['gbm1_c1']), 
           f1_score(comb_test['y_test'], comb_test['gbm1_c2']),
           f1_score(comb_test['y_test'], comb_test['gbm2_c1']), 
           f1_score(comb_test['y_test'], comb_test['gbm2_c2']),
           f1_score(comb_test['y_test'], comb_test['gbm1b_c1']), 
           f1_score(comb_test['y_test'], comb_test['gbm1b_c2']),
           f1_score(comb_test['y_test'], comb_test['gbm2b_c1']), 
           f1_score(comb_test['y_test'], comb_test['gbm2b_c2'])
          ])

for pop in pops: 
    subset = (comb_test[pop] == 1)
    f1.append([pop,
                f1_score(comb_test[subset]['y_test'], comb_test[subset]['glm1_c1']),
                f1_score(comb_test[subset]['y_test'], comb_test[subset]['glm1_c2']),
                f1_score(comb_test[subset]['y_test'], comb_test[subset]['glm2_c1']),
                f1_score(comb_test[subset]['y_test'], comb_test[subset]['glm2_c2']),
                f1_score(comb_test[subset]['y_test'], comb_test[subset]['glm1b_c1']),
                f1_score(comb_test[subset]['y_test'], comb_test[subset]['glm1b_c2']),
                f1_score(comb_test[subset]['y_test'], comb_test[subset]['glm2b_c1']),
                f1_score(comb_test[subset]['y_test'], comb_test[subset]['glm2b_c2']),
                f1_score(comb_test[subset]['y_test'], comb_test[subset]['net1_c1']),
                f1_score(comb_test[subset]['y_test'], comb_test[subset]['net1_c2']),
                f1_score(comb_test[subset]['y_test'], comb_test[subset]['net2_c1']),
                f1_score(comb_test[subset]['y_test'], comb_test[subset]['net2_c2']),
                f1_score(comb_test[subset]['y_test'], comb_test[subset]['net1b_c1']),
                f1_score(comb_test[subset]['y_test'], comb_test[subset]['net1b_c2']),
                f1_score(comb_test[subset]['y_test'], comb_test[subset]['net2b_c1']),
                f1_score(comb_test[subset]['y_test'], comb_test[subset]['net2b_c2']),
                f1_score(comb_test[subset]['y_test'], comb_test[subset]['rf1_c1']),
                f1_score(comb_test[subset]['y_test'], comb_test[subset]['rf1_c2']),
                f1_score(comb_test[subset]['y_test'], comb_test[subset]['rf2_c1']),
                f1_score(comb_test[subset]['y_test'], comb_test[subset]['rf2_c2']),
                f1_score(comb_test[subset]['y_test'], comb_test[subset]['rf1b_c1']),
                f1_score(comb_test[subset]['y_test'], comb_test[subset]['rf1b_c2']),
                f1_score(comb_test[subset]['y_test'], comb_test[subset]['rf2b_c1']),
                f1_score(comb_test[subset]['y_test'], comb_test[subset]['rf2b_c2']),
                f1_score(comb_test[subset]['y_test'], comb_test[subset]['gbm1_c1']),
                f1_score(comb_test[subset]['y_test'], comb_test[subset]['gbm1_c2']),
                f1_score(comb_test[subset]['y_test'], comb_test[subset]['gbm2_c1']),
                f1_score(comb_test[subset]['y_test'], comb_test[subset]['gbm2_c2']),
                f1_score(comb_test[subset]['y_test'], comb_test[subset]['gbm1b_c1']),
                f1_score(comb_test[subset]['y_test'], comb_test[subset]['gbm1b_c2']),
                f1_score(comb_test[subset]['y_test'], comb_test[subset]['gbm2b_c1']),
                f1_score(comb_test[subset]['y_test'], comb_test[subset]['gbm2b_c2'])
              ])

f1_group = pd.DataFrame(f1)

f1_group = f1_group.rename(columns={0: 'pop', 
                                    1: 'LR_1_c1', 2: 'LR_1_c2', 3: 'LR_2_c1', 4: 'LR_2_c2', 
                                    5: 'LR_1b_c1', 6: 'LR_1b_c2', 7: 'LR_2b_c1', 8: 'LR_2b_c2',
                                    9: 'PLR_1_c1', 10: 'PLR_1_c2', 11: 'PLR_2_c1', 12: 'PLR_2_c2', 
                                    13: 'PLR_1b_c1', 14: 'PLR_1b_c2', 15: 'PLR_2b_c1', 16: 'PLR_2b_c2',
                                    17: 'RF_1_c1', 18: 'RF_1_c2', 19: 'RF_2_c1', 20: 'RF_2_c2', 
                                    21: 'RF_1b_c1', 22: 'RF_1b_c2', 23: 'RF_2b_c1', 24: 'RF_2b_c2', 
                                    25: 'GBM_1_c1', 26: 'GBM_1_c2', 27: 'GBM_2_c1', 28: 'GBM_2_c2',
                                    29: 'GBM_1b_c1', 30: 'GBM_1b_c2', 31: 'GBM_2b_c1', 32: 'GBM_2b_c2'})
f1_group['pop'] = ['Overall', 'Female', 'Non-German', 'Non-Ger. M', 'Non-Ger. F']

f1_group_l1 = pd.wide_to_long(f1_group, stubnames=['LR', 'PLR', 'RF', 'GBM'], i=['pop'], j='model', sep='_', suffix='\w+')
f1_group_l1 = f1_group_l1.reset_index()
f1_group_l2 = f1_group_l1.melt(id_vars=['pop', 'model'], var_name='method')
f1_group_l2[['Type', 'Cutoff']] = f1_group_l2['model'].str.split("_", expand = True)

f1_group_l2.to_csv('./output/f1_group.csv', index = False)

acc = []

acc.append(['Overall',
            balanced_accuracy_score(comb_test['y_test'], comb_test['glm1_c1']),
            balanced_accuracy_score(comb_test['y_test'], comb_test['glm1_c2']),
            balanced_accuracy_score(comb_test['y_test'], comb_test['glm2_c1']),
            balanced_accuracy_score(comb_test['y_test'], comb_test['glm2_c2']),
            balanced_accuracy_score(comb_test['y_test'], comb_test['glm1b_c1']),
            balanced_accuracy_score(comb_test['y_test'], comb_test['glm1b_c2']),
            balanced_accuracy_score(comb_test['y_test'], comb_test['glm2b_c1']),
            balanced_accuracy_score(comb_test['y_test'], comb_test['glm2b_c2']),
            balanced_accuracy_score(comb_test['y_test'], comb_test['net1_c1']),
            balanced_accuracy_score(comb_test['y_test'], comb_test['net1_c2']),
            balanced_accuracy_score(comb_test['y_test'], comb_test['net2_c1']),
            balanced_accuracy_score(comb_test['y_test'], comb_test['net2_c2']),
            balanced_accuracy_score(comb_test['y_test'], comb_test['net1b_c1']),
            balanced_accuracy_score(comb_test['y_test'], comb_test['net1b_c2']),
            balanced_accuracy_score(comb_test['y_test'], comb_test['net2b_c1']),
            balanced_accuracy_score(comb_test['y_test'], comb_test['net2b_c2']),
            balanced_accuracy_score(comb_test['y_test'], comb_test['rf1_c1']),
            balanced_accuracy_score(comb_test['y_test'], comb_test['rf1_c2']),
            balanced_accuracy_score(comb_test['y_test'], comb_test['rf2_c1']),
            balanced_accuracy_score(comb_test['y_test'], comb_test['rf2_c2']),
            balanced_accuracy_score(comb_test['y_test'], comb_test['rf1b_c1']),
            balanced_accuracy_score(comb_test['y_test'], comb_test['rf1b_c2']),
            balanced_accuracy_score(comb_test['y_test'], comb_test['rf2b_c1']),
            balanced_accuracy_score(comb_test['y_test'], comb_test['rf2b_c2']),
            balanced_accuracy_score(comb_test['y_test'], comb_test['gbm1_c1']),
            balanced_accuracy_score(comb_test['y_test'], comb_test['gbm1_c2']),
            balanced_accuracy_score(comb_test['y_test'], comb_test['gbm2_c1']),
            balanced_accuracy_score(comb_test['y_test'], comb_test['gbm2_c2']),
            balanced_accuracy_score(comb_test['y_test'], comb_test['gbm1b_c1']),
            balanced_accuracy_score(comb_test['y_test'], comb_test['gbm1b_c2']),
            balanced_accuracy_score(comb_test['y_test'], comb_test['gbm2b_c1']),
            balanced_accuracy_score(comb_test['y_test'], comb_test['gbm2b_c2'])
          ])

for pop in pops: 
    subset = (comb_test[pop] == 1)
    acc.append([pop,
                balanced_accuracy_score(comb_test[subset]['y_test'], comb_test[subset]['glm1_c1']),
                balanced_accuracy_score(comb_test[subset]['y_test'], comb_test[subset]['glm1_c2']),
                balanced_accuracy_score(comb_test[subset]['y_test'], comb_test[subset]['glm2_c1']),
                balanced_accuracy_score(comb_test[subset]['y_test'], comb_test[subset]['glm2_c2']),
                balanced_accuracy_score(comb_test[subset]['y_test'], comb_test[subset]['glm1b_c1']),
                balanced_accuracy_score(comb_test[subset]['y_test'], comb_test[subset]['glm1b_c2']),
                balanced_accuracy_score(comb_test[subset]['y_test'], comb_test[subset]['glm2b_c1']),
                balanced_accuracy_score(comb_test[subset]['y_test'], comb_test[subset]['glm2b_c2']),
                balanced_accuracy_score(comb_test[subset]['y_test'], comb_test[subset]['net1_c1']),
                balanced_accuracy_score(comb_test[subset]['y_test'], comb_test[subset]['net1_c2']),
                balanced_accuracy_score(comb_test[subset]['y_test'], comb_test[subset]['net2_c1']),
                balanced_accuracy_score(comb_test[subset]['y_test'], comb_test[subset]['net2_c2']),
                balanced_accuracy_score(comb_test[subset]['y_test'], comb_test[subset]['net1b_c1']),
                balanced_accuracy_score(comb_test[subset]['y_test'], comb_test[subset]['net1b_c2']),
                balanced_accuracy_score(comb_test[subset]['y_test'], comb_test[subset]['net2b_c1']),
                balanced_accuracy_score(comb_test[subset]['y_test'], comb_test[subset]['net2b_c2']),
                balanced_accuracy_score(comb_test[subset]['y_test'], comb_test[subset]['rf1_c1']),
                balanced_accuracy_score(comb_test[subset]['y_test'], comb_test[subset]['rf1_c2']),
                balanced_accuracy_score(comb_test[subset]['y_test'], comb_test[subset]['rf2_c1']),
                balanced_accuracy_score(comb_test[subset]['y_test'], comb_test[subset]['rf2_c2']),
                balanced_accuracy_score(comb_test[subset]['y_test'], comb_test[subset]['rf1b_c1']),
                balanced_accuracy_score(comb_test[subset]['y_test'], comb_test[subset]['rf1b_c2']),
                balanced_accuracy_score(comb_test[subset]['y_test'], comb_test[subset]['rf2b_c1']),
                balanced_accuracy_score(comb_test[subset]['y_test'], comb_test[subset]['rf2b_c2']),
                balanced_accuracy_score(comb_test[subset]['y_test'], comb_test[subset]['gbm1_c1']),
                balanced_accuracy_score(comb_test[subset]['y_test'], comb_test[subset]['gbm1_c2']),
                balanced_accuracy_score(comb_test[subset]['y_test'], comb_test[subset]['gbm2_c1']),
                balanced_accuracy_score(comb_test[subset]['y_test'], comb_test[subset]['gbm2_c2']),
                balanced_accuracy_score(comb_test[subset]['y_test'], comb_test[subset]['gbm1b_c1']),
                balanced_accuracy_score(comb_test[subset]['y_test'], comb_test[subset]['gbm1b_c2']),
                balanced_accuracy_score(comb_test[subset]['y_test'], comb_test[subset]['gbm2b_c1']),
                balanced_accuracy_score(comb_test[subset]['y_test'], comb_test[subset]['gbm2b_c2'])
              ])

acc_group = pd.DataFrame(acc)

acc_group = acc_group.rename(columns={0: 'pop', 
                                      1: 'LR_1_c1', 2: 'LR_1_c2', 3: 'LR_2_c1', 4: 'LR_2_c2', 
                                      5: 'LR_1b_c1', 6: 'LR_1b_c2', 7: 'LR_2b_c1', 8: 'LR_2b_c2',
                                      9: 'PLR_1_c1', 10: 'PLR_1_c2', 11: 'PLR_2_c1', 12: 'PLR_2_c2', 
                                      13: 'PLR_1b_c1', 14: 'PLR_1b_c2', 15: 'PLR_2b_c1', 16: 'PLR_2b_c2',
                                      17: 'RF_1_c1', 18: 'RF_1_c2', 19: 'RF_2_c1', 20: 'RF_2_c2', 
                                      21: 'RF_1b_c1', 22: 'RF_1b_c2', 23: 'RF_2b_c1', 24: 'RF_2b_c2', 
                                      25: 'GBM_1_c1', 26: 'GBM_1_c2', 27: 'GBM_2_c1', 28: 'GBM_2_c2',
                                      29: 'GBM_1b_c1', 30: 'GBM_1b_c2', 31: 'GBM_2b_c1', 32: 'GBM_2b_c2'})
acc_group['pop'] = ['Overall', 'Female', 'Non-German', 'Non-Ger. M', 'Non-Ger. F']

acc_group_l1 = pd.wide_to_long(acc_group, stubnames=['LR', 'PLR', 'RF', 'GBM'], i=['pop'], j='model', sep='_', suffix='\w+')
acc_group_l1 = acc_group_l1.reset_index()
acc_group_l2 = acc_group_l1.melt(id_vars=['pop', 'model'], var_name='method')
acc_group_l2[['Type', 'Cutoff']] = acc_group_l2['model'].str.split("_", expand = True)

acc_group_l2.to_csv('./output/acc_group.csv', index = False)

# Plots

sns.stripplot(x = "value", y = "pop", data = f1_group_l2, alpha = .5, zorder = 1)
sns.stripplot(x = "value", y = "pop", hue = "method", dodge = True, data = f1_group_l2, zorder = 1)
sns.stripplot(x = "value", y = "pop", hue = "Cutoff", dodge = True, data = f1_group_l2, zorder = 1)

fig, ax1 = plt.subplots(figsize = (8.5, 7))
ax1 = sns.boxplot(x = "value", y = "pop", hue = "Cutoff", dodge = True, data = f1_group_l2, linewidth = 1.25, fliersize = 0)
ax1.xaxis.set_tick_params(labelsize = 15)
ax1.yaxis.set_tick_params(labelsize = 15)
ax2 = sns.stripplot(x = "value", y = "pop", hue = "Cutoff", dodge = True, data = f1_group_l2, palette = "muted")
ax2.set_xlabel("Performance Score", fontsize = 15)
ax2.set_ylabel("")
handles, labels = ax1.get_legend_handles_labels()
plt.legend(handles[2:4], ['Policy 1a', 'Policy 1b'], bbox_to_anchor = (0.975, 0.55), loc = 2, fontsize = 14)
plt.setp(ax1.artists, fill = False)
plt.tight_layout()
plt.savefig('./output/group_f1', dpi = 300)

sns.stripplot(x = "value", y = "pop", data = acc_group_l2, alpha = .5, zorder = 1)
sns.stripplot(x = "value", y = "pop", hue = "method", dodge = True, data = acc_group_l2, zorder = 1)
sns.stripplot(x = "value", y = "pop", hue = "Cutoff", dodge = True, data = acc_group_l2, zorder = 1)

fig, ax1 = plt.subplots(figsize = (8.5, 7))
ax1 = sns.boxplot(x = "value", y = "pop", hue = "Cutoff", dodge = True, data = acc_group_l2, linewidth = 1.25, fliersize = 0)
ax1.xaxis.set_tick_params(labelsize = 15)
ax1.yaxis.set_tick_params(labelsize = 15)
ax2 = sns.stripplot(x = "value", y = "pop", hue = "Cutoff", dodge = True, data = acc_group_l2, palette = "muted")
ax2.set_xlabel("Performance Score", fontsize = 15)
ax2.set_ylabel("")
handles, labels = ax1.get_legend_handles_labels()
plt.legend(handles[2:4], ['Policy 1a', 'Policy 1b'], bbox_to_anchor = (0.975, 0.55), loc = 2, fontsize = 14)
plt.setp(ax1.artists, fill = False)
plt.tight_layout()
plt.savefig('./output/group_acc', dpi = 300)

