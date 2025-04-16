"""
Fair Algorithmic Profiling
Train
"""

# Setup

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import precision_score, recall_score, make_scorer
from sklearn.model_selection import TimeSeriesSplit, GroupKFold, GridSearchCV, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from joblib import dump, load

from utils import precision_at_k, recall_at_k

# exec(open('01_setup.py').read())

X_train_f = pd.read_csv("./output/X_train_f.csv") # 2010 - 2015, w. protected attributes
X_train_fs = pd.read_csv("./output/X_train_fs.csv") # 2015, w. protected attributes
X_train_s = pd.read_csv("./output/X_train_s.csv") # 2010 - 2015, w/o protected attributes
X_train_ss = pd.read_csv("./output/X_train_ss.csv") # 2015, w/o protected attributes
y_train = pd.read_csv("./output/y_train.csv").iloc[:,0]
y_train_s = pd.read_csv("./output/y_train_s.csv").iloc[:,0]

X_test_f = pd.read_csv("./output/X_test_f.csv")
X_test_s = pd.read_csv("./output/X_test_s.csv")
y_test = pd.read_csv("./output/y_test.csv").iloc[:,0]

# 00 Setup

corrM = X_train_f.corr().abs() # Corr matrix of X
corrM = corrM.unstack()
corrMo = corrM.sort_values(kind = "quicksort")
corrMo[corrMo < 1].tail(20)

tscv = TimeSeriesSplit(5) # Create splits by year

for train_index, test_index in tscv.split(X_train_f):
    print("TRAIN:", train_index, "TEST:", test_index)
    
custom_precision25 = make_scorer(precision_at_k, needs_proba = True, k = 0.25) # Precision at top 25%
custom_precision10 = make_scorer(precision_at_k, needs_proba = True, k = 0.10) # Precision at top 10%

custom_recall25 = make_scorer(recall_at_k, needs_proba = True, k = 0.25) # Recall at top 25%
custom_recall10 = make_scorer(recall_at_k, needs_proba = True, k = 0.10) # Recall at top 10%

score = {'log_loss': 'neg_log_loss',
         'auc': 'roc_auc',
         'precision': 'precision',
         'recall': 'recall',
         'precision_at_k25': custom_precision25,
         'recall_at_k25': custom_recall25,
         'precision_at_k10': custom_precision10,
         'recall_at_k10': custom_recall10}

# 01 Logit Regression (w. protected attributes)

glm1 = LogisticRegression(penalty = 'none', solver = 'lbfgs', max_iter = 1000)
glm1.fit(X_train_f, y_train) # 2010 - 2015

glmcv1 = cross_validate(estimator = glm1, 
                       X = X_train_f,
                       y = y_train,
                       cv = tscv,
                       n_jobs = -1,
                       scoring = score)

glmcv1

coefs1 = pd.DataFrame(X_train_f.columns, columns = ['var'])
coefs1['coef'] = pd.DataFrame(glm1.coef_).transpose()

dump(glm1, 'glm1.joblib')

glm1b = LogisticRegression(penalty = 'none', solver = 'lbfgs', max_iter = 1000)
glm1b.fit(X_train_fs, y_train_s) # 2015

dump(glm1b, 'glm1b.joblib')

# 02 Logit Regression (w/o protected attributes)

glm2 = LogisticRegression(penalty = 'none', solver = 'lbfgs', max_iter = 1000)
glm2.fit(X_train_s, y_train) # 2010 - 2015

glmcv2 = cross_validate(estimator = glm1, 
                       X = X_train_s,
                       y = y_train,
                       cv = tscv,
                       n_jobs = -1,
                       scoring = score)

glmcv2

coefs2 = pd.DataFrame(X_train_s.columns, columns = ['var'])
coefs2['coef'] = pd.DataFrame(glm2.coef_).transpose()

dump(glm2, 'glm2.joblib')

glm2b = LogisticRegression(penalty = 'none', solver = 'lbfgs', max_iter = 1000)
glm2b.fit(X_train_ss, y_train_s) # 2015

dump(glm2b, 'glm2b.joblib')

# 01 Elastic Net (w. protected attributes)

grid = {'penalty': ['l1', 'l2'],
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'solver': ['liblinear']}

netm1 = LogisticRegression()

net1 = GridSearchCV(estimator = netm1, 
                   cv = tscv,
                   param_grid = grid,
                   n_jobs = -1, 
                   scoring = score,
                   refit = 'auc',
                   verbose = 2)
net1.fit(X_train_f, y_train) 

pd.concat([pd.DataFrame(net1.cv_results_['params']), 
           pd.DataFrame(net1.cv_results_['mean_test_auc'], columns = ['roc_auc'])],
           axis = 1)

dump(net1, 'net1.joblib')

net1b = LogisticRegression(**net1.best_params_)
net1b.fit(X_train_fs, y_train_s) # 2015

dump(net1b, 'net1b.joblib')

# 02 Elastic Net (w/o protected attributes)

netm2 = LogisticRegression()

net2 = GridSearchCV(estimator = netm2, 
                   cv = tscv,
                   param_grid = grid,
                   n_jobs = -1, 
                   scoring = score,
                   refit = 'auc',
                   verbose = 2)
net2.fit(X_train_s, y_train) # 2010 - 2015

pd.concat([pd.DataFrame(net2.cv_results_['params']), 
           pd.DataFrame(net2.cv_results_['mean_test_auc'], columns = ['roc_auc'])],
           axis = 1)

dump(net2, 'net2.joblib')

net2b = LogisticRegression(**net2.best_params_)
net2b.fit(X_train_ss, y_train_s) # 2015

dump(net2b, 'net2b.joblib')

# 01 Random Forest (w. protected attributes)

grid = {'n_estimators': [500, 750],
        'min_samples_leaf': [1, 5, 10],
        'max_features': ['sqrt', 'log2']}

rfm1 = RandomForestClassifier()

rf1 = GridSearchCV(estimator = rfm1, 
                  cv = tscv,
                  param_grid = grid,
                  n_jobs = -1, 
                  scoring = score,
                  refit = 'auc',
                  verbose = 2)
rf1.fit(X_train_f, y_train) # 2010 - 2015

pd.concat([pd.DataFrame(rf1.cv_results_['params']), 
           pd.DataFrame(rf1.cv_results_['mean_test_auc'], columns = ['roc_auc'])],
           axis = 1)

dump(rf1, 'rf1.joblib')

rf1b = RandomForestClassifier(**rf1.best_params_)
rf1b.fit(X_train_fs, y_train_s) # 2015

dump(rf1b, 'rf1b.joblib')

# 02 Random Forest (w/o protected attributes)

rfm2 = RandomForestClassifier()

rf2 = GridSearchCV(estimator = rfm2, 
                  cv = tscv,
                  param_grid = grid,
                  n_jobs = -1, 
                  scoring = score,
                  refit = 'auc',
                  verbose = 2)
rf2.fit(X_train_s, y_train) # 2010 - 2015

pd.concat([pd.DataFrame(rf2.cv_results_['params']), 
           pd.DataFrame(rf2.cv_results_['mean_test_auc'], columns = ['roc_auc'])],
           axis = 1)

dump(rf2, 'rf2.joblib')

rf2b = RandomForestClassifier(**rf2.best_params_)
rf2b.fit(X_train_ss, y_train_s) # 2015

dump(rf2b, 'rf2b.joblib')

# 01 Gradient Boosting (w. protected attributes)

grid = {'learning_rate': [0.01, 0.025, 0.05],
        'max_depth': [3, 5, 7],
        'max_features': ['log2', 'sqrt'],
        'subsample': [0.6, 0.8],
        'n_estimators': [250, 500, 750]
       }

gb1 = GradientBoostingClassifier()

gbm1 = GridSearchCV(estimator = gb1, 
                   cv = tscv,
                   param_grid = grid,
                   n_jobs = -1,
                   scoring = score,
                   refit = 'auc',
                   verbose = 2)
gbm1.fit(X_train_f, y_train) # 2010 - 2015

pd.concat([pd.DataFrame(gbm1.cv_results_['params']), 
           pd.DataFrame(gbm1.cv_results_['mean_test_auc'], columns = ['roc_auc'])],
           axis = 1).tail(50)

dump(gbm1, 'gbm1.joblib')

gbm1b = GradientBoostingClassifier(**gbm1.best_params_)
gbm1b.fit(X_train_fs, y_train_s) # 2015

dump(gbm1b, 'gbm1b.joblib')

# 02 Gradient Boosting (w/o protected attributes)

gb2 = GradientBoostingClassifier()

gbm2 = GridSearchCV(estimator = gb2, 
                   cv = tscv,
                   param_grid = grid,
                   n_jobs = -1,
                   scoring = score,
                   refit = 'auc',
                   verbose = 2)
gbm2.fit(X_train_s, y_train) # 2010 - 2015

pd.concat([pd.DataFrame(gbm2.cv_results_['params']), 
           pd.DataFrame(gbm2.cv_results_['mean_test_auc'], columns = ['roc_auc'])],
           axis = 1).tail(50)

dump(gbm2, 'gbm2.joblib')

gbm2b = GradientBoostingClassifier(**gbm2.best_params_)
gbm2b.fit(X_train_ss, y_train_s) # 2015

dump(gbm2b, 'gbm2b.joblib')

# Output CV results

# Logit

auc_glm1 = pd.DataFrame(glmcv1['test_auc']).transpose()
auc_glm1 = auc_glm1.rename(columns={0: "2011", 1: "2012", 2: "2013", 3: "2014", 4: "2015"})

prec10_glm1 = pd.DataFrame(glmcv1['test_precision_at_k10']).transpose()
prec10_glm1 = prec10_glm1.rename(columns={0: "2011", 1: "2012", 2: "2013", 3: "2014", 4: "2015"})

prec25_glm1 = pd.DataFrame(glmcv1['test_precision_at_k25']).transpose()
prec25_glm1 = prec25_glm1.rename(columns={0: "2011", 1: "2012", 2: "2013", 3: "2014", 4: "2015"})

rec10_glm1 = pd.DataFrame(glmcv1['test_recall_at_k10']).transpose()
rec10_glm1 = rec10_glm1.rename(columns={0: "2011", 1: "2012", 2: "2013", 3: "2014", 4: "2015"})

rec25_glm1 = pd.DataFrame(glmcv1['test_recall_at_k25']).transpose()
rec25_glm1 = rec25_glm1.rename(columns={0: "2011", 1: "2012", 2: "2013", 3: "2014", 4: "2015"})

auc_glm2 = pd.DataFrame(glmcv2['test_auc']).transpose()
auc_glm2 = auc_glm2.rename(columns={0: "2011", 1: "2012", 2: "2013", 3: "2014", 4: "2015"})

prec10_glm2 = pd.DataFrame(glmcv2['test_precision_at_k10']).transpose()
prec10_glm2 = prec10_glm2.rename(columns={0: "2011", 1: "2012", 2: "2013", 3: "2014", 4: "2015"})

prec25_glm2 = pd.DataFrame(glmcv2['test_precision_at_k25']).transpose()
prec25_glm2 = prec25_glm2.rename(columns={0: "2011", 1: "2012", 2: "2013", 3: "2014", 4: "2015"})

rec10_glm2 = pd.DataFrame(glmcv2['test_recall_at_k10']).transpose()
rec10_glm2 = rec10_glm2.rename(columns={0: "2011", 1: "2012", 2: "2013", 3: "2014", 4: "2015"})

rec25_glm2 = pd.DataFrame(glmcv2['test_recall_at_k25']).transpose()
rec25_glm2 = rec25_glm2.rename(columns={0: "2011", 1: "2012", 2: "2013", 3: "2014", 4: "2015"})

# Elastic net

auc_net1 = pd.concat([pd.DataFrame(net1.cv_results_['rank_test_auc'], columns = ['ranks']),
                      pd.DataFrame(net1.cv_results_['split0_test_auc'], columns = ['2011']), 
                      pd.DataFrame(net1.cv_results_['split1_test_auc'], columns = ['2012']),
                      pd.DataFrame(net1.cv_results_['split2_test_auc'], columns = ['2013']),
                      pd.DataFrame(net1.cv_results_['split3_test_auc'], columns = ['2014']),
                      pd.DataFrame(net1.cv_results_['split4_test_auc'], columns = ['2015'])],
                     axis = 1)

auc_best_net1 = auc_net1[auc_net1.ranks == 1].drop(columns = ['ranks'])

prec10_net1 = pd.concat([pd.DataFrame(net1.cv_results_['rank_test_auc'], columns = ['ranks']),
                         pd.DataFrame(net1.cv_results_['split0_test_precision_at_k10'], columns = ['2011']), 
                         pd.DataFrame(net1.cv_results_['split1_test_precision_at_k10'], columns = ['2012']),
                         pd.DataFrame(net1.cv_results_['split2_test_precision_at_k10'], columns = ['2013']),
                         pd.DataFrame(net1.cv_results_['split3_test_precision_at_k10'], columns = ['2014']),
                         pd.DataFrame(net1.cv_results_['split4_test_precision_at_k10'], columns = ['2015'])],
                        axis = 1)

prec10_best_net1 = prec10_net1[prec10_net1.ranks == 1].drop(columns = ['ranks'])

prec25_net1 = pd.concat([pd.DataFrame(net1.cv_results_['rank_test_auc'], columns = ['ranks']),
                         pd.DataFrame(net1.cv_results_['split0_test_precision_at_k25'], columns = ['2011']), 
                         pd.DataFrame(net1.cv_results_['split1_test_precision_at_k25'], columns = ['2012']),
                         pd.DataFrame(net1.cv_results_['split2_test_precision_at_k25'], columns = ['2013']),
                         pd.DataFrame(net1.cv_results_['split3_test_precision_at_k25'], columns = ['2014']),
                         pd.DataFrame(net1.cv_results_['split4_test_precision_at_k25'], columns = ['2015'])],
                        axis = 1)

prec25_best_net1 = prec25_net1[prec25_net1.ranks == 1].drop(columns = ['ranks'])

rec10_net1 = pd.concat([pd.DataFrame(net1.cv_results_['rank_test_auc'], columns = ['ranks']),
                        pd.DataFrame(net1.cv_results_['split0_test_recall_at_k10'], columns = ['2011']), 
                        pd.DataFrame(net1.cv_results_['split1_test_recall_at_k10'], columns = ['2012']),
                        pd.DataFrame(net1.cv_results_['split2_test_recall_at_k10'], columns = ['2013']),
                        pd.DataFrame(net1.cv_results_['split3_test_recall_at_k10'], columns = ['2014']),
                        pd.DataFrame(net1.cv_results_['split4_test_recall_at_k10'], columns = ['2015'])],
                       axis = 1)

rec10_best_net1 = rec10_net1[rec10_net1.ranks == 1].drop(columns = ['ranks'])

rec25_net1 = pd.concat([pd.DataFrame(net1.cv_results_['rank_test_auc'], columns = ['ranks']),
                        pd.DataFrame(net1.cv_results_['split0_test_recall_at_k25'], columns = ['2011']), 
                        pd.DataFrame(net1.cv_results_['split1_test_recall_at_k25'], columns = ['2012']),
                        pd.DataFrame(net1.cv_results_['split2_test_recall_at_k25'], columns = ['2013']),
                        pd.DataFrame(net1.cv_results_['split3_test_recall_at_k25'], columns = ['2014']),
                        pd.DataFrame(net1.cv_results_['split4_test_recall_at_k25'], columns = ['2015'])],
                       axis = 1)

rec25_best_net1 = rec25_net1[rec25_net1.ranks == 1].drop(columns = ['ranks'])

auc_net2 = pd.concat([pd.DataFrame(net2.cv_results_['rank_test_auc'], columns = ['ranks']),
                      pd.DataFrame(net2.cv_results_['split0_test_auc'], columns = ['2011']), 
                      pd.DataFrame(net2.cv_results_['split1_test_auc'], columns = ['2012']),
                      pd.DataFrame(net2.cv_results_['split2_test_auc'], columns = ['2013']),
                      pd.DataFrame(net2.cv_results_['split3_test_auc'], columns = ['2014']),
                      pd.DataFrame(net2.cv_results_['split4_test_auc'], columns = ['2015'])],
                     axis = 1)

auc_best_net2 = auc_net2[auc_net2.ranks == 1].drop(columns = ['ranks'])

prec10_net2 = pd.concat([pd.DataFrame(net2.cv_results_['rank_test_auc'], columns = ['ranks']),
                         pd.DataFrame(net2.cv_results_['split0_test_precision_at_k10'], columns = ['2011']), 
                         pd.DataFrame(net2.cv_results_['split1_test_precision_at_k10'], columns = ['2012']),
                         pd.DataFrame(net2.cv_results_['split2_test_precision_at_k10'], columns = ['2013']),
                         pd.DataFrame(net2.cv_results_['split3_test_precision_at_k10'], columns = ['2014']),
                         pd.DataFrame(net2.cv_results_['split4_test_precision_at_k10'], columns = ['2015'])],
                        axis = 1)

prec10_best_net2 = prec10_net2[prec10_net2.ranks == 1].drop(columns = ['ranks'])

prec25_net2 = pd.concat([pd.DataFrame(net2.cv_results_['rank_test_auc'], columns = ['ranks']),
                         pd.DataFrame(net2.cv_results_['split0_test_precision_at_k25'], columns = ['2011']), 
                         pd.DataFrame(net2.cv_results_['split1_test_precision_at_k25'], columns = ['2012']),
                         pd.DataFrame(net2.cv_results_['split2_test_precision_at_k25'], columns = ['2013']),
                         pd.DataFrame(net2.cv_results_['split3_test_precision_at_k25'], columns = ['2014']),
                         pd.DataFrame(net2.cv_results_['split4_test_precision_at_k25'], columns = ['2015'])],
                        axis = 1)

prec25_best_net2 = prec25_net2[prec25_net2.ranks == 1].drop(columns = ['ranks'])

rec10_net2 = pd.concat([pd.DataFrame(net2.cv_results_['rank_test_auc'], columns = ['ranks']),
                        pd.DataFrame(net2.cv_results_['split0_test_recall_at_k10'], columns = ['2011']), 
                        pd.DataFrame(net2.cv_results_['split1_test_recall_at_k10'], columns = ['2012']),
                        pd.DataFrame(net2.cv_results_['split2_test_recall_at_k10'], columns = ['2013']),
                        pd.DataFrame(net2.cv_results_['split3_test_recall_at_k10'], columns = ['2014']),
                        pd.DataFrame(net2.cv_results_['split4_test_recall_at_k10'], columns = ['2015'])],
                       axis = 1)

rec10_best_net2 = rec10_net2[rec10_net2.ranks == 1].drop(columns = ['ranks'])

rec25_net2 = pd.concat([pd.DataFrame(net2.cv_results_['rank_test_auc'], columns = ['ranks']),
                        pd.DataFrame(net2.cv_results_['split0_test_recall_at_k25'], columns = ['2011']), 
                        pd.DataFrame(net2.cv_results_['split1_test_recall_at_k25'], columns = ['2012']),
                        pd.DataFrame(net2.cv_results_['split2_test_recall_at_k25'], columns = ['2013']),
                        pd.DataFrame(net2.cv_results_['split3_test_recall_at_k25'], columns = ['2014']),
                        pd.DataFrame(net2.cv_results_['split4_test_recall_at_k25'], columns = ['2015'])],
                       axis = 1)

rec25_best_net2 = rec25_net2[rec25_net2.ranks == 1].drop(columns = ['ranks'])

# RF

auc_rf1 = pd.concat([pd.DataFrame(rf1.cv_results_['rank_test_auc'], columns = ['ranks']),
                     pd.DataFrame(rf1.cv_results_['split0_test_auc'], columns = ['2011']), 
                     pd.DataFrame(rf1.cv_results_['split1_test_auc'], columns = ['2012']),
                     pd.DataFrame(rf1.cv_results_['split2_test_auc'], columns = ['2013']),
                     pd.DataFrame(rf1.cv_results_['split3_test_auc'], columns = ['2014']),
                     pd.DataFrame(rf1.cv_results_['split4_test_auc'], columns = ['2015'])],
                    axis = 1)

auc_best_rf1 = auc_rf1[auc_rf1.ranks == 1].drop(columns = ['ranks'])

prec10_rf1 = pd.concat([pd.DataFrame(rf1.cv_results_['rank_test_auc'], columns = ['ranks']),
                        pd.DataFrame(rf1.cv_results_['split0_test_precision_at_k10'], columns = ['2011']), 
                        pd.DataFrame(rf1.cv_results_['split1_test_precision_at_k10'], columns = ['2012']),
                        pd.DataFrame(rf1.cv_results_['split2_test_precision_at_k10'], columns = ['2013']),
                        pd.DataFrame(rf1.cv_results_['split3_test_precision_at_k10'], columns = ['2014']),
                        pd.DataFrame(rf1.cv_results_['split4_test_precision_at_k10'], columns = ['2015'])],
                       axis = 1)

prec10_best_rf1 = prec10_rf1[prec10_rf1.ranks == 1].drop(columns = ['ranks'])

prec25_rf1 = pd.concat([pd.DataFrame(rf1.cv_results_['rank_test_auc'], columns = ['ranks']),
                        pd.DataFrame(rf1.cv_results_['split0_test_precision_at_k25'], columns = ['2011']), 
                        pd.DataFrame(rf1.cv_results_['split1_test_precision_at_k25'], columns = ['2012']),
                        pd.DataFrame(rf1.cv_results_['split2_test_precision_at_k25'], columns = ['2013']),
                        pd.DataFrame(rf1.cv_results_['split3_test_precision_at_k25'], columns = ['2014']),
                        pd.DataFrame(rf1.cv_results_['split4_test_precision_at_k25'], columns = ['2015'])],
                       axis = 1)

prec25_best_rf1 = prec25_rf1[prec25_rf1.ranks == 1].drop(columns = ['ranks'])

rec10_rf1 = pd.concat([pd.DataFrame(rf1.cv_results_['rank_test_auc'], columns = ['ranks']),
                       pd.DataFrame(rf1.cv_results_['split0_test_recall_at_k10'], columns = ['2011']), 
                       pd.DataFrame(rf1.cv_results_['split1_test_recall_at_k10'], columns = ['2012']),
                       pd.DataFrame(rf1.cv_results_['split2_test_recall_at_k10'], columns = ['2013']),
                       pd.DataFrame(rf1.cv_results_['split3_test_recall_at_k10'], columns = ['2014']),
                       pd.DataFrame(rf1.cv_results_['split4_test_recall_at_k10'], columns = ['2015'])],
                      axis = 1)

rec10_best_rf1 = rec10_rf1[rec10_rf1.ranks == 1].drop(columns = ['ranks'])

rec25_rf1 = pd.concat([pd.DataFrame(rf1.cv_results_['rank_test_auc'], columns = ['ranks']),
                       pd.DataFrame(rf1.cv_results_['split0_test_recall_at_k25'], columns = ['2011']), 
                       pd.DataFrame(rf1.cv_results_['split1_test_recall_at_k25'], columns = ['2012']),
                       pd.DataFrame(rf1.cv_results_['split2_test_recall_at_k25'], columns = ['2013']),
                       pd.DataFrame(rf1.cv_results_['split3_test_recall_at_k25'], columns = ['2014']),
                       pd.DataFrame(rf1.cv_results_['split4_test_recall_at_k25'], columns = ['2015'])],
                      axis = 1)

rec25_best_rf1 = rec25_rf1[rec25_rf1.ranks == 1].drop(columns = ['ranks'])

auc_rf2 = pd.concat([pd.DataFrame(rf2.cv_results_['rank_test_auc'], columns = ['ranks']),
                     pd.DataFrame(rf2.cv_results_['split0_test_auc'], columns = ['2011']), 
                     pd.DataFrame(rf2.cv_results_['split1_test_auc'], columns = ['2012']),
                     pd.DataFrame(rf2.cv_results_['split2_test_auc'], columns = ['2013']),
                     pd.DataFrame(rf2.cv_results_['split3_test_auc'], columns = ['2014']),
                     pd.DataFrame(rf2.cv_results_['split4_test_auc'], columns = ['2015'])],
                    axis = 1)

auc_best_rf2 = auc_rf2[auc_rf2.ranks == 1].drop(columns = ['ranks'])

prec10_rf2 = pd.concat([pd.DataFrame(rf2.cv_results_['rank_test_auc'], columns = ['ranks']),
                        pd.DataFrame(rf2.cv_results_['split0_test_precision_at_k10'], columns = ['2011']), 
                        pd.DataFrame(rf2.cv_results_['split1_test_precision_at_k10'], columns = ['2012']),
                        pd.DataFrame(rf2.cv_results_['split2_test_precision_at_k10'], columns = ['2013']),
                        pd.DataFrame(rf2.cv_results_['split3_test_precision_at_k10'], columns = ['2014']),
                        pd.DataFrame(rf2.cv_results_['split4_test_precision_at_k10'], columns = ['2015'])],
                       axis = 1)

prec10_best_rf2 = prec10_rf2[prec10_rf2.ranks == 1].drop(columns = ['ranks'])

prec25_rf2 = pd.concat([pd.DataFrame(rf2.cv_results_['rank_test_auc'], columns = ['ranks']),
                        pd.DataFrame(rf2.cv_results_['split0_test_precision_at_k25'], columns = ['2011']), 
                        pd.DataFrame(rf2.cv_results_['split1_test_precision_at_k25'], columns = ['2012']),
                        pd.DataFrame(rf2.cv_results_['split2_test_precision_at_k25'], columns = ['2013']),
                        pd.DataFrame(rf2.cv_results_['split3_test_precision_at_k25'], columns = ['2014']),
                        pd.DataFrame(rf2.cv_results_['split4_test_precision_at_k25'], columns = ['2015'])],
                       axis = 1)

prec25_best_rf2 = prec25_rf2[prec25_rf2.ranks == 1].drop(columns = ['ranks'])

rec10_rf2 = pd.concat([pd.DataFrame(rf2.cv_results_['rank_test_auc'], columns = ['ranks']),
                       pd.DataFrame(rf2.cv_results_['split0_test_recall_at_k10'], columns = ['2011']), 
                       pd.DataFrame(rf2.cv_results_['split1_test_recall_at_k10'], columns = ['2012']),
                       pd.DataFrame(rf2.cv_results_['split2_test_recall_at_k10'], columns = ['2013']),
                       pd.DataFrame(rf2.cv_results_['split3_test_recall_at_k10'], columns = ['2014']),
                       pd.DataFrame(rf2.cv_results_['split4_test_recall_at_k10'], columns = ['2015'])],
                      axis = 1)

rec10_best_rf2 = rec10_rf2[rec10_rf2.ranks == 1].drop(columns = ['ranks'])

rec25_rf2 = pd.concat([pd.DataFrame(rf2.cv_results_['rank_test_auc'], columns = ['ranks']),
                       pd.DataFrame(rf2.cv_results_['split0_test_recall_at_k25'], columns = ['2011']), 
                       pd.DataFrame(rf2.cv_results_['split1_test_recall_at_k25'], columns = ['2012']),
                       pd.DataFrame(rf2.cv_results_['split2_test_recall_at_k25'], columns = ['2013']),
                       pd.DataFrame(rf2.cv_results_['split3_test_recall_at_k25'], columns = ['2014']),
                       pd.DataFrame(rf2.cv_results_['split4_test_recall_at_k25'], columns = ['2015'])],
                      axis = 1)

rec25_best_rf2 = rec25_rf2[rec25_rf2.ranks == 1].drop(columns = ['ranks'])

# GBM

auc_gbm1 = pd.concat([pd.DataFrame(gbm1.cv_results_['rank_test_auc'], columns = ['ranks']),
                      pd.DataFrame(gbm1.cv_results_['split0_test_auc'], columns = ['2011']), 
                      pd.DataFrame(gbm1.cv_results_['split1_test_auc'], columns = ['2012']),
                      pd.DataFrame(gbm1.cv_results_['split2_test_auc'], columns = ['2013']),
                      pd.DataFrame(gbm1.cv_results_['split3_test_auc'], columns = ['2014']),
                      pd.DataFrame(gbm1.cv_results_['split4_test_auc'], columns = ['2015'])],
                     axis = 1)

auc_best_gbm1 = auc_gbm1[auc_gbm1.ranks == 1].drop(columns = ['ranks'])

prec10_gbm1 = pd.concat([pd.DataFrame(gbm1.cv_results_['rank_test_auc'], columns = ['ranks']),
                         pd.DataFrame(gbm1.cv_results_['split0_test_precision_at_k10'], columns = ['2011']), 
                         pd.DataFrame(gbm1.cv_results_['split1_test_precision_at_k10'], columns = ['2012']),
                         pd.DataFrame(gbm1.cv_results_['split2_test_precision_at_k10'], columns = ['2013']),
                         pd.DataFrame(gbm1.cv_results_['split3_test_precision_at_k10'], columns = ['2014']),
                         pd.DataFrame(gbm1.cv_results_['split4_test_precision_at_k10'], columns = ['2015'])],
                        axis = 1)

prec10_best_gbm1 = prec10_gbm1[prec10_gbm1.ranks == 1].drop(columns = ['ranks'])

prec25_gbm1 = pd.concat([pd.DataFrame(gbm1.cv_results_['rank_test_auc'], columns = ['ranks']),
                         pd.DataFrame(gbm1.cv_results_['split0_test_precision_at_k25'], columns = ['2011']), 
                         pd.DataFrame(gbm1.cv_results_['split1_test_precision_at_k25'], columns = ['2012']),
                         pd.DataFrame(gbm1.cv_results_['split2_test_precision_at_k25'], columns = ['2013']),
                         pd.DataFrame(gbm1.cv_results_['split3_test_precision_at_k25'], columns = ['2014']),
                         pd.DataFrame(gbm1.cv_results_['split4_test_precision_at_k25'], columns = ['2015'])],
                        axis = 1)

prec25_best_gbm1 = prec25_gbm1[prec25_gbm1.ranks == 1].drop(columns = ['ranks'])

rec10_gbm1 = pd.concat([pd.DataFrame(gbm1.cv_results_['rank_test_auc'], columns = ['ranks']),
                        pd.DataFrame(gbm1.cv_results_['split0_test_recall_at_k10'], columns = ['2011']), 
                        pd.DataFrame(gbm1.cv_results_['split1_test_recall_at_k10'], columns = ['2012']),
                        pd.DataFrame(gbm1.cv_results_['split2_test_recall_at_k10'], columns = ['2013']),
                        pd.DataFrame(gbm1.cv_results_['split3_test_recall_at_k10'], columns = ['2014']),
                        pd.DataFrame(gbm1.cv_results_['split4_test_recall_at_k10'], columns = ['2015'])],
                       axis = 1)

rec10_best_gbm1 = rec10_gbm1[rec10_gbm1.ranks == 1].drop(columns = ['ranks'])

rec25_gbm1 = pd.concat([pd.DataFrame(gbm1.cv_results_['rank_test_auc'], columns = ['ranks']),
                        pd.DataFrame(gbm1.cv_results_['split0_test_recall_at_k25'], columns = ['2011']), 
                        pd.DataFrame(gbm1.cv_results_['split1_test_recall_at_k25'], columns = ['2012']),
                        pd.DataFrame(gbm1.cv_results_['split2_test_recall_at_k25'], columns = ['2013']),
                        pd.DataFrame(gbm1.cv_results_['split3_test_recall_at_k25'], columns = ['2014']),
                        pd.DataFrame(gbm1.cv_results_['split4_test_recall_at_k25'], columns = ['2015'])],
                       axis = 1)

rec25_best_gbm1 = rec25_gbm1[rec25_gbm1.ranks == 1].drop(columns = ['ranks'])

auc_gbm2 = pd.concat([pd.DataFrame(gbm2.cv_results_['rank_test_auc'], columns = ['ranks']),
                      pd.DataFrame(gbm2.cv_results_['split0_test_auc'], columns = ['2011']), 
                      pd.DataFrame(gbm2.cv_results_['split1_test_auc'], columns = ['2012']),
                      pd.DataFrame(gbm2.cv_results_['split2_test_auc'], columns = ['2013']),
                      pd.DataFrame(gbm2.cv_results_['split3_test_auc'], columns = ['2014']),
                      pd.DataFrame(gbm2.cv_results_['split4_test_auc'], columns = ['2015'])],
                     axis = 1)

auc_best_gbm2 = auc_gbm2[auc_gbm2.ranks == 1].drop(columns = ['ranks'])

prec10_gbm2 = pd.concat([pd.DataFrame(gbm2.cv_results_['rank_test_auc'], columns = ['ranks']),
                         pd.DataFrame(gbm2.cv_results_['split0_test_precision_at_k10'], columns = ['2011']), 
                         pd.DataFrame(gbm2.cv_results_['split1_test_precision_at_k10'], columns = ['2012']),
                         pd.DataFrame(gbm2.cv_results_['split2_test_precision_at_k10'], columns = ['2013']),
                         pd.DataFrame(gbm2.cv_results_['split3_test_precision_at_k10'], columns = ['2014']),
                         pd.DataFrame(gbm2.cv_results_['split4_test_precision_at_k10'], columns = ['2015'])],
                        axis = 1)

prec10_best_gbm2 = prec10_gbm2[prec10_gbm2.ranks == 1].drop(columns = ['ranks'])

prec25_gbm2 = pd.concat([pd.DataFrame(gbm2.cv_results_['rank_test_auc'], columns = ['ranks']),
                         pd.DataFrame(gbm2.cv_results_['split0_test_precision_at_k25'], columns = ['2011']), 
                         pd.DataFrame(gbm2.cv_results_['split1_test_precision_at_k25'], columns = ['2012']),
                         pd.DataFrame(gbm2.cv_results_['split2_test_precision_at_k25'], columns = ['2013']),
                         pd.DataFrame(gbm2.cv_results_['split3_test_precision_at_k25'], columns = ['2014']),
                         pd.DataFrame(gbm2.cv_results_['split4_test_precision_at_k25'], columns = ['2015'])],
                        axis = 1)

prec25_best_gbm2 = prec25_gbm2[prec25_gbm2.ranks == 1].drop(columns = ['ranks'])

rec10_gbm2 = pd.concat([pd.DataFrame(gbm2.cv_results_['rank_test_auc'], columns = ['ranks']),
                        pd.DataFrame(gbm2.cv_results_['split0_test_recall_at_k10'], columns = ['2011']), 
                        pd.DataFrame(gbm2.cv_results_['split1_test_recall_at_k10'], columns = ['2012']),
                        pd.DataFrame(gbm2.cv_results_['split2_test_recall_at_k10'], columns = ['2013']),
                        pd.DataFrame(gbm2.cv_results_['split3_test_recall_at_k10'], columns = ['2014']),
                        pd.DataFrame(gbm2.cv_results_['split4_test_recall_at_k10'], columns = ['2015'])],
                       axis = 1)

rec10_best_gbm2 = rec10_gbm2[rec10_gbm2.ranks == 1].drop(columns = ['ranks'])

rec25_gbm2 = pd.concat([pd.DataFrame(gbm2.cv_results_['rank_test_auc'], columns = ['ranks']),
                        pd.DataFrame(gbm2.cv_results_['split0_test_recall_at_k25'], columns = ['2011']), 
                        pd.DataFrame(gbm2.cv_results_['split1_test_recall_at_k25'], columns = ['2012']),
                        pd.DataFrame(gbm2.cv_results_['split2_test_recall_at_k25'], columns = ['2013']),
                        pd.DataFrame(gbm2.cv_results_['split3_test_recall_at_k25'], columns = ['2014']),
                        pd.DataFrame(gbm2.cv_results_['split4_test_recall_at_k25'], columns = ['2015'])],
                       axis = 1)

rec25_best_gbm2 = rec25_gbm2[rec25_gbm2.ranks == 1].drop(columns = ['ranks'])

# Combine

aucs1 = pd.concat([auc_glm1,
                   auc_best_net1,
                   auc_best_rf1,
                   auc_best_gbm1],
                  axis = 0)

aucs1.to_latex('./output/train_cv_auc1.tex', index = False, float_format = "%.3f")

precs1_10 = pd.concat([prec10_glm1,
                       prec10_best_net1,
                       prec10_best_rf1,
                       prec10_best_gbm1],
                      axis = 0)

precs1_10.to_latex('./output/train_cv_prec1_10.tex', index = False, float_format = "%.3f")

precs1_25 = pd.concat([prec25_glm1,
                       prec25_best_net1,
                       prec25_best_rf1,
                       prec25_best_gbm1],
                      axis = 0)

precs1_25.to_latex('./output/train_cv_prec1_25.tex', index = False, float_format = "%.3f")

recs1_10 = pd.concat([rec10_glm1,
                      rec10_best_net1,
                      rec10_best_rf1,
                      rec10_best_gbm1],
                     axis = 0)

recs1_10.to_latex('./output/train_cv_rec1_10.tex', index = False, float_format = "%.3f")

recs1_25 = pd.concat([rec25_glm1,
                      rec25_best_net1,
                      rec25_best_rf1,
                      rec25_best_gbm1],
                     axis = 0)

recs1_25.to_latex('./output/train_cv_rec1_25.tex', index = False, float_format = "%.3f")

aucs2 = pd.concat([auc_glm2,
                   auc_best_net2,
                   auc_best_rf2,
                   auc_best_gbm2],
                  axis = 0)

aucs2.to_latex('./output/train_cv_auc2.tex', index = False, float_format = "%.3f")

precs2_10 = pd.concat([prec10_glm2,
                       prec10_best_net2,
                       prec10_best_rf2,
                       prec10_best_gbm2],
                     axis = 0)

precs2_10.to_latex('./output/train_cv_prec2_10.tex', index = False, float_format = "%.3f")

precs2_25 = pd.concat([prec25_glm2,
                       prec25_best_net2,
                       prec25_best_rf2,
                       prec25_best_gbm2],
                     axis = 0)

precs2_25.to_latex('./output/train_cv_prec2_25.tex', index = False, float_format = "%.3f")

recs2_10 = pd.concat([rec10_glm2,
                      rec10_best_net2,
                      rec10_best_rf2,
                      rec10_best_gbm2],
                     axis = 0)

recs2_10.to_latex('./output/train_cv_rec2_10.tex', index = False, float_format = "%.3f")

recs2_25 = pd.concat([rec25_glm2,
                      rec25_best_net2,
                      rec25_best_rf2,
                      rec25_best_gbm2],
                     axis = 0)

recs2_25.to_latex('./output/train_cv_rec2_25.tex', index = False, float_format = "%.3f")

# Predict

k75 = 0.75 # Top 75% 
k25 = 0.25 # Top 25% 
k10 = 0.1 # Top 10% 

# Logit

glm1_p = glm1.predict_proba(X_test_f)[:,1] # glm1

threshold75 = np.sort(glm1_p)[::-1][int(k75*len(glm1_p))]
threshold25 = np.sort(glm1_p)[::-1][int(k25*len(glm1_p))]
threshold10 = np.sort(glm1_p)[::-1][int(k10*len(glm1_p))]

glm1_c1 = glm1_p.copy()
glm1_c1[glm1_c1 < threshold10] = 0
glm1_c1[glm1_c1 >= threshold10] = 1

glm1_c2 = glm1_p.copy()
glm1_c2[glm1_c2 < threshold25] = 0
glm1_c2[glm1_c2 >= threshold25] = 1

glm1_c3 = glm1_p.copy()
glm1_c3[(glm1_c3 <= threshold75) | (glm1_c3 >= threshold25)] = 0
glm1_c3[(glm1_c3 > threshold75) & (glm1_c3 < threshold25)] = 1

glm1b_p = glm1b.predict_proba(X_test_f)[:,1] # glm1b

threshold75 = np.sort(glm1b_p)[::-1][int(k75*len(glm1b_p))]
threshold25 = np.sort(glm1b_p)[::-1][int(k25*len(glm1b_p))]
threshold10 = np.sort(glm1b_p)[::-1][int(k10*len(glm1b_p))]

glm1b_c1 = glm1b_p.copy()
glm1b_c1[glm1b_c1 < threshold10] = 0
glm1b_c1[glm1b_c1 >= threshold10] = 1

glm1b_c2 = glm1b_p.copy()
glm1b_c2[glm1b_c2 < threshold25] = 0
glm1b_c2[glm1b_c2 >= threshold25] = 1

glm1b_c3 = glm1b_p.copy()
glm1b_c3[(glm1b_c3 <= threshold75) | (glm1b_c3 >= threshold25)] = 0
glm1b_c3[(glm1b_c3 > threshold75) & (glm1b_c3 < threshold25)] = 1

glm2_p = glm2.predict_proba(X_test_s)[:,1] # glm2

threshold75 = np.sort(glm2_p)[::-1][int(k75*len(glm2_p))]
threshold25 = np.sort(glm2_p)[::-1][int(k25*len(glm2_p))]
threshold10 = np.sort(glm2_p)[::-1][int(k10*len(glm2_p))]

glm2_c1 = glm2_p.copy()
glm2_c1[glm2_c1 < threshold10] = 0
glm2_c1[glm2_c1 >= threshold10] = 1

glm2_c2 = glm2_p.copy()
glm2_c2[glm2_c2 < threshold25] = 0
glm2_c2[glm2_c2 >= threshold25] = 1

glm2_c3 = glm2_p.copy()
glm2_c3[(glm2_c3 <= threshold75) | (glm2_c3 >= threshold25)] = 0
glm2_c3[(glm2_c3 > threshold75) & (glm2_c3 < threshold25)] = 1

glm2b_p = glm2b.predict_proba(X_test_s)[:,1] # glm2b

threshold75 = np.sort(glm2b_p)[::-1][int(k75*len(glm2b_p))]
threshold25 = np.sort(glm2b_p)[::-1][int(k25*len(glm2b_p))]
threshold10 = np.sort(glm2b_p)[::-1][int(k10*len(glm2b_p))]

glm2b_c1 = glm2b_p.copy()
glm2b_c1[glm2b_c1 < threshold10] = 0
glm2b_c1[glm2b_c1 >= threshold10] = 1

glm2b_c2 = glm2b_p.copy()
glm2b_c2[glm2b_c2 < threshold25] = 0
glm2b_c2[glm2b_c2 >= threshold25] = 1

glm2b_c3 = glm2b_p.copy()
glm2b_c3[(glm2b_c3 <= threshold75) | (glm2b_c3 >= threshold25)] = 0
glm2b_c3[(glm2b_c3 > threshold75) & (glm2b_c3 < threshold25)] = 1

# Elastic net

net1_p = net1.predict_proba(X_test_f)[:,1] # net1

threshold75 = np.sort(net1_p)[::-1][int(k75*len(net1_p))]
threshold25 = np.sort(net1_p)[::-1][int(k25*len(net1_p))]
threshold10 = np.sort(net1_p)[::-1][int(k10*len(net1_p))]

net1_c1 = net1_p.copy()
net1_c1[net1_c1 < threshold10] = 0
net1_c1[net1_c1 >= threshold10] = 1

net1_c2 = net1_p.copy()
net1_c2[net1_c2 < threshold25] = 0
net1_c2[net1_c2 >= threshold25] = 1

net1_c3 = net1_p.copy()
net1_c3[(net1_c3 <= threshold75) | (net1_c3 >= threshold25)] = 0
net1_c3[(net1_c3 > threshold75) & (net1_c3 < threshold25)] = 1

net1b_p = net1b.predict_proba(X_test_f)[:,1] # net1b

threshold75 = np.sort(net1b_p)[::-1][int(k75*len(net1b_p))]
threshold25 = np.sort(net1b_p)[::-1][int(k25*len(net1b_p))]
threshold10 = np.sort(net1b_p)[::-1][int(k10*len(net1b_p))]

net1b_c1 = net1b_p.copy()
net1b_c1[net1b_c1 < threshold10] = 0
net1b_c1[net1b_c1 >= threshold10] = 1

net1b_c2 = net1b_p.copy()
net1b_c2[net1b_c2 < threshold25] = 0
net1b_c2[net1b_c2 >= threshold25] = 1

net1b_c3 = net1b_p.copy()
net1b_c3[(net1b_c3 <= threshold75) | (net1b_c3 >= threshold25)] = 0
net1b_c3[(net1b_c3 > threshold75) & (net1b_c3 < threshold25)] = 1

net2_p = net2.predict_proba(X_test_s)[:,1] # net2

threshold75 = np.sort(net2_p)[::-1][int(k75*len(net2_p))]
threshold25 = np.sort(net2_p)[::-1][int(k25*len(net2_p))]
threshold10 = np.sort(net2_p)[::-1][int(k10*len(net2_p))]

net2_c1 = net2_p.copy()
net2_c1[net2_c1 < threshold10] = 0
net2_c1[net2_c1 >= threshold10] = 1

net2_c2 = net2_p.copy()
net2_c2[net2_c2 < threshold25] = 0
net2_c2[net2_c2 >= threshold25] = 1

net2_c3 = net2_p.copy()
net2_c3[(net2_c3 <= threshold75) | (net2_c3 >= threshold25)] = 0
net2_c3[(net2_c3 > threshold75) & (net2_c3 < threshold25)] = 1

net2b_p = net2b.predict_proba(X_test_s)[:,1] # net2b

threshold75 = np.sort(net2b_p)[::-1][int(k75*len(net2b_p))]
threshold25 = np.sort(net2b_p)[::-1][int(k25*len(net2b_p))]
threshold10 = np.sort(net2b_p)[::-1][int(k10*len(net2b_p))]

net2b_c1 = net2b_p.copy()
net2b_c1[net2b_c1 < threshold10] = 0
net2b_c1[net2b_c1 >= threshold10] = 1

net2b_c2 = net2b_p.copy()
net2b_c2[net2b_c2 < threshold25] = 0
net2b_c2[net2b_c2 >= threshold25] = 1

net2b_c3 = net2b_p.copy()
net2b_c3[(net2b_c3 <= threshold75) | (net2b_c3 >= threshold25)] = 0
net2b_c3[(net2b_c3 > threshold75) & (net2b_c3 < threshold25)] = 1

# RF

rf1_p = rf1.predict_proba(X_test_f)[:,1] # rf1

threshold75 = np.sort(rf1_p)[::-1][int(k75*len(rf1_p))]
threshold25 = np.sort(rf1_p)[::-1][int(k25*len(rf1_p))]
threshold10 = np.sort(rf1_p)[::-1][int(k10*len(rf1_p))]

rf1_c1 = rf1_p.copy()
rf1_c1[rf1_c1 < threshold10] = 0
rf1_c1[rf1_c1 >= threshold10] = 1

rf1_c2 = rf1_p.copy()
rf1_c2[rf1_c2 < threshold25] = 0
rf1_c2[rf1_c2 >= threshold25] = 1

rf1_c3 = rf1_p.copy()
rf1_c3[(rf1_c3 <= threshold75) | (rf1_c3 >= threshold25)] = 0
rf1_c3[(rf1_c3 > threshold75) & (rf1_c3 < threshold25)] = 1

rf1b_p = rf1b.predict_proba(X_test_f)[:,1] # rf1b

threshold75 = np.sort(rf1b_p)[::-1][int(k75*len(rf1b_p))]
threshold25 = np.sort(rf1b_p)[::-1][int(k25*len(rf1b_p))]
threshold10 = np.sort(rf1b_p)[::-1][int(k10*len(rf1b_p))]

rf1b_c1 = rf1b_p.copy()
rf1b_c1[rf1b_c1 < threshold10] = 0
rf1b_c1[rf1b_c1 >= threshold10] = 1

rf1b_c2 = rf1b_p.copy()
rf1b_c2[rf1b_c2 < threshold25] = 0
rf1b_c2[rf1b_c2 >= threshold25] = 1

rf1b_c3 = rf1b_p.copy()
rf1b_c3[(rf1b_c3 <= threshold75) | (rf1b_c3 >= threshold25)] = 0
rf1b_c3[(rf1b_c3 > threshold75) & (rf1b_c3 < threshold25)] = 1

rf2_p = rf2.predict_proba(X_test_s)[:,1] # rf2

threshold75 = np.sort(rf2_p)[::-1][int(k75*len(rf2_p))]
threshold25 = np.sort(rf2_p)[::-1][int(k25*len(rf2_p))]
threshold10 = np.sort(rf2_p)[::-1][int(k10*len(rf2_p))]

rf2_c1 = rf2_p.copy()
rf2_c1[rf2_c1 < threshold10] = 0
rf2_c1[rf2_c1 >= threshold10] = 1

rf2_c2 = rf2_p.copy()
rf2_c2[rf2_c2 < threshold25] = 0
rf2_c2[rf2_c2 >= threshold25] = 1

rf2_c3 = rf2_p.copy()
rf2_c3[(rf2_c3 <= threshold75) | (rf2_c3 >= threshold25)] = 0
rf2_c3[(rf2_c3 > threshold75) & (rf2_c3 < threshold25)] = 1

rf2b_p = rf2b.predict_proba(X_test_s)[:,1] # rf2b

threshold75 = np.sort(rf2b_p)[::-1][int(k75*len(rf2b_p))]
threshold25 = np.sort(rf2b_p)[::-1][int(k25*len(rf2b_p))]
threshold10 = np.sort(rf2b_p)[::-1][int(k10*len(rf2b_p))]

rf2b_c1 = rf2b_p.copy()
rf2b_c1[rf2b_c1 < threshold10] = 0
rf2b_c1[rf2b_c1 >= threshold10] = 1

rf2b_c2 = rf2b_p.copy()
rf2b_c2[rf2b_c2 < threshold25] = 0
rf2b_c2[rf2b_c2 >= threshold25] = 1

rf2b_c3 = rf2b_p.copy()
rf2b_c3[(rf2b_c3 <= threshold75) | (rf2b_c3 >= threshold25)] = 0
rf2b_c3[(rf2b_c3 > threshold75) & (rf2b_c3 < threshold25)] = 1

# GBM

gbm1_p = gbm1.predict_proba(X_test_f)[:,1] # gbm1

threshold75 = np.sort(gbm1_p)[::-1][int(k75*len(gbm1_p))]
threshold25 = np.sort(gbm1_p)[::-1][int(k25*len(gbm1_p))]
threshold10 = np.sort(gbm1_p)[::-1][int(k10*len(gbm1_p))]

gbm1_c1 = gbm1_p.copy()
gbm1_c1[gbm1_c1 < threshold10] = 0
gbm1_c1[gbm1_c1 >= threshold10] = 1

gbm1_c2 = gbm1_p.copy()
gbm1_c2[gbm1_c2 < threshold25] = 0
gbm1_c2[gbm1_c2 >= threshold25] = 1

gbm1_c3 = gbm1_p.copy()
gbm1_c3[(gbm1_c3 <= threshold75) | (gbm1_c3 >= threshold25)] = 0
gbm1_c3[(gbm1_c3 > threshold75) & (gbm1_c3 < threshold25)] = 1

gbm1b_p = gbm1b.predict_proba(X_test_f)[:,1] # gbm1b

threshold75 = np.sort(gbm1b_p)[::-1][int(k75*len(gbm1b_p))]
threshold25 = np.sort(gbm1b_p)[::-1][int(k25*len(gbm1b_p))]
threshold10 = np.sort(gbm1b_p)[::-1][int(k10*len(gbm1b_p))]

gbm1b_c1 = gbm1b_p.copy()
gbm1b_c1[gbm1b_c1 < threshold10] = 0
gbm1b_c1[gbm1b_c1 >= threshold10] = 1

gbm1b_c2 = gbm1b_p.copy()
gbm1b_c2[gbm1b_c2 < threshold25] = 0
gbm1b_c2[gbm1b_c2 >= threshold25] = 1

gbm1b_c3 = gbm1b_p.copy()
gbm1b_c3[(gbm1b_c3 <= threshold75) | (gbm1b_c3 >= threshold25)] = 0
gbm1b_c3[(gbm1b_c3 > threshold75) & (gbm1b_c3 < threshold25)] = 1

gbm2_p = gbm2.predict_proba(X_test_s)[:,1] # gbm2

threshold75 = np.sort(gbm2_p)[::-1][int(k75*len(gbm2_p))]
threshold25 = np.sort(gbm2_p)[::-1][int(k25*len(gbm2_p))]
threshold10 = np.sort(gbm2_p)[::-1][int(k10*len(gbm2_p))]

gbm2_c1 = gbm2_p.copy()
gbm2_c1[gbm2_c1 < threshold10] = 0
gbm2_c1[gbm2_c1 >= threshold10] = 1

gbm2_c2 = gbm2_p.copy()
gbm2_c2[gbm2_c2 < threshold25] = 0
gbm2_c2[gbm2_c2 >= threshold25] = 1

gbm2_c3 = gbm2_p.copy()
gbm2_c3[(gbm2_c3 <= threshold75) | (gbm2_c3 >= threshold25)] = 0
gbm2_c3[(gbm2_c3 > threshold75) & (gbm2_c3 < threshold25)] = 1

gbm2b_p = gbm2b.predict_proba(X_test_s)[:,1] # gbm2b

threshold75 = np.sort(gbm2b_p)[::-1][int(k75*len(gbm2b_p))]
threshold25 = np.sort(gbm2b_p)[::-1][int(k25*len(gbm2b_p))]
threshold10 = np.sort(gbm2b_p)[::-1][int(k10*len(gbm2b_p))]

gbm2b_c1 = gbm2b_p.copy()
gbm2b_c1[gbm2b_c1 < threshold10] = 0
gbm2b_c1[gbm2b_c1 >= threshold10] = 1

gbm2b_c2 = gbm2b_p.copy()
gbm2b_c2[gbm2b_c2 < threshold25] = 0
gbm2b_c2[gbm2b_c2 >= threshold25] = 1

gbm2b_c3 = gbm2b_p.copy()
gbm2b_c3[(gbm2b_c3 <= threshold75) | (gbm2b_c3 >= threshold25)] = 0
gbm2b_c3[(gbm2b_c3 > threshold75) & (gbm2b_c3 < threshold25)] = 1

# Combine and save

preds_test = pd.concat([pd.DataFrame(np.array(y_test), columns = ['y_test']),
                         pd.DataFrame(glm1_p, columns = ['glm1_p']),
                         pd.DataFrame(glm1_c1, columns = ['glm1_c1']),
                         pd.DataFrame(glm1_c2, columns = ['glm1_c2']),
                         pd.DataFrame(glm1_c3, columns = ['glm1_c3']),
                         pd.DataFrame(glm1b_p, columns = ['glm1b_p']),
                         pd.DataFrame(glm1b_c1, columns = ['glm1b_c1']),
                         pd.DataFrame(glm1b_c2, columns = ['glm1b_c2']),
                         pd.DataFrame(glm1b_c3, columns = ['glm1b_c3']),
                         pd.DataFrame(glm2_p, columns = ['glm2_p']),
                         pd.DataFrame(glm2_c1, columns = ['glm2_c1']),
                         pd.DataFrame(glm2_c2, columns = ['glm2_c2']),
                         pd.DataFrame(glm2_c3, columns = ['glm2_c3']),
                         pd.DataFrame(glm2b_p, columns = ['glm2b_p']),
                         pd.DataFrame(glm2b_c1, columns = ['glm2b_c1']),
                         pd.DataFrame(glm2b_c2, columns = ['glm2b_c2']),
                         pd.DataFrame(glm2b_c3, columns = ['glm2b_c3']),
                         pd.DataFrame(net1_p, columns = ['net1_p']),
                         pd.DataFrame(net1_c1, columns = ['net1_c1']),
                         pd.DataFrame(net1_c2, columns = ['net1_c2']),
                         pd.DataFrame(net1_c3, columns = ['net1_c3']),
                         pd.DataFrame(net1b_p, columns = ['net1b_p']),
                         pd.DataFrame(net1b_c1, columns = ['net1b_c1']),
                         pd.DataFrame(net1b_c2, columns = ['net1b_c2']),
                         pd.DataFrame(net1b_c3, columns = ['net1b_c3']),
                         pd.DataFrame(net2_p, columns = ['net2_p']),
                         pd.DataFrame(net2_c1, columns = ['net2_c1']),
                         pd.DataFrame(net2_c2, columns = ['net2_c2']),
                         pd.DataFrame(net2_c3, columns = ['net2_c3']),
                         pd.DataFrame(net2b_p, columns = ['net2b_p']),
                         pd.DataFrame(net2b_c1, columns = ['net2b_c1']),
                         pd.DataFrame(net2b_c2, columns = ['net2b_c2']),
                         pd.DataFrame(net2b_c3, columns = ['net2b_c3']),
                         pd.DataFrame(rf1_p, columns = ['rf1_p']),
                         pd.DataFrame(rf1_c1, columns = ['rf1_c1']),
                         pd.DataFrame(rf1_c2, columns = ['rf1_c2']),
                         pd.DataFrame(rf1_c3, columns = ['rf1_c3']),
                         pd.DataFrame(rf1b_p, columns = ['rf1b_p']),
                         pd.DataFrame(rf1b_c1, columns = ['rf1b_c1']),
                         pd.DataFrame(rf1b_c2, columns = ['rf1b_c2']),
                         pd.DataFrame(rf1b_c3, columns = ['rf1b_c3']),
                         pd.DataFrame(rf2_p, columns = ['rf2_p']),
                         pd.DataFrame(rf2_c1, columns = ['rf2_c1']),
                         pd.DataFrame(rf2_c2, columns = ['rf2_c2']),
                         pd.DataFrame(rf2_c3, columns = ['rf2_c3']),
                         pd.DataFrame(rf2b_p, columns = ['rf2b_p']),
                         pd.DataFrame(rf2b_c1, columns = ['rf2b_c1']),
                         pd.DataFrame(rf2b_c2, columns = ['rf2b_c2']),
                         pd.DataFrame(rf2b_c3, columns = ['rf2b_c3']),
                         pd.DataFrame(gbm1_p, columns = ['gbm1_p']),
                         pd.DataFrame(gbm1_c1, columns = ['gbm1_c1']),
                         pd.DataFrame(gbm1_c2, columns = ['gbm1_c2']),
                         pd.DataFrame(gbm1_c3, columns = ['gbm1_c3']),
                         pd.DataFrame(gbm1b_p, columns = ['gbm1b_p']),
                         pd.DataFrame(gbm1b_c1, columns = ['gbm1b_c1']),
                         pd.DataFrame(gbm1b_c2, columns = ['gbm1b_c2']),
                         pd.DataFrame(gbm1b_c3, columns = ['gbm1b_c3']),
                         pd.DataFrame(gbm2_p, columns = ['gbm2_p']),
                         pd.DataFrame(gbm2_c1, columns = ['gbm2_c1']),
                         pd.DataFrame(gbm2_c2, columns = ['gbm2_c2']),
                         pd.DataFrame(gbm2_c3, columns = ['gbm2_c3']),
                         pd.DataFrame(gbm2b_p, columns = ['gbm2b_p']),
                         pd.DataFrame(gbm2b_c1, columns = ['gbm2b_c1']),
                         pd.DataFrame(gbm2b_c2, columns = ['gbm2b_c2']),
                         pd.DataFrame(gbm2b_c3, columns = ['gbm2b_c3'])],
                    axis = 1)

preds_test.to_csv('./output/preds_test.csv', index = False)

