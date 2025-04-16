import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-darkgrid')

from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, make_scorer
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric


def precision_at_k(y_true, y_score, k):
    threshold = np.sort(y_score)[::-1][int(k*len(y_score))]
    y_pred = np.asarray([1 if i >= threshold else 0 for i in y_score])
    return precision_score(y_true, y_pred)

def recall_at_k(y_true, y_score, k):
    threshold = np.sort(y_score)[::-1][int(k*len(y_score))]
    y_pred = np.asarray([1 if i >= threshold else 0 for i in y_score])
    return recall_score(y_true, y_pred)

def aif_test(dataset, scores, thresh_arr, unpriv_group, priv_group):
    
    metric_arrs = defaultdict(list)
    for thresh in thresh_arr:
        y_val_pred = (scores > thresh).astype(np.float64)

        dataset_pred = dataset.copy()
        dataset_pred.labels = y_val_pred
        metric = ClassificationMetric(
                dataset, dataset_pred,
                unprivileged_groups = unpriv_group,
                privileged_groups = priv_group)

        metric_arrs['bal_acc'].append((metric.true_positive_rate()
                                     + metric.true_negative_rate()) / 2)
        metric_arrs['f1'].append(2*(metric.precision() * metric.recall()) / 
                                   (metric.precision() + metric.recall()))
        metric_arrs['acc'].append(metric.accuracy())
        metric_arrs['prec'].append(metric.precision())
        metric_arrs['rec'].append(metric.recall())
        metric_arrs['err_rate'].append(metric.error_rate())
        metric_arrs['err_rate_diff'].append(metric.error_rate_difference())
        metric_arrs['disp_imp'].append(metric.disparate_impact())
        metric_arrs['stat_par_diff'].append(metric.statistical_parity_difference())
        metric_arrs['eq_opp_diff'].append(metric.equal_opportunity_difference())
        metric_arrs['avg_odds_diff'].append(metric.average_odds_difference())
    
    return metric_arrs

def aif_plot(x, x_name, y_left, y_left_name, y_right, y_right_name, cutoff1, cutoff2, ax1min = 0, ax1max = 1, ax2min = 0, ax2max = 1):
    fig, ax1 = plt.subplots(figsize=(10,7))
    ax1.plot(x, y_left, color='steelblue')
    ax1.set_xlabel(x_name, fontsize=18)
    ax1.set_ylabel(y_left_name, color='steelblue', fontsize=18)
    ax1.xaxis.set_tick_params(labelsize=16)
    ax1.yaxis.set_tick_params(labelsize=16)
    ax1.set_ylim(ax1min, ax1max)

    ax2 = ax1.twinx()
    ax2.plot(x, y_right, color='r')
    ax2.set_ylabel(y_right_name, color='r', fontsize=18)
    ax2.set_ylim(ax2min, ax2max)

    ax2.axvline(cutoff1, color='k', linestyle='dashed')
    ax2.axvline(cutoff2, color='k', linestyle='dotted')
    ax2.text(cutoff1 - 0.015, ax1max + 0.015, 'P1b', fontsize=16)
    ax2.text(cutoff2 - 0.015, ax1max + 0.015, 'P1a', fontsize=16)

    ax2.yaxis.set_tick_params(labelsize=16)
    ax2.grid(True)

def aif_plot2(x, x_name, y_left, y_left2, y_left_name, y_right, y_right2, y_right_name, cutoff1, cutoff12, cutoff2, cutoff22, ax1min = 0, ax1max = 1, ax2min = 0, ax2max = 1):
    fig, ax1 = plt.subplots(figsize=(10,7))
    ax1.plot(x, y_left, color='steelblue')
    ax1.plot(x, y_left2, color='steelblue', linestyle='dashdot')
    ax1.set_xlabel(x_name, fontsize=18)
    ax1.set_ylabel(y_left_name, color='steelblue', fontsize=18)
    ax1.xaxis.set_tick_params(labelsize=16)
    ax1.yaxis.set_tick_params(labelsize=16)
    ax1.set_ylim(ax1min, ax1max)

    ax1.legend(handles = legend_elements, fontsize=16)

    ax2 = ax1.twinx()
    ax2.plot(x, y_right, color='r')
    ax2.plot(x, y_right2, color='r', linestyle='dashdot')
    ax2.set_ylabel(y_right_name, color='r', fontsize=18)
    ax2.set_ylim(ax2min, ax2max)

    ax2.axvline(cutoff1, color='k', linestyle='dashed')
    ax2.axvline(cutoff2, color='k', linestyle='dotted')
    ax2.text(cutoff1, ax1max + 0.015, 'P1b-l', fontsize=16)
    ax2.text(cutoff2, ax1max + 0.015, 'P1a-l', fontsize=16)

    ax2.axvline(cutoff12, color='gray', linestyle='dashed')
    ax2.axvline(cutoff22, color='gray', linestyle='dotted')
    ax2.text(cutoff12 - 0.055, ax1max + 0.015, 'P1b-s', fontsize=16, color='gray') # cutoff12 - 0.025
    ax2.text(cutoff22 - 0.025, ax1max + 0.015, 'P1a-s', fontsize=16, color='gray')

    ax2.yaxis.set_tick_params(labelsize=16)
    ax2.grid(True)



