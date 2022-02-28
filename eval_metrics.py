import json
import os
import numpy as np
from sklearn import metrics
from scipy import stats
import matplotlib.pyplot as plt


def calculate_stats(target, output):
    AP_list = []
    AUC_list = []
    for k in range(target.shape[0]):
        avg_precision = metrics.average_precision_score(target[k, :], output[k, :])
        auc = metrics.roc_auc_score(target[k, :], output[k, :])
        AP_list.append(avg_precision)
        AUC_list.append(auc)
    # d_prime
    standard_normal = stats.norm()
    d_prime = standard_normal.ppf(np.mean([AUC_list])) * np.sqrt(2.0)

    return AP_list, AUC_list, d_prime


def calculate_overall_lwlrap(truth, scores):
    """Calculate the overall lwlrap using sklearn.metrics.lrap()"""
    overall_lwlrap = metrics.label_ranking_average_precision_score(truth, scores, sample_weight=np.sum(truth > 0, axis=1))
    return overall_lwlrap


def barplot_APs(par, APs, start, end):
    labels_map_path = os.path.join(par['meta_root'], par['labels_map'])
    with open(labels_map_path, 'r') as fd:
        lbls_string = json.load(fd)
    lbls_list = [key for key, _ in lbls_string.items()]
    x = [i for i in range(end - start)]
    APs = APs[start:end]
    plt.subplots(figsize=(18, 6))
    plt.bar(x, APs, color='y')
    plt.ylim(0, 1)
    plt.yticks([.25, .5, .75])
    plt.xticks(x, lbls_list[start:end], rotation=45, ha='right')
    plt.grid(axis='y', linestyle='dotted')
    for i in range(end - start):
        plt.annotate(str(APs[i])[0:4], xy=(x[i], APs[i]), ha='center', va='bottom')
    plt.tight_layout()
    plt.show()
