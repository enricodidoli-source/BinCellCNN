'''
from classification import robust_prediction_labelling, seed_level_robust_prediction_labelling
from classification import find_robust_threshold, compute_metrics, compute_mean_std_metrics
'''

from utils import flatten
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score 
from sklearn.metrics import matthews_corrcoef

def robust_prediction_labelling(trials_pred_lists, threshold, pred = False):
    """
    Assigns binary labels to each sample based on averaged prediction probabilities across trials and samples.
    """
    patient_timepoints_labels = []
    patient_timepoints_preds = []

    for patient in trials_pred_lists:
        file_labels = []
        file_preds = []
        for sample in patient:
            positive_probs_mean = np.mean(sample , axis=0) #column mean (seeds)
            mean_timepoint_probs = np.mean(positive_probs_mean) #row mean (subsets) to get the positive scores
            file_preds.append(mean_timepoint_probs)

            if mean_timepoint_probs >= threshold*0.01:
                file_labels.append(1)
            else:
                file_labels.append(0)
        patient_timepoints_labels.append(file_labels)
        patient_timepoints_preds.append(file_preds)
      
    if pred:
        return patient_timepoints_labels, patient_timepoints_preds

    return patient_timepoints_labels



def seed_level_robust_prediction_labelling(trials_pred_lists, threshold, pred = False):
    """
    Assigns binary labels to each sample at seed level, based on mean prediction probability
    labelling with a threshold.
    """
    patient_timepoints_labels = []
    patient_timepoints_preds = []
    for patient in trials_pred_lists:

        file_labels = []
        file_preds = []
        for sample in patient:

            seed_labels = []
            seed_preds = []
            for seed in sample:
                positive_probs_mean = np.mean(seed)
                seed_preds.append(positive_probs_mean)
                    
                if positive_probs_mean >= threshold*0.01:
                    seed_labels.append(1)
                else:
                    seed_labels.append(0)

            file_labels.append(seed_labels)
            file_preds.append(seed_preds)
          
        patient_timepoints_labels.append(file_labels)
        patient_timepoints_preds.append(file_preds)
    if pred:
        return patient_timepoints_labels, patient_timepoints_preds

    return patient_timepoints_labels




def find_robust_threshold(mean_probs_per_patient, metric = 'f1', closest = False):
    """
    Finds the optimal classification threshold (1-100) by maximizing a chosen metric.
    If multiple thresholds tie, picks the median or the one closest to 50 if closest=True.
    """

    if metric == 'f1':
        metric_score = f1_score
    elif metric == 'recall':
        metric_score = recall_score
    elif metric == 'roc':
        metric_score = roc_auc_score
    elif metric == 'accuracy':
        metric_score = accuracy_score
    elif metric == 'precision':
        metric_score = precision_score

    # Concatenate mean probs and labels into two nsubset x timepoints lists
    probs = []
    resampled_ys = []
    for patient_probs_tuple in mean_probs_per_patient:

        for timep, timep_res_y in patient_probs_tuple:

            # get mean columns predicted probabilities
            probs += list(timep)

            # get the resampled ys
            resampled_ys += list(timep_res_y)

    best_f1 = -1
    best_thr = -1
    tot_per_tr_f1_scores = []
    threshold_predictions = []

    for threshold in list(range(1,101)):
        y_pred = (np.array(probs) >= threshold*0.01).astype(int) #checks column by column if the element is > than the threshold and converts it in 1 or 0

        # compute f1 score on the concatenated timepoints results
        if metric in ['f1', 'recall', 'precision']:
            total_f1_score = metric_score(resampled_ys, y_pred, pos_label = 1, zero_division=1)
        else:
            total_f1_score = metric_score(resampled_ys, y_pred)

        tot_per_tr_f1_scores.append(total_f1_score) # Visualization purposes

        if total_f1_score > best_f1:
            best_f1 = total_f1_score
            best_thr = threshold*0.01

    # Best Threshold selection section
    #find threshold
    max_f1 = max(tot_per_tr_f1_scores)

    best_thresholds_idx = [i for i, f1 in enumerate(tot_per_tr_f1_scores) if f1 == max_f1]

    # whether multiple threholds provides the maximum f1_score, the median is taken
    best_threshold = np.median(best_thresholds_idx) + 1

    lowest_distance = 100
    thr_distances = []
    if closest:

        for thr_idx in np.array(list(range(1,101)))[best_thresholds_idx]:
            thr_dist = abs(thr_idx - 50)
            if thr_dist < lowest_distance:
                thr_distances.append(thr_dist)

        min_dist_thr_idx = thr_distances.index(np.min(thr_distances))
        best_threshold = np.array(list(range(1,101)))[best_thresholds_idx][min_dist_thr_idx]

    print(f'Chosen threshold: {best_threshold}. Associated F1_score: {tot_per_tr_f1_scores[int(best_threshold - 1)]:.4f}' )
    return best_threshold*0.01, tot_per_tr_f1_scores



# performance metrics section

def compute_metrics(true_labels, pred_probs, thr, mcc = False):
    """
    Computes classification metrics (f1, recall, precision, accuracy, optionally MCC)
    by classifying pred_probs with thr.
    """
    pred_labels = (np.array(pred_probs) >= thr).astype(int)

    f1 = f1_score(true_labels, pred_labels, pos_label=1, zero_division=0)
    rec = recall_score(true_labels, pred_labels, pos_label=1, zero_division=0)
    pre = precision_score(true_labels, pred_labels, pos_label=1, zero_division=0)
    acc = accuracy_score(true_labels, pred_labels)

    met_dict = {'f1': f1, 'rec': rec, 'pre': pre, 'acc': acc}

    if mcc:
        mcc_score = matthews_corrcoef(true_labels, pred_labels)
        met_dict['mcc'] = mcc_score
    return met_dict

def compute_mean_std_metrics(all_dir_metrics_across_LOPO, decimals = None, return_df = False):
    """
    Computes mean and std for each metric across all LOPO folds.
    Optionally rounds results or returns the raw DataFrame instead.
    """

    df = pd.DataFrame(all_dir_metrics_across_LOPO)

    if return_df:
        return df

    mean_std_dict = {}
    for metric in df.columns:
        all_met_list = flatten(df[metric].to_list())
        all_met_list_mean = np.mean(all_met_list)
        all_met_list_std = np.std(all_met_list, ddof=1)

        if decimals is not None:
            all_met_list_mean = round(all_met_list_mean, decimals)
            all_met_list_std = round(all_met_list_std, decimals)
        mean_std_dict[metric] = [all_met_list_mean, all_met_list_std]

    else:
        return mean_std_dict

