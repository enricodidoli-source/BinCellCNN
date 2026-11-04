from sklearn.metrics import fbeta_score, f1_score, accuracy_score, recall_score, precision_score, auc
from sklearn.metrics import matthews_corrcoef
import numpy as np
import pandas as pd
import os
import pickle as pkl


import json
import random
import seaborn as sns
import matplotlib.pyplot as plt



from src.shared.utils import flatten
from src.sample_level.classification import phenotype_prediction


def elaborate_predictions(predictions_list, test_y, results = True, beta = 1, mcc = False):
    """
    Computes F1, recall, precision, accuracy and optionally MCC across multiple prediction trials.
    If results=True, prints per-trial and mean metrics. Returns per-trial scores and a prediction DataFrame.
    """
    pred_phenotypes_dict = {}

    f1_scores_list = []
    recall_scores_list = []
    precision_scores_list = []
    accuracy_list = []
    mcc_list = []

    for i, w_init_pred_list in enumerate(predictions_list): # a prediction for every different seed

        pred_phenotypes = phenotype_prediction(w_init_pred_list, thr = 0.5)

        f1 = fbeta_score(test_y, pred_phenotypes, beta = beta, pos_label=1, zero_division = 0) # file-level
        f1_scores_list.append(f1)

        recall = recall_score(test_y, pred_phenotypes, pos_label=1, zero_division = 0) # file-level
        recall_scores_list.append(recall)
        
        precision = precision_score(test_y, pred_phenotypes, pos_label=1, zero_division = 0) # file-level
        precision_scores_list.append(precision)
        
        tot_correct = np.array(pred_phenotypes) ==  test_y #checks differencies in prediction
        accuracy = np.sum(tot_correct)/ len(test_y)  #compute accuracy
        accuracy_list.append(accuracy)
    
        if mcc:
            mcc_score = matthews_corrcoef(test_y, pred_phenotypes)
            mcc_list.append(mcc_score)

        pred_phenotypes_dict[i] = pred_phenotypes
        if results:
            print(f'Trial {i} Accuracy: {accuracy}')
            print(f'Trial {i} F1_score: {f1}')
            print(f'Trial {i} Recall: {recall}')
            print(f'Trial {i} Precision: {precision}')
            if mcc:
                print(f'Trial {i} MCC: {mcc_score}')

    pred_phenotypes_dict['True Labels'] = test_y
    
    pred_phenotype_df = pd.DataFrame(pred_phenotypes_dict)

    if results:
        print(pred_phenotype_df.T)
        mean_accuracy = np.mean(accuracy_list)
        print(f'mean_accuracy over the ten trials: {mean_accuracy}')
        accuracy_std = np.std(accuracy_list)
        print(f'accuracy_std over the ten trials: {accuracy_std }')

        mean_f1 = np.mean(f1_scores_list)
        print(f'mean_f1 over the ten trials: {mean_f1}')
        std_f1 = np.std(f1_scores_list)
        print(f'std_f1 over the ten trials: {std_f1}')

        mean_recall = np.mean(recall_scores_list)
        print(f'mean_recall over the ten trials: {mean_recall}')
        std_recall = np.std(recall_scores_list)
        print(f'std_recall over the ten trials: {std_recall}')       
        
        mean_precision = np.mean(precision_scores_list)
        print(f'mean_precision over the ten trials: {mean_precision}')
        std_precision = np.std(precision_scores_list)
        print(f'std_precision over the ten trials: {std_precision}')
    
        if mcc:
            mean_mcc = np.mean(mcc_list)
            print(f'mean_mcc over the ten trials: {mean_mcc}')
            std_mcc = np.std(mcc_list)
            print(f'std_mcc over the ten trials: {std_mcc}')
    if mcc:
        return pred_phenotype_df.T, accuracy_list , f1_scores_list, recall_scores_list, precision_scores_list, mcc_list
    
    return pred_phenotype_df.T, accuracy_list , f1_scores_list, recall_scores_list, precision_scores_list












def show_blast_distribution_perc(ALL_DATASETS, multiple_donations, return_perc = False, log = False, decimals = 4, mem = None, label_column = 'IsBlast'):
    """
    Computes and shows the blast cell percentage for each sample across all patients.
    Optionally returns the percentage list.
    """
    if mem:
        (ALL_DATASETS,) = mem.load_variables(['ALL_DATASETS'])

    tot_perc_list = []
    for pat_idx, samples_idx in multiple_donations.items():
        if log:
            print(samples_idx)
        for sample in samples_idx:

            dataset = ALL_DATASETS[sample]
            blast_n = (dataset[label_column] == 1).sum()
            tot_perc_list.append(round((blast_n/len(dataset))*100, decimals))
            if log:
                print(f'sample: {sample}: {round((blast_n/len(dataset))*100, 2)}')

    positions = range(1, len(tot_perc_list) + 1)

    fig, ax1 = plt.subplots(figsize = [len(positions)/2,4])
    ax1.bar(list(range(1, len(tot_perc_list) + 1)), [max(tot_perc_list)]*len(tot_perc_list), alpha = 0.5)
    ax1.bar(list(range(1, len(tot_perc_list) + 1)), tot_perc_list)
    ax1.set_xticks(positions)

    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(ax1.get_xticks())

    ax2.set_xticklabels(tot_perc_list, size = 'x-small')
    plt.show()

    if return_perc:
        return tot_perc_list

    return



def show_all_LOPO_boxplots(boxplot_data, timepoints_blast_perc, thresholds_list, num_samples_cum,
                        save_dir = None, file_name = None, title = None, fold_or_seed = 'Fold', y_labels = None):
    """
    Plots per-sample score distributions as boxplots across all LOPO folds, with per-fold
    threshold lines, blast percentage on the top axis, and misclassification markers.
    """
    sns.set_style("ticks")

    round_timepoints_blast_perc = [round(perc, 3) for perc in timepoints_blast_perc]

    timepoints_labels = [timepoint["True_Label"] for i, timepoint in enumerate(boxplot_data)]
    timepoints_trials_scores = [timepoint["CV_Fold_Preds"] for timepoint in boxplot_data]

    if y_labels:
        x_axis_1_labels = y_labels[:len(timepoints_trials_scores)]
    else:
        x_axis_1_labels = timepoints_labels

    n_box = len(timepoints_labels)
    fig, ax = plt.subplots(figsize=(max(0.5*n_box, 4), 4))

    # Plot the boxplot on the first axes (ax1)
    bp = ax.boxplot(timepoints_trials_scores, tick_labels=x_axis_1_labels, patch_artist=True,
                                                  showmeans=True,      # shows the mean
                                                  meanline=True, showfliers=False)
    for patch in bp['boxes']:
        patch.set_alpha(0.3)
        patch.set_facecolor('none')

    for whisker in bp['whiskers']:
        whisker.set_alpha(0.3)

    for cap in bp['caps']:
        cap.set_alpha(0.3)

    for median in bp['medians']:
        median.set_visible(False)

    for i, mean in enumerate(bp['means']):
        mean.set_color('red')
        mean.set_linestyle('--')
        mean.set_linewidth(2)
        mean.set_alpha(0.8)
        if i == 0:
            mean.set_label('Mean')

    for s, sample in enumerate(timepoints_trials_scores):
        for p, point in enumerate(sample):
            label = f'i-th {fold_or_seed} Prediction' if s == 0 and p == 0 else None
            ax.scatter(s+1, point, color='grey', label=label, s =10)

    ax.set_xlabel("True Samples Labels")
    ax.set_ylabel("Positivity Score")
    ax.set_xticklabels(x_axis_1_labels, rotation=45, ha='right', fontsize=7)


    ax2 = ax.twiny()

    # Make ax2 have the same limits and ticks as ax1
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(ax.get_xticks()) # Positions [1, 2, 3, 4]

    # Set the new labels for the top axis
    ax2.set_xticklabels(round_timepoints_blast_perc, fontsize=7)
    ax2.set_xlabel("% of Unhealthy cells per Sample")

    # tuned thresholds
    for i, threshold in enumerate(thresholds_list):
        if i == 0:
            x_min = 0.5
        else:
            x_min = i + 0.5
        x_max = num_samples_cum[i] + 0.5
        print(x_max)
        box_label = f'j-th LOPO Threshold' if i == 0 else None

        ax.hlines(y=threshold, xmin=x_min, xmax=x_max, color='red',
               linestyles='solid', label=box_label, linewidth=1.5, alpha = 0.3)

        ax.vlines(x=x_max, ymin = -0.05, ymax = 1.05, color = 'grey', linestyle='--', alpha = 0.5)

    # default threshold = 0.5
    ax.hlines(y=0.5, xmin=0.5, xmax=x_max, color='black',
               linestyles='dotted', label='Default Threshold = 0.5', linewidth=1.5, alpha = 0.3)

    # misprediction marks
    
    mis_label = True
    for s, sample in enumerate(timepoints_trials_scores):

        thr_index = num_samples_cum.index(next(x for x in num_samples_cum if x > s))

        thr = thresholds_list[thr_index]
        true_label = timepoints_labels[s]        # 0 o 1
        pred_label = 1 if np.mean(sample) > thr else 0    # 0 o 1

        if true_label != pred_label:
            ax.scatter(s + 1, 1.025, color='red', marker='x', s=50, linewidths=1.5, zorder=5,
                      label='Misclassified' if mis_label else None)

            if mis_label:
                mis_label = False
    
    plt.ylim(-0.05, 1.05)
    if title is  None:
        title = "Positivity Score Distribution across LOPOCV Iterations"

    plt.title(title)
    ax.legend()
    if save_dir:
        if file_name is None:
            file_name = '5.2.all_box_straight.pdf'
        fig.savefig(f'{save_dir}/{file_name}', format = 'pdf', bbox_inches='tight')

    plt.show()
    plt.close()







