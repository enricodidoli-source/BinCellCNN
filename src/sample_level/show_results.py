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



from src.shared.utils import flatten, load_element
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

def build_results_df(results_dir, n_folds):
    rows = []
    for i in range(n_folds):
        try:
            
            fold_data = load_element(results_dir, f'fold_box_data_{i}')
        except FileNotFoundError:
            print(f'Fold {i} not found')
            continue
        print(f'Fold {i} found')
        for box in fold_data:
            rows.append({**box, 'fold': i})
    return pd.DataFrame(rows)

def show_all_LOPO_boxplots(results_df, save_dir=None, file_name=None, title=None,
                           fold_or_seed='Fold', y_labels=None):
    sns.set_style("ticks")
    df = results_df.reset_index(drop=True)

    timepoints_labels = df['True_Label'].tolist()
    timepoints_trials_scores = df['CV_Fold_Preds'].tolist()

    sample_thr = df['best_thr'].to_numpy()                      # soglia per sample

    thresholds_list = np.unique(df['best_thr'].to_numpy())
    by_fold = df.groupby('fold', sort=True)
    num_samples_cum = by_fold.size().cumsum().tolist()
    x_axis_1_labels = df['samples_ids'].tolist()

    if 'perc' in df.columns:
        timepoints_blast_perc = df['perc'].tolist()
        round_timepoints_blast_perc = [round(perc, 3) for perc in timepoints_blast_perc]
        has_perc =  True
    else:
        has_perc = False



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

    if has_perc:
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(ax.get_xticks())
        ax2.set_xticklabels(round_timepoints_blast_perc, fontsize=7)
        ax2.set_xlabel("% of Unhealthy cells per Sample")

    # tuned thresholds
    for i, threshold in enumerate(thresholds_list):
        if i == 0:
            x_min = 0.5
        else:
            x_min = num_samples_cum[i-1] + 0.5
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








