from sklearn.metrics import fbeta_score, f1_score, accuracy_score, recall_score, precision_score, auc
from sklearn.metrics import matthews_corrcoef
import numpy as np
import pandas as pd
import os
import pickle as pkl

from classification import robust_prediction_labelling

from utils import flatten
import json
import random
import seaborn as sns
import matplotlib.pyplot as plt


'''
retrieve_samples_info, load_tuning_data, retrieve_all_LOPO_thresholds, retrieve_all_LOPO_ensemble_thresholds
retrieve_all_LOPO_ensemble_metrics, retrieve_mean_std, rebuild_dataset_predictions
generate_heatmap_dict, generate_dict_comb_3d, predict_trials,final_trials_prediction
elaborate_data_for_box_violin, elaborate_ens_data_for_box_violin, from_orig_to_res_structure
direct_prediction_across_seeds, scores_from_robust_labelling, cumulative_num_samples_sum, retrieve_blast_perc
'''

def retrieve_samples_info(data_folder_dir, multiple_donations, ALL_DATASETS):
    """
    Checks sample files from a folder, extracts cell counts and blast percentages,
    and returns a DataFrame sorted by blast count.
    """

    file_names = os.listdir(data_folder_dir)

    samples_info_dict = {}

    last_patient = '1'
    counter = -1
    for element in file_names:

        if 'csv' in element:
            element = element[:-4]
            splitted_element = element.split('GHE')[1]
            patient_id = splitted_element.split('_')[0]
            time_point_days = splitted_element.split('_')[1]


            if last_patient == str(patient_id):
                counter += 1
            else:
                counter = 0


            file_idx = multiple_donations[str(patient_id)][counter]

            sample = ALL_DATASETS[file_idx]

            blast_n = len(sample[sample['IsBlast'] == 1])
            cells_n = len(sample)
            healthy_n = len(sample[sample['IsBlast'] == 0])
            blast_p = round((blast_n/cells_n)*100,4)
            healthy_p = 100-blast_p
            samples_info_dict[file_idx] = [cells_n, healthy_n, blast_n, healthy_p, blast_p, patient_id, time_point_days]

            last_patient = str(patient_id)

    df = pd.DataFrame.from_dict(samples_info_dict)
    df = df.transpose()
    df.columns = ['cells_n', 'healthy_n', 'blast_n', 'healthy_p', 'blast_p', 'patient_id', 'time_point_days']
    df['Sample'] = df.index
    df = df.sort_values(by='blast_n')


    return df


def load_tuning_data(tuning_load_dir):
    """
    Loads tuning results from disk.
    """
    keys = ['best_ncells', 'best_nsub', 'robust_threshold', 'roc_threshold', 'tested_par', 'val_predicted_for_roc', 'ensemble_mean_probs_per_patient' ]
    threshold_data = {}
    for key in keys:
        with open(os.path.join(tuning_load_dir, f'{key}.pkl'), 'rb') as f:
            threshold_data[key] = pkl.load(f)


    return threshold_data

def retrieve_all_LOPO_thresholds(LOPO_folds, cellcnn_path, tuning_exp):
    """
    Loads ROC and RES thresholds for each LOPO fold from disk.
    """
    roc_thr_per_fold = []
    rob_thr_per_fold = []

    for i in range(LOPO_folds):
        tuning_load_dir = f'{cellcnn_path}/experiments/experiment_{tuning_exp}/outer_fold_{i}/tuning/results'


        with open(os.path.join(tuning_load_dir, 'robust_threshold.pkl'), 'rb') as f:
                                robust_threshold = pkl.load(f)

        with open(os.path.join(tuning_load_dir, 'roc_threshold.pkl'), 'rb') as f:
                                roc_threshold = pkl.load(f)
        roc_thr_per_fold.append(roc_threshold)
        rob_thr_per_fold.append(robust_threshold)
    return roc_thr_per_fold, rob_thr_per_fold


def retrieve_all_LOPO_ensemble_thresholds(LOPO_folds, cellcnn_path, tuning_exp):
    """
    Loads ensemble ROC and RES thresholds for each LOPO fold from disk.
    """    
    ens_roc_thr_per_fold = []
    ens_rob_thr_per_fold = []
    for i in range(LOPO_folds):
        tuning_load_dir = f'{cellcnn_path}/experiments/experiment_{tuning_exp}/outer_fold_{i}/ensemble/tuning/resampling/'
        with open(os.path.join(tuning_load_dir, 'ensemble_robust_threshold.pkl'), 'rb') as f:
                    ensemble_robust_threshold = pkl.load(f) 

        tuning_load_dir = f'{cellcnn_path}/experiments/experiment_{tuning_exp}/outer_fold_{i}/ensemble/tuning/roc/'
        with open(os.path.join(tuning_load_dir, 'ensemble_roc_threshold.pkl'), 'rb') as f:
                    ensemble_roc_threshold = pkl.load(f) 
        
        ens_roc_thr_per_fold.append(ensemble_roc_threshold)
        ens_rob_thr_per_fold.append(ensemble_robust_threshold)

    return ens_roc_thr_per_fold, ens_rob_thr_per_fold

def retrieve_all_LOPO_ensemble_metrics(save_ensemble_dir, roc_TUNED_THRESHOLD, pred = False):
    """
    Loads ensemble predictions for each inner fold, computes classification metrics
    and returns them aggregated in a dictionary.
    """
    ens_folds = len([fold_name for fold_name in list(os.listdir(save_ensemble_dir)) if 'fold' in fold_name])
    
    roc_ens_metrics_dicts = []
    rob_ens_metrics_dicts = []

    for i in range(ens_folds):
            

        save_robust_dir = f'{save_ensemble_dir}/inner_fold_{i}/predictions/robust/'

        with open(os.path.join(save_robust_dir, 'test_total_trial_pred_lists.pkl'), 'rb') as f:
            test_total_trial_pred_lists = pkl.load(f)
        with open(os.path.join(save_robust_dir, 'per_donor_original_test_y.pkl'), 'rb') as f:
            per_donor_original_test_y = pkl.load(f)

        if pred:
            ens_roc_test_total_labels, ens_roc_test_total_preds = robust_prediction_labelling(test_total_trial_pred_lists, roc_TUNED_THRESHOLD*100, pred = True)
            
        else:
            ens_roc_test_total_labels = robust_prediction_labelling(test_total_trial_pred_lists, roc_TUNED_THRESHOLD*100)

        per_donor_original_test_y_flat = flatten(per_donor_original_test_y)
        ens_roc_test_total_labels_flat = flatten(ens_roc_test_total_labels)
            
        if pred:  
            rounded_pred = [round(pr, 6) for pr in  flatten(ens_roc_test_total_preds)] 
 
        roc_robust_metrics_across_trials = scores_from_robust_labelling(per_donor_original_test_y_flat, ens_roc_test_total_labels_flat)
        roc_ens_metrics_dicts.append(roc_robust_metrics_across_trials)
 
    df = pd.DataFrame(roc_ens_metrics_dicts)
    new = {}
    for key in df.columns:
        new[key] = df[key].to_list()
    return new

def retrieve_mean_std(thesis_images_dir, exp_names, metric = 'f1', function = 'mean'):
    """
    Loads per-fold and across-fold metric statistics for each experiment and classification type,
    returning mean or std values depending on the function parameter.
    across_lopo_mean_std_dict and all_lopo_mean_std_dict are dictionaty containing metrics as keys and [mean, std] as values
    """
    type_classification = ['DIR', 'ROC', 'RES']
    if function == 'mean':
        pos = 0
    else:
        pos = 1

    tot_exp_std_list = []
    for exp_name in exp_names:
        std_f1_list = []
        for class_type in type_classification:
            tables_dir = f'{thesis_images_dir}/mean_std_tables_{exp_name}/{class_type}/'

            with open(os.path.join(tables_dir, 'all_lopo_mean_std_dict.pkl'), 'rb') as f:
                    all_lopo_mean_std_dict = pkl.load(f)
            with open(os.path.join(tables_dir, 'across_lopo_mean_std_dict.pkl'), 'rb') as f:
                    across_lopo_mean_std_dict = pkl.load(f)

            f1_std_dir = all_lopo_mean_std_dict[metric][pos]
            std_f1_list.append(f1_std_dir)

            f1_std_ens = across_lopo_mean_std_dict[metric][pos]
            std_f1_list.append(f1_std_ens)

        tot_exp_std_list.append(std_f1_list)
    return tot_exp_std_list



def rebuild_dataset_predictions(val_predicted_for_roc, best_idx):
    """
    Flattens predictions and labels from the best fold configuration
    into two flat lists for ROC curve computation.
    val_predicted_for_roc is a list of lists, where each sublist contains k tuples (one per fold),
    each tuple contains per-sample predictions and the corresponding ground truth labels.
    """
    val_predicted_for_roc_folds = val_predicted_for_roc[best_idx]

    predictions_list, new_val_y = [], []
    for fold in val_predicted_for_roc_folds:
        pred_list, val_y = fold
        new_val_y += val_y

        for neg_prob, pos_prob in pred_list[0]:
            predictions_list.append(pos_prob)
    return predictions_list, new_val_y


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

    for i, pred in enumerate(predictions_list): # a prediction for every different seed

        pred_phenotypes = phenotype_prediction(pred)

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

def phenotype_prediction(test_pred):
    """
    Converts raw prediction probabilities into binary labels using a fixed 0.5 threshold on the first class.
    """
    pred_phenotypes = []
    for sample_pred in test_pred:
        if sample_pred[0] < 0.5: # if  the first class < 0.5 => second class > 0.5
            pred_phenotypes.append(1)
        else:
            pred_phenotypes.append(0)
    return pred_phenotypes

"==============================================================================="

def generate_heatmap_dict(tested_par, val_predicted_for_roc):
    """
    Computes mean F1 score for each (ncells, nsubs) combination across CV folds,
    returning a dictionary ready for heatmap plotting.
    """
    f1_score_to_plot = []

    tested_ncells, tested_nsubs = [], []
    heatmap_dict = {}
    for (ncells, nsubs), fold in zip(tested_par, val_predicted_for_roc):
        f1_fold = []
        for pred_list, val_y in fold:
            _, _, f1_score_list, _, _ = elaborate_predictions(
                                        pred_list, val_y, results=False)

            f1_fold.append(f1_score_list)

        f1_score_to_plot.append(np.mean(f1_fold))

        tested_ncells.append(ncells)
        tested_nsubs.append(nsubs)

    heatmap_dict['ncells'] = tested_ncells
    heatmap_dict['nsubs'] = tested_nsubs
    heatmap_dict['f1'] = f1_score_to_plot

    return heatmap_dict


def generate_dict_comb_3d(LOPO_folds, tuning_exp_pat):
    """
    Loads tuning results for each LOPO fold and collects the best (ncells, nsubs)
    combination and its mean F1 score into a dictionary for 3D plotting.
    """
    heatmap_dict = {}
    ncells_per_fold = []
    nsubs_per_fold = []
    f1_score_per_fold = []
    for LOPO_idx in range(LOPO_folds):

        tuning_load_dir = f'{tuning_exp_pat}/outer_fold_{LOPO_idx}/tuning/results'

        threshold_data = load_tuning_data(tuning_load_dir)

        best_ncells =threshold_data['best_ncells']
        best_nsub =threshold_data['best_nsub']
        tested_par =threshold_data['tested_par']
        val_predicted_for_roc = threshold_data['val_predicted_for_roc']

        chosen_combination = (best_ncells, best_nsub)
        best_idx = tested_par.index(chosen_combination)
        
        f1_fold = []
        for pred_list, val_y in val_predicted_for_roc[best_idx]:
            _, _, f1_score_list, _, _ = elaborate_predictions(
                                            pred_list, val_y, results=False
                                        )
            f1_fold.append(f1_score_list)
        mean_f1 = np.mean(f1_fold)

        ncells_per_fold.append(best_ncells)
        nsubs_per_fold.append(best_nsub)
        f1_score_per_fold.append(mean_f1)

    heatmap_dict['ncells'] = ncells_per_fold
    heatmap_dict['nsubs'] = nsubs_per_fold
    heatmap_dict['f1'] = f1_score_per_fold

    return heatmap_dict



def predict_trials(patient, patient_ys, patient_resampled_y, best_threshold):
    """
    For each timepoint of a patient, computes mean probability and F1 score across trials,
    returning data structured for violin, boxplot and F1 plots.
    """
    sample_prob_data = []

    f1_data = []
    boxplot_data = []
    for timepoint, true_timepoint_y, true_resampled_y in zip(patient, patient_ys, patient_resampled_y):
        label_str = "Positive" if true_timepoint_y == 1 else "Negative"

        timepoint_boxplot_data = []
        timepoint_f1_data = []

        for trial in timepoint: 

            trial_prob = np.mean(trial) # mean of the probabilities of the subsets
            trial_pred = (np.array(trial) >= best_threshold).astype(int)

            trial_f1 =  f1_score(true_resampled_y, trial_pred, pos_label = 1, zero_division=1)

            prob_dict = {
                "True_Label": label_str,
                "Timepoint_Score": trial_prob
                }

            sample_prob_data.append(prob_dict)

            timepoint_boxplot_data.append(trial_prob)
            timepoint_f1_data.append(trial_f1)

        box_dict = {
                "True_Label": true_timepoint_y,
                "Timepoint_trials_scores": timepoint_boxplot_data
                }

        f1_dict = {
                "True_Label": true_timepoint_y,
                "Timepoint_trials_scores": timepoint_f1_data
                }

        boxplot_data.append(box_dict)
        f1_data.append(f1_dict)
    return sample_prob_data, boxplot_data, f1_data


def final_trials_prediction(total_trial_pred_lists, per_donor_original_test_y, per_donor_resampled_test_y, best_threshold):
    """
    Aggregates trial predictions across all patients and timepoints by calling predict_trials,
    returning flat lists of probability, boxplot and F1 data.
    """
    prob_data = []
    f1_data = []
    boxplot_data = []

    for i, (patient, patient_ys, patient_resampled_y) in enumerate(zip(total_trial_pred_lists, per_donor_original_test_y, per_donor_resampled_test_y)):
        sample_prob_data, box_dict, f1_dict = predict_trials(patient, patient_ys, patient_resampled_y, best_threshold)

        prob_data += sample_prob_data

        boxplot_data.extend(box_dict)
        f1_data.extend(f1_dict)

    return prob_data, boxplot_data, f1_data


def elaborate_data_for_box_violin(save_robust_dir, threshold = 0.5):
    """
    Loads prediction and label files from disk and returns data structured
    for violin and boxplot visualization.
    """
    with open(os.path.join(save_robust_dir, 'test_total_trial_pred_lists.pkl'), 'rb') as f:
                            test_total_trial_pred_lists = pkl.load(f)

    with open(os.path.join(save_robust_dir, 'per_donor_resampled_test_y.pkl'), 'rb') as f:
                            per_donor_resampled_test_y = pkl.load(f)

    with open(os.path.join(save_robust_dir, 'per_donor_original_test_y.pkl'), 'rb') as f:
                            per_donor_original_test_y = pkl.load(f)

    plot_data, boxplot_data, _ = final_trials_prediction(test_total_trial_pred_lists,
                            per_donor_original_test_y, per_donor_resampled_test_y, threshold)
    return plot_data, boxplot_data


def elaborate_ens_data_for_box_violin(save_ensemble_dir, thresholds_list, num_samples):
    """
    Loads ensemble predictions for each inner fold and aggregates data
    for violin and boxplot visualization across folds.
    """
    ens_folds = len([fold_name for fold_name in list(os.listdir(save_ensemble_dir)) if 'fold' in fold_name])

    roc_per_cv_fold_violin = []

    roc_per_cv_fold_box = [[] for _ in range(num_samples)]

    for i in range(ens_folds):

        roc_TUNED_THRESHOLD = thresholds_list[i]
        save_robust_dir = f'{save_ensemble_dir}/inner_fold_{i}/predictions/robust/'

        with open(os.path.join(save_robust_dir, 'test_total_trial_pred_lists.pkl'), 'rb') as f:
                                test_total_trial_pred_lists = pkl.load(f)

        with open(os.path.join(save_robust_dir, 'per_donor_original_test_y.pkl'), 'rb') as f:
                                per_donor_original_test_y = pkl.load(f)

        with open(os.path.join(save_robust_dir, 'per_donor_resampled_test_y.pkl'), 'rb') as f:
                                per_donor_resampled_test_y = pkl.load(f)

        plot_data, boxplot_data, f1_data = final_trials_prediction(test_total_trial_pred_lists,
                            per_donor_original_test_y, per_donor_resampled_test_y, roc_TUNED_THRESHOLD)

        roc_per_cv_fold_violin += plot_data

        for s, sample in enumerate(boxplot_data):
            roc_per_cv_fold_box[s].append(np.mean(sample['Timepoint_trials_scores']))


    roc_cv_box_mean_across_folds = []
    for s, sample in enumerate(boxplot_data):

        dict_across_folds = {}
        dict_across_folds['True_Label'] = sample['True_Label']
        dict_across_folds['Timepoint_trials_scores'] = roc_per_cv_fold_box[s]
        roc_cv_box_mean_across_folds.append(dict_across_folds)

    return roc_per_cv_fold_violin, roc_cv_box_mean_across_folds




def from_orig_to_res_structure(original_predictions_list, per_donor_original_test_y):
    """
    Converts a trial-first prediction structure into a patient/sample/trial nested structure
    compatible with the violin plot and labelling pipeline.
    """
    patients = []
    samples_trials = []
    num_samples = len(original_predictions_list[0])
    for s in range(num_samples):
        sample_trial = []
        for t, trial in enumerate(original_predictions_list):
            sample_trial.append(original_predictions_list[t][s])
        samples_trials.append(sample_trial)

    orig_pred_for_violin = []
    for sample in samples_trials:
        trials = []
        for neg, pos in sample:
            subsets_pred = [pos]
            trials.append(subsets_pred)
        orig_pred_for_violin.append(trials)
    patients.append(orig_pred_for_violin)

    patients_y = []
    for p, pat in enumerate(patients):
        pat_y = []
        for s, sample in enumerate(pat):
            sample_label = per_donor_original_test_y[p][s]

            pat_y.append([sample_label])
        patients_y.append(pat_y)

    return patients, patients_y



def direct_prediction_across_seeds(direct_pred_across_seeds, thr = 0.5):
    """
    Averages predictions across seeds for each patient and sample,
    then assigns binary labels based on the given threshold.
    """
    final_labels_per_patient = []
    mean_prob_per_patient = []
    for p, patient in enumerate(direct_pred_across_seeds[0]):
        mean_probs_per_sample = []
        final_labels_per_sample = []
        for  s, sample in enumerate(patient):
            sample_pred = []

            for t, seed in enumerate(direct_pred_across_seeds):

                sample_pred.append(direct_pred_across_seeds[t][p][s][1])

            mean_prob = np.mean(sample_pred)
            mean_probs_per_sample.append(mean_prob)
            label = 1 if mean_prob >= thr else 0
            final_labels_per_sample.append(label)
            print(f'patient: {p}, sample: {s}, mean_prob: {mean_prob}, pred_label: {label}')

        final_labels_per_patient.append(final_labels_per_sample)
        mean_prob_per_patient.append(mean_probs_per_sample)
    return final_labels_per_patient, mean_prob_per_patient

"==============================================================================="


def scores_from_robust_labelling(per_donor_original_test_y_flat, test_total_labels_flat):
    """
    Computes F1, recall, precision and accuracy from flat ground truth and predicted labels.
    """

    robust_metrics_across_trials= {}
    robust_metrics_across_trials['f1'] = f1_score(per_donor_original_test_y_flat, test_total_labels_flat, pos_label = 1, zero_division = 0 )
    robust_metrics_across_trials['rec'] = recall_score(per_donor_original_test_y_flat, test_total_labels_flat, pos_label = 1, zero_division = 0 )
    robust_metrics_across_trials['pre'] = precision_score(per_donor_original_test_y_flat, test_total_labels_flat, pos_label = 1, zero_division = 0 )
    robust_metrics_across_trials['acc'] = accuracy_score(per_donor_original_test_y_flat, test_total_labels_flat)

    return robust_metrics_across_trials

def cumulative_num_samples_sum(data):
    """
    Computes the cumulative sample count across patients.
    """
    num_samples_cum = []
    cum_sum = 0
    for patient in data: 
        cum_sum += len(patient)
        num_samples_cum.append(cum_sum)

    return num_samples_cum

def retrieve_blast_perc(per_donor_original_test_datasets):
    """
    Computes the blast cell percentage for each sample across all donors.
    """
    patient_blast_perc = []
    for donor in per_donor_original_test_datasets:
        for i, dataset in enumerate(donor):
            blast_n = (dataset['IsBlast'] == 1).sum()/len(dataset)
            patient_blast_perc.append(round(blast_n*100,4))
    return patient_blast_perc

"==============================================================================="


def show_sample_healthy_blast_distribution(samples_info_dict, axis=None, save_dir=None, numbers = False):
    """
    Plots a horizontal bar chart of healthy vs blast cells per sample on a symlog scale.
    Supports embedding in an existing axis, optional numeric annotations, and PDF export.
    """
    df = samples_info_dict
    df = df.sort_values(by='blast_n').reset_index(drop=True)
    y_labels = [str(f'Pat_{patient_id}_{time_point}') for sample_id, time_point, patient_id in zip(df['Sample'], df['time_point_days'], df['patient_id'])]

    max_val = max(df['healthy_n'].max(), df['blast_n'].max())

    if axis is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        ax = axis

    ax.barh(y_labels, -df['blast_n'],  color='tomato',    label='Unhealthy cells')
    ax.barh(y_labels,  df['healthy_n'], color='steelblue', label='Healthy cells', alpha = 0.7)

    ax.axvline(0, color='black', linewidth=0.8)

    if numbers:
        # numbers of cells on bars
        for i, (blast, healthy) in enumerate(zip(df['blast_n'], df['healthy_n'])):
            # unhealthy cells (left)
            ax.text(-blast, i, f'{blast:,.0f}', 
                    va='center', ha='right', fontsize=6, color='black')
            # healthy cells (right)
            ax.text(healthy/10, i, f'{healthy:,.0f}', 
                    va='center', ha='left', fontsize=6, color='black')

    ax.set_xscale('symlog', linthresh=1000)

    # 10^* notation on axis x
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: '' if v == 0 else f'$10^{{{int(np.log10(abs(v)))}}}$'))
    ax.tick_params(axis='x', labelsize=7)

    ax.set_title('Healthy vs Unhealthy cells per sample (symlog scale)', fontsize=13)
    ax.set_ylabel('Patient-ID-Time Point Day', fontsize=10)
    ax.set_xlabel('Number of Unhealthy/Healthy cells (symlog scale)', fontsize=10)
    ax.set_xlim(-max_val * 1.1, max_val * 1.1)

    ax.legend()
    if axis is None:
        plt.tight_layout()
        if save_dir:
            plt.savefig(f'{save_dir}/0.1.healthy_unhealthy_samples_distribution.pdf', format='pdf')
            plt.close()
        plt.show()

def show_sample_blast_perc_distribution(samples_info_dict, axis = None, save_dir = None):
    """
    Plots a horizontal bar chart of blast cell percentage per sample,
    with a secondary y-axis showing the exact percentage values.
    """

    df = samples_info_dict
    df = df.sort_values(by='blast_n').reset_index(drop=True)
    y_labels = [str(f'Pat_{patient_id}_{time_point}') for sample_id, time_point, patient_id in zip(df['Sample'], df['time_point_days'], df['patient_id'])]

    df['pct_unhealthy'] = -df['blast_p']  # negative --> left

    if axis is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        ax = axis

    ax.barh(y_labels, df['pct_unhealthy'], color='tomato',    label='Blast cells')

    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('% of total cells')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{abs(v):.0f}%'))
    plt.draw()  

    ax2 = ax.twinx()

    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(ax.get_yticks())
    ax2.set_yticklabels([f'{v:.4f}%' for v in df['blast_p']])

    ax.set_title('Unhealthy cells % per sample', fontsize=13)
    ax.set_ylabel('Patient-ID-Time Point Day',  fontsize=10)
    ax.set_xlabel('Unhealthy cells %',  fontsize=10)
    ax2.set_ylabel('Unhealthy cells %',  fontsize=10)

    ax.legend()
    if axis is None:
        plt.tight_layout()
        if save_dir:
            plt.savefig(f'{save_dir}/0.2.samples_unhealthy_cells_percentages.pdf', format = 'pdf')
            plt.close()
        plt.show()


def show_patients_samples_info(samples_info_dict, save_dir = None, numbers = False): 
    """
    Combines show_sample_healthy_blast_distribution and show_sample_blast_perc_distribution
    into a single two-panel figure. Optionally saves as PDF.
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'wspace': 0.3})

    show_sample_healthy_blast_distribution(samples_info_dict, axis = axs[0], numbers = numbers)
    show_sample_blast_perc_distribution(samples_info_dict, axis = axs[1])

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.suptitle('Samples Informations', fontsize = 15)

    if save_dir:
        fig.savefig(f'{save_dir}/0.samples_info.pdf', format = 'pdf')
    plt.show()
    plt.close()

def show_roc_thresholds(fpr, tpr, thresholds, save_dir = None):
    """
    Combines ROC curve and sensitivity/specificity plots into a single two-panel figure.
    Optionally saves as PDF.
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    plot_roc_curve(fpr, tpr, thresholds, axis = axs[0])
    sensitivity_vs_specificity(fpr, tpr, thresholds, axis = axs[1])

    plt.tight_layout()

    if save_dir:
        fig.savefig(f'{save_dir}/1.roc_thresholds.pdf', format = 'pdf')

    plt.show()
    plt.close()

   



def plot_roc_curve(fpr, tpr, thresholds, axis = None, save_dir = None):
    """
    Plots the ROC curve with AUC and highlights the optimal threshold point (TPR >= 0.95).
    Supports embedding in an existing axis and PDF export.
    """    
    # index of the threshold that has the highest tpr
    if np.argmax(np.array(tpr) >= 0.95) == 0:
        opt_idx = np.argmax(np.array(tpr))
    else:
        opt_idx = np.argmax(np.array(tpr) >= 0.95)  

    roc_threshold = thresholds[opt_idx] 
    opt_fpr, opt_tpr = fpr[opt_idx], tpr[opt_idx]
    roc_auc = auc(fpr, tpr)

    if axis is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        ax = axis

    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
    ax.plot(opt_fpr, opt_tpr, 'ro', label=f'Optimal (thr={roc_threshold:.2f})')
    ax.plot([0,1], [0,1], 'k--', lw=1)
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title('ROC Curve')
    ax.legend()
    ax.grid(alpha=0.3)

    if axis is None:
        plt.tight_layout()
        if save_dir:
            plt.savefig(f'{save_dir}/1.1.roc_curve.pdf', format = 'pdf')
        plt.show()
        plt.close()


def sensitivity_vs_specificity(fpr, tpr, thresholds, axis = None, save_dir = None):
    """
    Plots sensitivity (TPR) and specificity (TNR) against threshold values,
    highlighting the optimal threshold. Supports embedding in an existing axis and PDF export.
    """
    # index of the threshold that has the highest tpr
    if np.argmax(np.array(tpr) >= 0.95) == 0:
        opt_idx = np.argmax(np.array(tpr))
    else:
        opt_idx = np.argmax(np.array(tpr) >= 0.95) 
    
    roc_threshold = thresholds[opt_idx] 

    if axis is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        ax = axis

    ax.plot(thresholds, tpr, label='Sensitivity (TPR)', color='green')
    ax.plot(thresholds, 1 - fpr, label='Specificity (TNR)', color='purple')
    ax.axvline(roc_threshold, color='red', linestyle='--', label=f'Optimal thr = {roc_threshold:.2f}')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Rate')
    ax.set_title('Sensitivity & Specificity vs Threshold')
    ax.legend()
    ax.grid(alpha=0.3)

    if axis is None:
        plt.tight_layout()
        if save_dir:
            plt.savefig(f'{save_dir}/1.2.sensitivity_vs_specificity.pdf', format = 'pdf')
        plt.show()
        plt.close()


def show_heat_map_combinations(heatmap_dict, best_idx = None, save_dir = None):
    """
    Plots explored (ncells, nsubs) combinations colored by F1 score.
    Optionally highlights the best combination with a red circle marker.
    """
    df = pd.DataFrame(heatmap_dict)

    sns.set_style("ticks")  # or "whitegrid"

    plt.figure(figsize=(6, 4))

    if best_idx is not None:
      best_ncells, best_nsub = heatmap_dict['ncells'][best_idx], heatmap_dict['nsubs'][best_idx]
      best_f1 = df['f1'].iloc[best_idx]

      plt.scatter(best_nsub, best_ncells,
                  s=200,
                  facecolors='none',
                  edgecolors='red', # red edge
                  linewidth=2.5,
                  label=f'Comb: {best_ncells}, {best_nsub}. F1-score: {best_f1*100:.2f}%'
                          )

    scatter = plt.scatter(df['nsubs'], df['ncells'], c=df['f1'], cmap='viridis',
                s=120,                 # dot size
                edgecolor='black',     # black enge
                linewidth=1)

    plt.colorbar(scatter, label='F1 Score')
    plt.ylabel('ncells')
    plt.xlabel('nsubs')
    plt.title('F1 Score - Explored combinations (Bayesian Trials)')

    if best_idx is not None:
        plt.legend(fontsize = 10)

    if save_dir is not None:
        plt.savefig(f'{save_dir}/1.2.tested_combinations.pdf', format = 'pdf')

    plt.show()



def show_best_com_3d(heatmap_dict, save_dir = None):
    """
    Plots the best (ncells, nsubs) combinations and their F1 scores in a 3D scatter plot,
    with each point labeled by its combination. Optionally saves as PDF.
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    n = 100

    comb_text = [f'({ncells},{nsubs})' for ncells, nsubs in zip(heatmap_dict['ncells'], heatmap_dict['nsubs'])]

    print(comb_text)
    xs = heatmap_dict['ncells']
    ys = heatmap_dict['nsubs']
    zs = heatmap_dict['f1']

    # Add slight offset to prevent text overlapping with points
    for i, txt in enumerate(comb_text):
        ax.text(xs[i], ys[i], zs[i] + 0.0075, txt, fontsize=6)

    ax.scatter(xs, ys, zs, marker='o', s = 100)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    if save_dir is not None:
        plt.savefig(f'{save_dir}/0.1.best_comb_3D.pdf', format = 'pdf')

    plt.show()




def show_violin_plot(plot_data, TUNED_THRESHOLD, axis = None, save_dir = None, LOPO_fold = None):
    """
    Plots a violin chart showing the distribution of timepoint scores
    by true label, with a horizontal line marking the tuned threshold.
    """
    sns.set_style("ticks")

    df = pd.DataFrame(plot_data)

    if axis is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        ax = axis

    sns.violinplot(
        x="True_Label",
        y="Timepoint_Score",
        data=df,
        order=["Negative", "Positive"],
        ax=ax # Ensure this order
    )
    sns.swarmplot(
        x="True_Label",
        y="Timepoint_Score",
        data=df,
        color="black",
        alpha=0.2, 
        size=2,
        order=["Negative", "Positive"],
        ax=ax
        )

    ax.axhline(y=TUNED_THRESHOLD, color='red', linestyle='--', label=f'Tuned Threshold ({TUNED_THRESHOLD:.2f})')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('Sample Score Distribution vs. True Label (Across All Seeds)')
    ax.set_ylabel('Timepoint Blast Score')
    ax.set_xlabel('True Timepoint Label')

    handles, labels = ax.get_legend_handles_labels()
    if handles:  
        ax.legend(handles=handles, labels=labels)

    if axis is None:
        plt.tight_layout()

        if save_dir:
            if LOPO_fold is None:
                print('Warning: same name for each barplot. Overwrite: True!')
                LOPO_fold = ''
            plt.savefig(f'{save_dir}/4.1.violin_LOPO_fold_{LOPO_fold}.pdf', format = 'pdf')
            plt.show()
            plt.close() 




def show_all_LOPO_boxplots(boxplot_data, timepoints_blast_perc, thresholds_list, num_samples_cum, 
                        save_dir = None, file_name = None, title = None, fold_or_seed = 'Fold', y_labels = None):
    """
    Plots per-sample score distributions as boxplots across all LOPO folds, with per-fold
    threshold lines, blast percentage on the top axis, and misclassification markers.
    """
    sns.set_style("ticks")

    round_timepoints_blast_perc = [round(perc, 3) for perc in timepoints_blast_perc]

    timepoints_labels = [timepoint["True_Label"] for i, timepoint in enumerate(boxplot_data)]
    timepoints_trials_scores = [timepoint["Timepoint_trials_scores"] for timepoint in boxplot_data]

    if y_labels:
        x_axis_1_labels = y_labels
    else:
        x_axis_1_labels = timepoints_labels

    n_box = len(timepoints_labels)
    fig, ax = plt.subplots(figsize=(0.5*n_box, 4))

    # Plot the boxplot on the first axes (ax1)
    bp = ax.boxplot(timepoints_trials_scores, labels=x_axis_1_labels, patch_artist=True,
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
            x_min = num_samples_cum[i-1] + 0.5
        x_max = num_samples_cum[i] +0.5

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

show_sample_healthy_blast_distribution, show_sample_blast_perc_distribution, show_patients_samples_info
show_roc_thresholds, plot_roc_curve, sensitivity_vs_specificity, show_heat_map_combinations
show_best_com_3d, show_violin_plot, show_all_LOPO_boxplots




def show_dot_boxplot_metrics(ens_all_roc_metrics, axis = None, file_name = None, save_dir = None, mean = True, title = None, dots = True,  figsize = (6,5), 
                                  legend_pos = 1.15, fold_or_seed = 'Fold'):
    """
    Plots a boxplot of classification metrics (F1, recall, precision, accuracy) with optional
    dots showing per-fold/seed values. Supports mean aggregation across folds.
    """
    random.seed(42)
    if title is None:
        title = 'Ensemble Boxpot Metrics'

    mean_across_folds = {met: [] for met, _ in ens_all_roc_metrics[0].items()}
    if mean:
        for fold in ens_all_roc_metrics:
            for met, values in fold.items():
                values = np.mean(values)
                mean_across_folds[met].append(values)
    else:
        mean_across_folds = ens_all_roc_metrics[0]

    df = pd.DataFrame(mean_across_folds)

    metrics_names = df.columns.to_list()
    metrics_means = df.mean(axis=0).to_list()

    if axis is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        ax = axis

    colors = ['blue', 'orange', 'green', 'red' ]
    full_metrics_names = ['F1-score', 'Recall', 'Precision', 'Accuracy']

    df.columns = full_metrics_names

    bp = ax.boxplot(df, patch_artist=True, showmeans=True, meanline=True, showfliers=False)
    for i, patch in enumerate(bp['boxes']):
        patch.set_alpha(0.5)
        patch.set_facecolor(colors[i])
        
    for median in bp['medians']:
        median.set_visible(False)

    for i, m in enumerate(bp['means']):
        m.set_color('black')
        m.set_linestyle('--')
        m.set_linewidth(2)
        m.set_alpha(0.8)
        if i == 0:
          round_mean = [round(met, 4) for met in metrics_means]
          m.set_label(f'Metrics Means: {round_mean}')

    #draw dots
    if dots:
        for c, col in enumerate(df.columns):
            sorted_points = np.sort(df[col])

            #ensure the points are not overlapped
            for p, point in enumerate(sorted_points):
                if p == 0:
                    x_pos = c+1
                    counter = 1
                else:
                    if point == sorted_points[p-1]:
                        x_pos = c+1 + random.uniform(-0.2, 0.2)
                        counter = (-1)*counter if counter > 0 else (-1)*counter + 0.5

                    else:
                        x_pos = c+1
                        counter = 1

                label = f'i-th {fold_or_seed} Prediction' if c == 0 and p == 0 else None
                ax.scatter(x_pos, point, color='grey', label=label, s =15)

    ax.set_title(title, fontsize=13, pad=40)
    ax.set_ylabel('Score')
    ax.set_xlabel('Performance Metrics')
    ax.set_ylim(-0.05,1.05)
    ax.set_xticklabels(df.columns, fontsize=10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, legend_pos), 
              frameon=True, bbox_transform=ax.transAxes)

    if axis is None:
        plt.tight_layout()
        if save_dir is not None:
            if file_name is None:
                file_name = '2.1.dot_boxplot'

            plt.savefig(f'{save_dir}/{file_name}.pdf', format = 'pdf', bbox_inches='tight')
            plt.show()
            plt.close()


def show_ensemble_heatmap(ens_all_roc_metrics, axis = None, save_dir = None, mean = True, file_name = None, title = None, left_out_pat_list = None, figsize = (6,5)):
    """
    Plots a heatmap of classification metrics per fold or per patient,
    with optional mean aggregation across folds.
    """
    mean_across_folds = {'f1':[], 'rec': [], 'pre': [],  'acc': []}

    if title is None:
        title = 'Ensemble Boxpot Metrics'
    if mean:
        for fold in ens_all_roc_metrics:
            for met, values in fold.items():
                values = np.mean(values)
                mean_across_folds[met].append(values)
    else:
        mean_across_folds = ens_all_roc_metrics

    df = pd.DataFrame(mean_across_folds)

    if axis is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        ax = axis

    folds = len(df.index)

    if left_out_pat_list is None:
        heat_yticks = [f'Fold {i+1}' for i in range(folds)]
    else:
        heat_yticks = [f'Pat. {pat}' for pat in left_out_pat_list]

    sns.heatmap(df, annot=True, fmt='.3f', cmap='YlGnBu',
            yticklabels=heat_yticks, ax = ax)

    ax.set_title(title, fontsize=13, pad = 40)

    if axis is None:
        plt.tight_layout()
        if save_dir is not None:
            plt.savefig(f'{save_dir}/2.2.predictions_heatmap.pdf', format = 'pdf', bbox_inches='tight')
            plt.show()
            plt.close()


def show_dotbox_heat(samples_info_dict, save_dir = None, mean = True, dots = True, file_name = None,  title_1 = None, title_2 = None, sup_title = None, left_out_pat_list = None, sub_figsize = (6,5),
                     legend_pos = 1.15, fold_or_seed = 'Fold'):
    """
    Combines show_dot_boxplot_metrics and show_ensamble_heatmap into a single two-panel figure.
    Optionally saves as PDF.
    """ 

    w = sub_figsize[0]
    l = sub_figsize[1]
    fig, axs = plt.subplots(1, 2, figsize=(2*w, l+1))

    show_dot_boxplot_metrics(samples_info_dict, axis = axs[0], mean = mean, dots = dots, 
                      title = title_1, figsize = sub_figsize, legend_pos = legend_pos, fold_or_seed = fold_or_seed)
    show_ensemble_heatmap(samples_info_dict, axis = axs[1], mean = mean,  title = title_2, 
                      left_out_pat_list = left_out_pat_list, figsize = sub_figsize )

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    if sup_title is None:
        sup_title = 'Metrics from Ensemble Procedure'
    fig.suptitle(sup_title, fontsize = 15, y=1.05)

    if save_dir:
        if file_name is None:
            file_name = '2.dotbox_heat'

        fig.savefig(f'{save_dir}/{file_name}.pdf', format = 'pdf', bbox_inches='tight')

    plt.show()
    plt.close()

def show_dotbox_dotboxheat(samples_info_dict, all_dir_metrics_across_LOPO, fold_or_seed = 'Fold',
                           save_dir = None, mean = True, dots = True, file_name = None,  
                           title_1 = None, title_2 = None, title_3 = None, sup_title = None, 
                           left_out_pat_list = None, sub_figsize = (6,5), legend_pos = 1.15):
    """
    Combines two dot-boxplots and one heatmap into a single three-panel figure.
    The first panel shows LOPO-level metrics, the second ensemble metrics, the third a heatmap.
    """
  
    w = sub_figsize[0]
    l = sub_figsize[1]
    fig, axs = plt.subplots(1, 3, figsize=(3*w, l+1))

    show_dot_boxplot_metrics(all_dir_metrics_across_LOPO,  axis = axs[0], mean = mean, dots = dots, 
                          title = title_1, figsize = sub_figsize, legend_pos = legend_pos, fold_or_seed = fold_or_seed)
    show_dot_boxplot_metrics(samples_info_dict, axis = axs[1], mean = mean, dots = dots, title = title_2, 
                            figsize = sub_figsize, legend_pos = legend_pos, fold_or_seed = fold_or_seed)

    show_ensemble_heatmap(samples_info_dict, axis = axs[2], mean = mean,  title = title_3, left_out_pat_list = left_out_pat_list, figsize = sub_figsize )

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    if sup_title is None:
        sup_title = 'Metrics from Ensemble Procedure'
    fig.suptitle(sup_title, fontsize = 15, y=1.05)

    if save_dir:
        if file_name is None:
            file_name = '2.dotbox_dotbox_heat'

        fig.savefig(f'{save_dir}/{file_name}.pdf', format = 'pdf', bbox_inches='tight')

    plt.show()
    plt.close()


def show_dotbox_dotbox(samples_info_dict, all_dir_metrics_across_LOPO, 
                           save_dir = None, mean = True, dots = True, file_name = None,  
                           title_1 = None, title_2 = None, sup_title = None,
                           fold_or_seed = 'Fold', 
                           sub_figsize = (6,5), legend_pos = 1.15):
    """
    Combines two dot-boxplots into a single two-panel figure.
    The first panel shows LOPO-level metrics, the second ensemble metrics.
    """

    w = sub_figsize[0]
    l = sub_figsize[1]
    fig, axs = plt.subplots(1, 2, figsize=(2*w, l+1))

    show_dot_boxplot_metrics(all_dir_metrics_across_LOPO,  axis = axs[0], mean = mean, dots = dots, title = title_1, 
                            figsize = sub_figsize, legend_pos = legend_pos, fold_or_seed = fold_or_seed)
    
    show_dot_boxplot_metrics(samples_info_dict, axis = axs[1], mean = mean, dots = dots, title = title_2, 
                            figsize = sub_figsize, legend_pos = legend_pos, fold_or_seed = fold_or_seed)
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    if sup_title is None:
        sup_title = 'Metrics from Ensemble Procedure'
    fig.suptitle(sup_title, fontsize = 15, y=1.05)

    if save_dir:
        if file_name is None:
            file_name = '2.dotbox_dotbox'

        fig.savefig(f'{save_dir}/{file_name}.pdf', format = 'pdf', bbox_inches='tight')

    plt.show()
    plt.close()



def show_metrics_mean_std(tot_exp_mean_list, tot_exp_std_list, title=None, name_fig=None, save_dir=None, y_label = None):
    """
    Plots a grouped bar chart comparing mean metric scores with std error bars across experiments.
    Labels, colors and grouping are hardcoded for 4 experiments and 6 bar types.
    """
    x_labels = ['Exp 1\n(AS, Single-Split)', 'Exp 2\n(NO AS, Single-Split)', 'Exp 3\n(AS, Ensemble)', 'Exp 4\n(NO AS, Ensemble)']
    bar_labels = ['Seed/Fold-Level (0.5)', 'Single-Split/Ensemble (0.5)', 'Seed/Fold-Level (ROC)', 'Single-Split/Ensemble (ROC)', 'Seed/Fold-Level (RES)', 'Single-Split/Ensemble (RES)']
    colors = ['steelblue', 'skyblue', 'tomato', 'lightsalmon', 'mediumseagreen', 'lightgreen']

    n_groups = len(tot_exp_mean_list)
    n_bars = len(bar_labels)
    bar_width = 0.135
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(12, 5))

    for j in range(n_bars):
        values = [tot_exp_mean_list[i][j] for i in range(n_groups)]
        errors = [tot_exp_std_list[i][j]  for i in range(n_groups)]
        offset = (j - n_bars / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, values, width=bar_width,
                      color=colors[j], label=bar_labels[j],
                      edgecolor='black', linewidth=0.8,
                      yerr=errors, capsize=3,
                      error_kw=dict(ecolor='black', elinewidth=0.8, capthick=0.8))

        for bar, err in zip(bars, errors):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + err + 0.02,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=6, rotation=90)

    ax.set_ylim(0, 1.25)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=10)
    if y_label is None:
        y_label = 'Score'
    ax.set_ylabel(y_label, fontsize=11)

    if title is None:
        title = 'Metric across Experiments'

    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=8, loc='lower right', borderaxespad=1, framealpha = 1)
    plt.tight_layout()
    if save_dir is not None:
        if name_fig is None:
            name_fig = 'metric_across_experiments'
        plt.savefig(f'{save_dir}/{name_fig}.pdf', format = 'pdf')
    plt.show()



def elaborate_direct_prediction(imported_predictions, imported_test_y, metric = 'acc', pred = False, mcc = False):
    """
    Extracts a specific metric list from direct predictions, or returns raw predictions if pred=True.
    Maps metric name to its position in the output of elaborate_metrics.
    """
    if metric == 'acc':
        pos = 0
    elif metric == 'f1':
        pos = 1
    elif metric == 'rec':
        pos = 2
    elif metric == 'pre':
        pos = 3
    elif metric == 'mcc':
        pos = 4
        mcc = True

    labels_flat = flatten(imported_test_y)
    # elaborate metric data from imported predictions
    metrics = elaborate_metrics(imported_predictions, labels_flat, mcc = mcc)

    if pred:
        el_preds = elaborate_metrics(imported_predictions, labels_flat, pred = pred)

        return el_preds
    else:
        return metrics[pos] # extract the metric in the chosen position
    


def elaborate_metrics(imported_predictions, imported_test_y, results = False, pred = False, mcc = False):
    """
    Thin wrapper around elaborate_predictions that returns either the prediction DataFrame
    or the metric lists (excluding the DataFrame).
    """
    
    metric_lists = elaborate_predictions(imported_predictions, imported_test_y, results = results, mcc = mcc)
    
    if pred:
        new_pred_phenotype_df = metric_lists 
        return new_pred_phenotype_df
    else:
    
        return metric_lists[1:]





