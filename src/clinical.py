import os
import time

import optuna
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
import time

from src.shared.timepoints_elaboration import donation_extraction
from src.shared.labels import retrieve_labels

from src.sample_level.bayesian_tuning import BO_tuning
from src.sample_level.training import fit
from src.sample_level.testing import predict, trials_test_BinCellCNN
from src.sample_level.configs import (
    build_tuning_config, build_training_config, get_param_list_dict)
from src.shared.utils import save_element

from seed_manager import SeedManager
from memory_utils import clear_RAM

from src.shared.cv_folds import remove_test_patients_from_categories, classic_CV_train_val_splits
from src.shared.timepoints_elaboration import donor_division
from src.shared.utils import load_element


def generate_clinical_folds(multiple_donations, ALL_DATASETS = None, mem = None, seed_manager = None, folds = 5, label_column = 'IsBlast'):
    """
    Builds the full LOPOCV structure: for each left-out patient, generates the inner
    k-fold CV splits for hyperparameter tuning.

    """

    if mem:
        (ALL_DATASETS,) = mem.load_variables(['ALL_DATASETS'])

    if seed_manager is None:
            CV_seed = 42
    else:
            CV_seed = seed_manager.get_seed(f'Patient_division')
    print('CV_seed', CV_seed)
    healthy_donors, blast_donors, mixed_donors = donor_division(multiple_donations, ALL_DATASETS, label_column = label_column)

    # defined the External validation set, remove the patients from the categories
    h_pat, b_pat, m_pat, tot_pat = remove_test_patients_from_categories(healthy_donors, blast_donors, mixed_donors)

    # remove the Left Out patient and generate the folds for the 5 folds CV
    k_folds = classic_CV_train_val_splits(h_pat, b_pat, m_pat, tot_pat, folds = folds, shuffle_seed = CV_seed)

    clear_RAM(['ALL_DATASETS'])
    return k_folds
   
def tuning_and_train_model(BinCellCnn, config, experiment_path, train_kfolds, multiple_donations, ALL_DATASETS, mem = None, seed_manager = None):
    
    # --- Seed parameters ---
    #starting_seed = config['starting_seed']

    param_list_dict = get_param_list_dict(config)

    print(param_list_dict)
    
    full_process_time_start = time.time()

    tuning_save_dir = f'{experiment_path}/tuning/'
    os.makedirs(tuning_save_dir, exist_ok=True)
    weights_outdir = f'{tuning_save_dir}/weights'
    os.makedirs(weights_outdir, exist_ok=True)

    ## === load the samples === ##

    tuning_config = build_tuning_config(config)

    best_ncells, best_nsubs, best_bs, _ = BO_tuning(
            LOPOCV_fold_idx=0,
            train_kfolds=train_kfolds,
            param_list_dict=param_list_dict,
            BinCellCnn=BinCellCnn,
            multiple_donations=multiple_donations,
            ALL_DATASETS=ALL_DATASETS,
            outdir=weights_outdir,
            mem=mem,
            seed_manager=seed_manager,
            tuning_config = tuning_config,
            tuning_save_dir = tuning_save_dir
        )

        # save seeds and timing
    bayesian_LOPOCV_time = time.time() - full_process_time_start
    print(f'bayesian_LOPOCV_time: {bayesian_LOPOCV_time}')


    # === Training === #
    print(best_bs)
    training_config = build_training_config(config, best_ncells, best_nsubs, best_bs)

    print(f'Theta*: {best_ncells, best_nsubs, best_bs}')

    training_save_dir = f'{experiment_path}/training/'
    os.makedirs(training_save_dir, exist_ok=True)

    weights_outdir = f'{training_save_dir}/weights'
    os.makedirs(weights_outdir, exist_ok=True)

    CV_models, best_thr = fit(multiple_donations = multiple_donations,
                                               ALL_DATASETS = ALL_DATASETS,
                                               BinCellCnn = BinCellCnn,
                                               config = training_config,
                                               mem = mem,
                                               seed_manager = seed_manager,
                                               save_dir = training_save_dir,
                                               outdir = weights_outdir,
                                               train_kfolds = train_kfolds)
    fold_end = time.time()
    print(f'\n Time elapsed process started: {fold_end - full_process_time_start}\n')

    seed_manager.save_state(experiment_path)

    return CV_models, best_thr



def clinical_predict(new_unlabelled_samples, training_save_dir, test_save_dir, seed_manager = None):
            # Predict and Classify test samples

            CV_models = load_element(training_save_dir, 'CV_models')
            test_samples_preds = []

            for models_list in CV_models:
                w_init_preds_per_sample_list, original_results_list = trials_test_BinCellCNN(models_list, new_unlabelled_samples, mem = None, seed_manager = seed_manager)
                print('w_init_preds_per_sample_list', w_init_preds_per_sample_list)

                pos_preds_per_sample = np.mean(w_init_preds_per_sample_list, axis =0)

                test_samples_preds.append(pos_preds_per_sample)

            save_element(test_save_dir, 'test_samples_preds', test_samples_preds)


            return test_samples_preds


def classify_new_samples(experiment_path, new_unlabelled_samples, samples_ids, seed_manager = None):
    
    training_save_dir = f'{experiment_path}/training/'
    classification_save_dir = f'{experiment_path}/classification_results/'

    
    CV_models = load_element(training_save_dir, 'CV_models')
    best_thr = load_element(training_save_dir, 'best_thr')

    # === Testing === #

    test_samples_preds = clinical_predict(new_unlabelled_samples = new_unlabelled_samples,
                                                  training_save_dir = training_save_dir,
                                                  test_save_dir = classification_save_dir,
                                                  seed_manager = seed_manager)

    print('test_samples_preds', test_samples_preds)

    seed_manager.save_state(experiment_path)
    
    
    plot_data = prepare_boxplot_data(classification_save_dir, test_samples_preds,  best_thr = best_thr, samples_ids = samples_ids)
    print('plot_data', plot_data)
    show_clinical_boxplots(plot_data = plot_data,
                          save_dir = classification_save_dir, file_name = 'Positivity_Score_boxplots.pdf',
                          title = None,
                          fold_or_seed = 'Fold')
    
    return test_samples_preds, best_thr


def _temp_test_extraction(test_pat, multiple_donations, ALL_DATASETS = None, mem = None):
    (ALL_DATASETS,) = mem.load_variables(['ALL_DATASETS'])

    test_datasets_extracted = donation_extraction(test_pat, multiple_donations, ALL_DATASETS = ALL_DATASETS)
    original_test_datasets, original_test_y = retrieve_labels(test_datasets_extracted, remove=True, flat=True)
    return original_test_datasets, original_test_y

import seaborn as sns


def show_clinical_boxplots(plot_data,
                        save_dir = None, file_name = None, title = None, fold_or_seed = 'Fold'):
    """
    Plots per-sample score distributions as boxplots across all LOPO folds, with per-fold
    threshold lines, blast percentage on the top axis, and misclassification markers.
    """
    sns.set_style("ticks")


    boxplot_data = plot_data['boxplot_data']
    thresholds_list  = plot_data['thr_list']
    num_samples_cum = plot_data['num_samples_cum']
    y_predicted = plot_data['y_predicted']
    ID_labels = plot_data['samples_ID']
    
    # timepoints_labels = [timepoint["True_Label"] for i, timepoint in enumerate(boxplot_data)]
    timepoints_trials_scores = [timepoint["CV_Fold_Preds"] for timepoint in boxplot_data]

    if ID_labels:
        x_axis_1_labels = ID_labels[:len(y_predicted)]


    n_box = len(timepoints_trials_scores)
    fig, ax = plt.subplots(figsize=(max(0.5*n_box, 4), 4))

    print('timepoints_trials_scores', timepoints_trials_scores)
    print('x_axis_1_labels', ID_labels)
    # Plot the boxplot on the first axes (ax1)
    bp = ax.boxplot(timepoints_trials_scores, tick_labels =  ID_labels, patch_artist=True,
                                                  showmeans=True,      # shows the mean
                                                  meanline=True, showfliers=False)
    
    ax.set_xticklabels(ID_labels, rotate = 60)
    
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

    ax.set_xlabel("Samples ID")
    ax.set_ylabel("Positivity Score")

    ax2 = ax.twiny()

    # Make ax2 have the same limits and ticks as ax1
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(ax.get_xticks()) # Positions [1, 2, 3, 4]

    print(len(ax.get_xticks()))
    print(len(y_predicted), y_predicted)
    
    # Set the new labels for the top axis
    ax2.set_xticklabels(y_predicted, fontsize=7)
    ax2.set_xlabel("Predicted Labels")

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

    # default threshold = 0.5
    ax.hlines(y=0.5, xmin=0.5, xmax=x_max, color='black',
               linestyles='dotted', label='Default Threshold = 0.5', linewidth=1.5, alpha = 0.3)
    
    plt.ylim(-0.05, 1.05)
    if title is  None:
        title = "Positivity Score Distribution across samples"

    plt.title(title)
    ax.legend()
    if save_dir:
        if file_name is None:
            file_name = '5.2.clinical_boxplots.pdf'
        fig.savefig(f'{save_dir}/{file_name}', format = 'pdf', bbox_inches='tight')

    plt.show()
    plt.close()


def prepare_boxplot_data(results_save_dir, test_samples_preds, best_thr = None, samples_ids = None):

    per_samples_preds = np.array(test_samples_preds).T.tolist()

    if best_thr is None:
        best_thr = 0.5
    if samples_ids is None:
        samples_ids = list(range(len(per_samples_preds)))

    print(per_samples_preds)
    per_samples_mean_preds = np.mean(per_samples_preds, axis = 1)
    print(per_samples_mean_preds)
    y_predicted = (per_samples_mean_preds > best_thr).astype(int)
    print(y_predicted)

    num_samples_cum = [len(per_samples_preds)]
    thr_list = [best_thr]
    boxplot_data = []
    for fold_samples in per_samples_preds:
        box_data = {'CV_Fold_Preds': fold_samples}

        boxplot_data.append(box_data)



    plot_data = {'boxplot_data': boxplot_data, 
                  'thr_list': thr_list, 
                  'num_samples_cum': num_samples_cum,
                  'samples_ID': samples_ids,
                  'y_predicted': y_predicted}

    with open(f'{results_save_dir}/plot_data.pkl', 'wb') as f:
        pickle.dump(plot_data, f)

    return plot_data

