'''
from training import run_training, val_res_pred, train_val_finalizing, test_res_pred, find_theta_best
'''

import pandas as pd
import numpy as np

from utils import flatten, remove_labels, subset_sampling, generate_seeds, sub_resampling_list
from run_models import trials_train_CellCNN_old, trials_test_CellCNN_old



def run_training(CellCnn, new_train_datasets, new_train_y, new_val_datasets,
                     new_val_y, seed_list, hyper,
                                 grid = True, labels = False, trials = 1, cells_per_sub = 200, best_nsub = 200,
                                 max_epochs = 100, nrun = 15, generate = False, no_val = False, outdir = None):
    """
    Prepares datasets and runs CellCNN training, depending on on no_val and generate (validation set) flags.
    """
    if no_val:
            train = train_val_finalizing(new_train_datasets, new_val_datasets, grid, labels, no_val)
    else:
            train, val = train_val_finalizing(new_train_datasets, new_val_datasets, grid, labels, no_val)

    if no_val and not generate:
            models_lists = trials_train_CellCNN_old(CellCnn, train,
                                          new_train_y,
                                          trials = trials,
                                          n_cell = cells_per_sub, nsubset = best_nsub,
                                          max_epochs= max_epochs, ### 100,
                                          nrun= 15, ### 15,
                                          seed_list = seed_list, hyper = hyper,
                                          generate = generate, grid = grid, outdir = outdir)
    elif generate:
            models_lists = trials_train_CellCNN_old(CellCnn, train,
                                          new_train_y,
                                          trials = trials,
                                          n_cell = cells_per_sub, nsubset = best_nsub,
                                          max_epochs= max_epochs, ### 100,
                                          nrun= 15, ### 15,
                                          seed_list = seed_list, hyper = hyper,
                                            generate = generate, grid = grid, outdir = outdir)
    else:
            models_lists = trials_train_CellCNN_old(CellCnn, train, new_train_y,
                                              val_datasets = val, val_y = new_val_y,
                                              trials = trials,
                                              n_cell = cells_per_sub, nsubset = best_nsub,
                                              max_epochs= max_epochs, ### 100,
                                              nrun= 15, ### 15,
                                              seed_list = seed_list, hyper = hyper,
                                              generate = generate, grid = grid, outdir = outdir)

    return models_lists



def val_res_pred(models_lists, per_donor_original_val_datasets, n, k, seed):
    """
    For each patient file, generates k drawn subsets of n cells and runs predictions
    across all seeds. Returns mean probabilities and per-trial probabilities.
    """
    mean_probs_per_patient = []
    total_pred_lists = []
    total_trial_pred_lists = []

    counter = 1
    for patient in per_donor_original_val_datasets:

        timepoints_mean_probs = []
        patient_pred_list = []
        patient_trial_pred_list = []
        for file in patient:
            trials = len(models_lists)
            thr_res_seed_list = generate_seeds(trials, seed = seed + counter)

            # for each dataset, multiple subsets of n cells are resampled
            resampled_datasets,  resampled_y , _, seed =  subset_sampling(file, ncells = n, nsubsets = k, seed = thr_res_seed_list[0])

            counter += 1
            new_datasets_predictions_list, _ = trials_test_CellCNN_old(models_lists, resampled_datasets, thr_res_seed_list)
            positive_probs = []

            for trial in new_datasets_predictions_list:
                positive_probs.append(pd.DataFrame(trial)[1]) # appends the probability of positive classification

            all_trials_probs_array = np.array(positive_probs) #it converts the list of (list of) probabilities into an array of lists of probs

            # Computes the mean over the columns ( it takes the first element of all arrays and make the mean, then the second and so on)
            positive_probs_mean = np.mean(all_trials_probs_array, axis=0).tolist()

            patient_pred_list.append(positive_probs_mean) # stores the mean subset probabilities
            patient_trial_pred_list.append(positive_probs) # stores all probabilities af all trials

            ## assigns mean probabilities to its true resampled subsets labels
            timepoints_mean_probs.append((positive_probs_mean, resampled_y))

        mean_probs_per_patient.append(timepoints_mean_probs)

        total_pred_lists.append(patient_pred_list)
        total_trial_pred_lists.append(patient_trial_pred_list)


    return total_pred_lists, total_trial_pred_lists, mean_probs_per_patient





def train_val_finalizing(train_datasets, val_datasets, grid, labels, no_val = False):
    """
    Prepares train and validation datasets by removing labels based on the search mode.
    """
    
    if grid:
        if not no_val:
            val = remove_labels(val_datasets)
        train = remove_labels(train_datasets)

        print(f'Labels from Train and Validations Sets have been REMOVED.\nGrid Search Ready!')
    else:
        if not labels: # if random search but we don't want use labels
            if not no_val:
                val = remove_labels(val_datasets)
            train = remove_labels(train_datasets)

            print(f'Labels from Train and Validations Sets have been REMOVED.\nRandom Search Ready!')

        else:
            if not no_val:
                val = remove_labels(val_datasets)
            else:
                val = val_datasets
            train = train_datasets
            print(f'Labels from Train and Validations Sets have been PRESERVED.\nRandom Search Ready!')
    if not no_val:
        return train, val
    else:
        return train


def test_res_pred(models_lists, per_donor_original_test_datasets, n, k, best_threshold, trials, seed):
    """
    Runs resampled predictions on test datasets. Subsets are generated in chunks via sub_resampling_list,
    predictions aggregated across seeds, and a binary label assigned per file via best_threshold.
    """
    counter = 1

    per_donor_resampled_test_y = []
    test_total_labels = []
    test_total_pred_lists = []
    test_total_trial_pred_lists = []

    for patient in per_donor_original_test_datasets:
        per_dataset_resampled_ys = []
        patient_timepoints_labels = []
        patient_pred_list = []
        patient_trial_pred_list = []

        for file in patient:
            sub_division = sub_resampling_list(k, nsub_per_sub = 50)
            total_resampled_y = []
            total_positive_probs = [[] for _ in range(trials)]
            division_positive_probs_mean = []

            rob_res_seed_list = generate_seeds(trials + 1, seed = seed + counter)
            counter += 1
            for _, division in enumerate(sub_division):
                resampled_datasets, resampled_y, blast_perc, seed = subset_sampling(dataset = file, ncells = n, nsubsets = division, seed = rob_res_seed_list[0])

                # predict labels
                new_datasets_predictions_list, _ = trials_test_CellCNN_old(models_lists, resampled_datasets, rob_res_seed_list[1:])
                positive_probs = []

                # Extraction and mean probabilities section 
                for iterat, trial in enumerate(new_datasets_predictions_list):
                    positive_probs.append(pd.DataFrame(trial)[1].values) # appends the probability of positive classification
                    total_positive_probs[iterat] += list(pd.DataFrame(trial)[1].values)

                # Computes the mean over the columns ( it takes the first element of all arrays and make the mean, then the second and so on)
                division_positive_probs_mean.append(np.mean(positive_probs , axis=0))

                total_resampled_y += resampled_y

            positive_probs_mean = np.concatenate(division_positive_probs_mean)

            # Threshold prediction section 
            mean_timepoint_probs = np.mean(positive_probs_mean) # mean percentage of timepoints labelled as with blast cells

            if mean_timepoint_probs >= best_threshold*0.01:
                    patient_timepoints_labels.append(1)
            else:
                    patient_timepoints_labels.append(0)

            patient_pred_list.append(positive_probs_mean) # stores the mean subset probabilities
            patient_trial_pred_list.append(total_positive_probs) # stores all probabilities af all trials

            per_dataset_resampled_ys.append(total_resampled_y)

        test_total_labels.append(patient_timepoints_labels)
        test_total_pred_lists.append(patient_pred_list)

        test_total_trial_pred_lists.append(patient_trial_pred_list)
        per_donor_resampled_test_y.append(per_dataset_resampled_ys)

    return test_total_labels, test_total_pred_lists, test_total_trial_pred_lists, per_donor_resampled_test_y




def find_theta_best(f1_tested_par, tested_par):
    """
    Finds the best parameter combination by selecting the configuration with the highest F--score.
    If multiple values scores the max(f1), the median of the top-scoring
    combinations, sorted by their parameter sum is chosen
    """
    best_combinations = []
    max_f1 = np.max(f1_tested_par)
    for i, f1_par in enumerate(f1_tested_par):
        if f1_par == max_f1:
            best_combinations.append(tested_par[i])


    sorted_sum_best_combinations = [np.sum(par) for par in best_combinations]
    sort_idx = np.argsort(sorted_sum_best_combinations)
    sorted_best_combinations = np.array(best_combinations)[sort_idx].tolist()
            

    if len(sorted_best_combinations) == 1:
        chosen_par = sorted_best_combinations[0]

    else:  #if even number
        middle_position = int(len(sorted_best_combinations)/2)
        print(f'Position in the middle of best cobinations: {middle_position}')
        half_sorted_best_combinations = sorted_best_combinations[-middle_position:]
        print(f'Remaining half best comvinations: {half_sorted_best_combinations}')

        if len(half_sorted_best_combinations) %2:
            chosen_par = half_sorted_best_combinations[int(len(half_sorted_best_combinations)/2)]
        else:
            new_middle_position = max(1,int(len(half_sorted_best_combinations)/2))
            chosen_par = half_sorted_best_combinations[new_middle_position]

    chosen_par = (chosen_par[0], chosen_par[1])

    return chosen_par


