'''
from function import retrieve_labels, prepare_results_to_save, train_val_finalizing, grid_or_rand_model 
from function import remove_labels, show_blast_distribution_num, show_blast_distribution_perc
from function importval_res_pred, find_best_nsubs, retrieve_nsub_models_names, final_trials_prediction
from function import chosen_folds, CV_train_val_splits, flatten
from function import find_threshold, compute_timepoint_best_f1, get_trial_pred_per_timepoint
from function import get_timepoints_predictions, subset_sampling, generate_seeds, LOOCV_division
from function import remove_test_patients_from_categories, test_res_pred, sub_resampling_list, find_robust_threshold
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from itertools import zip_longest

from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score 
from sklearn.metrics import matthews_corrcoef
from timepoints_elaboration import donor_division, patient_code_extraction, remove_from_cache
from run_models import trials_train_CellCNN_old, trials_test_CellCNN_old


"==============================================================================="
# CV FUNCTIONS
# classical CV 3 fold
def remove_test_patients_from_categories(healthy_donors, blast_donors, mixed_donors,  test_donors_idx):
    """
    Removes test patients from the three patients categories and generates a list of the remaining patients (train + validation).

    Input:
        healthy_donors   (list): healthy patients
        blast_donors     (list): blast patients
        mixed_donors     (list): mixed patients
        test_donors_idx  (list): indices of patients to remove

    Output:
        h_pat   (list): healthy donors without test patients
        b_pat   (list): blast donors without test patients
        m_pat   (list): mixed donors without test patients
        tot_pat (list): all remaining patients, interleaved by category
    """
    h_pat, b_pat, m_pat = healthy_donors.copy(), blast_donors.copy(), mixed_donors.copy()

    # remove patients in the external test set from the three patient categories
    for pat in test_donors_idx:
        if pat in healthy_donors:
            h_pat.remove(pat)
        if pat in blast_donors:
             b_pat.remove(pat)
        if pat in mixed_donors:
            m_pat.remove(pat)

    # generate the entire list of patients to split
    tot_pat = []
    for pat_1, pat_2, pat_3 in zip_longest(h_pat, b_pat, m_pat, fillvalue=None):
        if pat_1 is not None:
            tot_pat.append(pat_1)
        if pat_2 is not None:
            tot_pat.append(pat_2)
        if pat_3 is not None:
            tot_pat.append(pat_3)

    print(f'Remaining Patients: healthy = {h_pat}, unhealthy = {b_pat}, mixed = {m_pat}')
    print(f'All Patients in Train + Validation set: {tot_pat}')

    return h_pat, b_pat, m_pat, tot_pat


'''
def LOOCV_division(tot_pat):
    LOOCV_folds = []
    for i, fold in enumerate(tot_pat):
        test = [fold]
        train = [sub_fold for sub_fold in tot_pat if fold != sub_fold]
        LOOCV_folds.append((train, test))
        print(train, test)
    print('')
    return LOOCV_folds
'''


def classic_CV_train_val_splits(h_pat, b_pat, m_pat, tot_pat, folds = 3, LOOCV_test = None, shuffle_seed = None):
    """
    Generates train/validation splits for cross-validation.
    Optionally removes a single LOOCV patient before splitting, and supports reproducible shuffling.

    Input:
        h_pat         (list): healthy patients
        b_pat         (list): blast patients
        m_pat         (list): mixed patients
        tot_pat       (list): all patients
        folds         (int) : number of CV folds (default: 3)
        LOOCV_test    (any) : single patient to exclude before splitting (default: None)
        shuffle_seed  (int) : seed for reproducible shuffling (default: None)

    Output:
        tot_folds (list of [list, list]): each element is a [train_fold, val_fold] pair
    """
    # if present, remove the test patient from h,b,m sets
    if LOOCV_test is not None:
        if str(LOOCV_test) in h_pat: #healthy
            h_pat = [pat for pat in h_pat if pat != str(LOOCV_test)]
        elif str(LOOCV_test) in b_pat: #unhealthy
            b_pat = [pat for pat in b_pat if pat != str(LOOCV_test)]
        elif str(LOOCV_test) in m_pat: #mixed
            m_pat = [pat for pat in m_pat if pat != str(LOOCV_test)]

        tot_pat = []
        for pat_1, pat_2, pat_3 in zip_longest(h_pat, b_pat, m_pat, fillvalue=None):
            if pat_1 is not None:
                tot_pat.append(pat_1)
            if pat_2 is not None:
                tot_pat.append(pat_2)
            if pat_3 is not None:
                tot_pat.append(pat_3)

    if shuffle_seed is not None:
        np.random.seed(shuffle_seed)
        np.random.shuffle(tot_pat)
    # store the three different patietns' folds
    single_folds = []
    counter = 0
    len_counter = 0
    for fold in range(folds - 1):
        len_counter += len(tot_pat)/folds
        diff =  int(len_counter - counter)
        single_folds.append(tot_pat[counter: counter + diff])
        counter += diff
    single_folds.append(tot_pat[counter:])
    print(f'Folds: {single_folds}')

    # finally make the n train-validation combinations
    tot_folds = []
    for n_fold in range(folds):
        val_fold = single_folds[n_fold - 1]
        train_fold = []
        for fold in single_folds:
            if fold != val_fold:
                train_fold += fold

        print(f'Combination {n_fold + 1}: {train_fold, val_fold}')
        tot_folds.append([train_fold, val_fold])
    return tot_folds



def extract_fold_features(chosen_perm, multiple_donations, tot_perc_list):
    """
    Extracts and aggregates file-level features for each train/validation fold.

    Input:
        chosen_perm        (list): list of [train_set, val_set] pairs (output of CV split)
        multiple_donations (dict): maps patient id and patient's samples
        tot_perc_list      (list): percentages of unhealthy cells or each file

    Output:
        fold_features (dict): maps fold index -> [train_features, val_features]
                              each features entry: [patient_set, file_percentages, mean, std]
    """

    fold_features = {}
    for i, (train_set, val_set) in enumerate(chosen_perm):

        ##### Train #####
        # extract files from patients' id
        set_train_files, set_val_files = [], []
        for train_pat in train_set:
            train_files = multiple_donations[train_pat] # indexes of files in ALL_DATASETS list
            set_train_files += train_files 

        train_file_perc = list(np.array(tot_perc_list)[set_train_files]) # extracts only the interested file pecentages
        train_features = [train_set, train_file_perc, np.mean(train_file_perc), np.std(train_file_perc)]

        ##### Validation #####
        for val_pat in val_set:
            val_files = multiple_donations[val_pat]
            set_val_files += val_files

        val_file_perc = list(np.array(tot_perc_list)[set_val_files])
        val_features = [val_set, val_file_perc, np.mean(val_file_perc), np.std(val_file_perc)]

        # assignes train and validation features to the i-th fold combination
        fold_features[i] = [train_features, val_features]
    return fold_features

def generate_LOPOCV_dicts(multiple_donations, ALL_DATASETS = None):
    """
    Generates Leave-One-Patient-Out Cross-Validation (LOPOCV) fold pairs.
    For each patient, creates a (train_dict, test_dict) tuple where the patient is the test set.
    Optionally computes blast percentages for the test patient's samples.

    Input:
        multiple_donations (dict): maps patient id -> list of file indices
        ALL_DATASETS       (list, optional): list of sample dataframes/objects with an 'IsBlast' column

    Output:
        full_LOPOCV_dicts (list of tuples): each element is (train_dict, test_dict),
                                            both in the same format as multiple_donations
    """
    
    full_LOPOCV_dicts = []
    for patient in multiple_donations.keys():#
        prov_multiple = multiple_donations.copy()
        prov_multiple.pop(patient) # remove patient

        test = {} #store patient
        test[patient] = multiple_donations[patient]
        if ALL_DATASETS is not None:
            percs = []
            for sample_idx in multiple_donations[patient]:
                sample = ALL_DATASETS[sample_idx]
                p = round(((sample['IsBlast'] == 1).sum()/len(sample))*100, 5) # perc 
                percs.append(p)
            print(prov_multiple.keys(), test.keys(), percs)

        full_LOPOCV_dicts.append((prov_multiple, test)) # append folds
    return full_LOPOCV_dicts



def generate_LOPOCV_folds(full_LOPOCV_dicts, ALL_DATASETS, starting_seed):
    """
    Builds the full LOPOCV structure: for each left-out patient, generates the inner
    k-fold CV splits for hyperparameter tuning.

    Input:
        full_LOPOCV_dicts (list of tuples): output of generate_LOPOCV_dicts,
                                            each element is (train_dict, test_dict)
        ALL_DATASETS      (list): list of sample dataframes/objects
        starting_seed     (int): base seed for shuffling; each fold uses starting_seed + 10*i

    Output:
        LOPOCV_patients_folds (list of tuples): each element is (k_folds, left_out_patient),
                                                where k_folds is the inner CV split
                                                and left_out_patient
    """

    LOPOCV_patients_folds = []
    for i, (train_set_dict, external_val_set_dict) in enumerate(full_LOPOCV_dicts):
        CV_seed = starting_seed + 10*i
        healthy_donors, blast_donors, mixed_donors = donor_division(train_set_dict, ALL_DATASETS)

        left_out_patient = list(external_val_set_dict.keys())

        # defined the External validation set, remove the patients from the categories
        h_pat, b_pat, m_pat, tot_pat = remove_test_patients_from_categories(healthy_donors, blast_donors, mixed_donors, left_out_patient)

        # remove the Left Out patient and generate the folds for the 5 folds CV
        k_folds = classic_CV_train_val_splits(h_pat, b_pat, m_pat, tot_pat, folds = 5, LOOCV_test = left_out_patient[0], shuffle_seed = CV_seed)

        # k_folds is for the tuning. left_out_patient for the final training
        LOPOCV_patients_folds.append((k_folds, left_out_patient))
    return LOPOCV_patients_folds


"==============================================================================="




def robust_prediction_labelling(trials_pred_lists, threshold, pred = False):
    """
    Assigns binary labels to each sample based on averaged prediction probabilities across trials and samples.

    Input:
        trials_pred_lists (list): nested list [patient][sample][trial_probs],
                                  where trial_probs is an array of probabilities across seeds/subsets
        threshold         (float): classification threshold expressed as a percentage (e.g. 50 = 50%)
        pred              (bool) : if True, also returns the raw mean probabilities (default: False)

    Output:
        patient_timepoints_labels (list): binary labels per sample per patient
        patient_timepoints_preds  (list): mean probabilities per sample per patient (only if pred=True)
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




def retireve_sorted_pat_sample_ids(samples_info_dict):
    pat_ids = samples_info_dict['patient_id'].to_list()
    int_pat_ids = [int(i) for i in pat_ids]

    pat_sample_ids = []
    for i in range(1, np.max(int_pat_ids) + 1):

      if str(i) in pat_ids:
          df = samples_info_dict[samples_info_dict['patient_id'] == str(i)]
          df = df.sort_index()
          for s in range(df.shape[0]):
              time_point = df.iloc[s]['time_point_days']
              pat_sample_ids.append(f'Pat_{i}_{time_point}')

    return pat_sample_ids

"=============================================================================="
#utils
def remove_labels(new_test_datasets):
    """
    Removes the 'IsBlast' column from each dataset, if present.

    Input:
        new_test_datasets (list): list of DataFrames

    Output:
        new_no_label_test_datasets (list): same DataFrames without the 'IsBlast' column
    """
    new_no_label_test_datasets = []
    for dataset in new_test_datasets:
        if 'IsBlast' in dataset.columns:
            dataset = dataset.drop(columns = ['IsBlast'])

        new_no_label_test_datasets.append(dataset)
    return new_no_label_test_datasets

def show_blast_distribution_perc(ALL_DATASETS, multiple_donations, return_perc = False, log = False):
    """
    Computes and shows the blast cell percentage for each sample across all patients.
    Optionally returns the percentage list.

    Input:
        ALL_DATASETS       (list): list of sample DataFrames with an 'IsBlast' column
        multiple_donations (dict): maps patient id -> list of sample indices (in ALL_DATASETS)
        return_perc        (bool): if True, returns the percentage list (default: False)
        log                (bool): if True, prints per-sample details (default: False)

    Output:
        tot_perc_list (list): blast percentages per sample (only if return_perc=True)
    """
    tot_perc_list = []
    for pat_idx, samples_idx in multiple_donations.items():
        if log:
            print(samples_idx)
        for sample in samples_idx:

            dataset = ALL_DATASETS[sample]
            blast_n = (dataset['IsBlast'] == 1).sum()
            tot_perc_list.append(round((blast_n/len(dataset))*100, 2))
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

def show_blast_distribution_num(ALL_DATASETS, tot_num_list, return_num = False):
    tot_perc_list = []
    for i, dataset in enumerate(ALL_DATASETS):
        blast_n = (dataset['IsBlast'] == 1).sum()
        tot_num_list.append(blast_n)

    positions = range(1, len(tot_num_list) + 1)

    fig, ax1 = plt.subplots(figsize = [len(positions)/2,4])
    ax1.bar(list(range(1, len(tot_num_list) + 1)), [max(tot_num_list)]*len(tot_num_list), alpha = 0.5)
    ax1.bar(list(range(1, len(tot_num_list) + 1)), tot_num_list)
    ax1.set_xticks(positions)

    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(ax1.get_xticks()) # Positions [1, 2, 3, 4]

    ax2.set_xticklabels(tot_num_list, size = 'x-small')
    plt.show()

    if return_num:
        return tot_num_list


def train_val_finalizing(train_datasets, val_datasets, grid, labels, no_val = False):
    """
    Prepares train and validation datasets by removing labels based on the search mode.

    Input:
        train_datasets (list): list of train DataFrames
        val_datasets   (list): list of validation DataFrames
        grid           (bool): if True, runs in grid search mode (labels always removed)
        labels         (bool): if True, preserves labels in random search mode
        no_val         (bool): if True, skips validation set processing (default: False)

    Output:
        train         (list): processed train datasets
        val           (list): processed validation datasets (only if no_val=False)
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

def prepare_results_to_save(results_list, par_list = ['config', 'model_sorted_idx']):
    """
    Filters a list of result dictionaries, keeping only the specified keys.

    Input:
        results_list (list of dict): full result entries
        par_list     (list)        : keys to keep in each entry (default: ['config', 'model_sorted_idx'])

    Output:
        tot_trials_results (list of dict): filtered result entries containing only the keys in par_list
    """
    tot_trials_results = []

    for res in results_list:
        needed_results = {}
        for key, value in res.items():
            if key in par_list:
                needed_results[key] = value

        tot_trials_results.append(needed_results)
    return tot_trials_results

def retrieve_labels(datasets_extracted, remove = False, flat = False):
    """
    Extracts binary labels from the 'IsBlast' column of each dataset.
    Optionally removes the label column or flattens the output.

    Input:
        datasets_extracted (list of list): nested list [donor][sample] of DataFrames with 'IsBlast' column
        remove             (bool): if True, drops 'IsBlast' column from each DataFrame (default: False)
        flat               (bool): if True, applies flatten() to both output lists (default: False)

    Output:
        per_donor_original_datasets (list): DataFrames, nested by donor or flat depending on flat flag
        per_donor_original_y        (list): binary labels (1 if any blast present, 0 otherwise),
                                            same structure as per_donor_original_datasets
    """
    per_donor_original_datasets = []
    per_donor_original_y = []

    for donor in datasets_extracted:
        donor_datasets = []
        donor_ys = []
        for dataset in donor:
            if (dataset['IsBlast'] == 1).sum() > 0:
                donor_ys.append(1)
            else:
                donor_ys.append(0)

            if remove:
                dataset = dataset.drop(columns = ['IsBlast'])

            donor_datasets.append(dataset)

        per_donor_original_datasets.append(donor_datasets)
        per_donor_original_y.append(donor_ys)


    if flat:
        per_donor_original_datasets = flatten(per_donor_original_datasets)
        per_donor_original_y = flatten(per_donor_original_y)

    return per_donor_original_datasets, per_donor_original_y


def subset_sampling(dataset, ncells, nsubsets, seed):
    """
    Generates multiple random subsets (with replacement) from a single dataset.

    Input:
        dataset  (DataFrame): input data with an 'IsBlast' column
        ncells   (int)      : number of cells to sample per subset
        nsubsets (int)      : number of subsets to generate
        seed     (int)      : starting seed; incremented by 10 at each iteration

    Output:
        resampled_datasets (list): sampled DataFrames without 'IsBlast' column
        resampled_y        (list): binary label per subset (1 if any blast present, 0 otherwise)
        blast_perc         (list): blast percentage per subset
        seed               (int) : last seed used
    """
    
    resampled_datasets = []
    resampled_y = []
    blast_perc = []

    for i in range(nsubsets):
        seed += 10
        #print(f'seed:{seed}')
        resampled_cells = dataset.sample(ncells, replace = True, random_state = seed).reset_index(drop=True) # sample cells

        if (resampled_cells['IsBlast'] == 1).sum() > 0: #check label
            resampled_y.append(1)
            blast_perc.append((resampled_cells['IsBlast'] == 1).sum() / len(resampled_cells))
        else:
            resampled_y.append(0)
            blast_perc.append(0)
        resampled_cells = resampled_cells.drop(columns = ['IsBlast']) #remove isblast column

        resampled_datasets.append(resampled_cells)
    return    resampled_datasets,  resampled_y , blast_perc, seed


def flatten(nested):
    """
    Recursively flattens a nested list or tuple into a flat list.

    Input:
        nested (list | tuple | any): arbitrarily nested structure, or a single value, or None

    Output:
        result (list): flat list of all non-list/tuple elements
    """
    
    if nested is None:
        return []
    if not isinstance(nested, (list, tuple)):
        return [nested]
    nested = list(nested)
    result = []
    for item in nested:
        result.extend(flatten(item))
    return result



def val_res_pred(models_lists, per_donor_original_val_datasets, n, k, seed):
    per_donor_resampled_datasets =[]
    per_donor_resampled_y = []
    per_donor_perc = []

    mean_probs_per_patient = []
    total_pred_lists = []
    total_trial_pred_lists = []

    counter = 1
    for patient in per_donor_original_val_datasets:
        per_dataset_resampled_datasets = []
        per_dataset_resampled_ys = []
        per_dataset_perc = []

        timepoints_mean_probs = []
        patient_pred_list = []
        patient_trial_pred_list = []
        for file in patient:
            trials = len(models_lists)
            thr_res_seed_list = generate_seeds(trials, seed = seed + counter)
            print(thr_res_seed_list)

            # for each dataset, multiple subsets of n cells are resampled
            resampled_datasets,  resampled_y , _, seed =  subset_sampling(file, ncells = n, nsubsets = k, seed = thr_res_seed_list[0])

            print(f'Prediction {counter}')
            counter += 1
            new_datasets_predictions_list, new_datasets_results_list = trials_test_CellCNN_old(models_lists, resampled_datasets, thr_res_seed_list)
            positive_probs = []
            positive_probs_mean = []

            for trial in new_datasets_predictions_list:
                positive_probs.append(pd.DataFrame(trial)[1]) # appends the probability of positive classification

            all_trials_probs_array = np.array(positive_probs) #it converts the list of (list of) probabilities into an array of lists of probs

            # Computes the mean over the columns ( it takes the first element of all arrays and make the mean, then the second and so on)
            positive_probs_mean = np.mean(all_trials_probs_array, axis=0).tolist()

            print(f'Len of Mean: {len(positive_probs_mean)}')

            patient_pred_list.append(positive_probs_mean) # stores the mean subset probabilities
            patient_trial_pred_list.append(positive_probs) # stores all probabilities af all trials

            ## assigns mean probabilities to its true resampled subsets labels
            timepoints_mean_probs.append((positive_probs_mean, resampled_y))

        mean_probs_per_patient.append(timepoints_mean_probs)

        total_pred_lists.append(patient_pred_list)
        total_trial_pred_lists.append(patient_trial_pred_list)


    return total_pred_lists, total_trial_pred_lists, mean_probs_per_patient



def find_robust_threshold(mean_probs_per_patient, metric = 'f1', closest = False):

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

        """ Concatenate mean probs and labels into two nsubset x timepoints lists """
    probs = []
    resampled_ys = []
    for patient_probs_tuple in mean_probs_per_patient:

            for timep, timep_res_y in patient_probs_tuple:

                # get mean columns predicted probabilities
                probs += list(timep)

                # get the resampled ys
                resampled_ys += list(timep_res_y)

        #print('Log: Concatenation: Done!')

    best_f1 = -1
    best_thr = -1
    tot_per_tr_f1_scores = []
    threshold_predictions = []

    for threshold in list(range(1,101)):
        y_pred = []
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



    """ Best Threshold selection section """
    #find threshold
    max_f1 = max(tot_per_tr_f1_scores)
    best_thresholds_idx = []

    best_thresholds_idx = [i for i, f1 in enumerate(tot_per_tr_f1_scores) if f1 == max_f1]

    # whether multiple threholds provides the maximum f1_score, the median is taken
    best_threshold = np.median(best_thresholds_idx) + 1

    print(tot_per_tr_f1_scores)

    lowest_distance = 100
    thr_distances = []
    if closest:
        print(np.array(tot_per_tr_f1_scores)[best_thresholds_idx])
        for thr_idx in np.array(list(range(1,101)))[best_thresholds_idx]:
            thr_dist = abs(thr_idx - 50)
            if thr_dist < lowest_distance:
                thr_distances.append(thr_dist)

        min_dist_thr_idx = thr_distances.index(np.min(thr_distances))
        best_threshold = np.array(list(range(1,101)))[best_thresholds_idx][min_dist_thr_idx]

    print(f'Chosen threshold: {best_threshold}. Associated F1_score: {tot_per_tr_f1_scores[int(best_threshold - 1)]:.4f}' )
    return best_threshold*0.01, tot_per_tr_f1_scores


def sub_resampling_list(k, nsub_per_sub = 50):
    remaining_k = k
    sub_division = []

    while remaining_k > 0:
        print(remaining_k)
        if remaining_k >= nsub_per_sub:
            sub_division.append(nsub_per_sub)
            remaining_k -= nsub_per_sub
            print(sub_division)
        else:
            sub_division.append(remaining_k)
            return sub_division
    return sub_division


def test_res_pred(models_lists, per_donor_original_test_datasets, n, k, best_threshold, trials, seed):

    counter = 1
    per_donor_resampled_test_datasets =[]
    per_donor_resampled_test_y = []
    per_donor_perc = []

    test_total_labels = []
    test_total_pred_lists = []
    test_total_trial_pred_lists = []

    for patient in per_donor_original_test_datasets:
        per_dataset_resampled_datasets = []
        per_dataset_resampled_ys = []
        per_dataset_perc = []
        patient_timepoints_labels = []
        patient_pred_list = []
        patient_trial_pred_list = []

        for file in patient:
            sub_division = sub_resampling_list(k, nsub_per_sub = 50)
            print(sub_division)
            total_resampled_y = []
            total_positive_probs = [[] for _ in range(trials)]
            print(total_positive_probs)
            division_positive_probs_mean = []
            print(f'Prediction {counter}')

            rob_res_seed_list = generate_seeds(trials + 1, seed = seed + counter)
            counter += 1
            for _, division in enumerate(sub_division):
                print(division)
                resampled_datasets, resampled_y, blast_perc, seed = subset_sampling(dataset = file, ncells = n, nsubsets = division, seed = rob_res_seed_list[0])

                # predict labels
                new_datasets_predictions_list, new_datasets_results_list = trials_test_CellCNN_old(models_lists, resampled_datasets, rob_res_seed_list[1:])
                positive_probs = []

                """ Extraction and mean probabilities section """
                for iterat, trial in enumerate(new_datasets_predictions_list):
                    positive_probs.append(pd.DataFrame(trial)[1].values) # appends the probability of positive classification
                    total_positive_probs[iterat] += list(pd.DataFrame(trial)[1].values)

                 #it converts the list of (list of) probabilities into an array of lists of prob

                # Computes the mean over the columns ( it takes the first element of all arrays and make the mean, then the second and so on)
                division_positive_probs_mean.append(np.mean(positive_probs , axis=0))

                total_resampled_y += resampled_y

            positive_probs_mean = np.concatenate(division_positive_probs_mean)
            #print(f'Len of Mean: {len(positive_probs_mean)}')
            #print(f'Len of probs: {len(total_positive_probs)}')
            """ Threshold prediction section """
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




def generate_seeds(n=10, seed=None):
    """
    Generate n unique random seeds.

    Parameters:
    -----------
    n : int
        Number of seeds to generate
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    list of int
        List of n unique random seeds
    """
    if seed is None:
        print('Warning: random generation. No Reproducibility!')
        print('Reproducibility -> add "seed:int()" parameter!')
    else:
        np.random.seed(seed)

    if n > 10**6:
        raise ValueError(f"Cannot generate {n} unique seeds from range [0, 10^6)")
    return np.random.choice(10**6, n, replace=False)

"""======================================================="""


def save_models(model, save_dir):
        import pickle
        import os

        metadata_to_save = model.all_params
        # 3. Salva i metadati in un file pickle
        with open(os.path.join(save_dir, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata_to_save, f)
            print(f"âœ… Metadata salvati in {save_dir}/metadata.pkl")




# all possible combinations of parameters
def nsub_ncells_comb(ncells_step, max_ncells, blocks, nsub_step):
    ncells_list = list(range(ncells_step, max_ncells + ncells_step, ncells_step))
    nsub_list = list(range(100, blocks*nsub_step + nsub_step, nsub_step))
    
    all_nsub_ncells_comb = []
    for ncells_value in ncells_list:
        for nsub_value in nsub_list:
            all_nsub_ncells_comb.append([ncells_value, nsub_value])
    return all_nsub_ncells_comb

def run_training(CellCnn, new_train_datasets, new_train_y, new_val_datasets,
                     new_val_y, new_test_datasets, seed_list, hyper,
                                 grid = True, labels = False, trials = 1, cells_per_sub = 200, best_nsub = 200,
                                 max_epochs = 100, nrun = 15, generate = False, no_val = False):

        if no_val:
            train = train_val_finalizing(new_train_datasets, new_val_datasets, grid, labels, no_val)
        else:
            train, val = train_val_finalizing(new_train_datasets, new_val_datasets, grid, labels, no_val)


        new_test_datasets = flatten(new_test_datasets)
        test = remove_labels(new_test_datasets)
        #no_label_val = remove_labels(new_val_datasets)

        if no_val and not generate:
            models_lists = trials_train_CellCNN_old(CellCnn, train,
                                          new_train_y, test,
                                          trials = trials,
                                          n_cell = cells_per_sub, nsubset = best_nsub,
                                          max_epochs= max_epochs, ### 100,
                                          nrun= 15, ### 15,
                                          seed_list = seed_list, hyper = hyper,
                                          generate = generate, grid = grid)
        elif generate:
            models_lists = trials_train_CellCNN_old(CellCnn, train,
                                          new_train_y, test,
                                          trials = trials,
                                          n_cell = cells_per_sub, nsubset = best_nsub,
                                          max_epochs= max_epochs, ### 100,
                                          nrun= 15, ### 15,
                                          seed_list = seed_list, hyper = hyper,
                                          #val_datasets = val, val_y = new_val_y,
                                                    generate = generate, grid = grid)
        else:
            models_lists = trials_train_CellCNN_old(CellCnn, train, new_train_y,
                                              val_datasets = val, val_y = new_val_y,
                                              test_datasets_no_labels = test,
                                              trials = trials,
                                              n_cell = cells_per_sub, nsubset = best_nsub,
                                              max_epochs= max_epochs, ### 100,
                                              nrun= 15, ### 15,
                                              seed_list = seed_list, hyper = hyper,
                                              generate = generate, grid = grid)


        return models_lists


"=========================================================================="

def find_theta_best(f1_tested_par, tested_par):
            best_combinations = []
            for i, f1_par in enumerate(f1_tested_par):
                if f1_par == np.max(f1_tested_par):
                    best_combinations.append(tested_par[i])
            print(f'Best combinations found: {best_combinations}')

            sorted_sum_best_combinations = [np.sum(par) for par in best_combinations]
            sort_idx = np.argsort(sorted_sum_best_combinations)
            sorted_best_combinations = np.array(best_combinations)[sort_idx].tolist()
            print(f'Sort best combinations based on their var sum: {sorted_best_combinations}')


            if len(sorted_best_combinations) == 1:
                chosen_par = sorted_best_combinations[0]
                print(chosen_par)
            else:# len(sorted_best_combinations) %2: #if even number
                middle_position = int(len(sorted_best_combinations)/2)
                print(f'Position in the middle of best cobinations: {middle_position}')
                half_sorted_best_combinations = sorted_best_combinations[-middle_position:]
                print(f'Remaining half best comvinations: {half_sorted_best_combinations}')

                if len(half_sorted_best_combinations) %2:
                    chosen_par = half_sorted_best_combinations[int(len(half_sorted_best_combinations)/2)]
                else:
                    new_middle_position = max(1,int(len(half_sorted_best_combinations)/2))
                    chosen_par = half_sorted_best_combinations[new_middle_position]

            print(chosen_par)
            print(tuple(chosen_par))

            chosen_par = (chosen_par[0], chosen_par[1])
            print(chosen_par)
            return chosen_par



def compute_metrics(true_labels, pred_probs, thr, mcc = False):
    pred_labels = (np.array(pred_probs) >= thr).astype(int)
    print(pred_labels)
    f1 = f1_score(true_labels, pred_labels, pos_label=1, zero_division=0)
    rec = recall_score(true_labels, pred_labels, pos_label=1, zero_division=0)
    pre = precision_score(true_labels, pred_labels, pos_label=1, zero_division=0)
    acc = accuracy_score(true_labels, pred_labels)
    met_dict = {'f1': f1, 'rec': rec, 'pre': pre, 'acc': acc}

    if mcc:
        mcc = matthews_corrcoef(true_labels, pred_labels)
        met_dict['mcc'] = mcc
    return met_dict

def compute_mean_std_metrics(all_dir_metrics_across_LOPO, decimals = None, return_df = False):
    mean_std_dict = {}
    df = pd.DataFrame(all_dir_metrics_across_LOPO)

    for metric in df.columns:
        all_met_list = flatten(df[metric].to_list())
        all_met_list_mean = np.mean(all_met_list)
        all_met_list_std = np.std(all_met_list, ddof=1)

        if decimals:
            all_met_list_mean = round(all_met_list_mean, decimals)
            all_met_list_std = round(all_met_list_std, decimals)
        mean_std_dict[metric] = [all_met_list_mean, all_met_list_std]


    if return_df:
        return df
    else:
        return mean_std_dict

'''
def get_timepoints_predictions(total_pred_lists, per_donor_original_test_y, threshold):
    total_labels = []

    for patient in total_pred_lists:
        patient_timepoints_labels = []

        for timepoint in patient:
            timepoint_f1_scores = []
            blast_labelled_timepoint_scores = []

            true_y = timepoint.iloc[-1]

            for trial in range(len(timepoint.iloc[:-1])): # from each timepoint multiple samples has been generated. each sample has been predicted 10 times using the same 'best' model but using a different seed each time
                sub = timepoint.iloc[trial]               # each sub is a pediction of the 20 samples of a single timepoint
                blast_labelled_perc = sub.sum()/len(true_y)
                blast_labelled_timepoint_scores.append(blast_labelled_perc) #percentage of timepoints labelled as with blast cells

            mean_timepoint_blast_score = np.mean(blast_labelled_timepoint_scores) # mean percentage of timepoints labelled as with blast cells

            if mean_timepoint_blast_score >= threshold*0.01:
                patient_timepoints_labels.append(1)
            else:
                patient_timepoints_labels.append(0)

        total_labels.append(patient_timepoints_labels)
    return total_labels


def get_trial_pred_per_timepoint(total_pred_lists, per_donor_original_test_y):
    plot_data = []
    boxplot_data = []
    total_timepoint_prediction_per_trial = []

    for patient, patient_ys in zip(total_pred_lists, per_donor_original_test_y):
        timepoint_prediction_per_trial = []

        for timepoint, true_timepoint_y in zip(patient, patient_ys):
            label_str = "Positive" if true_timepoint_y == 1 else "Negative"

            trial_level_label_prediction = []
            true_y = timepoint.iloc[-1]
            timepoint_boxplot_data = []

            for trial in range(len(timepoint.iloc[:-1])): # from each timepoint multiple samples has been generated. each sample has been predicted 10 times using the same 'best' model but using a different seed each time
                sub = timepoint.iloc[trial]               # each sub is a pediction of the 20 samples of a single timepoint

                trial_perc = sub.sum()/len(true_y)
                #trial_perc = sub.sum()/min(1, true_y.sum())

                #trial_level_label_prediction.append((blast_labelled_perc >= threshold*0.01).astype(int))
                plot_data.append({
                "True_Label": label_str,
                "Timepoint_Score": trial_perc
                })

                timepoint_boxplot_data.append(trial_perc)

            boxplot_data.append({
                "True_Label": true_timepoint_y,
                "Timepoint_trials_scores": timepoint_boxplot_data
                })

    return plot_data, boxplot_data


def compute_timepoint_best_f1(timepoint_preds):
        """we are taking the best f1 score. because we are not tuning the model. we are just trying to predict the label of the timepoint"""
        timepoint_score = []

        resampled_true_y = timepoint_preds.iloc[-1] # get labels of resampled subsets
        #print(timepoint_preds.iloc[:-1]) #labels

        # initialize variables
        best_f1 = -1
        counter = 0

        for h in range(len(timepoint_preds) - 1):
            sub = timepoint_preds.iloc[h] # extrsct trial pediction

            f1 = f1_score(resampled_true_y, sub, pos_label = 1) # compute the f1_score
            if f1 > best_f1:
                best_f1_idx = counter
            counter += 1

        print(best_f1_idx)
        best_sub = timepoint_preds.iloc[best_f1_idx]
        #print(len(true_y))
        blast_score = best_sub.sum()
        print('')
        #print(best_sub)
        #print(f'blast_score: {blast_score}\n')
        timepoint_score = blast_score.sum()/len(resampled_true_y)
        return (timepoint_score, list(resampled_true_y))


def find_threshold(total_scores, per_donor_original_val_y):
    best_f1 = -1
    best_thr = -1
    tot_mean_f1_scores = []
    threshold_predictions = []
    for threshold in list(range(1,101)):
        f1_scores = []
        patient_predictions = []
        for patient_score, patient_y in zip(total_scores, per_donor_original_val_y):
            scores = []

            for timep in patient_score:
                # get mean columns predicted probabilities
                scores.append(timep[0])

            y_pred = []
            y_pred = (np.array(scores) >= threshold*0.01).astype(int) #checks column by column if the element is > than the threshold and converts it in 1 or 0
            #print(f'Threshold: {threshold*0.01}. Preds: {y_pred}')
            #print(patient_y)
            #print(y_pred)
            timepoint_f1_score = f1_score(patient_y, y_pred, pos_label = 1, zero_division=1)
            #print(f'f1_score: {timepoint_f1_score}\n')

            f1_scores.append(timepoint_f1_score)

            patient_predictions.append(list(y_pred))

        threshold_predictions.append(patient_predictions)

        #print(f1_scores)
        mean_f1_score = np.mean(f1_scores)
        tot_mean_f1_scores.append(mean_f1_score)

        if mean_f1_score > best_f1:
                best_f1 = mean_f1_score
                best_thr = threshold*0.01
        #print('')

        #patient_f1_scores.append(timepoint_f1_score)
    #plot thresholds
    plt.plot(list(range(1,101)), tot_mean_f1_scores)
    #for th in threshold_predictions:
        #print(th)

    #find threshold
    max_mean_f1 = max(tot_mean_f1_scores)
    best_thresholds_idx = []

    best_thresholds_idx = [i for i, f1 in enumerate(tot_mean_f1_scores) if f1 == max_mean_f1]
    #print(best_thresholds_idx)

    robust_best_thr_idx = np.median(best_thresholds_idx)
    #print(np.median(list(range(5))))
    print(f'Chosen threshold: {robust_best_thr_idx}' )
    return robust_best_thr_idx
    
    
    

def chosen_folds(iterations, train_perm, val_perm, seed = 42):
    print(seed)
    np.random.seed(seed)
    tot_perm = len(train_perm)

    idx = np.arange(tot_perm)
    np.random.shuffle(idx)

    chosen_folds = []
    for index in idx[:iterations]:
        chosen_folds.append([train_perm[index], val_perm[index]])

    return chosen_folds


def final_trials_prediction(total_trial_pred_lists, per_donor_original_test_y, per_donor_resampled_test_y, best_threshold):
    
    """ Elaborate data to show distribution of trial results over the entire set of timepoints and patiets"""
    prob_data = []
    f1_data = []
    boxplot_data = []

    for patient, patient_ys, patient_resampled_y in zip(total_trial_pred_lists, per_donor_original_test_y, per_donor_resampled_test_y):

        for timepoint, true_timepoint_y, true_resampled_y in zip(patient, patient_ys, patient_resampled_y):
            label_str = "Positive" if true_timepoint_y == 1 else "Negative"

            timepoint_boxplot_data = []
            timepoint_f1_data = []
            #print(timepoint)
            for trial in timepoint: # from each timepoint multiple samples has been generated. each sample has been predicted 10 times using the same 'best' model but using a different seed each time
                             # each sub is a pediction of the 20 samples of a single timepoint

                #print(trial)
                trial_prob = np.mean(trial) # mean of the probabilities of the subsets

                trial_pred = (np.array(trial) >= best_threshold*0.01).astype(int)
                #print(trial_pred)
                #print(true_resampled_y)

                trial_f1 =  f1_score(true_resampled_y, trial_pred, pos_label = 1, zero_division=1)
                #print(rqver)

                prob_data.append({
                "True_Label": label_str,
                "Timepoint_Score": trial_prob
                })


                timepoint_boxplot_data.append(trial_prob)
                timepoint_f1_data.append(trial_f1)

            boxplot_data.append({
                "True_Label": true_timepoint_y,
                "Timepoint_trials_scores": timepoint_boxplot_data
                })

            f1_data.append({
                "True_Label": true_timepoint_y,
                "Timepoint_trials_scores": timepoint_f1_data
                })

    return prob_data, boxplot_data, f1_data




def retrieve_nsub_models_names(n_sub_tuning_path):
    """ Retrieve nsub folders names """

    def extract_num(folder_name):
        if 'model_' in folder_name:
            return folder_name
    # get folder_names
    n_sub_cartelle = [
        nome for nome in os.listdir(n_sub_tuning_path)
        if os.path.isdir(os.path.join(n_sub_tuning_path , nome))]

    n_sub_models = [
        extract_num(model) for model in n_sub_cartelle
        if extract_num(model) is not None]

    return n_sub_models


def find_best_nsubs(f1_step_1, nsub_list, n = 5, indices = False):
    """ Returns the n nsub values that performed best. If multiple values have the highest f1_score, higher values are prefered"""

    idx = np.argsort(f1_step_1) # fi_step_1 elements' indices ordered in acscending order (the last element is the index of the best element in f1_step_1)
    unique_values = np.sort(list(set(f1_step_1.copy()))) #sort function returns the sorted uniques values of a list

    f1_5_best, best_5_idx = [], []
    for value in unique_values[::-1]:
        counter = len(f1_step_1) - 1

        for element in f1_step_1[::-1]:

            if element == value:
                f1_5_best.append(element)
                best_5_idx.append(counter)

            if len(best_5_idx) == n:
                print(f'5 best f1_score values: {f1_5_best} at indexes: {best_5_idx}')

                if indices:
                    return np.sort(np.array(nsub_list)[best_5_idx]), best_5_idx
                else:
                    return np.sort(np.array(nsub_list)[best_5_idx])

            counter -= 1
    return





'''