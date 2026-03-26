''' 
from cv_folds import remove_test_patients_from_categories, classic_CV_train_val_splits
from cv_folds import generate_LOPOCV_dicts, generate_LOPOCV_folds, extract_fold_features
''' 

from itertools import zip_longest
import pandas as pd
import numpy as np

from timepoints_elaboration import donor_division

def remove_test_patients_from_categories(healthy_donors, blast_donors, mixed_donors,  test_donors_idx):
    """
    Removes test patients from the three patients categories and generates a list of the remaining patients (train + validation).
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


# k-fold CV
def classic_CV_train_val_splits(h_pat, b_pat, m_pat, tot_pat, folds = 3, LOOCV_test = None, shuffle_seed = None):
    """
    Generates train/validation splits for cross-validation.
    Optionally removes a single LOOCV patient before splitting, and supports reproducible shuffling.
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


# LOPOCV 


def generate_LOPOCV_dicts(multiple_donations, ALL_DATASETS = None):
    """
    Generates Leave-One-Patient-Out Cross-Validation (LOPOCV) fold pairs.
    For each patient, creates a (train_dict, test_dict) tuple where the patient is the test set.
    Optionally computes blast percentages for the test patient's samples.

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



def extract_fold_features(chosen_perm, multiple_donations, tot_perc_list):
    """
    Extracts and aggregates file-level features for each train/validation fold.
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