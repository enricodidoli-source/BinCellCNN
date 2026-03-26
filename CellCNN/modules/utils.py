'''
from utils import flatten, remove_labels, retrieve_labels, show_blast_distribution_perc
from utils import prepare_results_to_save, subset_sampling, sub_resampling_list, generate_seeds
from utils import retireve_sorted_pat_sample_ids, nsub_ncells_comb, save_models, load_models
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import sys

def remove_from_cache(files: list):
    for file in files:
        if file in sys.modules:
            del sys.modules[file]
            print(f"{file} removed from cache")
        else:
            print(f"{file} not found in cache")

def generate_seeds(n=10, seed=None):
    """
    Generate n unique random seeds.
    """
    if seed is None:
        print('Warning: random generation. No Reproducibility!')
        print('Reproducibility -> add "seed:int()" parameter!')
    else:
        np.random.seed(seed)

    if n > 10**6:
        raise ValueError(f"Cannot generate {n} unique seeds from range [0, 10^6)")
    return np.random.choice(10**6, n, replace=False)
    
def flatten(nested):
    """
    Recursively flattens a nested list or tuple into a flat list.
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

def remove_labels(new_test_datasets):
    """
    Removes the 'IsBlast' column from each dataset, if present.
    """
    new_no_label_test_datasets = []
    for dataset in new_test_datasets:
        if 'IsBlast' in dataset.columns:
            dataset = dataset.drop(columns = ['IsBlast'])

        new_no_label_test_datasets.append(dataset)
    return new_no_label_test_datasets



def retrieve_labels(datasets_extracted, remove = False, flat = False):
    """
    Extracts binary labels from the 'IsBlast' column of each dataset.
    Optionally removes the label column or flattens the output.
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


def show_blast_distribution_perc(ALL_DATASETS, multiple_donations, return_perc = False, log = False):
    """
    Computes and shows the blast cell percentage for each sample across all patients.
    Optionally returns the percentage list.
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



def prepare_results_to_save(results_list, par_list = ['config', 'model_sorted_idx']):
    """
    Filters a list of result dictionaries, keeping only the specified keys.
    """
    tot_trials_results = []

    for res in results_list:
        needed_results = {}
        for key, value in res.items():
            if key in par_list:
                needed_results[key] = value

        tot_trials_results.append(needed_results)
    return tot_trials_results


def subset_sampling(dataset, ncells, nsubsets, seed):
    """
    Generates multiple random subsets (with replacement) from a single dataset.
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

def sub_resampling_list(k, nsub_per_sub = 50):
    """
    Splits k into chunks of nsub_per_sub, returning the list of chunk sizes. 
    Used to save memory
    """
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




def retireve_sorted_pat_sample_ids(samples_info_dict):
    """
    Builds a sorted list of sample IDs in the format 'Pat_{id}_{timepoint}',
    ordered by patient id and then by index within each patient.
    """
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

def nsub_ncells_comb(ncells_step, max_ncells, blocks, nsub_step):
    """
    Generates all combinations of ncells and nsubsets values from their respective ranges.
    """
    ncells_list = list(range(ncells_step, max_ncells + ncells_step, ncells_step))
    nsub_list = list(range(100, blocks*nsub_step + nsub_step, nsub_step))
    
    all_nsub_ncells_comb = []
    for ncells_value in ncells_list:
        for nsub_value in nsub_list:
            all_nsub_ncells_comb.append([ncells_value, nsub_value])
    return all_nsub_ncells_comb


# save and load CellCNN models
def save_models(model, save_dir):
    """
    Saves model parameters to a pickle file in the specified directory.
    """
    metadata_to_save = model.all_params
    # Salva i metadati in un file pickle
    with open(os.path.join(save_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata_to_save, f)


def load_models(CellCnn, meta):
        """
        Instantiates a CellCnn model and restores its parameters from a metadata dictionary.
        """
        model = CellCnn()

        model.scale = meta['scale']
        model.nsubset = meta['nsubset']
        model.ncell = meta['ncell']
        model.per_sample = meta['per_sample']
        model.maxpool_percentages = meta['maxpool_percentages']
        model.nfilter_choice = meta['nfilter_choice']
        model.learning_rate = meta['learning_rate']
        model.coeff_l1 = meta['coeff_l1']
        model.coeff_l2 = meta['coeff_l2']
        model.dropout = meta['dropout']
        model.dropout_p = meta['dropout_p']
        model.max_epochs = meta['max_epochs']
        model.patience = meta['patience']
        model.dendrogram_cutoff = meta['dendrogram_cutoff']
        model.accur_thres = meta['accur_thres']
        model.verbose = meta['verbose']
        model.seed = meta['seed']
        model.results = meta['results']
        model.grid = meta['grid']

        return model




