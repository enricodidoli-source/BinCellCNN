
import os
import gc
import time
import pickle
import ctypes
import matplotlib.pyplot as plt
from IPython.display import display
from pathlib import Path
import pandas as pd

from model_grid import CellCnn
from timepoints_elaboration import load_data, donation_extraction
from utils import load_models, retrieve_labels


def clear_RAM():
    plt.close('all')
    gc.collect()
    try:
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except Exception:
        pass


def save_all_results_singularly(single_run_plotdir, saving_list, name_list, LOPOCV_fold_idx = ''):
    for element, name in zip(saving_list, name_list):
        with open(f'{single_run_plotdir}/{name}{LOPOCV_fold_idx}.pkl', 'wb') as f:
            pickle.dump(element, f)



class TimeCalculator():
    def __init__(self):
        self.time_dict = {}

    def _get_time(self):
        return time.time()

    def add_time(self, column):
        self.time_dict[column] = self._get_time()

    def return_time(self):
        return self.time_dict

    def show_time(self):
        print(self.time_dict)


def analyse_working_time(time_df):
    filter_time = time_df['end_filter_time'] - time_df['start_time']
    working_time = time_df['end_test_time'] - time_df['end_filter_time'] + filter_time
    logistic_working_time = time_df['end_logostic_test_time'] - time_df['end_logostic_thr_time'] + filter_time
    best_3_filt_working_time = time_df['end_3_filt_test_time'] - time_df['end_3_filt_thr_time']

    fig = plt.figure(figsize=(10, 6))
    plt.boxplot([filter_time, working_time, logistic_working_time, best_3_filt_working_time])
    display(fig)
    plt.close(fig)



# =================================




def load_training_models(model_cvfolds_dir ):
    """Loads all cross-validation models using safe path resolution."""
    all_trials = os.listdir(model_cvfolds_dir)

    cross_val_models_list = []
    for t, trial in enumerate(all_trials): # cross validation folds

        seeds_dir = f'{model_cvfolds_dir}/inner_fold_{t}/models'
        all_seeds = os.listdir(seeds_dir)

        loaded_models_lists = []
        for s in range(len(all_seeds)):

            model_dir = f'{seeds_dir}/seed_{s}/'

            with open(os.path.join(model_dir, 'metadata.pkl'), 'rb') as f:
                meta = pickle.load(f)

            model = load_models(CellCnn, meta)
            loaded_models_lists.append(model)
        cross_val_models_list.append(loaded_models_lists)
    clear_RAM()
    return cross_val_models_list


def extract_data_from_fold(val_pat, multiple_donations, ALL_DATASETS):
    """Extracts, formats, and separates labels from the validation fold data."""

    val_datasets_extracted = donation_extraction(val_pat, multiple_donations, ALL_DATASETS)
    val_datasets_extracted_flat_with_labels, _ = retrieve_labels(val_datasets_extracted, remove = False, flat = True)

    comprehensive_val_df = pd.concat(val_datasets_extracted_flat_with_labels, ignore_index = True) # concatenate all df within the list

    del val_datasets_extracted_flat_with_labels
    gc.collect()

    true_cells_labels = comprehensive_val_df['IsBlast'].to_numpy()
    comprehensive_val_df.drop(columns=['IsBlast'], inplace=True)

    clear_RAM()
    return comprehensive_val_df, true_cells_labels


