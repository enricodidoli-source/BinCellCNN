


'''save_test_temp, save_train_val_in_temp, load_test_from_temp'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import sys


from src.shared.timepoints_elaboration import sample_label_extraction
from src.shared.labels import remove_labels, retrieve_labels
from src.shared.utils import flatten


from memory_utils import clear_RAM

def _prepare_samples(config, multiple_donations = None, train_donors_idx = None, val_donors_idx = None, 
                    ALL_DATASETS  = None, ALL_DATASETS_LABELS  = None, mem = None,
                         ):
    
    if mem:
        (ALL_DATASETS,) = mem.load_variables(['ALL_DATASETS'])
        (ALL_DATASETS_LABELS,) = mem.load_variables(['ALL_DATASETS_LABELS'])

    if True:
        print('config', config)
        remove_columns = config['remove_columns']
        label_column = config['label_column']

        train_datasets_extracted, train_labels_extracted = sample_label_extraction(
                    donors_idx = train_donors_idx, 
                    multiple_donations = multiple_donations, 
                    ALL_DATASETS = ALL_DATASETS, 
                    ALL_DATASETS_LABELS = ALL_DATASETS_LABELS, 
                    mem = mem,
                    remove_columns = remove_columns)

        train_datasets_extracted, original_train_y = flatten(train_datasets_extracted), flatten(train_labels_extracted)
        original_train_datasets = remove_labels(train_datasets_extracted, label_column = label_column)

        clear_RAM(variables_names=['train_datasets_extracted'])

        val_datasets_extracted, val_labels_extracted = sample_label_extraction(
                    donors_idx = val_donors_idx, 
                    multiple_donations = multiple_donations, 
                    ALL_DATASETS = ALL_DATASETS, 
                    ALL_DATASETS_LABELS = ALL_DATASETS_LABELS, 
                    mem = mem,
                    remove_columns = remove_columns)

        val_datasets_extracted, original_val_y = flatten(val_datasets_extracted), flatten(val_labels_extracted)
        original_val_datasets = remove_labels(val_datasets_extracted, label_column = label_column)

        clear_RAM(variables_names=['val_datasets_extracted'])

    print(f'Labels from Train and Validations Sets have been REMOVED')
    return original_train_datasets, original_train_y, original_val_datasets, original_val_y


def train_val_finalizing(config, multiple_donations = None, train_donors_idx = None, val_donors_idx = None, 
                    ALL_DATASETS  = None, ALL_DATASETS_LABELS  = None, mem = None,
                         ):
    """
    Prepares train and validation datasets by removing labels based on the search mode.
    """
    train, original_train_y, val, original_val_y = _prepare_samples(config, 
                    multiple_donations = multiple_donations, 
                    train_donors_idx = train_donors_idx, 
                    val_donors_idx = val_donors_idx, 
                    ALL_DATASETS  = ALL_DATASETS, 
                    ALL_DATASETS_LABELS  = ALL_DATASETS_LABELS,
                    mem = mem,
                         )

    train, val = save_train_val_in_temp(
                            train, original_train_y, val, original_val_y, mem=mem)

    return train, val, original_train_y, original_val_y







def save_test_temp(test_config, test_pat, multiple_donations, ALL_DATASETS = None, ALL_DATASETS_LABELS = None, mem = None):

    if mem:
        (ALL_DATASETS,) = mem.load_variables(['ALL_DATASETS'])
        (ALL_DATASETS_LABELS,) = mem.load_variables(['ALL_DATASETS_LABELS'])
    
    remove_columns = test_config['remove_columns']
    label_column = test_config['label_column']
    test_datasets_extracted, test_labels_extracted = sample_label_extraction(
                    donors_idx = test_pat, 
                    multiple_donations = multiple_donations, 
                    ALL_DATASETS = ALL_DATASETS, 
                    ALL_DATASETS_LABELS = ALL_DATASETS_LABELS, 
                    mem = mem,
                    remove_columns = remove_columns)
    
    test_datasets_extracted, original_test_y = flatten(test_datasets_extracted), flatten(test_labels_extracted)           
    original_test_datasets = remove_labels(test_datasets_extracted, label_column = label_column)

    #test_datasets_extracted = donation_extraction(test_pat, multiple_donations, ALL_DATASETS)
    #original_test_datasets, original_test_y = retrieve_labels(test_datasets_extracted, remove = True, flat = True)
    
    if mem:
        print('Saving to Temp')

        mem.save_and_clear([original_test_datasets, original_test_y], ['test_samples', 'test_y'], force = True)
        original_test_datasets, original_test_y = None, None

    return original_test_datasets, original_test_y


def save_train_val_in_temp(finalized_train, train_y, finalized_val, val_y, mem = None):
    if mem:
        mem.save_and_clear([finalized_train, train_y], ['train_samples', 'train_y'], force = True)

        mem.save_and_clear([finalized_val, val_y], ['valid_samples', 'valid_y'], force = True)

        return None, None
    else:
        return finalized_train, finalized_val

def load_test_from_temp(test_samples=None, test_phenotypes=None, validation = False, mem=None):

    if mem is not None:
        print('Loading from Temp')

        if validation:
            (test_samples, test_phenotypes) = mem.load_variables(['valid_samples', 'valid_y'])
            print('len(test_samples)', len(test_samples))
        else:
            (test_samples, test_phenotypes) = mem.load_variables(['test_samples', 'test_y'])
    else:

        # copy the list of samples so that they are not modified in place
        test_samples = test_samples.copy()

    return test_samples, test_phenotypes
