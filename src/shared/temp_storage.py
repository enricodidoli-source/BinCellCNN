


'''save_test_temp, save_train_val_in_temp, load_test_from_temp'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import sys


from src.shared.timepoints_elaboration import donation_extraction
from src.shared.labels import remove_labels, retrieve_labels


from memory_utils import clear_RAM


def train_val_finalizing(train_datasets, original_train_y, val_datasets, original_val_y, mem = None,
                         multiple_donations = None, train_donors_idx = None, val_donors_idx = None):
    """
    Prepares train and validation datasets by removing labels based on the search mode.
    """
    if mem:
            train_datasets_extracted = donation_extraction(
            train_donors_idx,
            multiple_donations,
            ALL_DATASETS=None,
            mem=mem)

            train_datasets, original_train_y = retrieve_labels(
            train_datasets_extracted,
            remove=True,
            flat=True)

            clear_RAM(variables_names=['train_datasets_extracted'])

            val_datasets_extracted = donation_extraction(
                val_donors_idx,
                multiple_donations,
                ALL_DATASETS=None,
                mem=mem)

            val_datasets, original_val_y = retrieve_labels(
                  val_datasets_extracted,
                  remove=True,
                  flat=True)

            clear_RAM(variables_names=['val_datasets_extracted'])


    val = remove_labels(val_datasets)
    train = remove_labels(train_datasets)
    print(f'Labels from Train and Validations Sets have been REMOVED')

    train, val = save_train_val_in_temp(
                            train, original_train_y, val, original_val_y, mem=mem)

    return train, val, original_train_y, original_val_y



def save_test_temp(test_pat, multiple_donations, ALL_DATASETS = None, mem = None):

    if mem:
        (ALL_DATASETS,) = mem.load_variables(['ALL_DATASETS'])
    test_datasets_extracted = donation_extraction(test_pat, multiple_donations, ALL_DATASETS)
    original_test_datasets, original_test_y = retrieve_labels(test_datasets_extracted, remove = True, flat = True)
    if mem:
        print('Saving to Temp')

        mem.save_and_clear([original_test_datasets, original_test_y], ['test_samples', 'test_phenotypes'], force = True)
        original_test_datasets, original_test_y = None, None

    return original_test_datasets, original_test_y



def save_train_val_in_temp(finalized_train, train_y, finalized_val, val_y, mem = None):
    if mem:
        mem.save_and_clear([finalized_train, train_y], ['train_samples', 'train_phenotypes'], force = True)

        mem.save_and_clear([finalized_val, val_y], ['valid_samples', 'valid_phenotypes'], force = True)

        return None, None
    else:
        return finalized_train, finalized_val

def load_test_from_temp(test_samples=None, test_phenotypes=None, validation = False, mem=None):

    if mem is not None:
        print('Loading from Temp')

        if validation:
            (test_samples, test_phenotypes) = mem.load_variables(['valid_samples', 'valid_phenotypes'])
            print('len(test_samples)', len(test_samples))
        else:
            (test_samples, test_phenotypes) = mem.load_variables(['test_samples', 'test_phenotypes'])
    else:

        # copy the list of samples so that they are not modified in place
        test_samples = test_samples.copy()

    return test_samples, test_phenotypes
