
'''
test_CellCNN_old, trials_test_CellCNN_old, predict
'''

import numpy as np
import pandas as pd
import random
import time
from memory_utils import clear_RAM, MemorySave


from src.shared.timepoints_elaboration import donation_extraction
from src.shared.temp_storage import load_test_from_temp

from src.shared.utils import save_element, load_element
from src.shared.labels import retrieve_labels


def test_BinCellCNN(model, test_datasets_no_labels = None, seed = None, validation = False, mem = None):
    """
    Runs prediction on test datasets using a trained BinCellCnn model.
    Optionally passes a seed to model.predict.
    """
    if mem is not None:
        print('Loading from Temp')
        test_datasets_no_labels, _ = load_test_from_temp(test_samples=None, test_phenotypes=None, validation = validation, mem = mem)
        print('len(test_datasets_no_labels)', len(test_datasets_no_labels))
    test_pred_list = {}
    #print(test_datasets_no_labels)

    for ds_idx, ds in enumerate(test_datasets_no_labels):
        if seed is None:
            seed = 42
        test_pred = model.predict(test_samples = [ds], seed = seed, mem = mem)

        print(f'Done')

        print('test_pred', test_pred )
        print('test_pred[0]', test_pred[0])

        test_pred_list[f'sample_{ds_idx}'] = test_pred[0][1]  # retrieve only the positive class prediction
    return test_pred_list, model.results




def trials_test_BinCellCNN(models_lists, test_datasets_no_labels = None, seed_manager = None, trials = 1, mem = None):

    """
    ONLY FOR TESTING.
    Runs prediction across multiple trained BinCellCnn models (seeds).
    If seed_list is missing or too short, generates one randomly.
    """
    predictions_list = []
    results_list = []
    tot_trials = len(models_lists)

    if seed_manager is None:

        seed_list = random.choices(list(range(1, 100000)), k=trials)
        print(f'seed_manager is randomly set to {seed_list}.\n')
        print(f'To control randomization, please generate seed_list outside the function.')
    else:
        seed_list = [seed_manager.get_seed(f'test_weight_init') for _ in range(trials)]


    for i in range(tot_trials):
        print(f'Weight Initialization {i+1} out of {tot_trials} started!')
        predictions, result = test_BinCellCNN(models_lists[i], test_datasets_no_labels, seed = seed_list[i], validation = False, mem = mem)

        predictions_list.append(list(predictions.values()))
        results_list.append(result)

        print(f'Trial {i+1} Done!\n')

    return predictions_list, results_list



def predict(config, test_pat, multiple_donations, training_save_dir, test_save_dir, ALL_DATASETS = None, mem = None, seed_manager = None):
            # Predict and Classify test samples
            original_test_datasets = None

            if not mem:
                test_datasets_extracted = donation_extraction(test_pat, multiple_donations, ALL_DATASETS = ALL_DATASETS)
                original_test_datasets, original_test_y = retrieve_labels(test_datasets_extracted, remove=True, flat=True)


            CV_models = load_element(training_save_dir, 'CV_models')
            test_samples_preds = []

            for models_list in CV_models:
                w_init_preds_per_sample_list, original_results_list = trials_test_BinCellCNN(models_list, original_test_datasets, mem = mem, seed_manager = seed_manager)
                if mem:
                    (original_test_y, ) = mem.load_variables(['test_phenotypes'])
                
                print('w_init_preds_per_sample_list', w_init_preds_per_sample_list)
                pos_preds_per_sample = np.mean(w_init_preds_per_sample_list, axis =0)

                test_samples_preds.append(pos_preds_per_sample)

            save_element(test_save_dir, 'test_samples_preds', test_samples_preds)
            save_element(test_save_dir, 'test_y', original_test_y)


            return test_samples_preds, original_test_y











            