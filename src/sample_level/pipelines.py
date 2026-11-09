import os
import time

import numpy as np

from src.shared.temp_storage import save_test_temp
from src.sample_level.bayesian_tuning import BO_tuning
from src.sample_level.training import fit
from src.sample_level.testing import predict
from src.sample_level.configs import (
    build_tuning_config, build_training_config, get_param_list_dict)

from seed_manager import SeedManager

def run_lopocv_fold(BinCellCnn, config, experiment_path, LOPOCV_fold_idx, train_kfolds, test_pat , multiple_donations, ALL_DATASETS, mem = None):
    # --- Seed parameters ---
    starting_seed = config['starting_seed']

    param_list_dict = get_param_list_dict(config)

    print(param_list_dict)
    
    full_process_time_start = time.time()

    tuning_save_dir = f'{experiment_path}/outer_fold_{LOPOCV_fold_idx}/tuning/'
    os.makedirs(tuning_save_dir, exist_ok=True)
    weights_outdir = f'{tuning_save_dir}/weights'
    os.makedirs(weights_outdir, exist_ok=True)

    seed_manager = SeedManager(starting_seed=starting_seed, LOPO_fold_idx=LOPOCV_fold_idx)

    ## === load the samples === ##

    tuning_config = build_tuning_config(config)

    best_ncells, best_nsubs, best_bs, _ = BO_tuning(
            LOPOCV_fold_idx=LOPOCV_fold_idx,
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

    _, original_test_y = save_test_temp(test_pat, multiple_donations, ALL_DATASETS = ALL_DATASETS, mem = mem)

    training_save_dir = f'{experiment_path}/outer_fold_{LOPOCV_fold_idx}/training/'
    os.makedirs(training_save_dir, exist_ok=True)

    weights_outdir = f'{training_save_dir}/weights'
    os.makedirs(weights_outdir, exist_ok=True)

    _, best_thr = fit(multiple_donations = multiple_donations,
                                               ALL_DATASETS = ALL_DATASETS,
                                               BinCellCnn = BinCellCnn,
                                               config = training_config,
                                               mem = mem,
                                               seed_manager = seed_manager,
                                               save_dir = training_save_dir,
                                               outdir = weights_outdir,
                                               train_kfolds = train_kfolds)

    # === Testing === #

    test_save_dir = f'{experiment_path}/outer_fold_{LOPOCV_fold_idx}/test/'
    os.makedirs(test_save_dir, exist_ok=True)

    test_samples_preds, original_test_y = predict(test_pat = test_pat,
                                                  multiple_donations = multiple_donations,
                                                  ALL_DATASETS = ALL_DATASETS,
                                                  mem = mem,
                                                  training_save_dir = training_save_dir,
                                                  test_save_dir = test_save_dir)

    print('test_samples_preds', test_samples_preds)
    print('original_test_y', original_test_y)

    print((np.mean(test_samples_preds, axis = 0) > best_thr).astype(int))

    fold_end = time.time()
    print(f'\n Time elapsed since {LOPOCV_fold_idx+1}-th iteration started: {fold_end - full_process_time_start}\n')

    seed_manager_dir = f'{experiment_path}/outer_fold_{LOPOCV_fold_idx}/'
    seed_manager.save_state(seed_manager_dir)
    
    return test_samples_preds, original_test_y, best_thr