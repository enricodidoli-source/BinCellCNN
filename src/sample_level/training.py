'''
from fit, inner_fold_iteration, train_CellCNN_old, run_trials
'''

import numpy as np
import pandas as pd
import random
import time
from memory_utils import clear_RAM, MemorySave


from src.sample_level.testing import test_BinCellCNN
from src.shared.timepoints_elaboration import donation_extraction
from src.shared.labels import retrieve_labels
from src.shared.temp_storage import train_val_finalizing
from src.sample_level.classification import get_best_thr
from src.sample_level.show_results import elaborate_predictions
from src.shared.utils import save_element

from src.BinCellCNN.model import BinCellCNN




def train_BinCellCNN(BinCellCnn, train_datasets, train_y, config=None,
                     val_datasets=None, val_y=None, outdir=None, mem=None, seed=None):
    """
    Trains BinCellCnn models. Uses default hyperparameters if none are provided,
    otherwise unpacks nfilter, maxpool_percentages and learning_rate from hyper.
    Supports optional validation set and generate mode.
    """
    hyper = config['hyper']
    if hyper is None:
        print('Warning: no hyperparameters passed. Hyperparameters fixed:')
        print('nfilter = [3,5,7,9]')
        print('maxpool_p = [1., 5., 20., 100.]')
        print('learning_r = [0.01, 0.001]')
        nfilter = [3,5,7,9]
        maxpool_p = [1., 5., 20., 100.]
        learning_r = [0.01, 0.001]
    else:
        nfilter, maxpool_p, learning_r = hyper


    print('bs = config["bs"]', config['bs'])
    
    model = BinCellCNN(
        ncell = config['ncells'], #200                # Number of cells per multi-cell input (sampled from the 'patient' datasets)
        nsubset = config['nsubs'],                  # Total number of multi-cell inputs generated per class (or sample and class)
        bs = config['bs'],   
        per_sample = True,                  # For each sample, nsubset samples of ncell are performed
        nfilter_choice = nfilter,           # Range of possible number of filters
        maxpool_percentages = maxpool_p,    # list of k-percentage max_pooling
        learning_rate = learning_r,         # Learning rate list
        max_epochs = config['tuning_epochs'], #50
        patience=5,                         # Early stopping patience
        nrun = config['tuning_nruns'],  #15                   # Number of neural network configurations to try (for Hyperparameter optimization)
        scale=True,                         # Z-score normalization
        verbose=1,
        seed = int(seed),
        grid = config['grid']
        
    )

    if outdir is None:
        outdir = f'/content/BinCellCnn_results'  # Results Directory


    model.fit(
                train_samples = train_datasets,
                train_phenotypes = np.array(train_y),
                outdir=outdir,
                valid_samples = val_datasets,
                valid_phenotypes = np.array(val_y),
                mem = mem,
                metric = config['eval_metric']
            )

    return model

def run_trials(BinCellCnn,  config, train_datasets = None, train_y = None,
                     val_datasets = None, val_y = None,  outdir = None, mem = None, seed_manager = None):
    """
    Runs multiple training weights initializations of BinCellCnn, each with a different seed.
    If seed_list is missing or too short, generates one randomly.
    """
    models_lists = []
    predictions_list, results_list = [], []

    trials = config['trials']
    for i in range(trials):

        if seed_manager is None:

            seed_list = random.choices(list(range(1, 10**10)), k=1)
            print(f'seed_manager is randomly set to {seed_list}.\n')
            print(f'To control randomization, please pass seed_manager as parameter.')

        print(f'Weight initialization {i+1} started')
        training_seed = seed_manager.get_seed(f'train_weight_init')
        testing_seed = seed_manager.get_seed(f'test_weight_init')

        print(f'Seed used: {training_seed}')
        model = train_BinCellCNN(BinCellCnn = BinCellCnn,
                                  train_datasets = train_datasets,
                                  train_y = train_y,
                                  config = config,
                                  seed = training_seed,

                                  val_datasets = val_datasets,
                                  val_y = val_y,
                                  outdir = outdir,
                                  mem = mem)


        models_lists.append(model)

        predictions, result = test_BinCellCNN(model, test_datasets_no_labels = val_datasets, seed = testing_seed, mem = mem, validation = True)
        predictions_list.append(list(predictions.values())) # retrieve only the samples predictions per w_initialization
        results_list.append(result)

        clear_RAM(['model'])
        print(f'Trial {i+1} Done!\n')

    return models_lists, predictions_list, results_list






def inner_fold_iteration(BinCellCnn, config, train_donors_idx, val_donors_idx, multiple_donations, ALL_DATASETS, mem, seed_manager, outdir, return_models = False, label_column = 'IsBlast'):


    # extraxt samples using pre-slitted indexes
    original_train_datasets, original_train_y, original_val_datasets, original_val_y = None, None, None, None
    if not mem:
        train_datasets_extracted = donation_extraction(train_donors_idx, multiple_donations, ALL_DATASETS=ALL_DATASETS, mem=mem)
        original_train_datasets, original_train_y = retrieve_labels(train_datasets_extracted, remove=True, flat=True, label_column = label_column)
        clear_RAM(variables_names=['train_datasets_extracted'])

        val_datasets_extracted = donation_extraction(val_donors_idx, multiple_donations, ALL_DATASETS=ALL_DATASETS, mem=mem)
        original_val_datasets, original_val_y = retrieve_labels(val_datasets_extracted, remove=True, flat=True,  label_column = label_column)
        clear_RAM(variables_names=['val_datasets_extracted'])


    # remove labels (random search without labels is required)
    train, val, original_train_y, original_val_y = train_val_finalizing(
        train_datasets=original_train_datasets,
        original_train_y=original_train_y,
        val_datasets=original_val_datasets,
        original_val_y=original_val_y,
        mem=mem,
        multiple_donations = multiple_donations,
        train_donors_idx = train_donors_idx,
        val_donors_idx = val_donors_idx)

    print(original_train_y)
    print(original_val_y)
    print(train)
    print(val)

    if mem:
        clear_RAM(variables_names=['original_train_datasets', 'original_val_datasets'])

    # === Train Model=== #
    (models_lists, predictions_list, results_list) = run_trials(
        BinCellCnn,
        train_datasets=train,
        train_y=original_train_y,                         # train set and labels
        val_datasets=val,
        val_y=original_val_y,                             # val set and labels
        config = config,

        outdir= outdir ,
        mem=mem,
        seed_manager=seed_manager)

    #save models
    print('predictions_list', predictions_list)
    print('original_val_y', original_val_y)

    # Elaborate results
    (_, _, f1_scores_list, _, _) = elaborate_predictions(
        predictions_list,
        original_val_y,
        results=True, # show results
        beta=1        # if 2 recall weight twice the precision weight
    )



    if return_models:
        return f1_scores_list, predictions_list, original_val_y, results_list, models_lists
    return f1_scores_list, predictions_list, original_val_y, results_list


def fit(BinCellCnn, multiple_donations, ALL_DATASETS, config, mem, seed_manager, save_dir, outdir, train_kfolds):

    across_folds_preds = []
    test_samples_preds = []
    across_folds_metrics = []
    CV_models = []
    for i in range(1):


            # prepare storing variables
            best_thresholds = []       # save thresholds
            roc_metrics = []
            val_predicted_for_roc = [] # store predictions from tuning for threshold tuning

            print(f'Model {i+1}. Testing params -> ncells: {config['ncells']}, nsubs: {config['nsubs']}, bs: {config['bs']}')
            print(f'5 Fold CV started')

            tuning_predictions_folds = []
            val_predicted_for_roc_folds = []
            f1_across_folds = []
            tuning_results = []

            #model_save_dir = f'{save_dir}/inner_fold_{config['fold_idx']}/models/'

            for fold_idx, (train_donors_idx, val_donors_idx) in enumerate(train_kfolds):
                print(f'\nFold: {fold_idx}\n')
                fold_start = time.time()

                #config['fold_idx'] = fold_idx
                f1_scores_list, predictions_list, original_val_y, results_list, models_list = inner_fold_iteration(
                    BinCellCnn=BinCellCnn,
                    config=config,
                    train_donors_idx=train_donors_idx,
                    val_donors_idx=val_donors_idx,
                    multiple_donations=multiple_donations,
                    ALL_DATASETS=ALL_DATASETS,
                    mem=mem,
                    seed_manager=seed_manager,
                    return_models=True,
                    outdir= outdir ,

                )

                f1_across_folds.append(np.mean(f1_scores_list))  # append the average across seeds
                tuning_results.append(results_list)             # hyperparameters results
                val_predicted_for_roc_folds.append((predictions_list, original_val_y))  # store full prediction
                CV_models.append(models_list)

                fold_end = time.time()
                print(f'Fold: {fold_idx}: Time spent for the {(config['ncells'], config['nsubs'], config['bs'])} combination: {(fold_end- fold_start)}')


            best_thr = get_best_thr(val_predicted_for_roc_folds)
            #training_save_dir = f'{BinCellCnn_path}/experiments/experiment_{tuning_exp}/outer_fold_{LOPOCV_fold_idx}/training/'

            save_element(save_dir, 'best_thr', best_thr)
            save_element(save_dir, 'CV_models', CV_models)

    return CV_models, best_thr













