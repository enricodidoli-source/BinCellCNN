import copy
import pandas as pd
import random
import numpy as np
import sys




def train_CellCNN_old(CellCnn, train_datasets, train_y,
                      n_cell = 10, nsubset = 50, max_epochs=5, nrun=1, seed = 42, hyper = None, 
                      val_datasets = None, val_y = None, 
                      generate = False, grid = False, outdir = None):
    """
    Trains CellCNN models. Uses default hyperparameters if none are provided,
    otherwise unpacks nfilter, maxpool_percentages and learning_rate from hyper.
    Supports optional validation set and generate mode.
    """
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

    model = CellCnn(
        ncell = n_cell, #200                # Number of cells per multi-cell input (sampled from the 'patient' datasets)
        nsubset = nsubset,                  # Total number of multi-cell inputs generated per class (or sample and class)
        per_sample = True,                  # For each sample, nsubset samples of ncell are performed
        nfilter_choice = nfilter,           # Range of possible number of filters
        maxpool_percentages = maxpool_p,    # list of k-percentage max_pooling
        learning_rate = learning_r,         # Learning rate list
        max_epochs = max_epochs, #50
        patience=5,                         # Early stopping patience
        nrun = nrun,  #15                   # Number of neural network configurations to try (for Hyperparameter optimization)
        scale=True,                         # Z-score normalization
        verbose=1,
        seed = int(seed),
        grid = grid
    )

    if outdir is None:
        outdir = f'/content/cellcnn_results'  # Results Directory
        
    if val_datasets is not None:
        model.fit(
                train_samples = train_datasets,
                train_phenotypes = np.array(train_y),
                outdir=outdir,
                valid_samples = val_datasets,
                valid_phenotypes = np.array(val_y),
                generate_valid_set = False
            )
    else:
        model.fit(
                train_samples = train_datasets,
                train_phenotypes = np.array(train_y),
                outdir=outdir,
                generate_valid_set = generate
            )
    return model
    
def test_CellCNN_old(model, test_datasets_no_labels, seed = None):
    """
    Runs prediction on test datasets using a trained CellCNN model.
    Optionally passes a seed to model.predict.
    """
    if seed is not None:
        test_pred = model.predict((test_datasets_no_labels), seed = seed)
    else:
        test_pred = model.predict((test_datasets_no_labels))
    print(f'Done')
    return list(test_pred), model.results



'========================================================================================================================================'


def trials_train_CellCNN_old(CellCnn, train_datasets, train_y, 
                    trials = 10, max_epochs = 50, nrun = 15, 
                    n_cell = 100000, nsubset = 50, seed_list = None, hyper = None, val_datasets = None, val_y = None, 
                             generate = False, grid = False, outdir = None):   
    """
    Runs multiple training weights initializations of CellCNN, each with a different seed.
    If seed_list is missing or too short, generates one randomly.
    """
    models_lists = []
    if (seed_list is None) or (len(seed_list) < trials):
        if seed_list is not None and len(seed_list) < trials:
            print(f'Warning: seed_list is None or has not the adeguate length: {len(seed_list)} instead of {trials}.\n')
            
        seed_list = random.choices(list(range(1, 100000)), k=trials)
        print(f'seed_list is randomly set = {seed_list}.\n')
        print(f'To control randomization, please generate seed_list outside the function.')
    

    for i in range(trials):
        print(f'Weight initialization {i+1} started')
        training_seed = seed_list[i]
        print(f'Seed used: {training_seed}')
        if val_datasets is not None:
            model = train_CellCNN_old(CellCnn, train_datasets, train_y, 
                                      n_cell, nsubset, max_epochs, nrun, training_seed, hyper = hyper, 
                                      val_datasets = val_datasets, val_y = val_y, generate = False,  grid = grid, outdir = outdir)
    
        else:
            model = train_CellCNN_old(CellCnn, train_datasets, train_y,
                                      n_cell, nsubset, max_epochs, nrun, training_seed, hyper = hyper, generate = generate, grid = grid, outdir = outdir)
        models_lists.append(model)
    return models_lists
    
def trials_test_CellCNN_old(models_lists, test_datasets_no_labels, seed_list = None):
    
    """
    Runs prediction across multiple trained CellCNN models (seeds).
    If seed_list is missing or too short, generates one randomly.
    """
    predictions_list = []
    results_list = [] 
    tot_trials = len(models_lists)
    
    if (seed_list is None) or (len(seed_list) < tot_trials):
        
        if seed_list is not None and len(seed_list) < tot_trials:
            print(f'Warning: seed_list is None or has not the adequate length: {len(seed_list)} instead of {tot_trials}.\n')
            
        seed_list = random.choices(list(range(1, 100000)), k=tot_trials)
        print(f'seed_list is randomly set = {seed_list}.\n')
        print(f'To control randomization, please generate seed_list outside the function.')
    
        
                                   
    for i in range(tot_trials):
        print(f'Weight Initialization {i+1} out of {tot_trials} started!')
        prediction, result = test_CellCNN_old(models_lists[i], test_datasets_no_labels, seed = seed_list[i])
        
        predictions_list.append(prediction)
        results_list.append(result)
        
        print(f'Trial {i+1} Done!\n')
    
    return predictions_list, results_list
