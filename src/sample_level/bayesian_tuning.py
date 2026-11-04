''' 
BO_trial, save_tuning_results, save_metadata_and_timing, BO_tuning
find_theta_best, alternative_elaborate_BO_results, create_BO_study  
'''
import os
import pickle
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler 

import random
import time

#from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve

from src.sample_level.training import inner_fold_iteration
from src.sample_level.classification import get_best_thr, phenotype_prediction
from src.shared.utils import save_element


def find_theta_best(f1_tested_par, tested_par):
    """
    Finds the best parameter combination by selecting the configuration with the highest F--score.
    If multiple values scores the max(f1), the 75th percentile of the top-scoring combinations,
     sorted by their parameter sum is chosen
    """
    tested_par = {i: par[:2] for i, par in tested_par.items()}
    best_combinations = {}
    print(f1_tested_par.values())
    max_f1 = np.max(list(f1_tested_par.values()))
    print('max_f1', max_f1)
    for i, f1_par in f1_tested_par.items():
        if f1_par == max_f1:
            best_combinations[i] = tested_par[i]

    if len(best_combinations) == 1:
            return list(best_combinations.keys())[0]


    print('best_combinations', best_combinations)

    sum_best_combinations = [np.sum(par) for par in best_combinations.values()]

    print('sum_best_combinations', sum_best_combinations)
    sort_idx = np.argsort(sum_best_combinations)

    sorted_best_combinations = np.array(list(best_combinations.values()))[sort_idx]
    sorted_best_combinations_idx = np.array(list(best_combinations.keys()))[sort_idx]
    print('sorted_best_combinations', sorted_best_combinations)
    print('sorted_best_combinations_idx', sorted_best_combinations_idx)
    #print('sorted_sum_best_combinations', sum_best_combinations[sort_idx])

    q3_idx = int(len(sorted_best_combinations_idx) * 0.75)
    chosen_par_idx = sorted_best_combinations_idx[q3_idx]
    print('chosen_par_idx', chosen_par_idx)
    return chosen_par_idx




def alternative_elaborate_BO_results(val_predicted_for_roc, pruned_par_idxs, tested_par):

    f1_tested_par = {}
    for bayesian_idx, val_predicted_for_roc_folds in val_predicted_for_roc.items(): # BO trial
        if bayesian_idx not in pruned_par_idxs:

            pred_list_list = []
            val_y_list = []
            f1_mean_list = []
            for fold_pred_list, val_y in val_predicted_for_roc_folds: # fold
                    print('fold_pred_list', fold_pred_list)
                    seed_pred_list = np.mean(fold_pred_list, axis = 0) # average across weigths initialization
                    print('seed_pred_list', seed_pred_list)
                    predictions_list = []
                    for pos_sample_pred, y in zip(seed_pred_list, val_y):
                        print('sample_pred', pos_sample_pred)
                        predictions_list.append(pos_sample_pred)

                    #prec, rec, thr = metrics.roc_curve(val_y, predictions_list, pos_label=1)# compute ROC curve
                    prec, rec, thr = roc_curve(val_y, predictions_list, pos_label=1)# compute ROC curve
                    f1_curve = 2 * prec * rec / (prec + rec + 1e-12)
                    best_idx = np.argmax(f1_curve[:-1])
                    best_thr = thr[best_idx]

                    print('best_thr', best_thr)
                    if np.isinf(best_thr):
                        best_thr = 0.5

                    pred_phenotypes = phenotype_prediction(predictions_list, thr = best_thr)
                    f1 = f1_score(val_y, pred_phenotypes)
                    f1_mean_list.append(f1)
                    print('')

            f1_tested_par[bayesian_idx] = np.mean(f1_mean_list)

    print('f1_tested_par', f1_tested_par)
    print('tested_par', tested_par)
    chosen_par_idx = find_theta_best(f1_tested_par, tested_par) # deterministic algorithm for \theta
    best_ncells, best_nsub, best_bs = tested_par[chosen_par_idx]
    print('chosen_par_idx', chosen_par_idx)
    print('best_ncells, best_nsub, best_bs', best_ncells, best_nsub, best_bs)
    return chosen_par_idx, best_ncells, best_nsub, best_bs



def create_BO_study(LOPOCV_fold_idx, seed_manager):
        if seed_manager:
            sampler_seed = seed_manager.get_seed(f'sampler_seed_LOPOCV_fold_{LOPOCV_fold_idx}')
            print(f"LOPO fold {LOPOCV_fold_idx}: Optuna sampler seed = {sampler_seed}")
            sampler = TPESampler(seed = sampler_seed)
        else:
            sampler = TPESampler()
            print('No seed manager provided. Reproducibility is not guaranteed.')

        study = optuna.create_study(direction="maximize", sampler = sampler) # direction="maximize"because we want to maximize the f1-score
        return study

def BO_trial(tuning_config, train_kfolds, study, param_list_dict, benchmark_list,
                    BinCellCnn, multiple_donations, ALL_DATASETS,
                    outdir, mem, seed_manager, tuning_save_dir):

    is_pruned = False  # pruning flag
    ncell_start = time.time()

    param_dict = {}
    trial = study.ask()  # Ask parameters
    for name, (min_n, max_n, step_n) in param_list_dict.items():
        value_n = trial.suggest_int(name, min_n, max_n, step=step_n)
        param_dict[name] = value_n

    ncells, nsubs, bs = param_dict['ncells'], param_dict['nsubs'], param_dict['bs']
    print(f'Testing params -> ncells: {ncells}, nsubs: {nsubs}, bs: {bs}')

    print('5 Fold CV started')

    tuning_predictions_folds = []
    val_predicted_for_roc_folds = []
    f1_across_folds = []
    tuning_results = []

    trial_config = {**tuning_config, 'ncells': ncells, 'nsubs': nsubs, 'bs': bs}
    CV_models = []
    for fold_idx, (train_donors_idx, val_donors_idx) in enumerate(train_kfolds):
        print(f'\nFold: {fold_idx}\n')
        fold_start = time.time()

        f1_scores_list, predictions_list, original_val_y, results_list, models_list = inner_fold_iteration(
            BinCellCnn=BinCellCnn,
            train_donors_idx=train_donors_idx,
            val_donors_idx=val_donors_idx,
            config = trial_config,
            multiple_donations=multiple_donations,
            ALL_DATASETS=ALL_DATASETS,
            outdir=outdir,
            mem=mem,
            seed_manager=seed_manager,
            return_models=True
        )

        CV_models.append(models_list)

        f1_across_folds.append(np.mean(f1_scores_list))  # append the average across seeds
        tuning_results.append(results_list)             # hyperparameters results
        val_predicted_for_roc_folds.append((predictions_list, original_val_y))  # store full prediction

        fold_end = time.time()
        print(f'Fold: {fold_idx}: Time spent for the {(ncells, nsubs)} combination: {(fold_end- fold_start)}')


        # === Pruning at the third fold === #
        if fold_idx == 2:
            mean_3_folds = np.mean(f1_across_folds)  # compute folds f1_score
            trial.report(mean_3_folds, step=2)       # Always report so the Bayesian Sampler sees the intermediate value

            # prune after 3 complete 5 fold cv
            if len(benchmark_list) >= 3 and mean_3_folds < np.median(benchmark_list):
                print("Pruning criteria met at fold 3. Stopping loop, but saving data...")
                is_pruned = True
                break  # break the fold loop

    tuning_CV_models_dir = os.path.join(tuning_save_dir, f'bayesian/combination_{ncells}_{nsubs}/')
    os.makedirs(tuning_CV_models_dir, exist_ok=True)
    save_element(tuning_CV_models_dir, 'CV_models', CV_models)

    print('Cross Validation Ended! Computing thresholds, and saving results...')

    # === Finalize with Optuna based on the flag === #
    if is_pruned:
        study.tell(trial, state=optuna.trial.TrialState.PRUNED)  # Tell the study it was pruned, but we still reach this line!
    else:
        study.tell(trial, np.mean(f1_across_folds))  # Tell to Bayesian search the results.

    print(f'Mean F1 Score across folds: {f1_across_folds}')
    print(f'Time Spent for this trial: {time.time() - ncell_start}')

    return val_predicted_for_roc_folds, f1_across_folds, tuning_results, ncells, nsubs, bs, is_pruned



def save_metadata_and_timing(tuning_save_dir, starting_seed, LOPOCV_fold_idx,
                              seed_manager, bayesian_LOPOCV_time):
    os.makedirs(tuning_save_dir, exist_ok=True)

    seed_info = {
        'starting_seed': starting_seed,
        'LOPOCV_fold_idx': LOPOCV_fold_idx,
        'fold_base_seed': starting_seed + (LOPOCV_fold_idx * 100000)
    }

    with open(os.path.join(tuning_save_dir, 'tuning_seed_info.pkl'), 'wb') as f:
        pickle.dump(seed_info, f)

    seed_manager.save_state(os.path.join(tuning_save_dir, 'save_manager_tuning_seed_info.pkl'))

    with open(os.path.join(tuning_save_dir, 'bayesian_LOPOCV_time.pkl'), 'wb') as f:
        pickle.dump(bayesian_LOPOCV_time, f)

    return

def BO_tuning(LOPOCV_fold_idx, train_kfolds, param_list_dict, BinCellCnn, multiple_donations,
    ALL_DATASETS, outdir, mem, seed_manager, tuning_config, tuning_save_dir):

    print('Start Bayesian Search for optimal ncells and nsubs parameters')

    # prepare storing variables
    tested_par = {}               # save parameter explored
    val_predicted_for_roc = {}    # store predictions from tuning for threshold tuning
    pruned_combinations = {}      # save the pruned combination for further analysis
    tuning_results_folds = []
    benchmark_list = []

    # Create the study outside the inner CV loop
    study = create_BO_study(LOPOCV_fold_idx, seed_manager)

    for bo_idx in range(tuning_config['bayesian_comb']):
        print(f'Model {bo_idx+1}')

        val_predicted_for_roc_folds, f1_across_folds, tuning_results, ncells, nsubs, bs, is_pruned = BO_trial(
            train_kfolds=train_kfolds,
            study=study,
            param_list_dict=param_list_dict,
            benchmark_list=benchmark_list,
            BinCellCnn=BinCellCnn,
            multiple_donations=multiple_donations,
            ALL_DATASETS=ALL_DATASETS,
            outdir=outdir,
            mem=mem,
            seed_manager=seed_manager,
            tuning_config = tuning_config,
            tuning_save_dir = tuning_save_dir
        )

        val_predicted_for_roc[bo_idx] = val_predicted_for_roc_folds
        tested_par[bo_idx] = [ncells, nsubs, bs]
        tuning_results_folds.append(tuning_results)

        if is_pruned:
            pruned_combinations[bo_idx] = (ncells, nsubs, bs)
            
            
        else:
            benchmark_list.append(np.mean(f1_across_folds[:3]))  # Update benchmark only for successful 5-fold runs

    print("Tuning Ended!")
    print('Start check the best combination of parameters')
    print(f'tested_par: {tested_par}')
    print('End check the best combination of parameters')

    pruned_par_idxs = list(pruned_combinations.keys())
    print(f'pruned_par_idx: {pruned_par_idxs}')

    chosen_par_idx, best_ncells, best_nsub, best_bs = alternative_elaborate_BO_results(
        val_predicted_for_roc,
        pruned_par_idxs,
        tested_par
    )
    val_predicted_for_roc_folds = val_predicted_for_roc[chosen_par_idx]
    
    # === Tune ROC threshold === #
    best_thr = get_best_thr(val_predicted_for_roc_folds)

    data_to_dump = {
        'tuning_results_folds.pkl': tuning_results_folds,
        'best_ncells.pkl': best_ncells,
        'best_nsub.pkl': best_nsub,
        'best_bs.pkl': best_bs,
        'tested_par.pkl': tested_par,
        'val_predicted_for_roc.pkl': val_predicted_for_roc,
        'pruned_combinations.pkl': pruned_combinations,
        'best_thr.pkl': best_thr
    }

    save_element(tuning_save_dir, 'threshold_data', data_to_dump)

    return best_ncells, best_nsub, best_bs, best_thr
