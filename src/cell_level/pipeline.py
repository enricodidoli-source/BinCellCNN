import os
import gc
import time
import pickle
import ctypes
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression

from memory_utils import clear_RAM, MemorySave

from cell_level.cell_level_utils import (
    extract_data_from_fold, 
    reload_from_temp, temp_key, 
    remove_temp, 
    TimeCalculator,
    save_all_results_singularly, 
    load_training_models
)


from cell_level.networks_ops import (
    get_full_network_prob, 
    get_filter_prob, 
    logit
)
from cell_level.filter_selection import (
    retrieve_filters_scalers,
    select_clustered_filters, 
    rebuild_metadata,         
    apply_backup_if_needed
)
from cell_level.evaluation import (
    evaluate_single_filter, 
    get_best_thr, 
    get_metrics_df,
    add_classification_metrics
)

from sample_level.timepoints_elaboration import donation_extraction
from sample_level.utils import flatten, retrieve_labels



def filters_selection_phase(cross_val_models_list, train_kfolds, multiple_donations, ALL_DATASETS, LOPOCV_fold_idx, DIST_THRESHOLD, FILTER_SEL_THRESHOLD, all_filter_metadata_df = None ):
    """Orchestrates filter selection across all cross-validation folds."""
    all_selected_filters_metadata = []
    all_filters_metadata = []

    
    for cv, fold_loaded_models_lists in enumerate(cross_val_models_list): # cross validation fold
        print(f'=========== Analysing Fold {cv} ==============')
        
        if all_filter_metadata_df is None:
            selected_filters_metadata, filters_metadata = evaluate_folds_filters(train_kfolds, cv, multiple_donations, ALL_DATASETS,
                                                                                    fold_loaded_models_lists, LOPOCV_fold_idx, DIST_THRESHOLD, FILTER_SEL_THRESHOLD)

            all_filters_metadata += filters_metadata

        else:
        
            selected_filters_metadata = rebuild_metadata(all_filter_metadata_df, cv,  DIST_THRESHOLD, FILTER_SEL_THRESHOLD)

        all_selected_filters_metadata += selected_filters_metadata
        
    filter_metadata_df = apply_backup_if_needed(pd.DataFrame(all_selected_filters_metadata))

    clear_RAM()

    if all_filter_metadata_df is None:
        all_filter_metadata_df = pd.DataFrame(all_filters_metadata)

        return filter_metadata_df, all_filter_metadata_df
        
    return filter_metadata_df





def evaluate_folds_filters(train_kfolds, cv, multiple_donations, ALL_DATASETS, fold_loaded_models_lists, LOPOCV_fold_idx, DIST_THRESHOLD, FILTER_SEL_THRESHOLD):
    """Pipeline to evaluate filters for a specific fold and select the best ones."""

    _, val_pat = train_kfolds[cv]
    comprehensive_val_df, true_cells_labels = extract_data_from_fold(val_pat, multiple_donations, ALL_DATASETS)

    all_best_filters, all_scalers = retrieve_filters_scalers(fold_loaded_models_lists) # seed/net/filter

    filters_metadata = []
    selected_filters_metadata = []

    for m, (model_filters, model_scalers) in enumerate(zip(all_best_filters, all_scalers)): # for each seed

        model_best_thr, model_max_f1 = evaluate_single_filter(comprehensive_val_df, true_cells_labels, model_filters, model_scalers)

        metadata = {
            'lopo_fold': LOPOCV_fold_idx,
            'cv_fold': cv,
            'model_ID': m,
            'filter_weights': model_filters,
            'scaler': model_scalers,
            'best_thr': model_best_thr,
            'max_f1': model_max_f1
        } 
        filters_metadata.append(metadata)


        selected_metadata = select_clustered_filters(metadata, DIST_THRESHOLD, FILTER_SEL_THRESHOLD)
        selected_filters_metadata.append(selected_metadata)

    clear_RAM()
    return selected_filters_metadata, filters_metadata


# ==================================



def best_cells_threshold_phase(filter_metadata_df, train_kfolds, multiple_donations, ALL_DATASETS, method='mean', show_plot = True,  max_cells = 10000000, random_state = 0, mem = None):

    all_cells_filtered = []
    all_cells_labels = []
    clf_list = []

    # make this elaboraition for each loaded model
    cv_folds = len(filter_metadata_df['cv_fold'].unique()) # cross validation fold
    print(f'Number of CV folds: {cv_folds}')
    for cv in range(cv_folds): # cross validation fold
        print(f'=========== Analysing Fold {cv} ==============')

        selected_cvfold_df = filter_metadata_df[filter_metadata_df['cv_fold'] == cv]
        selected_filters = selected_cvfold_df['filter_weights'].to_list()
        selected_scalers = selected_cvfold_df['scaler'].to_list()

        if len(flatten(selected_filters)) == 0:
            print(f'No filter for {cv}-th fold')
            if method == 'logistic':
                clf_list.append(None)
            continue

        train_pat, val_pat = train_kfolds[cv]
        comprehensive_val_df, true_cells_labels = extract_data_from_fold(val_pat, multiple_donations, ALL_DATASETS)
        if 'IsBlast' in comprehensive_val_df.columns:
            comprehensive_val_df = comprehensive_val_df.drop(columns=['IsBlast'])

        if method == 'logistic':
            clf = train_clf(train_pat, multiple_donations, ALL_DATASETS, selected_filters, selected_scalers, max_cells = max_cells, random_state = random_state)
            clf_list.append(clf)

            val_cells_probs = compute_mean_cells_probs(selected_filters, selected_scalers, comprehensive_val_df, compute_mean = False)
            val_cells_probs = logit(pd.DataFrame(val_cells_probs).T)
            val_scores = clf.predict_proba(val_cells_probs)[:, 1]
            try:
                ALL_DATASETS, = mem.load_variables(['ALL_DATASETS'])

            except:
                print('Variable not deleted')
                pass
                
        elif method == 'mean':
            val_scores = compute_mean_cells_probs(selected_filters, selected_scalers, comprehensive_val_df)

        else:
            raise ValueError("Parameter 'method' must be 'mean' or 'logistic'")

        all_cells_filtered.append(val_scores)
        all_cells_labels += true_cells_labels.tolist()

    all_cells_filtered = np.concatenate(all_cells_filtered)

    best_thr, max_f1  = get_best_thr(all_cells_labels, all_cells_filtered, show = show_plot)

    if method == 'logistic':
        return best_thr, max_f1, clf_list

    clear_RAM()
    return best_thr, max_f1



def test_phase(filter_metadata_df, test_pat, multiple_donations, ALL_DATASETS,
               max_f1, best_thr, LOPOCV_fold_idx, clf_list = None):

    """Executes the final test phase using the globally determined thresholds and filters."""
    all_selected_filters = flatten(filter_metadata_df['filter_weights'].to_list())
    all_selected_scalers = flatten(filter_metadata_df['scaler'].to_list())

    test_datasets_extracted = donation_extraction(test_pat, multiple_donations, ALL_DATASETS)
    test_datasets_extracted_flat_with_labels, val_y = retrieve_labels(test_datasets_extracted, remove = False, flat = True)
    del test_datasets_extracted

    LOPOCV_results = []
    all_samples_test = []

    for idx_x, x_df in enumerate(test_datasets_extracted_flat_with_labels):

        # Extract labels safely without destroying original DataFrame if needed later
        y_true = x_df['IsBlast'].to_numpy()
        x_features = x_df.drop(columns=['IsBlast'])

        if clf_list is not None:
            test_scores = compute_test_logistic(filter_metadata_df, x_features, clf_list)
        else:
            test_scores = compute_mean_cells_probs(all_selected_filters, all_selected_scalers, x_features)
        y_pred = (test_scores >= best_thr).astype(int)

        results_dict = get_metrics_df(y_true, y_pred, max_f1, LOPOCV_fold_idx, idx_x)
        LOPOCV_results.append(results_dict)

    clear_RAM()
    return LOPOCV_results



# =================================



def best_3_filt_threshold_phase(cross_val_models_list, train_kfolds,  multiple_donations, ALL_DATASETS, show_plot = True):
    all_preds = []
    all_labels = []

    for cv, fold_loaded_models_lists in enumerate(cross_val_models_list): # cross validation fold
        print(f'=========== Analysing Fold {cv} ==============')

        _, val_pat = train_kfolds[cv]
        val_datasets_extracted = donation_extraction(val_pat, multiple_donations, ALL_DATASETS)
        val_datasets_extracted_flat, val_y = retrieve_labels(val_datasets_extracted, remove = False, flat = True)
        del val_datasets_extracted
        gc.collect()

        for s, x_df in enumerate(val_datasets_extracted_flat):
            y_true = x_df['IsBlast'].to_numpy()
            x_features = x_df.drop(columns=['IsBlast'])


            preds = retrieve_flags(x_features, fold_loaded_models_lists) # adding x_name we save the dataset

            all_preds.append(preds)
            all_labels.append(y_true)

    del val_datasets_extracted_flat
    gc.collect()

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    print('=========== Finding Best Threshold ==============')

    best_thr, max_f1 = get_best_thr(all_labels, all_preds, show = show_plot)

    clear_RAM()
    return best_thr, max_f1


def test_3_filt_phase(test_pat, multiple_donations, ALL_DATASETS, cross_val_models_list, max_3_filt_f1, best_3_filt_thr, LOPOCV_fold_idx):
    test_datasets_extracted = donation_extraction(test_pat, multiple_donations, ALL_DATASETS)
    test_datasets_extracted_flat, test_y = retrieve_labels(test_datasets_extracted, remove = False, flat = True)

    LOPOCV_3_filt_results = []
    for idx_x, x_df in enumerate(test_datasets_extracted_flat):
        y_true = x_df['IsBlast'].to_numpy()
        x_features = x_df.drop(columns=['IsBlast'])

        sample_preds_across_folds = []
        for cv, fold_loaded_models_lists in enumerate(cross_val_models_list):

            print(f'=========== Analysing Sample {idx_x} ==============')
            preds = retrieve_flags(x_features, fold_loaded_models_lists)
            sample_preds_across_folds.append(preds)

        final_preds = np.mean(sample_preds_across_folds, axis=0)

        y_pred = (final_preds >= best_3_filt_thr).astype(int)

        results_dict = get_metrics_df(y_true, y_pred, max_3_filt_f1, LOPOCV_fold_idx, idx_x)
        LOPOCV_3_filt_results.append(results_dict)

    clear_RAM()
    return LOPOCV_3_filt_results



def analyse_single_filters(all_filter_metadata_df, test_pat, multiple_donations, ALL_DATASETS, LOPOCV_fold_idx):
    single_filters_test_results = []
    cols = ['filter_weights', 'scaler', 'best_thr', 'max_f1']
    all_df = all_filter_metadata_df[cols].explode(cols, ignore_index=True)

    for index, row in all_df.iterrows(): # iterrows() is more efficient than for loop using range(len())

        temp_metadata_df = pd.DataFrame({
            'filter_weights': [row['filter_weights']],
            'scaler': [row['scaler']]
        })

        test_results = test_phase(filter_metadata_df = temp_metadata_df, test_pat = test_pat, multiple_donations = multiple_donations,
                                  ALL_DATASETS = ALL_DATASETS, max_f1 = row['max_f1'], best_thr = row['best_thr'], LOPOCV_fold_idx = LOPOCV_fold_idx)

        for res_dict in test_results:
            res_dict['Filter_F1'] = row['max_f1']
            single_filters_test_results.append(res_dict)

    clear_RAM()
    return single_filters_test_results



# ============================================


def train_clf(train_pat, multiple_donations, ALL_DATASETS, selected_filters, selected_scalers, max_cells = None, random_state = 0, mem = None):

    train_comprehensive_train_df, true_cells_labels = extract_data_from_fold(train_pat, multiple_donations, ALL_DATASETS)
    try:
        mem.save_and_clear([ALL_DATASETS], ['ALL_DATASETS'], )
    except:
        print('Variable not deleted')
        pass

    print( len(train_comprehensive_train_df))

    if max_cells is None:
        max_cells = len(train_comprehensive_train_df)

    limitation = min(len(train_comprehensive_train_df), max_cells)
    train_comprehensive_train_df = train_comprehensive_train_df.iloc[:limitation]
    true_cells_labels = true_cells_labels[:limitation]


    if 'IsBlast' in train_comprehensive_train_df.columns:
        train_comprehensive_train_df = train_comprehensive_train_df.drop(columns=['IsBlast'])

    train_cells_probs = compute_mean_cells_probs(selected_filters, selected_scalers, train_comprehensive_train_df, compute_mean = False)

    del train_comprehensive_train_df #, ALL_DATASETS
    clear_RAM()

    train_cells_probs =logit(pd.DataFrame(train_cells_probs).T)

    clf = LogisticRegression(class_weight="balanced", random_state=random_state).fit(train_cells_probs, true_cells_labels)

    clear_RAM()
    return clf




def compute_test_logistic(filter_metadata_df, x, clf_list):

        test_all_cells_filtered = []
        cv_folds = len(filter_metadata_df['cv_fold'].unique()) # cross validation fold
        print(f'Number of CV folds: {cv_folds}')
        for cv in range(cv_folds): # cross validation fold
            print(f'=========== Analysing Fold {cv} ==============')

            selected_cvfold_df = filter_metadata_df[filter_metadata_df['cv_fold'] == cv]
            selected_filters = selected_cvfold_df['filter_weights'].to_list()
            selected_scalers = selected_cvfold_df['scaler'].to_list()

            if len(flatten(selected_filters)) == 0:
                print(f'No filter for {cv}-th fold')
                continue

            test_cells_probs = compute_mean_cells_probs(selected_filters, selected_scalers, x, compute_mean = False)
            test_cells_probs = logit(pd.DataFrame(test_cells_probs).T)
            test_scores = clf_list[cv].predict_proba(test_cells_probs)[:, 1]

            test_all_cells_filtered.append(test_scores)

        test_scores = np.mean(test_all_cells_filtered, axis = 0)

        clear_RAM()
        return test_scores




def compute_mean_cells_probs(selected_filters, selected_scalers, comprehensive_val_df, compute_mean = True):
    """Calculates predictions across multiple filters. Averaged if compute_mean is True."""

    filters_flat = flatten(selected_filters)
    scalers_flat = flatten(selected_scalers)

    if not filters_flat:
        return np.array([])

    val_all_cells = []
    for filter, scaler in zip(filters_flat, scalers_flat):
        pred_probs = get_filter_prob(filter, comprehensive_val_df, scaler=scaler)
        val_all_cells.append(pred_probs)

    del comprehensive_val_df
    gc.collect()

    if compute_mean == True:
        return np.mean(val_all_cells, axis=0).ravel()


    clear_RAM()
    return val_all_cells




def retrieve_flags(x, fold_loaded_models_lists):

    all_probs_list =  []
    for m, model in enumerate(fold_loaded_models_lists): # several seeds (our case 1 seed only)

        for net in model.results['best_3_nets']:

            all_probs = get_full_network_prob(net, x,  model.results['scaler'])
            all_probs_list.append(all_probs)

    all_probs_mean = np.mean(all_probs_list, axis = 0).ravel()

    clear_RAM()
    return all_probs_mean



# ============================================



def run_logistic_analysis(filter_metadata_df, train_kfolds, test_pat, multiple_donations, ALL_DATASETS, LOPOCV_fold_idx, max_cells = 2000000, random_state = 0, mem = None):

    best_logistic_thr, max_logistic_f1, clf_list = best_cells_threshold_phase(filter_metadata_df, train_kfolds,
                                                                                  multiple_donations, ALL_DATASETS,
                                                                                  method='logistic',  max_cells = max_cells,
                                                                                  random_state = random_state, mem = mem)

    try: # reload ALL_DATASETS becuase it has been deleted during best_cells_threshold_phase
        (ALL_DATASETS,) = mem.load_variables(['ALL_DATASETS'])
    except:
        print('Variable not loaded')
        pass

    LOPOCV_logistic_results = test_phase(filter_metadata_df, test_pat, multiple_donations, ALL_DATASETS,
                  max_logistic_f1, best_logistic_thr, LOPOCV_fold_idx, clf_list = clf_list)

    clear_RAM()
    return LOPOCV_logistic_results


def run_mean_analysis(filter_metadata_df, train_kfolds, test_pat, multiple_donations, ALL_DATASETS, LOPOCV_fold_idx, mem = None):

    best_avg_thr, max_f1 = best_cells_threshold_phase(
                                filter_metadata_df, train_kfolds,
                                multiple_donations, ALL_DATASETS, mem = mem)

    try:
        (ALL_DATASETS,) = mem.load_variables(['ALL_DATASETS'])
    except:
        print('Variable not loaded')
        pass

    LOPOCV_results = test_phase(filter_metadata_df, test_pat, multiple_donations, ALL_DATASETS,
                                    max_f1, best_avg_thr, LOPOCV_fold_idx)
    clear_RAM()
    return LOPOCV_results


def run_top3_filters_analysis(cross_val_models_list, train_kfolds, test_pat, multiple_donations, ALL_DATASETS, LOPOCV_fold_idx, mem = None):

    best_3_filt_thr, max_3_filt_f1 = best_3_filt_threshold_phase(cross_val_models_list, train_kfolds,
                                                                 multiple_donations, ALL_DATASETS)
    try:
        (ALL_DATASETS,) = mem.load_variables(['ALL_DATASETS'])
    except:
        print('Variable not loaded')
        pass

    LOPOCV_3_filt_results = test_3_filt_phase(test_pat, multiple_donations, ALL_DATASETS,
                                              cross_val_models_list, max_3_filt_f1, best_3_filt_thr, LOPOCV_fold_idx)
    clear_RAM()
    return LOPOCV_3_filt_results



def fold_iteration(cellcnn_path, single_run_plotdir, LOPOCV_patients_folds, LOPOCV_fold_idx, exp, filter_sel_thr_list, multiple_donations, ALL_DATASETS, DIST_THRESHOLD, random_state = 0, max_cells = 2000000, mem = None,
                   method = 'mean', DEFAULT_KEY = 'Default'):
    clear_RAM()

    SINGLE_RUN_METHODS = ('single', 'top3')
    VALID_METHODS = SINGLE_RUN_METHODS +  ('mean', 'logistic')
    if method not in VALID_METHODS:
        raise ValueError(f"Unknown method: {method}. Available methods: {VALID_METHODS}")


    print(f'\n======= Analysing LOPOCV fold {LOPOCV_fold_idx} =======\n')

    model_cvfolds_dir = f'{cellcnn_path}/experiments/experiment_{exp}/outer_fold_{LOPOCV_fold_idx}/ensemble/training/'

    cross_val_models_list = load_training_models(model_cvfolds_dir)
    train_kfolds, test_pat = LOPOCV_patients_folds[LOPOCV_fold_idx] # for each LOPO fold

    metadata_dir = os.path.join(single_run_plotdir, 'filters_metadata')
    os.makedirs(metadata_dir, exist_ok = True)

    metadata_list_in_dir = os.listdir(metadata_dir)
    print(metadata_list_in_dir)

    if method != 'top3':
        if f'all_filters_metadata{LOPOCV_fold_idx}.pkl' not in metadata_list_in_dir:

            filter_tim = TimeCalculator() # Reset Timer
            filter_tim.add_time('start_filters_extraction_time')

            _, all_filter_metadata_df = filters_selection_phase(cross_val_models_list, train_kfolds, multiple_donations,
                                                                                ALL_DATASETS, LOPOCV_fold_idx, DIST_THRESHOLD, FILTER_SEL_THRESHOLD = 0)
            filter_tim.add_time(f'end_filters_extraction_time')

            save_all_results_singularly(metadata_dir,
                                            saving_list = [all_filter_metadata_df],
                                            name_list = ['all_filters_metadata'],
                                            LOPOCV_fold_idx = LOPOCV_fold_idx)

            # save time required to select filters
            filters_time = filter_tim.return_time()

            metadata_time_dir = os.path.join(single_run_plotdir, 'time_results', 'filters_extraction')
            os.makedirs(metadata_time_dir, exist_ok = True)

            save_all_results_singularly(metadata_time_dir,
                                            saving_list = [filters_time],
                                            name_list = ['filters_time'],
                                            LOPOCV_fold_idx = LOPOCV_fold_idx)
            clear_RAM(['all_filter_metadata_df', 'filters_time'])


        tim = TimeCalculator() # Reset Timer

        if method == 'single':
            tim.add_time(f'start_{method}_time')

            with open(f'{metadata_dir}/all_filters_metadata{LOPOCV_fold_idx}.pkl', 'rb') as f:
                all_filter_metadata_df = pickle.load(f)

            print(f'\n======= Analyse all filters singularly =======\n')
            LOPOCV_analysis = analyse_single_filters(all_filter_metadata_df, test_pat, multiple_donations, ALL_DATASETS, LOPOCV_fold_idx)

            clear_RAM(['all_filter_metadata_df'])
        else:

            for FILTER_SEL_THRESHOLD in filter_sel_thr_list:
                tim.add_time(f'start_filter_selection_{FILTER_SEL_THRESHOLD}_time')

                print(f'Analysis using Filter Selection Thr = {FILTER_SEL_THRESHOLD}')

                with open(f'{metadata_dir}/all_filters_metadata{LOPOCV_fold_idx}.pkl', 'rb') as f:
                    all_filter_metadata_df = pickle.load(f)

                filter_metadata_df = filters_selection_phase(cross_val_models_list, train_kfolds, multiple_donations,
                                                                                    ALL_DATASETS, LOPOCV_fold_idx, DIST_THRESHOLD, FILTER_SEL_THRESHOLD, all_filter_metadata_df)
                clear_RAM(['all_filter_metadata_df'])

                tim.add_time(f'end_filter_selection_{FILTER_SEL_THRESHOLD}_time')

                tim.add_time(f'start_{FILTER_SEL_THRESHOLD}_{method}_time')

                if method == 'logistic':

                    print(f'\n======= BEST FILTERS: LOGISTIC REGRESSION using filter {FILTER_SEL_THRESHOLD} =======\n')
                    LOPOCV_analysis = run_logistic_analysis(filter_metadata_df, train_kfolds, test_pat, multiple_donations, ALL_DATASETS,
                                                                LOPOCV_fold_idx, max_cells = max_cells, random_state = random_state, mem = mem)
                elif method == 'mean':

                    print(f'\n======= BEST FILTERS: MEAN =======\n')
                    LOPOCV_analysis = run_mean_analysis(filter_metadata_df, train_kfolds, test_pat, multiple_donations,
                                                      ALL_DATASETS, LOPOCV_fold_idx, mem = mem)

                tim.add_time(f'end_{FILTER_SEL_THRESHOLD}_{method}_time')
                name_var = temp_key(method = method, thr = FILTER_SEL_THRESHOLD, fold_idx = LOPOCV_fold_idx)
                
                mem.save_and_clear([LOPOCV_analysis], [name_var], clear=True,  force = True)

                clear_RAM(['filter_metadata_df'])

                print('')
                print('')
                print('')
                print('')
                print(f'%%%%%% End filter {FILTER_SEL_THRESHOLD} %%%%%')
                print('')
                print('')
                print('')
                print('')

            LOPO_dict = reload_from_temp(filter_sel_thr_list, method = method, fold_idx = LOPOCV_fold_idx, mem = mem)

    elif method == 'top3':
        tim = TimeCalculator() # Reset Timer
        tim.add_time(f'start_{method}_time')

        print(f'\n======= BEST 3 MODELS =======\n')
        LOPOCV_analysis = run_top3_filters_analysis(cross_val_models_list, train_kfolds, test_pat, multiple_donations, ALL_DATASETS, LOPOCV_fold_idx, mem = mem)

    if method == 'top3' or method == 'single':
        LOPO_dict = {}
        LOPO_dict[DEFAULT_KEY] = LOPOCV_analysis
        clear_RAM([f'LOPOCV_analysis'])

        tim.add_time(f'end_{method}_time')

    # === Store results === #
    method_dir = os.path.join(single_run_plotdir, method)
    os.makedirs(method_dir, exist_ok = True)

    save_all_results_singularly(method_dir,
                                saving_list = [LOPO_dict],
                                name_list = [f'LOPO_{method}_dict'],
                                LOPOCV_fold_idx = LOPOCV_fold_idx)
    clear_RAM([f'LOPO_dict'])

    #tim.add_time('end_time')
    time_results = tim.return_time()

    time_results_dir = os.path.join(single_run_plotdir, 'time_results', method)
    os.makedirs(time_results_dir, exist_ok = True)

    save_all_results_singularly(time_results_dir,
                                saving_list = [time_results],
                                name_list = ['time_results'],
                                LOPOCV_fold_idx = LOPOCV_fold_idx)

    print(f'\n======= {LOPOCV_fold_idx} fold ended =======\n')

    remove_temp(filter_sel_thr_list, method, fold_idx = LOPOCV_fold_idx, mem = mem)


    clear_RAM(clear_all = True)
