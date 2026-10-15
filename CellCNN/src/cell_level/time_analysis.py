import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

from cell_level.cell_level_plotting import (
    plot_total_time
)

def analyse_method_time(time_df, filter_sel_thr_list, method_list, DEFAULT_KEY = 'Default'):
    comprehensive_time_dict = {}
    
    for method in method_list:
        if method == 'top3' or method == 'single':
            time_method_dict = {}
            time_method_dict[DEFAULT_KEY] = time_df[f'end_{method}_time'] - time_df[f'start_{method}_time']
            comprehensive_time_dict[method] = time_method_dict

        else:
            time_method_dict = extract_method_time(time_df, filter_sel_thr_list, method = method)
            comprehensive_time_dict[method] = time_method_dict

    return comprehensive_time_dict

def extract_method_time(time_df, filter_sel_thr_list, method):
    time_method_dict = {}

    filter_sel_times = [time_df[f'end_filter_selection_{thr}_time'] - time_df[f'start_filter_selection_{thr}_time']
                    for thr in filter_sel_thr_list]
    time_method_dict['filter_sel'] = np.mean(filter_sel_times)

    for thr in filter_sel_thr_list:
        time_method_dict[thr] = time_df[f'end_{thr}_{method}_time'] - time_df[f'start_{thr}_{method}_time']

    return time_method_dict



def analyse_time(single_run_plotdir, folds_to_test, method_list, filter_sel_thr_list):
    filters_extraction_time = []
    for LOPOCV_fold_idx in range(0,folds_to_test):
        with open(f'{single_run_plotdir}/time_results/filters_extraction/filters_time{LOPOCV_fold_idx}.pkl', 'rb') as f:
            filters_time = pickle.load(f)
            
        filters_extraction_time.append(filters_time)
    filters_extraction_time_df = pd.DataFrame(filters_extraction_time)
    filters_sel_time = filters_extraction_time_df['end_filters_extraction_time'] - filters_extraction_time_df['start_filters_extraction_time']
    filters_sel_time = filters_sel_time.to_numpy()
    
    all_time_dict = {}
    for i, method in enumerate(method_list):
        time_pipeline_results = []
        for LOPOCV_fold_idx in range(0,folds_to_test):
        
            with open(f'{single_run_plotdir}/time_results/{method}/time_results{LOPOCV_fold_idx}.pkl', 'rb') as f:
                time_results = pickle.load(f)
        
            time_pipeline_results.append(time_results)
            
        time_df = pd.DataFrame(time_pipeline_results)
    
        all_time_dict.update(analyse_method_time(time_df, filter_sel_thr_list, method_list=[method]))
    
    plot_total_time(all_time_dict, filters_sel_time, method_list)
    plt.tight_layout()
    plt.show()


