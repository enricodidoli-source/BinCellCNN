import copy
import pandas as pd
import random
import numpy as np
import sys

def B_H_data_extraction(dataset, blast = True):
    """
    Extracts blast or healthy cells from a dataset based on the 'IsBlast' column.
    """
    code = 1 if blast == True else 0 
    sub_data = dataset[dataset['IsBlast'] == code]
 
    data = sub_data.reset_index(drop=True)
    return data


def check_dataset_types(donor_datasets, log = False):
    """
    Returns 1 if any dataset in the donor contains blast cells, 0 otherwise.
    """
    for dataset in donor_datasets:
         # if there are blast cells in the timepoint
        if len(dataset[dataset['IsBlast'] == 1]) > 0:
            category = 1

            if log:
                condition = 'Unhealthy'
                print(f'Timepoint condition: {condition}')
            return category
            
    category = 0

    if log:
        condition = 'Healthy' 
        print(f'Timepoint condition: {condition}')
    return category
    
def sample_cells_new_dataset(data, per_dataset, seed):
    """
    Splits a dataset into subsets of specified sizes. If a subset requires more cells than available
    in its chunk, the remaining cells are resampled with replacement.
    """
    data = data.copy(deep=True) #copy data
    data_division = []
            
    # shuffle and divide the two datasets in chunks, from which extract final data
    data = data.sample(frac=1, random_state=seed).reset_index(drop=True)

    tot_sub = len(per_dataset)
    
    remaining_sub = tot_sub
    for s in range(tot_sub):
        mean_chunk_length = int(len(data)/remaining_sub)
       
        n_cells = int(per_dataset[s]) #retrieve number of blast cells to assign to the s-th subset
        if n_cells <= mean_chunk_length:
            cells = data[:n_cells]
                    
            data = data[n_cells:] #reduce the cells selection to those that have not been sampled yet
                    
        else: #n_blast_cells > mean_blast_chunk_length
            data_section = data[:mean_chunk_length]  # retrieve the subset assigned cells
            one_time_all_cells = data_section.copy(deep=True) # select all cells one time
                    
            remaining_cells_to_sample = int(n_cells - len(one_time_all_cells)) # n cells to be resampled
            resampled_cells = data_section.sample(remaining_cells_to_sample, replace = True, random_state = seed).reset_index(drop=True)
                    
            cells = pd.concat([one_time_all_cells, resampled_cells], ignore_index = True)
            data = data[mean_chunk_length :] #reduce the cells selection to those that have not been sampled yet

        remaining_sub = remaining_sub - 1
        data_division.append(cells)
                
    return data_division
    
def generate_new_datasets(donor_datasets_extracted, n_sub, n_cells, seed, per_perc = False, log = False, blast_perc = None):
    """
    Generates synthetic datasets from donor samples by mixing blast and healthy cells
    at random or fixed percentages. Blast donors produce both positive and negative subsets,
    healthy donors produce only negative subsets.
    """
    new_donor_datasets = []
    new_donor_y = []
    blast_per_dataset = []
    np.random.seed(seed)
    blast_data = pd.DataFrame()
    healthy_data = pd.DataFrame()

    if blast_perc is None:
        blast_percentages = [0.005, 0.01, 0.05, 0.1, 0.2] # blast percenteges
    else:
        blast_percentages = blast_perc
        

    condition = check_dataset_types(donor_datasets_extracted, log = log)
    
    # aggregate healthy and blast data form donor cells
    if log:
        print('Generation lists of healthy and blast cells per patient: Started...')
        
    for dt in donor_datasets_extracted:

        if condition == 1:
            blast_dataset_i = B_H_data_extraction(dt) #blast_data
            blast_data = pd.concat([blast_data, blast_dataset_i], ignore_index = True)
            print(f'Tot blast data in the donor timepoints: {len(blast_data)}')
        healthy_dataset_i = B_H_data_extraction(dt, False)  #healthy_data

        # create a single big dataset of blast or healthy cells
        healthy_data = pd.concat([healthy_data, healthy_dataset_i], ignore_index = True)
    print('Extraction Done')
    if log:
        print('Generation lists of healthy and blast cells per patient: Done!')
        
    if condition == 1:
        print(f'Condition: {condition}')

        #blast percentages per subset generation
        if not per_perc:
            blast_per_dataset = np.random.choice(blast_percentages, n_sub)  #chosed percenteges for each sub
             
        else:
            mod = len(blast_percentages)
            rem = int(n_sub % mod)
            n_times = int(n_sub / mod)

            if n_times != 0:
                blast_per_dataset = blast_percentages * n_times
                
            if rem != 0:
                
                blast_per_dataset = blast_per_dataset + blast_percentages[:rem]
                
        unsorted_blast_per_dataset = np.array(blast_per_dataset) * n_cells

        blast_per_dataset = np.sort(unsorted_blast_per_dataset) #total number of cells

        if log:
            print(f'Chosen # of blast cells: {unsorted_blast_per_dataset}')
            print(f'Chosen # of blast cells: {blast_per_dataset}')
            print('Percentages of Blast cells : Done!')

        # healthy numer of cells retriving
        healthy_per_dataset = []
        for b_c in blast_per_dataset:
            h_c = n_cells - int(b_c)
            healthy_per_dataset.append(h_c)
            if log:
                print(f'New Generated Dataset: healthy = {h_c}, blast = {b_c}')
                
        healthy_per_dataset = healthy_per_dataset + [n_cells]*n_sub # divisions

        # sampling section
        blast_data_division = sample_cells_new_dataset(blast_data, blast_per_dataset, seed)
        healthy_data_division = sample_cells_new_dataset(healthy_data, healthy_per_dataset, seed)

    
        for i in range(n_sub):
            new_dataset = pd.concat([healthy_data_division[i], blast_data_division[i]], ignore_index = True)
            new_donor_datasets.append(new_dataset)
            new_donor_y.append(1)
            
        for i in range(n_sub, n_sub*2):
            new_donor_datasets.append(healthy_data_division[i])
            new_donor_y.append(0)
            
    else:
        print(f'Condition: {condition}')
        healthy_per_dataset = [n_cells]*n_sub
        healthy_data_division = sample_cells_new_dataset(healthy_data, healthy_per_dataset, seed)
        
        for i in range(n_sub):
            new_donor_datasets.append(healthy_data_division[i])
            new_donor_y.append(0)
    return new_donor_datasets, new_donor_y


def splitting_and_dataset_elaboration(train_datasets_extracted, val_datasets_extracted, test_datasets_extracted, n_sub, n_cells, seed, 
                                      cv = False, per_perc = False, log = False, blast_perc = None):
    """
    Generates synthetic datasets for train, validation and test splits by calling generate_new_datasets
    per donor. If cv=True, returns data nested by donor instead of flattened.
    """
    new_train_datasets = []
    new_train_y = []
    
    new_val_datasets = []
    new_val_y = []
    
    new_test_datasets = []
    new_test_y = []

    print(f'New training datasets creation...')
    print(len(train_datasets_extracted))
    print(len(train_datasets_extracted[0]))

    if cv:
        cv_train = []
        cv_train_y = []
        cv_val = []
        cv_val_y = []
        
    for donor_datasets in train_datasets_extracted:

        gen_results = generate_new_datasets(donor_datasets, n_sub, n_cells, seed, per_perc = per_perc, log = log, blast_perc = blast_perc)
        new_train_datasets += gen_results[0]
        new_train_y += gen_results[1]

        if cv:
            cv_train.append(gen_results[0])
            cv_train_y.append(gen_results[1])
        seed += 1 
    
    print(f'New validation datasets creation...')
    for donor_datasets in val_datasets_extracted:

        gen_results = generate_new_datasets(donor_datasets, n_sub, n_cells, seed, per_perc = per_perc, log = log, blast_perc = blast_perc)
        new_val_datasets += gen_results[0]
        new_val_y += gen_results[1]

            
        if cv:
            cv_val.append(gen_results[0])
            cv_val_y.append(gen_results[1])
        seed += 1 
    
    print(f'New test datasets creation...')
    for donor_datasets in test_datasets_extracted:

        gen_results = generate_new_datasets(donor_datasets, n_sub, n_cells, seed, per_perc = per_perc, log = log, blast_perc = blast_perc)
    
        new_test_datasets += gen_results[0]
        new_test_y += gen_results[1]
        seed += 1 

    if cv:
        return cv_train, cv_train_y, cv_val, cv_val_y, new_test_datasets, new_test_y
        
    return new_train_datasets, new_train_y, new_val_datasets, new_val_y, new_test_datasets, new_test_y








