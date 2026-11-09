''' 
load_data, donation_extraction, donor_division, patient_code_extraction
'''

import copy
import pandas as pd
import random
import numpy as np
import sys 
import glob
import os




def load_data(data_path, ext = '*.csv', mem = None, remove_columns: list = None):
    """
    Loads CSV datasets from a folder, organizes them by patient via patient_code_extraction,
    reassigns datasets with no ID, and returns a sorted patient-to-file-index mapping.
    """
    print(data_path)
    #files_list = glob.glob(os.path.join(data_path, ext))
    files_list = glob.glob(os.path.join(data_path, ext))

    ALL_DATASETS = []
    multiple_donations = {}
    samples_ids = []
    patient_samples = []
    counter = 0
    for file_path in files_list:
        #dataset = pd.read_csv(file_path, sep = ';', decimal = ',').astype('float32')
        dataset = pd.read_csv(file_path, sep=',', decimal='.').astype('float32')
        print(dataset)

        if remove_columns:
            for col in remove_columns:
                if col in dataset.columns:
                    print(col)
                    dataset = dataset.drop(columns=[col])
        ALL_DATASETS.append(dataset) # list of all datasets

        # divide the datasets by donor
        sequence = 'sample_'
        ext_length = len(ext) - 1

        if sequence in file_path:
            idx = file_path.find(sequence)
            sample_name = file_path[idx:-ext_length]

        samples_ids.append(sample_name)
        splitted_name = sample_name.split('_')

        patient_id = splitted_name[1]
        #sample_id = int(splitted_name[2])

        if patient_id not in multiple_donations:
            multiple_donations[patient_id] = [counter]
        else:
            multiple_donations[patient_id].append(counter)


        print(multiple_donations)

        counter += 1
        print(f"Elaborating file {splitted_name}: {file_path}") # information about the process

    #return
    patients = [int(key) for key in list(multiple_donations.keys())]

    sorted_multiple_donations = {}
    sorted_ALL_DATASETS = []

    for pat_idx in range(1, max(patients) + 1):
        if pat_idx in patients:
            sorted_multiple_donations[str(pat_idx)] = multiple_donations[str(pat_idx)]

    if mem:
        mem.save_and_clear(variables = [ALL_DATASETS], variables_names = ['ALL_DATASETS'], clear = True, force = True)
        return sorted_multiple_donations, None, samples_ids
    else:
        print(len(ALL_DATASETS))
        return sorted_multiple_donations, ALL_DATASETS, samples_ids


def donation_extraction(donors_idx, multiple_donations, ALL_DATASETS = None, mem = None, remove_columns = ['Original_ID'] ):
    """ Retrieves specific donors datasets (for ex. donors for train) from all datasets list """

    if mem:
        (ALL_DATASETS,) = mem.load_variables(['ALL_DATASETS'])

    datasets_extracted = []
    for donor in donors_idx:
        donor_datasets = multiple_donations[donor]

        donor_donations = []
        
        if remove_columns:
            for donation in donor_datasets:
                donation_dataset = ALL_DATASETS[donation]
                for col in remove_columns:
                    if col in donation_dataset.columns:

                        donation_dataset = donation_dataset.drop(columns=[col])
                donor_donations.append(donation_dataset)
        else:
            donor_donations = donor_datasets

        datasets_extracted.append(donor_donations)
    return datasets_extracted





def donor_division(multiple_donations: dict, all_datasets, label_column = 'IsBlast'):
    """
    Classifies each donor as healthy (all samples blast-free), blast (all samples with blast),
    or mixed (both), based on the 'IsBlast' column of their samples.
    """
    donors_labels = {}
    for donor, donations in multiple_donations.items():
        donor_l = []
        for don in donations:
            dataset = all_datasets[don]
            blast_cells = (dataset[label_column] == 1).sum()
            if blast_cells > 0:
                donor_l.append(1)
            else:
                donor_l.append(0)
        donors_labels[donor] = donor_l
    
    healthy_donors = []
    blast_donors = []
    mixed_donors = []
    for donor, donations_labels in donors_labels.items():
        if 1 in donations_labels and 0 not in donations_labels:
            
            blast_donors.append(donor)
        elif 0 in donations_labels and 1 not in donations_labels:
            healthy_donors.append(donor)
        else:
            mixed_donors.append(donor)
    return healthy_donors, blast_donors, mixed_donors





def take_first_n_patients(tot_perc_list, multiple_donations, n_patients = None):
    if n_patients is None:
        n_patients = len(tot_perc_list)

    print(n_patients)
    new_multiple_donations = {}
    counter = 0
    for pat_idx, samples_idx in multiple_donations.items():
        new_multiple_donations[pat_idx] = samples_idx
        counter += 1
        if counter == n_patients:
            break
    print(new_multiple_donations)
    
    return new_multiple_donations