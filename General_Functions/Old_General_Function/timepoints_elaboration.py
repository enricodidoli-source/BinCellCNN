import copy
import pandas as pd
import random
import numpy as np

import sys

def remove_from_cache(files: list):
    for file in files:
        if file in sys.modules:
            del sys.modules[file]
            print(f"{file} rimosso dalla cache")
        else:
            print(f"{file} non trovato nella cache")


def patient_code_extraction(text, counter, multiple_donations):
    """Divides the donations by donor
        Inpus:  - text: file_path
                - counter: i-th file elaborated
                - multiple_donations: dict. each donor code is associated to its sample datasets
        Outputs:
                - multiple_donations: dict updated with the new dataset elaborated 
    """

    sequence = 'B-ALL_GHE'
    if sequence in text:
        idx = text.find(sequence)
        code = text[idx:-4]
        #print(code)
        if True: #code[-2] == '_' and code[-1].isdigit():
            identifier = ''
            idx = code.find('GHE')
            patient_code =code[idx+3:]
            #print(patient_code)
            for i, element in enumerate(patient_code):
                if element.isdigit():
                    identifier += element
                else:
                    break
            
            if identifier not in multiple_donations.keys():
                
                multiple_donations[identifier] = []
                multiple_donations[identifier].append(counter)
            else:
                
                multiple_donations[identifier].append(counter)
    else:
        
        if 'no_id' not in multiple_donations.keys():
                
                multiple_donations['no_id'] = []
                multiple_donations['no_id'].append(counter)
        else:
                
                multiple_donations['no_id'].append(counter)

    return multiple_donations

'========================================================================================================================================'
    
def donor_division(multiple_donations: dict, all_datasets):
    donors = len(multiple_donations)

    #dataset_label_extraction
    donors_labels = {}
    for donor, donations in multiple_donations.items():
        donor_l = []
        for don in donations:
            dataset = all_datasets[don]
            blast_cells = (dataset['IsBlast'] == 1).sum()
            if blast_cells > 0:
                donor_l.append(1)
            else:
                donor_l.append(0)
        donors_labels[donor] = donor_l
    print(donors_labels)
    healthy_donors = []
    blast_donors = []
    mixed_donors = []
    for donor, donations_labels in donors_labels.items():
        if 1 in donations_labels and 0 not in donations_labels:
            
            healthy_donors.append(donor)
        elif 0 in donations_labels and 1 not in donations_labels:
            blast_donors.append(donor)
        else:
            mixed_donors.append(donor)
    return healthy_donors, blast_donors, mixed_donors


'========================================================================================================================================'

def splitting(healthy_donors, blast_donors, mixed_donors, healthy_donors_idx, blast_donors_idx, mixed_donors_idx, set_division = [2,1,2]):
    """Splits donors in train, validation and test according to the decided division"""
    
    train_donors_idx = []
    val_donors_idx = []
    test_donors_idx = []

    # healthy donors
    for i, don in enumerate(healthy_donors_idx):
        if i in range(set_division[0]):                 # append first t donors to train donors set
            train_donors_idx.append(healthy_donors[don])
        elif i in range(set_division[0], set_division[0] + set_division[1]):  # append v donors to validation donors set
            val_donors_idx.append(healthy_donors[don])
        else:                                           # append last (n - t - v) donors to test donors set
            test_donors_idx.append(healthy_donors[don])

    # blast donors
    for i, don in enumerate(blast_donors_idx):
        if i == 0:
            train_donors_idx.append(blast_donors[don])
        elif i == 1:
            val_donors_idx.append(blast_donors[don])
        else:
            test_donors_idx.append(blast_donors[don])

    # mixed donors       
    for i, don in enumerate(mixed_donors_idx):
        if i in range(set_division[0]):
            train_donors_idx.append(mixed_donors[don])
        elif i in range(set_division[0], set_division[0] + set_division[1]):
            val_donors_idx.append(mixed_donors[don])
        else:
            test_donors_idx.append(mixed_donors[don])
    return train_donors_idx, val_donors_idx, test_donors_idx


def dataset_elaboration(multiple_donations, ALL_DATASETS, healthy_donors, blast_donors,
                        mixed_donors, n_sub = 3, seed = 42):
    """ Samples donors for Train, Validation and Test sets"""
    
    train_donors = []
    val_donors = []
    test_donors = []
    
    random.seed(seed)
    print(f'Precess starts. Dividing donors...')
    
    # sammple indexed for donor division
    healthy_donors_idx = random.sample(list(range(len(healthy_donors))), len(healthy_donors))
    blast_donors_idx = random.sample(list(range(len(blast_donors))), len(blast_donors))
    mixed_donors_idx = random.sample(list(range(len(mixed_donors))), len(mixed_donors))
    print(f'healthy_donors_idx, blast_donors_idx, mixed_donors_idx: {healthy_donors_idx}, {blast_donors_idx},{mixed_donors_idx}')

    print(f'Seting Train, Validation and Test idx...')
    # just divide accoding to the sampled indexes
    train_donors_idx, val_donors_idx, test_donors_idx = splitting(healthy_donors, blast_donors, mixed_donors, healthy_donors_idx, blast_donors_idx, mixed_donors_idx)
    print(train_donors_idx, val_donors_idx, test_donors_idx)

    return train_donors_idx, val_donors_idx, test_donors_idx

'========================================================================================================================================'

def donation_extraction(donors_idx, multiple_donations, ALL_DATASETS):
    """ Retrieves specific donors datasets (for ex. donors for train) from all datasets list """
    datasets_extracted = []
    for donor in donors_idx:
        donor_datasets = multiple_donations[donor]
        #print(donor_datasets)
        donor_donations = []
        for donation in donor_datasets:
            donation_dataset = ALL_DATASETS[donation].drop(columns = ['Original_ID'])
            
            donor_donations.append(donation_dataset)
            
        datasets_extracted.append(donor_donations)
    return datasets_extracted

