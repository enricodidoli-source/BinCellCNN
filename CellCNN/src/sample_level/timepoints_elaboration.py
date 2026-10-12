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



def load_data(data_path, ext = '*.csv', max_file = None, remove_control = None):
    """
    Loads CSV datasets from a folder, organizes them by patient via patient_code_extraction,
    reassigns datasets with no ID, and returns a sorted patient-to-file-index mapping.
    """
    files_list = glob.glob(os.path.join(data_path, ext))
    if max_file is None:
        max_file = len(files_list)
        
    ALL_DATASETS = []
    multiple_donations = {}
    no_id = []
    counter = 0
    if remove_control:
        files_list = [file_path for file_path in files_list if 'GHE' in file_path]
                
    for file_path in files_list[:max_file]:
        
        dataset = pd.read_csv(file_path, sep = ';', decimal = ',').astype('float32')
        ALL_DATASETS.append(dataset) # list of all datasets
        blast_n = (dataset['IsBlast'] == 1).sum()
        perc = round((blast_n/len(dataset))*100, 2)

        # divide the datasets by donors
        multiple_donations = patient_code_extraction(file_path, counter, multiple_donations)
        
        print(f"Elaborating file {counter}: {file_path}") # information about the process
        
        counter += 1 
    
    # Fix no_id datasets
    last_identifier = 0
    for element in multiple_donations.keys():
        if element.isdigit():
            if int(element) > int(last_identifier):
                last_identifier = int(element)

    if not remove_control:
      if 'no_id' in list(multiple_donations.keys()):
        for dataset in multiple_donations['no_id']:
            last_identifier += 1
            multiple_donations[str(last_identifier)] = [dataset]
        multiple_donations.pop('no_id')

    patients = [int(key) for key in list(multiple_donations.keys())]

    sorted_multiple_donations = {}
    sorted_ALL_DATASETS = []

    for pat_idx in range(1, max(patients) + 1):
        if pat_idx in patients:
            sorted_multiple_donations[str(pat_idx)] = multiple_donations[str(pat_idx)]

    return sorted_multiple_donations, ALL_DATASETS 


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
    """
    Classifies each donor as healthy (all samples blast-free), blast (all samples with blast),
    or mixed (both), based on the 'IsBlast' column of their samples.
    """
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


def donation_extraction(donors_idx, multiple_donations, ALL_DATASETS):
    """ Retrieves specific donors datasets (for ex. donors for train) from all datasets list """
    datasets_extracted = []
    for donor in donors_idx:
        donor_datasets = multiple_donations[donor]

        donor_donations = []
        for donation in donor_datasets:
            donation_dataset = ALL_DATASETS[donation].drop(columns = ['Original_ID'])
            
            donor_donations.append(donation_dataset)
            
        datasets_extracted.append(donor_donations)
    return datasets_extracted

