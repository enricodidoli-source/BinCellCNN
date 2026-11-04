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



def load_data(data_path, ext = '*.csv', max_file = None, remove_control = None, mem = None, label_column = 'IsBlast'):
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
        blast_n = (dataset[label_column] == 1).sum()
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

    if mem:
        mem.save_and_clear(variables = [ALL_DATASETS], variables_names = ['ALL_DATASETS'], clear = True, force = True)
        return sorted_multiple_donations, None
    else:
        return sorted_multiple_donations, ALL_DATASETS



def donation_extraction(donors_idx, multiple_donations, ALL_DATASETS = None, mem = None):
    """ Retrieves specific donors datasets (for ex. donors for train) from all datasets list """

    if mem:
        (ALL_DATASETS,) = mem.load_variables(['ALL_DATASETS'])

    datasets_extracted = []
    for donor in donors_idx:
        donor_datasets = multiple_donations[donor]

        donor_donations = []
        for donation in donor_datasets:
            donation_dataset = ALL_DATASETS[donation].drop(columns = ['Original_ID'])

            donor_donations.append(donation_dataset)

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




def retrieve_samples_info(data_folder_dir, multiple_donations, ALL_DATASETS = None, mem = None):
    """
    Checks sample files from a folder, extracts cell counts and blast percentages,
    and returns a DataFrame sorted by blast count.
    """

    if mem:
        (ALL_DATASETS,) = mem.load_variables(['ALL_DATASETS'])

    file_names = os.listdir(data_folder_dir)

    samples_info_dict = {}

    last_patient = '1'
    counter = -1
    for element in file_names:

        if 'csv' in element:
            element = element[:-4]
            splitted_element = element.split('GHE')[1]
            patient_id = splitted_element.split('_')[0]
            time_point_days = splitted_element.split('_')[1]


            if last_patient == str(patient_id):
                counter += 1
            else:
                counter = 0


            file_idx = multiple_donations[str(patient_id)][counter]

            sample = ALL_DATASETS[file_idx]

            blast_n = len(sample[sample['IsBlast'] == 1])
            cells_n = len(sample)
            healthy_n = len(sample[sample['IsBlast'] == 0])
            blast_p = round((blast_n/cells_n)*100,4)
            healthy_p = 100-blast_p
            samples_info_dict[file_idx] = [cells_n, healthy_n, blast_n, healthy_p, blast_p, patient_id, time_point_days]

            last_patient = str(patient_id)


    df = pd.DataFrame.from_dict(samples_info_dict)
    df = df.transpose()
    df.columns = ['cells_n', 'healthy_n', 'blast_n', 'healthy_p', 'blast_p', 'patient_id', 'time_point_days']
    df['Sample'] = df.index
    df = df.sort_values(by='blast_n')

    return df


def retrieve_sorted_pat_sample_ids(samples_info_dict):
    """
    Builds a sorted list of sample IDs in the format 'Pat_{id}_{timepoint}',
    ordered by patient id and then by index within each patient.
    """
    pat_ids = samples_info_dict['patient_id'].to_list()
    int_pat_ids = [int(i) for i in pat_ids]

    pat_sample_ids = []
    for i in range(1, np.max(int_pat_ids) + 1):

        if str(i) in pat_ids:
            df = samples_info_dict[samples_info_dict['patient_id'] == str(i)]
            df = df.sort_index()
            for s in range(df.shape[0]):
                time_point = df.iloc[s]['time_point_days']
                pat_sample_ids.append(f'Pat_{i}_{time_point}')

    return pat_sample_ids


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