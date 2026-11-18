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

import matplotlib.pyplot as plt




def get_sample_perc(samples_info_df):
    """
    Builds a sorted list of sample IDs in the format 'Pat_{id}_{timepoint}',
    ordered by patient id and then by index within each patient.
    """
    pat_sample_percs = []
    for row in samples_info_df.iterrows():
        row_values = row[1]

        perc = round(row_values['blast_p'],4)
        pat_sample_percs.append(perc)
    return pat_sample_percs





def show_blast_distribution_perc(samples_info_df, return_perc = False, decimals = 4, samples_ids = False):
    """
    Computes and shows the blast cell percentage for each sample across all patients.
    Optionally returns the percentage list.
    """
    if samples_ids:
        samples_ids = get_sample_ids(samples_info_df)

    tot_perc_list = samples_info_df['blast_p'].to_list()
    tot_perc_list = [round(perc, decimals) for perc in tot_perc_list]
    positions = range(1, len(tot_perc_list) + 1)

    fig, ax1 = plt.subplots(figsize = [len(positions)/2,4])
    ax1.bar(list(range(1, len(tot_perc_list) + 1)), [max(tot_perc_list)]*len(tot_perc_list), alpha = 0.5)
    ax1.bar(list(range(1, len(tot_perc_list) + 1)), tot_perc_list)

    ax1.set_xticks(positions)

    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(ax1.get_xticks())

    if samples_ids:
            ax1.set_xticklabels(samples_ids, size='x-small', rotation=60)#, ha='left')

    ax2.set_xticklabels(tot_perc_list, size = 'x-small')
    plt.show()

    if return_perc:
        return tot_perc_list

    return

def get_patient_identifier(file_path, sequence = 'sample_', ext = '*.csv'):
    """
    Extracts the patient identifier from a file name.
    """
    no_ext_path = file_path.split(ext[1:])[0]
    print('no_ext_path', no_ext_path)
    if sequence in no_ext_path:
        #idx = no_ext_path.find(sequence)
        sample_code = no_ext_path.split(sequence)[1]


        #splitted_element = sample_code.split(sequence)[1]

        splitted_code = sample_code.split('_')

        patient_id = splitted_code[0]
        time_point_days = splitted_code[1]

        return int(patient_id), int(time_point_days)
    else:
        return None, None

def get_sample_ids(samples_info_df):
    """
    Builds a sorted list of sample IDs in the format 'Pat_{id}_{timepoint}',
    ordered by patient id and then by index within each patient.
    """
    pat_sample_ids = []
    for row in samples_info_df.iterrows():
        row_values = row[1]

        pat_idx = int(row_values['patient_id'])
        time_point = int(row_values['time_point_days'])

        pat_sample_ids.append(f'sample_{pat_idx}_{time_point}')

    return pat_sample_ids



def load_data(config, data_folder_path, mem = None): #, remove_columns = None, label_column = 'IsBlast', sequence = 'sample_', labels_level = 'cell'):

    remove_columns = config['remove_columns']
    label_column = config['label_column']
    labels_level = config['labels_level']
    sequence = config['sequence']
    separator = config['separator']
    decimal = config['decimal']
    ext = config['extension']

    files_list = glob.glob(os.path.join(data_folder_path, ext))
    files_list = sorted(
        files_list,
        key=lambda f: tuple(
            float(x) for x in get_patient_identifier(f, sequence=sequence, ext=ext)
        )
    )
    print(files_list)
    ALL_DATASETS = []
    ALL_DATASETS_LABELS = []
    multiple_donations = {}
    samples_info_dict = {}
    counter = 0

    for file_path in files_list:

        patient_id, time_point_days = get_patient_identifier(file_path, sequence = sequence, ext = ext)

        str_patient_id = str(int(patient_id))
        if str_patient_id not in multiple_donations.keys():

                    multiple_donations[str_patient_id] = []
                    multiple_donations[str_patient_id].append(counter)
        else:
                    multiple_donations[str_patient_id].append(counter)


        sample = pd.read_csv(file_path, sep = separator, decimal = decimal).astype('float32')
        
        if labels_level == 'cell':
            if label_column not in sample.columns:
                raise ValueError(f"Label column '{label_column}' not found in the DataFrame. \nSet labels_level == 'sample'")

        if remove_columns:
            for col in remove_columns:
                if col in sample.columns:
                    print('col', col)
                    sample = sample.drop(columns=[col])

        ALL_DATASETS.append(sample)

        
        data_labels_folder_path = f'{data_folder_path}/samples_labels'
        label_path = os.path.join(data_labels_folder_path, f'{sequence}y_{patient_id}_{time_point_days}.csv')
        sample_y = pd.read_csv(label_path, sep = separator, decimal = decimal).astype('float32')
        label = int(sample_y['label'].iloc[0])
        print(label)

        ALL_DATASETS_LABELS.append(label)


        if labels_level == 'cell':
            sample_info = get_sample_info(sample, label_column)
            samples_info_dict[counter] = sample_info + [patient_id, time_point_days, label]
        else:
            samples_info_dict[counter] = [len(sample), patient_id, time_point_days, label]


        print(f"Elaborating file {counter}: {file_path}") # log
        counter += 1


    samples_info_df = get_sorted_df(samples_info_dict, labels_level = labels_level)


    sorted_patients = np.unique(samples_info_df['patient_id'])
    sorted_ALL_DATASETS, sorted_ALL_DATASETS_LABELS, sorted_multiple_donations = sort_dataset(ALL_DATASETS = ALL_DATASETS,
                                                                                              ALL_DATASETS_LABELS = ALL_DATASETS_LABELS,
                                                                                              multiple_donations = multiple_donations, 
                                                                                              sorted_patients = sorted_patients)

    if mem:
        mem.save_and_clear(variables = [sorted_ALL_DATASETS, sorted_ALL_DATASETS_LABELS], variables_names = ['ALL_DATASETS', 'ALL_DATASETS_LABELS'], clear = True, force = True)
        return None, None, sorted_multiple_donations, samples_info_df
    else:
        return sorted_ALL_DATASETS, sorted_ALL_DATASETS_LABELS, sorted_multiple_donations, samples_info_df



def get_sorted_df(samples_info_dict, labels_level = 'cell'):
    df = pd.DataFrame.from_dict(samples_info_dict)
    df = df.transpose()

    if labels_level == 'cell':
        df.columns = ['cells_n', 'healthy_n', 'blast_n', 'healthy_p', 'blast_p', 'patient_id', 'time_point_days', 'label']
    
    elif labels_level == 'sample':
        df.columns = ['cells_n', 'patient_id', 'time_point_days', 'label']

    df['Sample'] = df.index

    df = df.sort_values(['patient_id', 'time_point_days'], kind='stable')
    return df


def sort_dataset(ALL_DATASETS, multiple_donations, sorted_patients, ALL_DATASETS_LABELS = None):
    sorted_multiple_donations = {}
    sorted_ALL_DATASETS = []
    cond = ALL_DATASETS_LABELS is not None
    print('cond:', cond)
    if cond:
        sorted_ALL_DATASETS_LABELS = []

    for pat_idx in sorted_patients:
        str_patient_id = str(int(pat_idx))

        for sample_idx in multiple_donations[str_patient_id]:
            int_sample_idx = int(sample_idx)
            sorted_ALL_DATASETS.append(ALL_DATASETS[int_sample_idx])
            if cond:

                sorted_ALL_DATASETS_LABELS.append(ALL_DATASETS_LABELS[int_sample_idx])

        sorted_multiple_donations[str_patient_id] = multiple_donations[str_patient_id]

    if cond:
        return sorted_ALL_DATASETS, sorted_ALL_DATASETS_LABELS, sorted_multiple_donations
    else:
        return sorted_ALL_DATASETS, sorted_multiple_donations

def get_sample_info(sample, label_column):

    blast_n = (sample[label_column] == 1).sum()
    cells_n = len(sample)
    healthy_n = cells_n - blast_n

    blast_p = (blast_n/cells_n)*100

    healthy_p = 100-blast_p
    sample_info = [cells_n, healthy_n, blast_n, healthy_p, blast_p]
    return sample_info


def donation_extraction(donors_idx, multiple_donations, ALL_DATASETS = None, mem = None, remove_columns = ['Original_ID'] ):
    """ Retrieves specific donors datasets (for ex. donors for train) from all datasets list """

    if mem:
        (ALL_DATASETS,) = mem.load_variables(['ALL_DATASETS'])

    datasets_extracted = []
    for donor in donors_idx:
        donor_datasets = multiple_donations[donor]

        donor_donations = []

        for donation in donor_datasets:
            donation_dataset = ALL_DATASETS[donation]
            if remove_columns:
                for col in remove_columns:
                    if col in donation_dataset.columns:

                        donation_dataset = donation_dataset.drop(columns=[col])
            donor_donations.append(donation_dataset)

        datasets_extracted.append(donor_donations)

    return datasets_extracted

def label_extraction(donors_idx, multiple_donations, ALL_DATASETS_LABELS = None, mem = None):
    """ Retrieves specific donors datasets (for ex. donors for train) from all datasets list """

    if mem:
        (ALL_DATASETS_LABELS,) = mem.load_variables(['ALL_DATASETS_LABELS'])

    labels_extracted = []
    for donor in donors_idx:
        donor_labels = multiple_donations[donor]

        donor_donations = []

        for donation in donor_labels:
            donation_label = ALL_DATASETS_LABELS[donation]
            donor_donations.append(donation_label)

        labels_extracted.append(donor_donations)

    return labels_extracted

def sample_label_extraction(donors_idx, multiple_donations, ALL_DATASETS = None, ALL_DATASETS_LABELS = None, mem = None, remove_columns = None):
    """ Retrieves specific donors datasets (for ex. donors for train) from all datasets list """

    test_datasets_extracted = donation_extraction(donors_idx, multiple_donations, ALL_DATASETS, mem, remove_columns = remove_columns)
    test_labels_extracted = label_extraction(donors_idx, multiple_donations, ALL_DATASETS_LABELS, mem)

    return test_datasets_extracted, test_labels_extracted


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




def take_first_n_patients(config,  multiple_donations):
    n_patients = config['n_patients']
    
    if n_patients is None:
        return multiple_donations


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