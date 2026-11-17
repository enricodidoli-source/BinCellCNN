    
import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt

from src.shared.timepoints_elaboration import get_sample_info, get_sorted_df, sort_dataset

'''
patient_code_extraction, retrieve_samples_info, load_data, retrieve_sorted_pat_sample_ids, 
'''
def get_patient_identifier(file_path, sequence = 'B-ALL_GHE', ext = '*.csv'):
    """
    Extracts the patient identifier from a file name.
    """
    no_ext_path = file_path.split(ext[1:])[0]
    #print('no_ext_path', no_ext_path)
    if sequence in no_ext_path:
        sample_code = no_ext_path.split(sequence)[1]
        splitted_code = sample_code.split('_')

        patient_id = splitted_code[0]

        time_point_days = splitted_code[1][1:]
         #print('time_point_days', time_point_days)
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

        pat_sample_ids.append(f'Pat_{pat_idx}_D{time_point}')

    return pat_sample_ids


def load_data(data_folder_path, ext = '*.csv', max_file = None,  mem = None, remove_columns = None, label_column = 'IsBlast', sequence = 'sample_', labels_level = 'cell'):

    files_list = glob.glob(os.path.join(data_folder_path, ext))
    files_list = sorted(
        files_list,
        key=lambda f: tuple(
            float(x) for x in get_patient_identifier(f, sequence=sequence, ext=ext)
        )
    )

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


        sample = pd.read_csv(file_path, sep = ';', decimal = ',').astype('float32')
        
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
        label_path = os.path.join(data_labels_folder_path, f'sample_y_{sequence}{patient_id}_D{time_point_days}.csv')
        sample_y = pd.read_csv(label_path, sep = ';', decimal = ',').astype('float32')

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
    sorted_ALL_DATASETS, sorted_ALL_DATASETS_LABELS, sorted_multiple_donations = sort_dataset(ALL_DATASETS, ALL_DATASETS_LABELS, multiple_donations, sorted_patients)

    if mem:
        mem.save_and_clear(variables = [sorted_ALL_DATASETS, sorted_ALL_DATASETS_LABELS], variables_names = ['ALL_DATASETS', 'ALL_DATASETS_LABELS'], clear = True, force = True)
        return None, None, sorted_multiple_donations, samples_info_df
    else:
        return sorted_ALL_DATASETS, sorted_ALL_DATASETS_LABELS, sorted_multiple_donations, samples_info_df


'''
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
'''


'''



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
    

def get_sample_info(sample, label_column):
  
    blast_n = (sample[label_column] == 1).sum()
    cells_n = len(sample)
    healthy_n = cells_n - blast_n

    blast_p = (blast_n/cells_n)*100

    healthy_p = 100-blast_p
    sample_info = [cells_n, healthy_n, blast_n, healthy_p, blast_p]
    return sample_info



def sort_dataset(ALL_DATASETS, multiple_donations, sorted_patients):
    sorted_multiple_donations = {}
    sorted_ALL_DATASETS = []
    for pat_idx in sorted_patients:
        str_patient_id = str(int(pat_idx))

        #print('pat_idx', int(pat_idx))
        #print('samples', multiple_donations[str_patient_id])

        for sample_idx in multiple_donations[str_patient_id]:
            #print('sample_idx', sample_idx)
            #print(ALL_DATASETS[sample_idx])
            sorted_ALL_DATASETS.append(ALL_DATASETS[sample_idx])

        sorted_multiple_donations[str_patient_id] = multiple_donations[str_patient_id]
    return sorted_ALL_DATASETS, sorted_multiple_donations
'''
'''
def get_sorted_df(samples_info_dict):
    df = pd.DataFrame.from_dict(samples_info_dict)
    df = df.transpose()
    df.columns = ['cells_n', 'healthy_n', 'blast_n', 'healthy_p', 'blast_p', 'patient_id', 'time_point_days']
    df['Sample'] = df.index

    df = df.sort_values(by=['patient_id', 'time_point_days'])
    return df
'''