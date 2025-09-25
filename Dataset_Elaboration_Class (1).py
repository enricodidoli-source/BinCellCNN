import copy
import pandas as pd
import random
import numpy as np


class DataElaboration:
    def __init__(self):
        return


def read_data(data_path, sep = ';', dec = ','):
        '''For single datafile'''
        import pandas as pd

        # transform data to be in the right format
        raw_df = pd.read_csv(data_path, sep = ';', decimal = ',')
        df = raw_df.copy()

        print('### Data converted in a Padas Dataframe ###')
        return df, raw_df


def dataset_split(df, n_cells=20000, blast_perc=0.005, n_samples=20, blast_variable='IsBlast', seed=42):
    """
    Splits dataset into training, validation, and test sets with proper seeding for reproducibility.

    Args:
        df: Input dataframe
        n_cells: Number of cells per sample
        blast_perc: Percentage of blast cells
        n_samples: Number of samples to create
        blast_variable: Column name for blast indicator
        seed: Random seed for reproducibility
    """

    # Set all random seeds for full reproducibility
    random.seed(seed)
    np.random.seed(seed)

    var = blast_variable
    df = copy.deepcopy(df)

    blast_df = df[df[var] == 1]  # extract blast cells
    blast_df = blast_df.drop(columns=[var])

    healthy_df = df[df[var] == 0]  # extract healthy cells
    healthy_df = healthy_df.drop(columns=[var])

    #### healthy chunk division ####
    tot_healthy_cells = len(healthy_df)
    healthy_chunk_dim = int(tot_healthy_cells / n_samples)
    healthy_df_chunks = []
    counter = 0

    healthy_df = healthy_df.sample(frac=1, random_state=seed) # set the random state to reproduce the splitting
    for i in range(n_samples):
        sample_i = healthy_df.iloc[counter: counter + healthy_chunk_dim]
        counter += healthy_chunk_dim
        healthy_df_chunks.append(sample_i)

    #### Blast chunk division ####
    tot_blast_cells = len(blast_df)
    blast_chunk_dim = int(tot_blast_cells / int(n_samples/2))
    blast_df_chunks = []
    counter = 0

    blast_df = blast_df.sample(frac=1, random_state=seed)
    for i in range(int(n_samples/2)):
        sample_i = blast_df.iloc[counter: counter + blast_chunk_dim]
        counter += blast_chunk_dim
        blast_df_chunks.append(sample_i)

    ### Healthy and Blast Samples ###
    all_samples = []
    all_samples_labels = []
    healthy_dataset_1 = []
    healthy_y_1 = []
    blast_dataset_1 = []
    blast_y_1 = []

    blast_n = int(blast_perc * n_cells) # number of blast cells have to be added artificially
    healthy_n = n_cells - blast_n

    # create different healthy donor samples, artificially
    for i in range(int(n_samples/2)):
        healthy_chunk = healthy_df_chunks[i]

        healthy_sample = healthy_chunk.sample(n=n_cells, replace=False, random_state=seed+i) # modified random seed to avoid inteferences
        healthy_dataset_1.append(healthy_sample)
        healthy_y_1.append(0)

    # create different non-healthy donor samples artificially
    for i in range(int(n_samples/2)):
        healthy_chunk = healthy_df_chunks[i + int(n_samples/2)]
        blast_chunk = blast_df_chunks[i]

        # FIXED: Add random_state for reproducible sampling
        healthy_b_sample = healthy_chunk.sample(n=healthy_n, replace=False, random_state=seed+i+n_samples+100)
        blast_h_sample = blast_chunk.sample(n=blast_n, replace=False, random_state=seed+i+n_samples+200)

        blast_dataset_1.append(pd.concat([healthy_b_sample, blast_h_sample]))
        blast_y_1.append(1)

   ### Training, Validation and Test Splitting ###

    # sample indexes for train dataset
    dataset_idx = list(range(len(healthy_dataset_1)))
    train_perc = int(0.7 * len(healthy_dataset_1))

    # FIXED: Use random.Random with seed for consistent behavior
    rng = random.Random(seed)
    healthy_train_idx = rng.sample(dataset_idx, train_perc)
    blast_train_idx = rng.sample(dataset_idx, train_perc)

    # sample indexes for test dataset
    healthy_test_idx = list(set(dataset_idx) - set(healthy_train_idx))
    blast_test_idx = list(set(dataset_idx) - set(blast_train_idx))

    # sample indexes for val dataset
    val_perc = int(0.2 * len(healthy_train_idx))

    healthy_val_idx = rng.sample(healthy_train_idx, val_perc)
    blast_val_idx = rng.sample(blast_train_idx, val_perc)

    healthy_train_idx = list(set(healthy_train_idx) - set(healthy_val_idx))
    blast_train_idx = list(set(blast_train_idx) - set(blast_val_idx))

    # extract datasets and labels using indexes
    train_dataset_1 = []
    train_y_1 = []
    for i, j in zip(healthy_train_idx, blast_train_idx):
        train_dataset_1.append(healthy_dataset_1[i])
        train_dataset_1.append(blast_dataset_1[j])
        train_y_1.append(healthy_y_1[i])
        train_y_1.append(blast_y_1[j])

        all_samples.append(healthy_dataset_1[i])
        all_samples.append(blast_dataset_1[j])
        all_samples_labels.append(healthy_y_1[i])
        all_samples_labels.append(blast_y_1[j])

    val_dataset_1 = []
    val_y_1 = []
    for i, j in zip(healthy_val_idx, blast_val_idx):
        val_dataset_1.append(healthy_dataset_1[i])
        val_dataset_1.append(blast_dataset_1[j])
        val_y_1.append(healthy_y_1[i])
        val_y_1.append(blast_y_1[j])

        all_samples.append(healthy_dataset_1[i])
        all_samples.append(blast_dataset_1[j])
        all_samples_labels.append(healthy_y_1[i])
        all_samples_labels.append(blast_y_1[j])

    test_dataset_1 = []
    test_y_1 = []
    for i, j in zip(healthy_test_idx, blast_test_idx):
        test_dataset_1.append(healthy_dataset_1[i])
        test_dataset_1.append(blast_dataset_1[j])
        test_y_1.append(healthy_y_1[i])
        test_y_1.append(blast_y_1[j])

        all_samples.append(healthy_dataset_1[i])
        all_samples.append(blast_dataset_1[j])
        all_samples_labels.append(healthy_y_1[i])
        all_samples_labels.append(blast_y_1[j])

    return train_dataset_1, train_y_1, val_dataset_1, val_y_1, test_dataset_1, test_y_1, all_samples, all_samples_labels


def df_to_array(df):
    array = []
    for sample in df:
      array.append(sample.values)
    return array


def show_hyperparameters(model_hyperparameters):
    total_trial_parametes = []
    #print('### Hyperparameters ###\n')

    for idx, configuration in enumerate(model_hyperparameters):
        #print(f'Trial {idx+1}')

        trial_parameters = []
        for i in range (3):
            parameters = f'Model {i+1} -> '
            for key, value in configuration.items():
                if len(value) < 3:
                    par = 0
                else:
                    par = value[i]
                parameters += f'{key}: {par}, '
            parameters = parameters[:-2]
            if i == 0:
                parameters += ' <- Best Model of the trial'
            #print(parameters)
            trial_parameters.append(parameters)

        total_trial_parametes.append(trial_parameters)
            #print(f'Filters: {nfilter[i]}, Learning Rate: {learning_rate[i]}, Top-k Percentage Max Pooling: {maxpool_percentage[i]} ')
            #print(f'{key}: {value}')

        #print(f'\n')
    return total_trial_parametes

def phenotype_prediction(test_pred):
    pred_phenotypes = []
    for sample_pred in test_pred:
        if sample_pred[0] < 0.5:
            pred_phenotypes.append(1)
        else:
            pred_phenotypes.append(0)
    return pred_phenotypes


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

def divide_donations(multiple_donations: dict):
    """Divides the datasets to mantain divided donations from same donors
        Inputs: - multiple_donations: dict. each donor code is associated to its sample datasets  
        
        Outputs:- first: list of first donations datasets
                - second: list of datasets which are not first
    """
    first = []
    second = []
    
    for key, value in multiple_donations.items():
        if key == 'no_id': # if there are no information about donor, add it to training section
            for idx in value:
                first.append(idx)
                
        elif len(value) == 1:
            first.append(value[0])
        else:
            first.append(value[0])
            for idx in value[1:]:
                
                second.append(idx)
    return first, second


def idx_to_dataset(idx_list, all_datasets):
    """ Retrieve dataset indexes from the entire list of datasets.
        Prevents sampling datasets that comes form the same donors (for training)"""
    dataset = []
    for idx in idx_list:
        dataset.append(all_datasets[idx])
    return dataset


def retrieve_labels(dataset):
    """ Computes the number of blast cells in a sample returning the labels of the datasets
        Outputs:- clear_datasets: list of datasets without unnecessary columns
                - dataset_labels: list of binary lables. 1 for datasets with blast cells
    
    """
    
    clear_datasets = []
    dataset_labels = []
    #copy_all_datasets = copy.deepcopy(ALL_DATASETS)
    
    for i, dataset in enumerate(dataset):
        df = dataset.drop(columns = ['Original_ID'])
        blast = (dataset['IsBlast'] == 1).sum() #computes number of blast cells
        
        if blast > 0: #if blasts are present, then the donor is not healthy
            dataset_labels.append(1)
        else:
            dataset_labels.append(0)
        clear_datasets.append(df)
        
        df = dataset.drop(columns = ['IsBlast']) # drop not anymore necessary column
    #print(dataset_labels)
    return clear_datasets, dataset_labels

def class_division(datasets: list, dataset_labels: list):
    """ Separate datasets depending on their binary classification
    Outputs:- healthy_dataset: list
            - blast_dataset: list. List of datasets that contains at least 1 blast cell
    """
    healthy_dataset = []
    blast_dataset = []
    for i in range(len(dataset_labels)):
        if dataset_labels[i] == 1:
            blast_dataset.append(datasets[i])
        else:
            healthy_dataset.append(datasets[i])
    return healthy_dataset, blast_dataset

def train_val_samples(blast_datasets, healthy_datasets, train_perc, f_hb_ratio = None ):
    """ Elaborate the distribution of healthy and blast datasets
        Input:  - blast_dataset: list
                - healthy_dataset: list
                - train_perc: percentace splits for training and validation datasets
                - f_bh_ratio: number of blast datasets per healthy dataset 
        
        Output: - healthy_train_samples: number of samples for the healthy training dataset
                - healthy_val_samples: number of samples for the healthy training dataset (at least 1) 
                - blast_train_samples: number of samples for the blast training dataset
                - blast_val_samples: number of samples for the blast training dataset (at least 1)"""
    if f_hb_ratio is None:
        f_hb_ratio = len(blast_datasets) /len(healthy_datasets)

    healthy_val_samples = (1 - train_perc) *  f_hb_ratio
    healthy_val_samples = 1 if healthy_val_samples < 1 else int(healthy_val_samples)
      
    healthy_train_samples = len(healthy_datasets) - healthy_val_samples 

    blast_train_samples = int(f_hb_ratio * healthy_train_samples)
    blast_val_samples = int(f_hb_ratio * healthy_val_samples)

    return healthy_train_samples, healthy_val_samples, blast_train_samples, blast_val_samples

def concatenate_hb_datasets(healthy_idx, blast_idx, healthy_dataset, blast_dataset):
    """ Create merged dataset starting from indexs
        Outputs: - dataset:list. final list of both healthy and blast datasets
                 - y: list. labels of the datasets in the datasets list
    """
    dataset = []
    y = []
    for idx in healthy_idx:
        dataset.append(healthy_dataset[idx])
        y.append(0)
    for idx in blast_idx:
        dataset.append(blast_dataset[idx])
        y.append(1)
    return dataset, y


def default_serializer(obj):
    """ Let the results in numpy fomat being saved as .json file"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    #elif isinstance(obj, dict):
    #    return {k: converti_numpy_in_python(v) for k, v in obj.items()}
    #elif isinstance(obj, (list, tuple)):
    #    return [converti_numpy_in_python(v) for v in obj]
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
