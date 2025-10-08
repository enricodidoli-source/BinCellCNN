import copy
import pandas as pd
import random
import numpy as np



def read_data(data_path, sep = ';', dec = ','):
        '''For single datafile'''
        import pandas as pd

        # transform data to be in the right format
        raw_df = pd.read_csv(data_path, sep = ';', decimal = ',')
        df = raw_df.copy()

        print('### Data converted in a Padas Dataframe ###')
        return df, raw_df


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


def n_data_retriever(data):
    data_counter = 0
    for val in data.values():
        data_counter += len(val)
    return data_counter
    
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


def splitting(healthy_donors, blast_donors, mixed_donors, healthy_donors_idx, blast_donors_idx, mixed_donors_idx, set_division = [2,1,2]):
    train_donors_idx = []
    val_donors_idx = []
    test_donors_idx = []
    for i, don in enumerate(healthy_donors_idx):
        if i in range(set_division[0]):
            train_donors_idx.append(healthy_donors[don])
        elif i in range(set_division[0], set_division[0] + set_division[1]):
            val_donors_idx.append(healthy_donors[don])
        else:
            test_donors_idx.append(healthy_donors[don])
    
    for i, don in enumerate(blast_donors_idx):
        if i == 0:
            train_donors_idx.append(blast_donors[don])
        elif i == 1:
            val_donors_idx.append(blast_donors[don])
        else:
            test_donors_idx.append(blast_donors[don])
            
    for i, don in enumerate(mixed_donors_idx):
        if i in range(set_division[0]):
            train_donors_idx.append(mixed_donors[don])
        elif i in range(set_division[0], set_division[0] + set_division[1]):
            val_donors_idx.append(mixed_donors[don])
        else:
            test_donors_idx.append(mixed_donors[don])
    return train_donors_idx, val_donors_idx, test_donors_idx



def dataset_extraction(donors_dataset_idx, multiple_donations, all_datasets):
    
    raw_datasets = []
    raw_datasets_idx = []
    for donor_code in donors_dataset_idx:
        donor_datasets = multiple_donations[donor_code]
        for data in donor_datasets:
            raw_datasets_idx.append(data)
            raw_datasets.append(all_datasets[data])
    print(raw_datasets_idx)
    return raw_datasets

def generate_subsets_length(dataset, n_subsets,  seed = 42):
    
    random.seed(seed)
    
    total_dataset_cells = len(dataset)
    #print(total_dataset_cells)
    mean_sub_length = int(total_dataset_cells/n_subsets) 
    max_len = int(mean_sub_length * 1.1)
    min_len = int(mean_sub_length * 0.9)

    #print(min_len, max_len)
    lengths = random.choices(range(min_len, max_len), k=n_subsets)
    #print(lengths)
    return lengths

def idx_to_cells(cell_list, dataset):
    cell_dataset  = []
    for cell in cell_list:
        c = dataset[cell]
        cell_dataset.append(c)
    return cell_dataset

def create_subsets(dataset, n_subsets, seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    subsets_lengths = generate_subsets_length(dataset, n_subsets, seed) #Generate number of cells
    #print(len(subsets_lengths))

    # COMPUTE OERCENTAGE OF BLASTS AND SAMPLE FROM ORIGINAL DIVIDED  DATASETS THE SUBSETS LEN (WITH REPLACEMENT OR NOT)
    new_subsets = []
    total_dataset_cells = len(dataset)

    healthy_cells = dataset[dataset['IsBlast'] == 0]
    blast_cells = dataset[dataset['IsBlast'] == 1]
    n_healthy_cells = len(healthy_cells)
    n_blast_cells = len(blast_cells)
    #print(f'\nblast cells in the dataset: {blast_cells}')
    #n_cells = range
    
    blast_percentages = [0.005, 0.01, 0.05, 0.1, 0.2]
    
    for i in range(n_subsets):
        if n_blast_cells > 0:
            
            sampled_perc = random.choice(blast_percentages)
                               
            b_cells =int(sampled_perc * subsets_lengths[i])
            
            #print(f'Blast percentage in hte new dataset: {sampled_perc}')
            #print(f'total blast cells in the new dataset {i+1}: {b_cells}')
            if b_cells > n_blast_cells:
                replace = True
            else:
                replace = False
            sampled_blast_cells = np.random.choice(list(range(n_blast_cells)), size = b_cells, replace = replace)

            h_cells = subsets_lengths[i] - b_cells
            
        else:
            h_cells = subsets_lengths[i] 
            
        if h_cells > n_healthy_cells:
            replace = True
        else:
            replace = False
        sampled_healthy_cells = np.random.choice(list(range(n_healthy_cells)), size = h_cells, replace = replace)

        
        if n_blast_cells > 0:
            data_healthy_cells = healthy_cells.iloc[sampled_healthy_cells]
            #print(data_healthy_cells)
            data_blast_cells = blast_cells.iloc[sampled_blast_cells]
            
            combined = pd.concat([data_healthy_cells, data_blast_cells], ignore_index=True)
            new_subsets.append(combined)
            #print(f'{len(data_healthy_cells)}')
            #print(f'{len(data_blast_cells)}\n')
        else:
            data_healthy_cells = healthy_cells.iloc[sampled_healthy_cells]
            new_subsets.append(data_healthy_cells)
            #print(f'{len(data_healthy_cells)}\n')
    return new_subsets

def subsample_generation(raw_datasets, n_subsets, seed = 42):
    raw_sampled_subsets = []
    i = 0
    for ds in raw_datasets:
        subsets = create_subsets(ds, n_subsets, seed = seed)
        for sub in subsets:
            raw_sampled_subsets.append(sub)
        
        #print(f'\niteration: {i+1}\n')
        i += 1
    return raw_sampled_subsets


def retrieve_labels(datasets, log = False):
    """ Computes the number of blast cells in a sample returning the labels of the datasets
        Outputs:- clear_datasets: list of datasets without unnecessary columns
                - dataset_labels: list of binary lables. 1 for datasets with blast cells
    
    """
    #if dataset.isinstance(np.array):
        
    clear_datasets = []
    dataset_labels = []
    #copy_all_datasets = copy.deepcopy(ALL_DATASETS)
    
    for i, dataset in enumerate(datasets):
        df = dataset.drop(columns = ['Original_ID'])
        blast = (dataset['IsBlast'] == 1).sum() #computes number of blast cells
        
        if blast > 0: #if blasts are present, then the donor is not healthy
            dataset_labels.append(1)
        else:
            dataset_labels.append(0)
        clear_datasets.append(df)

        if log:
            print(f'Dataset {i} has: {blast} blast cells over {len(dataset)} healthy cells ({(blast/len(dataset))*100.:4f} % blast cells)')
        df = dataset.drop(columns = ['IsBlast']) # drop not anymore necessary column
        

    return clear_datasets, dataset_labels



def dataset_elaboration(multiple_donations, ALL_DATASETS, healthy_donors, blast_donors, mixed_donors, n_sub = 3, seed = 42):
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
    print(f'Extracting data...')


    
    # extract donors datasets from the entire natch od datasets and place them into correct dataset
    raw_train_datasets = dataset_extraction(train_donors_idx, multiple_donations, ALL_DATASETS)
    raw_val_datasets = dataset_extraction(val_donors_idx, multiple_donations, ALL_DATASETS)
    raw_test_datasets = dataset_extraction(test_donors_idx, multiple_donations, ALL_DATASETS)

    
    print(f'Generating Subsamples...')
    # creates subsets for each sample
    raw_train_sampled_subsets = subsample_generation(raw_train_datasets, n_sub, seed = seed)
    raw_val_sampled_subsets = subsample_generation(raw_val_datasets, n_sub, seed = seed)
    raw_test_sampled_subsets = subsample_generation(raw_test_datasets, n_sub, seed = seed)


    #for sub in raw_train_sampled_subsets:
    #    print(len(sub))
        
    # retrieve labels from each subset
    train_datasets, train_y = retrieve_labels(raw_train_sampled_subsets, log = False)
    val_datasets, val_y = retrieve_labels(raw_val_sampled_subsets, log = False)
    test_datasets, test_y = retrieve_labels(raw_test_sampled_subsets, log = False)

        # remove cells labels for training
    test_datasets_no_labels = []
    for df in test_datasets:
        test_datasets_no_labels.append(df.drop(columns = ['IsBlast']).values)

    
    # recreate training dataset for cross validation training
    cv_train_datasets = train_datasets + val_datasets
    cv_train_y = train_y + val_y

    return train_datasets, train_y, val_datasets, val_y, test_datasets_no_labels, test_y, cv_train_datasets, cv_train_y




def CellCNN_restructured(CellCnn, train_datasets, train_y, val_datasets, val_y, test_datasets_no_labels,
                      n_cell = 100000, max_epochs=50, nrun=15, seed = 42):
   
    # use n_cell equal to the minimum dimension over all datasets (subdatasets)
    if n_cell == 'min':
        n_cell = min_length(train_dataset)
        print(f'n_cell = {min_l}')
    
    model = CellCnn(
        ncell = n_cell,            #200                        # Number of cells per multi-cell input (sampled from the 'patient' datasets)
        nsubset = int(1),                                    # Total number of multi-cell inputs that will be generated per class (or sample and class)
        per_sample = True,                                # For each sample, nsubset samples of ncell are performed
        nfilter_choice= [3,5,7,9], #list(range(3,21)),                 # Range of possible number of filters
        maxpool_percentages=[0.01, 1., 5., 20., 100.],    # list of k-percentage max_pooling
        learning_rate=None, #[0.01], #, 0.1, 1, 10, 50, 100],                               # Learning rate
        max_epochs = max_epochs,  #50
        patience=5,                                       # Early stopping patience
        nrun = nrun,  #15                                     # Number of neural network configurations to try (for Hyperparameter optimization)
        regression=False,
        scale=True,                                       # Z-score normalization
        verbose=1,
        seed=seed
    )

    print(f'Model defined...')
    
    outdir = f'/content/cellcnn_results'  # Results Directory

    print(f'Fitting started...')
    model.fit(
            train_samples = train_datasets,
            train_phenotypes = np.array(train_y),
            outdir=outdir,
            valid_samples = val_datasets,
            valid_phenotypes = np.array(val_y),
            generate_valid_set = False
        )

    print(f'Prediction started...')
    test_pred = model.predict((test_datasets_no_labels))
    print(f'Done')
    return test_pred, model.results 


def run_CellCNN_res(CellCnn, train_datasets, train_y,
                    val_datasets, val_y, 
                    test_datasets_no_labels, 
                    trials = 10,
                    max_epochs = 50, nrun = 15, n_cell = 100000, seed_list = None):
    
    if (seed_list is None) or (len(seed_list) < trials): 
        seed_list = random.sample(list(range(10000)), 10)
        
        print(f'Warning: seed_list is None or has not the adeguate length.\n')
        print(f'seed_list is randomly set = {seed_list}.\n')
        print(f'To control randomization, please generate seed_list outside the function.')
    
    predictions_list = []
    results_list = [] 

    for i in range(trials):
        print(f'Trial {i+1} started')
        seed = seed_list[i] * i * 2
        print(f'Seed used: {seed}')
        test_pred, model_param = CellCNN_restructured(CellCnn, train_datasets, train_y, val_datasets, val_y, test_datasets_no_labels,
                                                    n_cell, max_epochs, nrun, seed)
        predictions_list.append(test_pred)
        
        results_list.append(model_param)
        #best_indices_list.append(best_indices)
        
        print(f'Trial {i+1} Done!\n')
    return predictions_list, results_list





def CV_CellCNN_restructured(CellCnn, cv_train_datasets, cv_train_y, k = 5, n_cell = 10, n_sub = 3, seed = 42, max_epochs=1, patience=5, nrun=1):
    


    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed) # define class
   
    cv_results = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(cv_train_datasets, cv_train_y)):
        #print(fold, (train_idx, val_idx))
        print(f"Fold {fold + 1}")
        
        print(f'Creating folds...')
        
        # Seleziona i dataset e le etichette per train e val
        train_datasets = [cv_train_datasets[i] for i in train_idx]
        val_datasets   = [cv_train_datasets[i] for i in val_idx]
        train_labels   = [cv_train_y[i] for i in train_idx]
        val_labels     = [cv_train_y[i] for i in val_idx]
        print(f'Folds Created!')

        fold_seed = seed*(fold + 1)*3
        print(f'seed: {fold_seed}')
        # use n_cell equal to the minimum dimension over all datasets (subdatasets)
        if n_cell == 'min':
            n_cell = min_length(train_datasets)
            print(f'n_cell = {min_l}')
        
        model = CellCnn(
            ncell = n_cell,            #200                        # Number of cells per multi-cell input (sampled from the 'patient' datasets)
            nsubset = int(1),                                    # Total number of multi-cell inputs that will be generated per class (or sample and class)
            per_sample = True,                                # For each sample, nsubset samples of ncell are performed
            nfilter_choice= [3,5,7,9],                  # Range of possible number of filters
            maxpool_percentages=[0.01, 1., 5., 20., 100.],    # list of k-percentage max_pooling
            learning_rate= None, # [0.01], #, 0.1, 1, 10, 50, 100],                               # Learning rate
            max_epochs = max_epochs,
            patience=10,                                       # Early stopping patience
            nrun = nrun,                                     # Number of neural network configurations to try (for Hyperparameter optimization)
            regression=False,
            scale=True,                                       # Z-score normalization
            verbose=1,
            seed = fold_seed
        )

        print(f'Model defined...')
    
        outdir = f'/content/cellcnn_results'  # Results Directory
    
        print(f'Fitting started...')

        model.fit(
            train_samples=train_datasets,
            train_phenotypes=np.array(train_labels),
            outdir=outdir,
            valid_samples = val_datasets,
            valid_phenotypes=np.array(val_labels),
            generate_valid_set = False
        )

        cv_results.append(model.results)

    print(f'Done')
    return cv_results, model



def CV_best_acc(cv_results):
    best_acc = 0
    counter = 0
    
    # find the best accuracy
    for fold in cv_results:
        #print(fold['accuracies'])
        for acc in fold['accuracies']: # for each run performed in the fold
            if acc > best_acc:
                best_acc = acc
                best_fold = counter
    
        counter += 1
    print(f'Best accuracy: {best_acc}')
    print(f'Fold that performed best: {best_fold}')
    return best_fold    










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

