import copy
import pandas as pd
import random
import numpy as np



'========================================================================================================================================'
def B_H_data_extraction(dataset, blast = True):
    """ Extracts blast or healthy cells from a specific dataset """
    code = 1 if blast == True else 0 
    data = pd.DataFrame()

    donation = dataset
    sub_data = donation[donation['IsBlast'] == code]
 
    data = pd.concat([data, sub_data], ignore_index = True)

    return data

def data_for_chunk(dataset, n_sub, blast = True):
    """ Compute the length of a chunk for Blast or Healthy datasets"""
    chunk_length = int(len(dataset) / n_sub)
              
    return chunk_length

def chunks_division(dataset, n_sub, blast = True, seed = 42):
    """ divides the dataset  in chunks creating a list of subsets of the dataset """
    if not blast:
        n_sub = n_sub*2

    
    chunk_length = data_for_chunk(dataset, n_sub = n_sub, blast = blast)
    
    #randomize data order
    dataset_shuffled = dataset.sample(frac=1, random_state=seed).reset_index(drop=True) # setting drop = False, it creates a new column with the original index of the ell

    data_chunks = [] 
    for i in range(n_sub):
        chunk_i = dataset_shuffled.iloc[i*chunk_length:  chunk_length * (i+1)]
        data_chunks.append(chunk_i)

    return data_chunks

def mixed_build_datasets(healthy_data_chunks, blast_data_chunks, n_cells = 100000, n_sub = 10, seed = 42):
    """ Builds the final datasets of a specific donor
        n_cells = number of healthy cells"""
    new_donor_datasets = []
    new_donor_y = []
    # blast_percentage_choice
    blast_percentages = [0.01, 0.05, 0.1, 0.2] # 
    

    # dataset with blast cells
    for i in range(n_sub):
        # set randomicity
        seed = seed + 3*(i)
        random.seed(seed)
        n_healthy_cells = len(healthy_data_chunks[i]) # number of healthy cells
        
        blast_perc = random.choice(blast_percentages)
        n_blast_cells = int(blast_perc * n_cells) # number of blast cells
        
        # if number of cells is too high, the sample with replacement is activated
        if n_blast_cells > len(blast_data_chunks[i]):
            b_rep = True
        else:
            b_rep = False
    
        blast_data = blast_data_chunks[i].sample(n = n_blast_cells, replace = b_rep, random_state = seed).reset_index(drop=True)

        
        if n_cells > len(healthy_data_chunks[i]):
            h_rep = True
        else:
            h_rep = False
        healthy_data = healthy_data_chunks[i].sample(n = n_cells, replace = h_rep, random_state = seed).reset_index(drop=True)
        new_blast_dataset_i = pd.concat([healthy_data, blast_data], ignore_index = True)
        new_donor_datasets.append(new_blast_dataset_i)
        new_donor_y.append(1)
        print(f'New Artificial Blast Donation {i + 1}: length = {len(new_blast_dataset_i)}, healthy cells:{n_healthy_cells}, blast cells: {len(blast_data)}')

    # dataset with only healthy cells
    for i in range(n_sub, int(n_sub*2)):
        n_healthy_cells = len(healthy_data_chunks[i]) # number of healthy cells
        
        if n_cells > len(healthy_data_chunks[i]):
            h_rep = True
        else:
            h_rep = False
        # set randomicity
        seed = seed + 2*(i)
        random.seed(seed)
        
        
        healthy_data_i = healthy_data_chunks[i].sample(n = n_cells, replace = h_rep, random_state = seed).reset_index(drop=True)
        new_donor_datasets.append(healthy_data_i)
        new_donor_y.append(0)
        print(f'New Artificial Healthy Donation {i - n_sub + 1}: length = {len(healthy_data_i)}')

    return new_donor_datasets, new_donor_y
    
def healthy_build_datasets(healthy_data_chunks, n_cells = 100000, n_sub = 10, seed = 42):
    """ Builds the final datasets of a specific donor """
    new_donor_datasets = []
    new_donor_y = []
    # dataset with only healthy cells
    for i in range(n_sub, int(n_sub*2)):
        # set randomicity
        seed = seed + 1*(i)
        random.seed(seed)
        
        n_healthy_cells = len(healthy_data_chunks[i]) # number of healthy cells
        
        if n_cells > len(healthy_data_chunks[i]):
            h_rep = True
        else:
            h_rep = False
        
        healthy_data_i = healthy_data_chunks[i].sample(n = n_cells, replace = h_rep, random_state = seed).reset_index(drop=True)
        new_donor_datasets.append(healthy_data_i)
        new_donor_y.append(0)
        print(f'New Artificial Healthy Donation {i - n_sub + 1}: length = {len(healthy_data_i)}')

    return new_donor_datasets, new_donor_y

def check_dataset_types(donor_datasets):
    counter = 0
    for dataset in donor_datasets:
        if len(dataset[dataset['IsBlast'] == 1]) > 0:
            counter += 1
    
    if counter > 0:
        type = 1
    else:
        type = 0
        
    #print(type)
    #print(f'\n')
    return type


def generate_new_datasets(donor_datasets_extracted, n_sub, n_cells, seed):
    """ generates new datasets from multiple donations of the same donor """
    
    blast_data = pd.DataFrame()
    healthy_data = pd.DataFrame()
   
    condition = check_dataset_types(donor_datasets_extracted)
    
    # aggregate healthy and blast data form donor cells
    for dt in donor_datasets_extracted:
        healthy = len(dt[dt['IsBlast'] == 0])
        if condition == 1:
            blast_dataset_i = B_H_data_extraction(dt) #blast_data
            blast_data = pd.concat([blast_data, blast_dataset_i], ignore_index = True)

        
        healthy_dataset_i = B_H_data_extraction(dt, False)  #healthy_data

        # create a single big dataset of blast or healthy cells
        healthy_data = pd.concat([healthy_data, healthy_dataset_i], ignore_index = True)

    
    # shuffle anf divide the two datasets in chunks, from which extract final data
    if condition == 1:
        blast_data_chunks = chunks_division(blast_data, n_sub = n_sub, blast = True, seed = seed)
        healthy_data_chunks = chunks_division(healthy_data, n_sub = n_sub, blast = False, seed = seed)
    
        # create the new datasets
        new_donor_datasets, new_donor_y = mixed_build_datasets(healthy_data_chunks, blast_data_chunks, n_cells, n_sub = n_sub, seed = seed)
        
    else:
        healthy_data_chunks = chunks_division(healthy_data, n_sub = n_sub, blast = False, seed = seed)
        new_donor_datasets, new_donor_y = healthy_build_datasets(healthy_data_chunks, n_cells, n_sub = n_sub, seed = seed)

    return new_donor_datasets, new_donor_y



def splitting_and_dataset_elaboration(train_datasets_extracted, val_datasets_extracted, test_datasets_extracted, n_sub, n_cells, seed):
    new_train_datasets = []
    new_train_y = []
    
    new_val_datasets = []
    new_val_y = []
    
    new_test_datasets = []
    new_test_y = []

    print(f'New training datasets creation...')
    for donor_datasets in train_datasets_extracted:
        print(f'\nNew Donor')
        
        gen_results = generate_new_datasets(donor_datasets, n_sub, n_cells, seed)
        
        new_train_datasets += gen_results[0]
        new_train_y += gen_results[1]
        seed += 1 
    print(new_train_y )
    print(f'Done!\n')
    
    
    print(f'New training datasets creation...')
    for donor_datasets in val_datasets_extracted:
        print(f'\nNew Donor')
        
        gen_results = generate_new_datasets(donor_datasets, n_sub, n_cells, seed)
        new_val_datasets += gen_results[0]
        new_val_y += gen_results[1]
        seed += 1 
    print(new_val_y )
    print(f'Done!\n')
    
    
    print(f'New training datasets creation...')
    for donor_datasets in test_datasets_extracted:
        print(f'\nNew Donor')


        gen_results = generate_new_datasets(donor_datasets, n_sub, n_cells, seed)
    
        new_test_datasets += gen_results[0]
        new_test_y += gen_results[1]
        seed += 1 
    print(new_test_y )
    print(f'Done!\n')

    return new_train_datasets, new_train_y, new_val_datasets, new_val_y, new_test_datasets, new_test_y
    
'========================================================================================================================================'

def remove_labels(new_test_datasets):
    no_labels = []
    for dataset in new_test_datasets:
        dataset = dataset.drop(columns = ['IsBlast'])
        no_labels.append(dataset)
    return no_labels
    
'========================================================================================================================================'





















    


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

