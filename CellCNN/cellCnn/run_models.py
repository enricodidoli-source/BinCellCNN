

def train_CellCNN_restructured(CellCnn, train_datasets, train_y, val_datasets, val_y, test_datasets_no_labels,
                      n_cell = 100000, max_epochs=50, nrun=15, seed = 42):
   
    # use n_cell equal to the minimum dimension over all datasets (subdatasets)
    if n_cell == 'min':
        n_cell = min_length(train_dataset)
        print(f'n_cell = {min_l}')
    
    model = CellCnn(
        ncell = n_cell,                                   # Number of cells per nsubset (sampled from the 'patient' datasets)
        nsubset = int(1),                                    
        per_sample = True,                                # For each sample, nsubset samples of ncell are created
        nfilter_choice= [3,5,7,9],                        # Range of possible number of filters
        maxpool_percentages=[0.01, 1., 5., 20., 100.],    # list of k-percentage max_pooling
        learning_rate=None,                               # Learning rate
        max_epochs = max_epochs,  #50
        patience=5,                                       # Early stopping patience
        nrun = nrun,  #15                                 # Number of neural network configurations (for Hyperparameter optimization)
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

    return model

def test_CellCNN_restructured(model, test_datasets_no_labels):
    print(f'Prediction started...')
    
    test_pred = model.predict((test_datasets_no_labels))
    print(f'Done')
    return test_pred, model.results 

'========================================================================================================================================'

def trials_train_CellCNN(CellCnn, train_datasets, train_y,
                    val_datasets, val_y, 
                    test_datasets_no_labels, 
                    trials = 10,
                    max_epochs = 50, nrun = 15, n_cell = 100000, seed_list = None):
    models_lists = []
    if (seed_list is None) or (len(seed_list) < trials): 
        seed_list = random.sample(list(range(10000)), 10)
        
        print(f'Warning: seed_list is None or has not the adeguate length.\n')
        print(f'seed_list is randomly set = {seed_list}.\n')
        print(f'To control randomization, please generate seed_list outside the function.')
    

    for i in range(trials):
        print(f'Trial {i+1} started')
        seed = seed_list[i] * (i + 1)
        print(f'Seed used: {seed}')
        model = train_CellCNN_restructured(CellCnn, train_datasets, train_y, val_datasets, val_y, test_datasets_no_labels,
                                                    n_cell, max_epochs, nrun, seed)

        models_lists.append(model)
        
    return models_lists
    
def trials_test_CellCNN(models_lists, test_datasets_no_labels):
    predictions_list = []
    results_list = [] 

    for i in range(len(models_lists)):
        print(len(models_lists))
        test_pred, model_param = test_CellCNN_restructured(models_lists[i], test_datasets_no_labels)
        predictions_list.append(test_pred)
        
        results_list.append(model_param)
        
        print(f'Trial {i+1} Done!\n')
    return predictions_list, results_list











