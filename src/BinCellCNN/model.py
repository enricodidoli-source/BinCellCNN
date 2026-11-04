""" Copyright 2016-2017 ETH Zurich, Eirini Arvaniti and Manfred Claassen.

This module contains functions for performing a CellCnn analysis.

"""

import sys
import os
import copy
import logging
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import random

from src.BinCellCNN.bincellcnn_utils import combine_samples, normalize_outliers_to_control
from src.BinCellCNN.bincellcnn_utils import cluster_profiles, keras_param_vector
from src.BinCellCNN.bincellcnn_utils import generate_subsets, generate_biased_subsets
from src.BinCellCNN.bincellcnn_utils import get_filters_classification, get_filters_regression
from src.BinCellCNN.bincellcnn_utils import mkdir_p


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, initializers, regularizers, optimizers, callbacks
from tensorflow.keras import backend as K

from memory_utils import clear_RAM

logger = logging.getLogger(__name__)


class BinCellCNN(object):
    """ Creates a BinCellCnn model.

    Args:
        - ncell :
            Number of cells per multi-cell input.
        - nsubset :
            Total number of multi-cell inputs that will be generated per class, if
            `per_sample` = `False`. Total number of multi-cell inputs that will be generated from
            each input sample, if `per_sample` = `True`.
        - per_sample :
            Whether the `nsubset` argument refers to each class or each input sample.

        - maxpool_percentages :
            A list specifying candidate percentages of cells that will be max-pooled per
            filter. For instance, mean pooling corresponds to `maxpool_percentages` = `[100]`.
        - nfilter_choice :
            A list specifying candidate numbers of filters for the neural network.
        - scale :
            Whether to z-transform each feature (mean = 0, standard deviation = 1) prior to
            training.

        - nrun :
            Number of neural network configurations to try (should be set >= 3).

        - learning_rate :
            Learning rate for the Adam optimization algorithm. If set to `None`,
            learning rates in the range [0.001, 0.01] will be tried out.
        - dropout :
            Whether to use dropout (at each epoch, set a neuron to zero with probability
            `dropout_p`). The default behavior 'auto' uses dropout when `nfilter` > 5.
        - dropout_p :
            The dropout probability.
        - coeff_l1 :
            Coefficient for L1 weight regularization.
        - coeff_l2 :
            Coefficient for L2 weight regularization.
        - max_epochs :
            Maximum number of iterations through the data.
        - patience :
            Number of epochs before early stopping (stops if the validation loss does not
            decrease anymore).
        - dendrogram_cutoff :
            Cutoff for hierarchical clustering of filter weights. Clustering is
            performed using cosine similarity, so the cutof should be in [0, 1]. A lower cutoff will
            generate more clusters.
        - accur_thres :
            Keep filters from models achieving at least this accuracy. If less than 3
            models pass the accuracy threshold, keep filters from the best 3 models.
    """

    def __init__(self, ncell=200, nsubset=1000, bs = 200, per_sample=False,
                 maxpool_percentages=None, scale=True,
                 nfilter_choice=None, dropout='auto', dropout_p=.5,
                 coeff_l1=0, coeff_l2=0.0001, learning_rate=None, 
                 max_epochs=20, patience=5, nrun=15, seed = 42, grid = False,
                 
                 dendrogram_cutoff=0.4, accur_thres=.95, verbose=1):

        # initialize model attributes
        self.scale = scale
        self.nrun = nrun
        self.ncell = ncell
        self.nsubset = nsubset
        self.bs = bs
        self.per_sample = per_sample
        self.maxpool_percentages = maxpool_percentages
        self.nfilter_choice = nfilter_choice
        self.learning_rate = learning_rate
        self.coeff_l1 = coeff_l1
        self.coeff_l2 = coeff_l2
        self.dropout = dropout
        self.dropout_p = dropout_p
        self.max_epochs = max_epochs
        self.patience = patience
        self.results = None
        self.model_sorted_idx = None
        self.seed = seed
        self.grid = grid

        self.dendrogram_cutoff = dendrogram_cutoff
        self.accur_thres = accur_thres
        self.verbose = verbose

        

        self.all_params = {}
        self.all_params['scale'] = self.scale
        self.all_params['nsubset'] = self.nsubset
        self.all_params['ncell'] = self.ncell
        self.all_params['bs'] = self.bs
        self.all_params['per_sample'] = self.per_sample
        self.all_params['maxpool_percentages'] = self.maxpool_percentages
        self.all_params['nfilter_choice'] = self.nfilter_choice
        self.all_params['learning_rate'] = self.learning_rate
        self.all_params['coeff_l1'] = self.coeff_l1
        self.all_params['coeff_l2'] = self.coeff_l2
        self.all_params['dropout'] = self.dropout
        self.all_params['dropout_p'] = self.dropout_p
        self.all_params['max_epochs'] = self.max_epochs
        self.all_params['patience'] = self.patience
        self.all_params['seed'] = self.seed
        self.all_params['grid'] = self.grid
        
        self.all_params['dendrogram_cutoff'] = self.dendrogram_cutoff
        self.all_params['accur_thres'] = self.accur_thres
        self.all_params['verbose'] = self.verbose     
        
        
        
    def fit(self, train_samples, train_phenotypes, outdir, valid_samples=None,
            valid_phenotypes=None, mem = None, metric = 'accuracy'):

        """ Trains a CellCnn model.

        Args:
            - train_samples :
                List with input samples (e.g. cytometry samples) as numpy arrays.
            - train_phenotypes :
                List of phenotypes associated with the samples in `train_samples`.
            - outdir :
                Directory where output will be generated.
            - valid_samples :
                List with samples to be used as validation set while training the network.
            - valid_phenotypes :
                List of phenotypes associated with the samples in `valid_samples`.
            - generate_valid_set :
                If `valid_samples` is not provided, generate a validation set
                from the `train_samples`.

        Returns:
            A trained CellCnn model with the additional attribute `results`. The attribute `results`
            is a dictionary with the following entries:

            - clustering_result : clustered filter weights from all runs achieving \
                validation accuracy above the specified threshold `accur_thres`
            - selected_filters : a consensus filter matrix from the above clustering result
            - best_3_nets : the 3 best models (achieving highest validation accuracy)
            - best_net : the best model
            - w_best_net : filter and output weights of the best model
            - accuracies : list of validation accuracies achieved by different models
            - best_model_index : list index of the best model
            - config : list of neural network configurations used
            - scaler : a z-transform scaler object fitted to the training data
            - n_classes : number of output classes
        """

        res = train_model(train_samples, train_phenotypes, outdir,
                          valid_samples, valid_phenotypes,
                          scale=self.scale, 
                          ncell=self.ncell, nsubset=self.nsubset, bs = self.bs, per_sample=self.per_sample,
                          maxpool_percentages=self.maxpool_percentages,
                          nfilter_choice=self.nfilter_choice,
                          learning_rate=self.learning_rate,
                          coeff_l1=self.coeff_l1, coeff_l2=self.coeff_l2,
                          dropout=self.dropout, dropout_p=self.dropout_p,
                          max_epochs=self.max_epochs,
                          patience=self.patience, dendrogram_cutoff=self.dendrogram_cutoff,
                          accur_thres=self.accur_thres, verbose=self.verbose, seed = self.seed,
                           grid = self.grid, nrun = self.nrun, mem = mem, metric = metric)
        
        self.results = res
        self.all_params['results'] = self.results
        return self

    def predict(self, test_samples, ncell_per_sample=None, seed = None, validation = False, mem = None):

        """ Makes predictions for new samples.

        Args:
            - new_samples :
                List with input samples (numpy arrays) for which predictions will be made.
            - ncell_per_sample :
                Size of the multi-cell inputs (only one multi-cell input is created
                per input sample). If set to None, the size of the multi-cell inputs equals the
                minimum size in `new_samples`.

        Returns:
            y_pred : Phenotype predictions for `new_samples`.
        """
        '''
        # Use a fixed seed for reproducible subsampling
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()  # fallback (non-deterministic)
        '''    
        print(f'seed: {seed}. Type: {type(seed)}')
        seed = int(seed)
        print(f'seed: {seed}. Type: {type(seed)}')
        self.all_params['prediction_seed'] = seed
            
        
        if ncell_per_sample is None:
        
            ncell_per_sample = np.min([len(x) for x in test_samples])
        logger.info(f"Predictions based on multi-cell inputs containing {ncell_per_sample} cells.")

        # z-transform the new samples if we did that for the training samples
        scaler = self.results['scaler']
        if scaler is not None:
            test_samples = [scaler.transform(x) for x in test_samples]

        print('samples length', [len(x) for x in test_samples])
        nmark = test_samples[0].shape[1]
        n_classes = self.results['n_classes']
 
        # get the configuration of the top 3 models
        scores = self.results['scores']
        sorted_idx = np.argsort(scores)[::-1][:3]
        config = self.results['config']

        
        y_pred = np.zeros((3, len(test_samples), n_classes))
        for i_enum, i in enumerate(sorted_idx):

            
            nfilter = config['nfilter'][i]
            maxpool_percentage = config['maxpool_percentage'][i]
            ncell_pooled = max(1, int(maxpool_percentage / 100. * ncell_per_sample))

            # build the model architecture
            model = build_model(ncell_per_sample, nmark,
                                nfilter=nfilter, coeff_l1=0, coeff_l2=0,
                                k=ncell_pooled, dropout=False, dropout_p=0, n_classes=n_classes, lr=0.01)

            # and load the learned filter and output weights
            weights = self.results['best_3_nets'][i_enum]
            
            model.set_weights(weights)

            # select a random subset of `ncell_per_sample` and make predictions
            if len(test_samples) > 1:
                keras.utils.set_random_seed(int(seed + i))

            for x in test_samples:
                test_samples = [shuffle(x)[:ncell_per_sample].reshape(1, ncell_per_sample, nmark)
                           for x in test_samples]
            print(f'test_samples_length: {len(test_samples)}')
            data_test = np.vstack(test_samples)

            y_pred[i_enum] = model.predict(data_test)
        print('y_pred', y_pred)
        print('np.mean(y_pred, axis=0)', np.mean(y_pred, axis=0))
        return np.mean(y_pred, axis=0) # computes the mean of the prediction results

def data_transformation(dataset, labels = False):
    if labels:
        # Copy the 'IsBlast' column
        dataset_labels = dataset[:, -1].copy()  # shape: (n_righe,)
        dataset_labels = dataset_labels.astype(int) # convert data in integers

        # Remove the 'IsBlast' column to not be included in the transformation
        dataset_no_labels = dataset[:, :-1]  # shape: (n_righe, n_colonne - 1)
        
        z_scaler = StandardScaler(with_mean=True, with_std=True) 
        z_scaler.fit(dataset_no_labels) #scale data

        transformed_dataset_no_labels = z_scaler.transform(dataset_no_labels)

        #append the 'IsBlast' column    
        dataset_after = np.column_stack([transformed_dataset_no_labels, dataset_labels])
   
        
    else:
        
        z_scaler = StandardScaler(with_mean=True, with_std=True) 
        z_scaler.fit(dataset) #scale data

        dataset_after = z_scaler.transform(dataset)
        
    return dataset_after, z_scaler
        

def grid_search_parameters(nfilter_choice, maxpool_percentages, learning_rate):
    param_grid = []
    for filter in nfilter_choice:
        for pool in maxpool_percentages:
                for rate in learning_rate:
                    hyper_selection = (filter, pool, rate)
                    param_grid.append(hyper_selection)
                    print(hyper_selection)
    return param_grid 

def fit_scaler(dataset):
    dataset_no_labels = dataset.copy()

    # Remove the 'IsBlast' column to not be included in the transformation
    z_scaler = StandardScaler(with_mean=True, with_std=True) 
    z_scaler.fit(dataset_no_labels) #scale data

    return z_scaler

def scale_data(dataset, z_scaler):

    dataset_no_labels = dataset.copy()
    scaled_dataset = z_scaler.transform(dataset_no_labels)
    dataset_after = scaled_dataset
    return dataset_after




def generate_train_val_identifier(train_samples, train_phenotypes, valid_samples, valid_phenotypes, eval_folds=5, seed=42, mem = None):
    
    if mem is not None:
        print('Loading from Temp')
        (train_samples, train_phenotypes, 
        valid_samples, valid_phenotypes) = load_train_valid_from_temp(train_samples=train_samples, 
                                                                   train_phenotypes=train_phenotypes, 
                                                                   valid_samples=valid_samples, 
                                                                   valid_phenotypes=valid_phenotypes, 
                                                                   mem=mem)

    train_sample_ids = np.arange(len(train_phenotypes))
    X_train, id_train = combine_samples(train_samples, train_sample_ids)
 
    valid_sample_ids = np.arange(len(valid_phenotypes))
    X_valid, id_valid = combine_samples(valid_samples, valid_sample_ids)

    # shuffling
    X_train, id_train = shuffle(X_train, id_train, random_state=seed)
    X_valid, id_valid = shuffle(X_valid, id_valid, random_state=seed)
        
    print('Identifier Done!')
    return X_train, id_train, X_valid, id_valid, train_phenotypes, valid_phenotypes


def load_train_valid_from_temp(train_samples=None, train_phenotypes=None, 
                               valid_samples=None, valid_phenotypes=None, mem=None):

    if mem is not None:
        print('Loading from Temp')
        (train_samples, train_phenotypes) = mem.load_variables(['train_samples', 'train_phenotypes'])

        (valid_samples, valid_phenotypes) = mem.load_variables(['valid_samples', 'valid_phenotypes'])
            
    else:

        # copy the list of samples so that they are not modified in place
        train_samples = train_samples.copy()
        valid_samples = valid_samples.copy()
        
    train_phenotypes = np.array(train_phenotypes)
    valid_phenotypes = np.array(valid_phenotypes)
    return train_samples, train_phenotypes, valid_samples, valid_phenotypes


def train_model(train_samples = None, train_phenotypes = None, outdir = None,
                valid_samples=None, valid_phenotypes=None,
                scale=True,
                ncell=200, nsubset=1000, bs = 200, per_sample=False,
                maxpool_percentages=None, nfilter_choice=None,
                learning_rate=None, coeff_l1=0, coeff_l2=1e-4, dropout='auto', dropout_p=.5,
                max_epochs=20, patience=5,
                dendrogram_cutoff=0.4, accur_thres=.95, verbose=1, seed = 42, grid = False, nrun = 3, mem = None, metric = 'accuracy'):
    

    """ Performs a CellCnn analysis """
    keras.utils.set_random_seed(seed)   # imposta random, numpy e tensorflow insieme


    if maxpool_percentages is None:
        maxpool_percentages = [0.01, 1., 5., 20., 100.]
    if nfilter_choice is None:
        nfilter_choice = list(range(3, 10))
    if learning_rate is None:
        learning_rate = [10 ** np.random.uniform(-3, -2)] # random choice 
        
    mkdir_p(outdir)
       
    print('')
    print('valid_phenotypes', valid_phenotypes)

    # train/val generation and shuffling
    X_train, id_train, X_valid, id_valid, train_phenotypes, valid_phenotypes = generate_train_val_identifier(
        train_samples=train_samples,
        train_phenotypes=train_phenotypes,
        valid_samples=valid_samples, 
        valid_phenotypes=valid_phenotypes,
        eval_folds=5,
        seed=seed,
        mem = mem
    )
    

    if mem is not None:
        clear_RAM(['valid_samples'])
    clear_RAM(['train_samples'])
  
    train_phenotypes = np.asarray(train_phenotypes)
    y_train = train_phenotypes[id_train]

    valid_phenotypes = np.asarray(valid_phenotypes)
    y_valid = valid_phenotypes[id_valid]


    # Scale data
    z_scaler = fit_scaler(X_train)
    X_train = scale_data(X_train, z_scaler)
    X_valid = scale_data(X_valid, z_scaler)


    # number of measured markers
    nmark = X_train.shape[1]

    # generate multi-cell inputs
    print("Generating multi-cell inputs...")

          
    X_tr, y_tr = generate_subsets(X_train, train_phenotypes, id_train,
                                            nsubset, ncell, per_sample, seed=seed)
    X_v, y_v = generate_subsets(X_valid, valid_phenotypes, id_valid,
                                                nsubset, ncell, per_sample, seed=seed)

    
    clear_RAM(['X_valid', 'X_train'])

    
    print('Subsets Generation concluded!')
    
    # neural network configuration
    # batch size

    train_sample_ids = []    
    #bs = 200

    # keras needs (nbatch, ncell, nmark)
    X_tr = np.swapaxes(X_tr, 2, 1)
    X_v = np.swapaxes(X_v, 2, 1)

    n_classes = len(np.unique(train_phenotypes))
    y_tr = keras.utils.to_categorical(y_tr, n_classes)
    y_v = keras.utils.to_categorical(y_v, n_classes)


    if grid:
        # define all cobinations of parameters
        param_grid = grid_search_parameters(nfilter_choice, maxpool_percentages, learning_rate)               
        nrun = len(param_grid)
        print(f'Total number of selections: {nrun}')

    if nrun < 3:
        logger.info(f"The nrun argument should be >= 3, setting it to 3.")
        nrun = 3
    scores = np.zeros(nrun)


    w_store = dict()
    config = dict()


    chosen_nfilter_list = []
    chosen_mp_list = []
    chosen_lr_list = []

    epochs_num = []
    histories = []

    for irun in range(nrun):
        
        print(f'\n# ============================================= #')
        if verbose:
            logger.info(f"Training network: {irun + 1}")
        
        # reproducible but different seed per run

        print(f'Seed: {seed}')


        # ============================================= #
        # hyperparameters selection
        # ============================================= #

        if grid:
            nfilter, mp, lr = param_grid[irun]
        else:
            lr = np.random.choice(learning_rate)
            nfilter = np.random.choice(nfilter_choice)
            mp = np.random.choice(maxpool_percentages)
            
        chosen_nfilter_list.append(nfilter)
        chosen_mp_list.append(mp)
        chosen_lr_list.append(lr)

        print(f'\n Learning Rate: {lr}\n')
        print(f'\n Number of filters: {nfilter}\n')
        print(f'\n k% max pooling: {mp}\n')

        k = max(1, int(mp / 100. * ncell)) 

        # ============================================= #
        # build the neural network
        # ============================================= #
        model = build_model(ncell, nmark, nfilter,
                            coeff_l1, coeff_l2, k,
                            dropout, dropout_p, n_classes, lr, metric = metric)
        
        #### weights saving ####
        filepath = os.path.join(outdir, f"nnet_run_{irun}.weights.h5")
        #try:
        for i in range(1):  
            check = callbacks.ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True,
                                      mode='auto', save_weights_only=True) 
                    
            earlyStopping = callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='auto')
                    
                    # keras.fit() trains the model
            history = model.fit(X_tr, y_tr,
                              epochs=max_epochs, batch_size=bs, callbacks=[check, earlyStopping],
                              validation_data=(X_v, y_v), verbose=verbose)


            epochs_num.append(len(history.epoch))
            clear_RAM(['history'])
            
            # load the model from the epoch with highest validation accuracy
            model.load_weights(filepath)

            valid_metric = model.evaluate(X_v, y_v, verbose=0)[-1] # test the model on the validation set
            
            try:
                valid_value = valid_metric[1] 
            except (TypeError, IndexError):
                valid_value = valid_metric

            logger.info(f"Best validation F1-score: {valid_value:.2f}")
            scores[irun] = valid_value

            # extract the network parameters
            w_store[irun] = model.get_weights()

        '''
        except Exception as e:
            sys.stderr.write('An exception was raised during training the network.\n')
            sys.stderr.write(str(e) + '\n')
        '''

    config['nfilter'] = chosen_nfilter_list
    config['learning_rate'] = chosen_lr_list
    config['maxpool_percentage'] = chosen_mp_list

    # the top 3 performing networks
    model_sorted_idx = np.argsort(scores)[::-1][:3]
    best_sorted_indices = np.argsort(scores)[::-1]
    
    best_3_nets = [w_store[int(i)] for i in model_sorted_idx if int(i) in w_store]
    
    if not best_3_nets:
        raise RuntimeError(
            "No valid model founf in w_store. "
            "Probably, weights were not saved correctly."
        )
    
    best_net = best_3_nets[0]
    best_accuracy_idx = model_sorted_idx[0]

    # weights from the best-performing network
    w_best_net = keras_param_vector(best_net)

    # post-process the learned filters
    # cluster weights from all networks that achieved accuracy above the specified thershold
    w_cons, cluster_res = cluster_profiles(w_store, nmark, scores, accur_thres,
                                           dendrogram_cutoff=dendrogram_cutoff)


    results = {
        'scaler': z_scaler,
        'n_classes': n_classes,
        'model_sorted_idx': best_sorted_indices,
        'config': config,
        'scores': scores,
        'best_3_nets': best_3_nets,
        'clustering_result': cluster_res,
        'selected_filters': w_cons,
        'best_net': best_net,
        'w_best_net': w_best_net,
        'best_model_index': best_accuracy_idx,
        'epochs_num': epochs_num,
        'w_store': w_store
    }
    

    maxpool_percentage = config['maxpool_percentage'][best_accuracy_idx]
        
    if mem is not None:
        (valid_samples,) = mem.load_variables(['valid_samples']) # bug if sample is generated from train sample because valid sample is not of the right format. TO fix later

    filter_diff = get_filters_classification(w_cons, z_scaler, valid_samples,
                                                     valid_phenotypes, maxpool_percentage) 
    results['filter_diff'] = filter_diff

    return results 

def build_model(ncell, nmark, nfilter, coeff_l1, coeff_l2,
                k, dropout, dropout_p, n_classes, lr=0.01, seed=42, metric = 'accuracy'):
    """ Builds the neural network architecture """

    # the input layer
    data_input = keras.Input(shape=(ncell, nmark))  # returns a vector of matrices
                                                    # each element is a matrix of ncell x nmark

    # the filters
    conv = layers.Conv1D(filters=nfilter,
                         kernel_size=1,
                         activation='relu',
                         kernel_initializer='random_uniform',
                         kernel_regularizer=regularizers.l1_l2(l1=coeff_l1, l2=coeff_l2), #reguarization
                         name='conv1')(data_input)

    print("=" * 50)
    # Converti nfilter da numpy.int64 a int Python
    nfilter = int(nfilter)
  
    # the cell grouping part (top-k pooling)
    pooled = layers.Lambda(pool_top_k, output_shape=(nfilter,), arguments={'k': k})(conv)
    
    
    # possibly add dropout
    if dropout or ((dropout == 'auto') and (nfilter > 5)):
        pooled = layers.Dropout(rate=dropout_p)(pooled)

    # network prediction output
    output = layers.Dense(units=n_classes, # one node for each class
                              activation='softmax',
                              kernel_initializer='random_uniform',
                              kernel_regularizer=regularizers.l1_l2(l1=coeff_l1, l2=coeff_l2),
                              name='output')(pooled)

    model = keras.Model(inputs=data_input, outputs=output)

    #f1_score = keras.metrics.F1Score(average='macro', threshold=None, name="f1_score", dtype=None)


    
    model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                      loss='categorical_crossentropy',
                      metrics = [metric])

    return model

@keras.utils.register_keras_serializable()
def pool_top_k(x, k=10):
    """
    Pooling che seleziona i top k valori per ogni feature lungo l'asse delle celle
    """
    # k must be an integer
    if isinstance(k, tf.Tensor):
        k = int(k.numpy())
    k = int(k)

    # k cannot be greater than the number of cells
    n_cells = tf.shape(x)[1] 
    k = tf.minimum(k, n_cells)

    # Finds top k elements
    top_k_values, _ = tf.nn.top_k(tf.transpose(x, [0, 2, 1]), k=k, sorted=True)

    result = tf.reduce_mean(top_k_values, axis=2)

    return result



