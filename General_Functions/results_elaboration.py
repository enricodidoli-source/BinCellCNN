import copy
import pandas as pd
import random
import numpy as np
import sys



def extract_hyper(configuration, best_models):
    
    trial_parameters = []
    for idx in best_models:
        model_parameters = {}  # Usa dizionario invece di lista
        for key, value in configuration.items():
            if len(value) == 0:
                v = 0
            else:
                v = value[idx]
            model_parameters[key] = v  # Aggiungi col nome del parametro
        trial_parameters.append(model_parameters)
    
    # Converti in DataFrame
    df = pd.DataFrame(trial_parameters)
    
    return df

def show_hyper(results_list, best_3 = False):
    best_per_trial = {}
    best_3_per_trial = []
    trials = len(results_list)
    for i in range(trials):
        #print(f'\nTrial {i}...')

        # retrieve the trial's results
        res = results_list[i]

        #extract best indexes
        best_models_i = res['model_sorted_idx'][:3]
        config = res['config']
        
        trial_par = extract_hyper(config, best_models_i)
        if best_3:
            best_3_per_trial.append(trial_par)

        #extract the best
        best = trial_par.iloc[0]

        # add the best in the dictionary
        best_per_trial[i] = (best)
        #show_hyperparameters(trial_par)
    
    best_per_trial = pd.DataFrame(best_per_trial).T
    if best_3:
        return best_per_trial, best_3_per_trial
        
    return best_per_trial

def elaborate_predictions(predictions_list, test_y, results = True):
    
    accuracy_list = []
    pred_phenotypes_dict = {}
    
    for i, pred in enumerate(predictions_list):
        pred_phenotypes = phenotype_prediction(pred)
    
        tot_correct = np.array(pred_phenotypes) ==  test_y #checks differencies in prediction
        accuracy = np.sum(tot_correct)/ len(test_y)  #compute accuracy
        accuracy_list.append(accuracy)
    
        pred_phenotypes_dict[i] = pred_phenotypes
        if results:
            print(f'Trial {i} Accuracy: {accuracy}')
            
    
    
    last_position = len(pred_phenotypes_dict)
    pred_phenotypes_dict['True Labels'] = test_y
    
    pred_phenotype_df = pd.DataFrame(pred_phenotypes_dict)

    if results:
        print(pred_phenotype_df.T)
        mean_accuracy = np.mean(accuracy_list)
        print(f'mean_accuracy over the ten trials: {mean_accuracy}')
        accuracy_std = np.std(accuracy_list)
        print(f'accuracy_std over the ten trials: {accuracy_std }')

    return pred_phenotype_df.T, accuracy_list


def show_hyperparameters(best_3_hyper):
    
    for i, model in enumerate(best_3_hyper):
        print(f'Model {i+1}')
        print(f'Filters: {model[0]}, Learning Rate: {model[1]}, Top-k Percentage Max Pooling: {model[2]} ')

def phenotype_prediction(test_pred):
    pred_phenotypes = []
    for sample_pred in test_pred:
        if sample_pred[0] < 0.5:
            pred_phenotypes.append(1)
        else:
            pred_phenotypes.append(0)
    return pred_phenotypes

def default_serializer(obj):
    """ Let the results in numpy fomat being saved as .json file"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: converti_numpy_in_python(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [converti_numpy_in_python(v) for v in obj]
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
