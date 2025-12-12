import copy
import pandas as pd
import random
import numpy as np
import sys

from sklearn.metrics import f1_score


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
    f1_scores_list = []
    for i, pred in enumerate(predictions_list):
        pred_phenotypes = phenotype_prediction(pred)

        f1 = f1_score(test_y, pred_phenotypes, pos_label=1)
        f1_scores_list.append(f1)
        
        tot_correct = np.array(pred_phenotypes) ==  test_y #checks differencies in prediction
        accuracy = np.sum(tot_correct)/ len(test_y)  #compute accuracy
        accuracy_list.append(accuracy)
    
        pred_phenotypes_dict[i] = pred_phenotypes
        if results:
            print(f'Trial {i} Accuracy: {accuracy}')
            print(f'Trial {i} F1_score: {f1}')
            
    
    
    last_position = len(pred_phenotypes_dict)
    pred_phenotypes_dict['True Labels'] = test_y
    
    pred_phenotype_df = pd.DataFrame(pred_phenotypes_dict)

    if results:
        print(pred_phenotype_df.T)
        mean_accuracy = np.mean(accuracy_list)
        print(f'mean_accuracy over the ten trials: {mean_accuracy}')
        accuracy_std = np.std(accuracy_list)
        print(f'accuracy_std over the ten trials: {accuracy_std }')

        mean_f1 = np.mean(f1_scores_list)
        print(f'mean_f1 over the ten trials: {mean_f1}')
        std_f1 = np.std(f1_scores_list)
        print(f'std_f1 over the ten trials: {std_f1}')
        
    return pred_phenotype_df.T, accuracy_list , f1_scores_list 


def show_hyperparameters(best_3_hyper):
    
    for i, model in enumerate(best_3_hyper):
        print(f'Model {i+1}')
        print(f'Filters: {model[0]}, Learning Rate: {model[1]}, Top-k Percentage Max Pooling: {model[2]} ')

def phenotype_prediction(test_pred):
    pred_phenotypes = []
    for sample_pred in test_pred:
        if sample_pred[0] < 0.5: # if  the first class < 0.5 => second class > 0.5
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
    # ADD THIS BLOCK
    elif isinstance(obj, pd.DataFrame):
        # Convert the DataFrame to a list of dictionaries (one per row)
        # This is a common and useful format for JSON.
        return obj.to_dict(orient='records')
        
        # --- Other common options ---
        # 1. Convert to a dictionary of {column: [values]}
        # return obj.to_dict(orient='list')
        #
        # 2. Convert to a dictionary of {index: {column: value}}
        # return obj.to_dict(orient='index')

    # Your original error line
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")







































'''



%%time
total_acc_lists = []
total_f1_lists = []
total_pred_lists = []

for patient, patient_y in zip(per_donor_resampled_datasets, per_donor_resampled_y):
    patient_acc_list = []
    patient_f1_list = []
    patient_pred_list = []

    
    for timepoint, timepoint_y in zip(patient, patient_y):
        print(len(timepoint))
        print(timepoint_y)
        new_datasets_predictions_list, new_datasets_results_list = trials_test_CellCNN_old(models_lists, timepoint)
        
        pred_phenotype_df, accuracy_list, f1_score_list = elaborate_predictions(new_datasets_predictions_list, timepoint_y, results = False)


        
        print(accuracy_list)
        print(f1_score_list)
        print(pred_phenotype_df)

        resampled_true_y = pred_phenotype_df.iloc[-1] # get labels of resampled subsets
        max_f1 = max(f1_score_list)
        best_f1_idx = f1_score_list.index(max_f1)
        best_sub = pred_phenotype_df.iloc[best_f1_idx]
        
        blast_score = best_sub.sum() 
        timepoint_score = blast_score.sum()/len(resampled_true_y)
        
        print('')
        patient_acc_list.append(accuracy_list)
        patient_f1_list.append(f1_score_list)
        patient_pred_list.append(pred_phenotype_df)
       
    print(patient_acc_list)
    print(patient_f1_list)
    total_acc_lists.append(patient_acc_list)
    total_f1_lists.append(patient_f1_list)
    total_pred_lists.append(patient_pred_list)



def compute_timepoint_best_f1(timepoint_preds):
        """we are taking the best f1 score. because we are not tuning the model. we are just trying to predict the label of the timepoint"""
        timepoint_score = []
        
        resampled_true_y = timepoint_preds.iloc[-1] # get labels of resampled subsets
        print(timepoint_preds.iloc[:-1]) #labels

        # initialize variables
        best_f1 = -1
        counter = 0
        
        for h in range(len(timepoint_preds) - 1):
            sub = timepoint_preds.iloc[h] # extrsct trial pediction
            
            f1 = f1_score(resampled_true_y, sub, pos_label = 1) # compute the f1_score
            if f1 > best_f1:
                best_f1_idx = counter
            counter += 1
        print(best_f1_idx)   
        best_sub = timepoint_preds.iloc[best_f1_idx]
        #print(len(true_y))
        blast_score = best_sub.sum() 
        print('')
        print(best_sub)
        print(f'blast_score: {blast_score}\n')
        timepoint_score = blast_score.sum()/len(resampled_true_y)
        return (timepoint_score, list(resampled_true_y))

       

'''



