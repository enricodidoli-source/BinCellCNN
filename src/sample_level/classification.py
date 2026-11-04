'''
from classification import robust_prediction_labelling, seed_level_robust_prediction_labelling
from classification import find_robust_threshold, compute_metrics, compute_mean_std_metrics
'''

from src.shared.utils import flatten
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score 
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_curve



def get_best_thr(val_predicted_for_roc_folds):
            # rebuild dataset predictions
            predictions_list = []

            for fold_pred_list, val_y in val_predicted_for_roc_folds: # fold
                print('fold_pred_list', fold_pred_list)
                seed_pred_list = np.mean(fold_pred_list, axis = 0) # average across weigths initialization
                print('seed_pred_list', seed_pred_list)

                avg_w_init_pred_list = np.mean(fold_pred_list, axis = 0) # average across weigths initialization
                print('avg_w_init_pred_list', avg_w_init_pred_list)

                #for sample_pred in fold_pred_list[0]: # check if this is the seed level
                predictions_list = np.hstack((predictions_list, avg_w_init_pred_list))

            new_val_y = flatten([val_y for _, val_y in val_predicted_for_roc_folds])
            print(new_val_y)
            prec, rec, thr = roc_curve(new_val_y, predictions_list, pos_label=1)# compute ROC curve

                # Calculate F1 score for all thresholds
            f1_curve = 2 * prec * rec / (prec + rec + 1e-12)
            best_idx = np.argmax(f1_curve[:-1])
            best_thr = thr[best_idx]
            max_f1 = f1_curve[:-1][best_idx]

            print(f'F1-score: { max_f1*100:.2f}%')

            return best_thr



def phenotype_prediction(predictions_list, thr):
    """
    Converts raw prediction probabilities into binary labels using a fixed 0.5 threshold on the first class.
    """

    pred_phenotypes = [1 if pred >= thr else 0 for pred in predictions_list]

    return pred_phenotypes





























# performance metrics section

def compute_metrics(true_labels, pred_probs, thr, mcc = False):
    """
    Computes classification metrics (f1, recall, precision, accuracy, optionally MCC)
    by classifying pred_probs with thr.
    """
    pred_labels = (np.array(pred_probs) >= thr).astype(int)

    f1 = f1_score(true_labels, pred_labels, pos_label=1, zero_division=0)
    rec = recall_score(true_labels, pred_labels, pos_label=1, zero_division=0)
    pre = precision_score(true_labels, pred_labels, pos_label=1, zero_division=0)
    acc = accuracy_score(true_labels, pred_labels)

    met_dict = {'f1': f1, 'rec': rec, 'pre': pre, 'acc': acc}

    if mcc:
        mcc_score = matthews_corrcoef(true_labels, pred_labels)
        met_dict['mcc'] = mcc_score
    return met_dict

def compute_mean_std_metrics(all_dir_metrics_across_LOPO, decimals = None, return_df = False):
    """
    Computes mean and std for each metric across all LOPO folds.
    Optionally rounds results or returns the raw DataFrame instead.
    """

    df = pd.DataFrame(all_dir_metrics_across_LOPO)

    if return_df:
        return df

    mean_std_dict = {}
    for metric in df.columns:
        all_met_list = flatten(df[metric].to_list())
        all_met_list_mean = np.mean(all_met_list)
        all_met_list_std = np.std(all_met_list, ddof=1)

        if decimals is not None:
            all_met_list_mean = round(all_met_list_mean, decimals)
            all_met_list_std = round(all_met_list_std, decimals)
        mean_std_dict[metric] = [all_met_list_mean, all_met_list_std]

    else:
        return mean_std_dict

