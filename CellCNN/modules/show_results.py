from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, auc
from sklearn.metrics import matthews_corrcoef
from results_elaboration import elaborate_predictions
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle as pkl
from functions import robust_prediction_labelling
import json


def retireve_samples_info(data_folder_dir, multiple_donations, ALL_DATASETS):
    file_names = os.listdir(data_folder_dir)

    samples_info_dict = {}

    last_patient = '1'
    counter = -1
    for element in file_names:

        if 'csv' in element:
            element = element[:-4]
            splitted_element = element.split('GHE')[1]
            patient_id = splitted_element.split('_')[0]
            time_point_days = splitted_element.split('_')[1]


            if last_patient == str(patient_id):
                counter += 1
            else:
                counter = 0


            file_idx = multiple_donations[str(patient_id)][counter]

            sample = ALL_DATASETS[file_idx]

            blast_n = len(sample[sample['IsBlast'] == 1])
            cells_n = len(sample)
            healthy_n = len(sample[sample['IsBlast'] == 0])
            blast_p = round((blast_n/cells_n)*100,4)
            healthy_p = 100-blast_p
            samples_info_dict[file_idx] = [cells_n, healthy_n, blast_n, healthy_p, blast_p, patient_id, time_point_days]

            last_patient = str(patient_id)

    df = pd.DataFrame.from_dict(samples_info_dict)
    df = df.transpose()
    df.columns = ['cells_n', 'healthy_n', 'blast_n', 'healthy_p', 'blast_p', 'patient_id', 'time_point_days']
    df['Sample'] = df.index
    df = df.sort_values(by='blast_n')


    return df

def show_sample_healthy_blast_distribution(samples_info_dict, axis=None, save_dir=None, numbers = False):
    df = samples_info_dict
    df = df.sort_values(by='blast_n').reset_index(drop=True)
    y_labels = [str(f'Pat_{patient_id}_{time_point}') for sample_id, time_point, patient_id in zip(df['Sample'], df['time_point_days'], df['patient_id'])]

    max_val = max(df['healthy_n'].max(), df['blast_n'].max())

    if axis is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        ax = axis

    ax.barh(y_labels, -df['blast_n'],  color='tomato',    label='Unhealthy cells')
    ax.barh(y_labels,  df['healthy_n'], color='steelblue', label='Healthy cells', alpha = 0.7)

    ax.axvline(0, color='black', linewidth=0.8)

    if numbers:
        # Annotazioni numeriche sulle barre
        for i, (blast, healthy) in enumerate(zip(df['blast_n'], df['healthy_n'])):
            # Numero cellule cancerogene (sinistra)
            ax.text(-blast, i, f'{blast:,.0f}', 
                    va='center', ha='right', fontsize=6, color='black')
            # Numero cellule sane (destra)
            ax.text(healthy/10, i, f'{healthy:,.0f}', 
                    va='center', ha='left', fontsize=6, color='black')

    ax.set_xscale('symlog', linthresh=1000)

    # Notazione scientifica sull'asse X
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: '' if v == 0 else f'$10^{{{int(np.log10(abs(v)))}}}$'))
    ax.tick_params(axis='x', labelsize=7)

    ax.set_title('Healthy vs Unhealthy cells per sample (symlog scale)', fontsize=13)
    ax.set_ylabel('Patient-ID-Time Point Day', fontsize=10)
    ax.set_xlabel('Number of Unhealthy/Healthy cells (symlog scale)', fontsize=10)
    ax.set_xlim(-max_val * 1.1, max_val * 1.1)

    ax.legend()
    if axis is None:
        plt.tight_layout()
        if save_dir:
            plt.savefig(f'{save_dir}/0.1.healthy_unhealthy_samples_distribution.pdf', format='pdf')
            plt.close()
        plt.show()

def show_sample_blast_perc_distribution(samples_info_dict, axis = None, save_dir = None):
    df = samples_info_dict
    df = df.sort_values(by='blast_n').reset_index(drop=True)
    y_labels = [str(f'Pat_{patient_id}_{time_point}') for sample_id, time_point, patient_id in zip(df['Sample'], df['time_point_days'], df['patient_id'])]

    df['pct_unhealthy'] = -df['blast_p']  # negativo → sinistra

    if axis is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        ax = axis

    #ax.barh(y_labels, df['pct_healthy'],   color='steelblue', label='Healthy cells')
    ax.barh(y_labels, df['pct_unhealthy'], color='tomato',    label='Blast cells')

    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('% of total cells')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{abs(v):.0f}%'))
    plt.draw()  # forza matplotlib a calcolare i tick finali

    ax2 = ax.twinx()

    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(ax.get_yticks())
    ax2.set_yticklabels([f'{v:.4f}%' for v in df['blast_p']])

    ax.set_title('Unhealthy cells % per sample', fontsize=13)
    ax.set_ylabel('Patient-ID-Time Point Day',  fontsize=10)
    ax.set_xlabel('Unhealthy cells %',  fontsize=10)
    ax2.set_ylabel('Unhealthy cells %',  fontsize=10)

    ax.legend()
    if axis is None:
        plt.tight_layout()
        if save_dir:
            plt.savefig(f'{save_dir}/0.2.samples_unhealthy_cells_percentages.pdf', format = 'pdf')
            plt.close()
        plt.show()

def show_patients_samples_info(samples_info_dict, save_dir = None, numbers = False): 

    fig, axs = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'wspace': 0.3})

    show_sample_healthy_blast_distribution(samples_info_dict, axis = axs[0], numbers = numbers)
    show_sample_blast_perc_distribution(samples_info_dict, axis = axs[1])

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.suptitle('Samples Informations', fontsize = 15)

    if save_dir:
        fig.savefig(f'{save_dir}/0.samples_info.pdf', format = 'pdf')
    plt.show()
    plt.close()





"==============================================================================="
def show_roc_thresholds(fpr, tpr, thresholds, save_dir = None):
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        plot_roc_curve(fpr, tpr, thresholds, axis = axs[0])
        sensitivity_vs_specificity(fpr, tpr, thresholds, axis = axs[1])

        plt.tight_layout()

        if save_dir:
            fig.savefig(f'{save_dir}/1.roc_thresholds.pdf', format = 'pdf')

        plt.show()
        plt.close()

    # rebuild dataset predictions

def rebuild_dataset_predictions(val_predicted_for_roc, best_idx):

        val_predicted_for_roc_folds = val_predicted_for_roc[best_idx]

        predictions_list, new_val_y = [], []
        for fold in val_predicted_for_roc_folds:
            pred_list, val_y = fold
            new_val_y += val_y

            #pred_list potentially contains multuple inner list (seeds)
            # but in tuning only one seed is performed
            for neg_prob, pos_prob in pred_list[0]:
                predictions_list.append(pos_prob)
        return predictions_list, new_val_y

def load_tuning_data(tuning_load_dir, colab = False):
    if not colab:
        keys = ['best_ncells', 'best_nsub', 'robust_threshold', 'roc_threshold', 'tested_par', 'val_predicted_for_roc', 'ensemble_mean_probs_per_patient' ]
        threshold_data = {}
        for key in keys:
            with open(os.path.join(tuning_load_dir, f'{key}.pkl'), 'rb') as f:
                threshold_data[key] = pkl.load(f)

    else:

        with open(f'{tuning_load_dir}/threshold_data.json', 'r') as f:
            threshold_data = json.load(f)

    return threshold_data




def generate_heatmap_dict(tested_par, val_predicted_for_roc):
        ncells_to_plot = []
        nsubs_to_plot = []
        f1_score_to_plot = []

        tested_ncells, tested_nsubs = [], []
        heatmap_dict = {}
        for (ncells, nsubs), fold in zip(tested_par, val_predicted_for_roc):
            f1_fold, rec_fold = [], []
            for pred_list, val_y in fold:
                _, _, f1_score_list, _, _ = elaborate_predictions(
                                        pred_list, val_y, results=False
                                    )
                f1_fold.append(f1_score_list)

            #print(f'(ncells, nsubs): {(ncells, nsubs)}, f1  scores: {np.mean(f1_fold)}')

            ncells_to_plot.append(ncells)
            nsubs_to_plot.append(nsubs)
            f1_score_to_plot.append(np.mean(f1_fold))

            tested_ncells.append(ncells)
            tested_nsubs.append(nsubs)

        heatmap_dict['ncells'] = tested_ncells
        heatmap_dict['nsubs'] = tested_nsubs
        heatmap_dict['f1'] = f1_score_to_plot

        return heatmap_dict


def show_heat_map_combinations(heatmap_dict, best_idx = None, save_dir = None):
    df = pd.DataFrame(heatmap_dict)

    sns.set_style("ticks")  # oppure "whitegrid", "ticks"

    plt.figure(figsize=(6, 4))

    if best_idx is not None:
      best_ncells, best_nsub = heatmap_dict['ncells'][best_idx], heatmap_dict['nsubs'][best_idx]
      best_f1 = df['f1'].iloc[best_idx]

      plt.scatter(best_nsub, best_ncells,
                  s=200,
                  facecolors='none',
                  edgecolors='red',         # ← bordo rosso
                  linewidth=2.5,
                  label=f'Comb: {best_ncells}, {best_nsub}. F1-score: {best_f1*100:.2f}%'
                          )

    scatter = plt.scatter(df['nsubs'], df['ncells'], c=df['f1'], cmap='viridis',
                s=120,                 # dimensione dei punti
                edgecolor='black',     # 👈 contorno nero
                linewidth=1            # 👈 spessore del contorno)
                        )

    plt.colorbar(scatter, label='F1 Score')
    plt.ylabel('ncells')
    plt.xlabel('nsubs')
    plt.title('F1 Score - Explored combinations (Bayesian Trials)')

    if best_idx is not None:
        plt.legend(fontsize = 10)

    if save_dir is not None:
        plt.savefig(f'{save_dir}/1.2.tested_combinations.pdf', format = 'pdf')

    plt.show()

def generate_dict_comb_3d(LOPO_folds, tuning_exp_pat):

    heatmap_dict = {}
    ncells_per_fold = []
    nsubs_per_fold = []
    f1_score_per_fold = []
    for LOPO_idx, _ in enumerate(range(LOPO_folds)):

        tuning_load_dir = f'{tuning_exp_pat}/outer_fold_{LOPO_idx}/tuning/results'

        threshold_data = load_tuning_data(tuning_load_dir, colab = False)

        best_ncells =threshold_data['best_ncells']
        best_nsub =threshold_data['best_nsub']
        tested_par =threshold_data['tested_par']
        val_predicted_for_roc = threshold_data['val_predicted_for_roc']

        chosen_combination = (best_ncells, best_nsub)
        best_idx = tested_par.index(chosen_combination)
        print(f'Best (ncells, nsub) = {chosen_combination} at index {best_idx}')

        f1_fold = []
        for pred_list, val_y in val_predicted_for_roc[best_idx]:
                    _, _, f1_score_list, _, _ = elaborate_predictions(
                                            pred_list, val_y, results=False
                                        )
                    f1_fold.append(f1_score_list)
        mean_f1 = np.mean(f1_fold)
        print(f'Mean F1 score: {mean_f1}')
        ncells_per_fold.append(best_ncells)
        nsubs_per_fold.append(best_nsub)
        f1_score_per_fold.append(mean_f1)

    heatmap_dict['ncells'] = ncells_per_fold
    heatmap_dict['nsubs'] = nsubs_per_fold
    heatmap_dict['f1'] = f1_score_per_fold

    return heatmap_dict


def show_best_com_3d(heatmap_dict, save_dir = None):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    n = 100

    comb_text = [f'({ncells},{nsubs})' for ncells, nsubs in zip(heatmap_dict['ncells'], heatmap_dict['nsubs'])]

    print(comb_text)
    xs = heatmap_dict['ncells']
    ys = heatmap_dict['nsubs']
    zs = heatmap_dict['f1']

    m = 'o'

    # Add slight offset to prevent text overlapping with points
    for i, txt in enumerate(comb_text):
        ax.text(xs[i], ys[i], zs[i] + 0.0075, txt, fontsize=6)

    ax.scatter(xs, ys, zs, marker=m, s = 100)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    if save_dir is not None:
        plt.savefig(f'{save_dir}/0.1.best_comb_3D.pdf', format = 'pdf')

    plt.show()

def flatten(nested):
    if nested is None:
        return []
    if not isinstance(nested, (list, tuple)):
        return [nested]
    nested = list(nested)
    result = []
    for item in nested:
        result.extend(flatten(item))
    return result

def elaborate_direct_prediction(imported_predictions, imported_test_y, metric = 'acc', pred = False):
    mcc = False
      
    if metric == 'acc':
        pos = 0
    elif metric == 'f1':
        pos = 1
    elif metric == 'rec':
        pos = 2
    elif metric == 'pre':
        pos = 3
    elif metric == 'mcc':
        pos = 4
        mcc = True

    labels_flat = flatten(imported_test_y)
    # elaborate metric data from imported predictions
    models_metric_lists = []
    metrics = elaborate_metrics(imported_predictions, labels_flat, mcc = mcc)
    if metric == 'pre':
        print(f'metric {metric}:', metrics[pos])
    if pred:
        el_preds = elaborate_metrics(imported_predictions, labels_flat, pred = pred)

        return el_preds
    else:
        return metrics[pos] # extract the metric in the chosen position
    

def elaborate_metrics(imported_predictions, imported_test_y, results = False, pred = False, mcc = False):

    
    #new_pred_phenotype_df, new_accuracy_list, new_f1_score_list, new_recall_score_list, new_precision_score_list, new_mcc_list 
    metric_lists = elaborate_predictions(imported_predictions, imported_test_y, results = results, mcc = mcc)
    
    if pred:
        new_pred_phenotype_df = metric_lists 
        return new_pred_phenotype_df
    else:
        #return new_accuracy_list, new_f1_score_list, new_recall_score_list, new_precision_score_list, new_mcc_list
        return metric_lists[1:]


def direct_model_comparison_barplot(direct, metric_name = None, subtitle = False, axis = None, save_dir = None, LOPO_fold = None):
    print(direct)
    models_values = list(direct.values())
    models_names = [name.replace('model_', '') for name in direct.keys()]

    errors = []
    means = []
    for val in models_values:
        m = np.mean(val)
        means.append(m)
        s = np.std(val)
        errors.append(s)

    # Colors matching the plot
    colors = ['#0077BB', '#FF8800', '#00BB00', '#DD0000', '#AA55BB']

    # Create bar plot
    if axis is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        ax = axis

    bars = ax.bar(models_names, means, yerr= errors, color=colors, edgecolor='black', capsize=5)

    for bar, mean, error in zip(bars, means, errors):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + error,
                f'{mean:.3f}',
                ha='center', va='bottom', fontsize=10)

    # Customize
    plt.suptitle(f'Direct Prediction {metric_name} per method', fontsize=14, fontweight='bold')
    plt.suptitle(f'Scores (Direct Prediction)', fontsize=14, fontweight='bold')

    if subtitle:
        ax.set_title(f'{subtitle}', fontsize=11)

    ax.set_ylabel(metric_name, fontsize=12)
    high_limit = np.max(np.array(means) + np.array(errors))

    ax.set_ylim(0, max(1, high_limit + 0.05)) # Adjust based on your data range
    ax.set_ylim(0, max(1, 1.1)) # Adjust based on your data range

    ax.grid(True, alpha=0.3, axis='y')

    plt.xticks(rotation=60)


    if axis is None:
        plt.tight_layout()

        if save_dir:
            if LOPO_fold is None:
                print('Warning: same name for each barplot. Overwrite: True!')
                LOPO_fold = ''
            plt.savefig(f'{save_dir}/3.1.metrics_barplot_LOPO_fold_{LOPO_fold}.pdf', format = 'pdf')
            plt.show()
            plt.close()

   


def direct_aggregated_barplot(all_dir_metrics, axis = None, save_dir = None):

    metrics_names = list(all_dir_metrics[0].keys())
    LOPO_folds = len(all_dir_metrics)
    width, ticks = 0.8/LOPO_folds, np.arange(len(metrics_names))
    colors = ['red', 'blue', 'green', 'black', 'yellow']
    
    num_neg_bars = LOPO_folds/2
    start_pos_bar = -width*num_neg_bars
    list_bar_positions = [start_pos_bar]

    for _ in range(LOPO_folds-1):
        start_pos_bar += width
        list_bar_positions.append(round(start_pos_bar,4))

    df = pd.DataFrame(all_dir_metrics)

    metrics_names = list(df.columns)
    mean_metrics_dict = {}
    for metric in metrics_names:
        values_folds = df[metric]
        mean_fold_metric = [np.mean(values) for values in values_folds]

        mean_metrics_dict[metric] = mean_fold_metric

    mean_df = pd.DataFrame(mean_metrics_dict)

    # Create bar plot
    if axis is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        ax = axis

    for f, fold in enumerate(mean_df.index): #iterate over indexes. else it iterates over columns

        metrics_values = mean_df.iloc[f]

        shift = list_bar_positions[f]

        ax.bar(ticks + shift, metrics_values, color = colors[f], width=width )
    ax.set_xticks(ticks)
    ax.set_xticklabels(metrics_names)

    if save_dir:
        plt.savefig(f'{save_dir}/3.2.mean_barplot_across_LOPO_folds.pdf', format = 'pdf')
    
    plt.show()            
    plt.close()



def compare_barplots_folds(all_dir_metrics, summary = None, save_dir = None):


    lopo_folds = len(all_dir_metrics)
    print(lopo_folds)
    if summary is not None:
        lopo_folds += 1
    sub_rows = lopo_folds/3 if lopo_folds%3 == 0 else int(lopo_folds/3) + 1

    if lopo_folds == 1:
        fig, axs = plt.subplots(int(sub_rows), 1, figsize = (10, sub_rows*3))
    elif lopo_folds == 2:
        fig, axs = plt.subplots(int(sub_rows), 2, figsize = (10, sub_rows*3))
    else:
        fig, axs = plt.subplots(int(sub_rows), 3, figsize = (10, sub_rows*3))

    for i, direct_metrics in enumerate(all_dir_metrics):
        if  lopo_folds > 1 and sub_rows == 1:
            ax = axs

        elif  lopo_folds > 1 and sub_rows == 1:
            ax = axs[i]

        else:
            ax = axs[int(i/3), i%3]

        direct_model_comparison_barplot(direct_metrics, axis = ax, LOPO_fold = i)

    if summary is not None:
        # Last summary or mean subset   
        i += 1
        ax = axs[int(i/3), i%3]

        if summary:
            direct_aggregated_barplot(all_dir_metrics, axis = ax)

        else:

          df = pd.DataFrame(all_dir_metrics)
          metric_names = list(df.columns)

          mean_metrics_dict = {}
          for metric in metric_names: 
              values_folds = df[metric]
              mean_values_folds = [np.mean(fold) for fold in values_folds]
              mean_metrics_dict[metric] = mean_values_folds

          direct_model_comparison_barplot(mean_metrics_dict, metric_name = None, subtitle = False, axis = ax, LOPO_fold = i)


    if save_dir:
        fig.savefig(f'{save_dir}/3.barplot_LOPO_comparison.pdf', format = 'pdf')
    
    plt.show()            
    plt.close()

  





import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def show_violin_plot(plot_data, TUNED_THRESHOLD, axis = None, save_dir = None, LOPO_fold = None):
    sns.set_style("ticks")

    # Convert your data into a DataFrame
    df = pd.DataFrame(plot_data)

    if axis is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        ax = axis

    # A Violin plot is like a boxplot but shows the full distribution
    sns.violinplot(
        x="True_Label",
        y="Timepoint_Score",
        data=df,
        order=["Negative", "Positive"],
        ax=ax # Ensure this order
    )

    # You can also add a swarmplot to see the individual points
    sns.swarmplot(
        x="True_Label",
        y="Timepoint_Score",
        data=df,
        color="black",
        alpha=0.2, # Make points transparent
        size=2,
        order=["Negative", "Positive"],
        ax=ax
        )

    # --- Add your threshold line! ---
    # This is the line that separates your predictions
    ax.axhline(y=TUNED_THRESHOLD, color='red', linestyle='--', label=f'Tuned Threshold ({TUNED_THRESHOLD:.2f})')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('Sample Score Distribution vs. True Label (Across All Seeds)')
    ax.set_ylabel('Timepoint Blast Score')
    ax.set_xlabel('True Timepoint Label')

    # Use ax.legend() and only show if there are labeled artists
    handles, labels = ax.get_legend_handles_labels()
    if handles:  # Only create legend if there are labeled items
        ax.legend(handles=handles, labels=labels)

    if axis is None:
        plt.tight_layout()

        if save_dir:
            if LOPO_fold is None:
                print('Warning: same name for each barplot. Overwrite: True!')
                LOPO_fold = ''
            plt.savefig(f'{save_dir}/4.1.violin_LOPO_fold_{LOPO_fold}.pdf', format = 'pdf')
            plt.show()
            plt.close() 



def predict_trials(patient, patient_ys, patient_resampled_y, best_threshold):
        sample_prob_data = []

        f1_data = []
        boxplot_data = []
        for timepoint, true_timepoint_y, true_resampled_y in zip(patient, patient_ys, patient_resampled_y):
            label_str = "Positive" if true_timepoint_y == 1 else "Negative"

            timepoint_boxplot_data = []
            timepoint_f1_data = []
            #print(timepoint)
            for trial in timepoint: # from each timepoint multiple samples has been generated. each sample has been predicted 10 times using the same 'best' model but using a different seed each time
                             # each sub is a pediction of the 20 samples of a single timepoint

                trial_prob = np.mean(trial) # mean of the probabilities of the subsets
                trial_pred = (np.array(trial) >= best_threshold).astype(int)

                trial_f1 =  f1_score(true_resampled_y, trial_pred, pos_label = 1, zero_division=1)

                prob_dict = {
                "True_Label": label_str,
                "Timepoint_Score": trial_prob
                }

                sample_prob_data.append(prob_dict)

                timepoint_boxplot_data.append(trial_prob)
                timepoint_f1_data.append(trial_f1)

            box_dict = {
                "True_Label": true_timepoint_y,
                "Timepoint_trials_scores": timepoint_boxplot_data
                }

            f1_dict = {
                "True_Label": true_timepoint_y,
                "Timepoint_trials_scores": timepoint_f1_data
                }

            boxplot_data.append(box_dict)
            f1_data.append(f1_dict)
        return sample_prob_data, boxplot_data, f1_data

def final_trials_prediction(total_trial_pred_lists, per_donor_original_test_y, per_donor_resampled_test_y, best_threshold):
    from sklearn.metrics import f1_score
    """ Elaborate data to show distribution of trial results over the entire set of timepoints and patiets"""
    prob_data = []
    f1_data = []
    boxplot_data = []

    for i, (patient, patient_ys, patient_resampled_y) in enumerate(zip(total_trial_pred_lists, per_donor_original_test_y, per_donor_resampled_test_y)):
        print(i)
        sample_prob_data, box_dict, f1_dict = predict_trials(patient, patient_ys, patient_resampled_y, best_threshold)


        # Add prob data
        prob_data += sample_prob_data

        # Extend the main lists with the lists returned from the helper
        # (Use extend, not append, to keep the flat structure of timepoints)
        boxplot_data.extend(box_dict)
        f1_data.extend(f1_dict)


    return prob_data, boxplot_data, f1_data



def compare_violin_plots_folds(per_LOPO_violin_data, roc_TUNED_THRESHOLDs, save_dir = None):

    lopo_folds = len(per_LOPO_violin_data)

    sub_rows = lopo_folds/3 if lopo_folds%3 == 0 else int(lopo_folds/3) + 1
    fig, axs = plt.subplots(int(sub_rows), 3, figsize = (10, sub_rows*3))

    for i, plot_data in enumerate(per_LOPO_violin_data):
        ax = axs[int(i/3), i%3]
        show_violin_plot(plot_data, roc_TUNED_THRESHOLDs[i], axis = ax, LOPO_fold= i)

    plt.tight_layout()
    if save_dir:
        fig.savefig(f'{save_dir}/4.comparison_violin_per_LOPO_fold.pdf', format = 'pdf')
        
    plt.show()            
    plt.close()



def compare_boxplots_folds(per_LOPO_boxplot_data, per_LOPO_blast_perc, roc_TUNED_THRESHOLDs, save_dir = None):

    lopo_folds = len(per_LOPO_boxplot_data)

    sub_rows = lopo_folds/3 if lopo_folds%3 == 0 else int(lopo_folds/3) + 1
    fig, axs = plt.subplots(int(sub_rows), 3, figsize = (10, sub_rows*3))


    for i, boxplot_data in enumerate(per_LOPO_boxplot_data):
        ax = axs[int(i/3), i%3]
        show_boxplots(boxplot_data, per_LOPO_blast_perc[i], roc_TUNED_THRESHOLDs[i], axis = ax, LOPO_fold = i)

    plt.tight_layout()
    if save_dir:
        fig.savefig(f'{save_dir}/5.comparison_boxplot_LOPO_fold.pdf', format = 'pdf')
    
    plt.show()            
    plt.close()




def show_boxplots(boxplot_data, timepoints_blast_perc, TUNED_THRESHOLD, axis = None, save_dir = None, LOPO_fold = None):
    sns.set_style("ticks")


    timepoints_labels = [timepoint["True_Label"] for i, timepoint in enumerate(boxplot_data)]
    timepoints_trials_scores = [timepoint["Timepoint_trials_scores"] for timepoint in boxplot_data]

    if axis is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        ax = axis


    # 2. Plot the boxplot on the first axes (ax1)
    #ax.boxplot(timepoints_trials_scores, labels=timepoints_labels, alpha= 0.5)
    bp = ax.boxplot(timepoints_trials_scores, labels=timepoints_labels, patch_artist=True)

    # patch_artist=True è necessario per poter modificare i box
    for patch in bp['boxes']:
        patch.set_alpha(0.7)
        patch.set_facecolor('none')

    for whisker in bp['whiskers']:
        whisker.set_alpha(0.7)

    for cap in bp['caps']:
        cap.set_alpha(0.7)

    for median in bp['medians']:
        median.set_alpha(0.7)

    for flier in bp['fliers']:
        flier.set_alpha(0.7)



    for s, sample in enumerate(timepoints_trials_scores):
        for p, point in enumerate(sample):
            ax.scatter(s+1, point, color = 'grey', alpha = 0.5)
        mean_pred = np.mean(sample)
        ax.scatter(s+1, mean_pred, color = 'red', s = 50)


    ax.set_xlabel("Timepoint (Bottom)")
    ax.set_ylabel("Scores")

    # 3. Create the SECOND axes (ax2)
    ax2 = ax.twiny()

    # 4. Configure the second axes
    # Make ax2 have the same limits and ticks as ax1
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(ax.get_xticks()) # Positions [1, 2, 3, 4]

    # Set the new labels for the top axis
    ax2.set_xticklabels(timepoints_blast_perc, fontsize=8)
    ax2.set_xlabel("Group (Top)")
    ax.axhline(y=TUNED_THRESHOLD, color='red', linestyle='--', label=f'Tuned Threshold ({TUNED_THRESHOLD})')
    ax.set_ylim(0, 1) #
    plt.title("Boxplots with Two X-Axes")

    if axis is None:
        plt.tight_layout()

        if save_dir:
            if LOPO_fold is None:
                print('Warning: same name for each barplot. Overwrite: True!')
                LOPO_fold = ''
            plt.savefig(f'{save_dir}/5.1.boxplot_LOPO_fold_{LOPO_fold}.pdf', format = 'pdf')
            plt.show()
            plt.close()

















def plot_roc_curve(fpr, tpr, thresholds, axis = None, save_dir = None):

        opt_idx = np.argmax(tpr >= 0.95)  # index of the threshold that has the highest tpr
        roc_threshold = thresholds[opt_idx] # extract the threshold
        opt_fpr, opt_tpr = fpr[opt_idx], tpr[opt_idx]
        roc_auc = auc(fpr, tpr)

        if axis is None:
            fig, ax = plt.subplots(figsize=(6, 5))
        else:
            ax = axis

        # ROC Curve
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
        ax.plot(opt_fpr, opt_tpr, 'ro', label=f'Optimal (thr={roc_threshold:.2f})')
        ax.plot([0,1], [0,1], 'k--', lw=1)
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.set_title('ROC Curve')
        ax.legend()
        ax.grid(alpha=0.3)

        if axis is None:
            plt.tight_layout()
            if save_dir:
                plt.savefig(f'{save_dir}/1.1.roc_curve.pdf', format = 'pdf')
            plt.show()
            plt.close()
        else:
            plt.show()

def sensitivity_vs_specificity(fpr, tpr, thresholds, axis = None, save_dir = None):
        opt_idx = np.argmax(tpr >= 0.95)  # index of the threshold that has the highest tpr
        roc_threshold = thresholds[opt_idx] # extract the threshold

        if axis is None:
            fig, ax = plt.subplots(figsize=(6, 5))
        else:
            ax = axis

        # Threshold plot
        ax.plot(thresholds, tpr, label='Sensitivity (TPR)', color='green')
        ax.plot(thresholds, 1 - fpr, label='Specificity (TNR)', color='purple')
        ax.axvline(roc_threshold, color='red', linestyle='--', label=f'Optimal thr = {roc_threshold:.2f}')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Rate')
        ax.set_title('Sensitivity & Specificity vs Threshold')
        ax.legend()
        ax.grid(alpha=0.3)

        if axis is None:
            plt.tight_layout()
            if save_dir:
                plt.savefig(f'{save_dir}/1.2.sensitivity_vs_specificity.pdf', format = 'pdf')
            plt.show()
            plt.close()
        else:
            plt.show()



def comulative_num_samples_sum(data):
    num_samples_cum = []
    cum_sum = 0
    for patient in data:#ens_roc_per_LOPO_boxplot_data:
        cum_sum += len(patient)
        num_samples_cum.append(cum_sum)
    print(num_samples_cum)
    return num_samples_cum

def retrieve_blast_perc(per_donor_original_test_datasets):

      patient_blast_perc = []
      for donor in per_donor_original_test_datasets:
          for i, dataset in enumerate(donor):
                    blast_n = (dataset['IsBlast'] == 1).sum()/len(dataset)
                    patient_blast_perc.append(round(blast_n*100,4))
      return patient_blast_perc


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import random

def show_dot_boxplot_metrics(ens_all_roc_metrics, axis = None, file_name = None, save_dir = None, mean = True, title = None, dots = True,  figsize = (6,5), 
                                  legend_pos = 1.15, fold_or_seed = 'Fold'):
    random.seed(42)
    if title is None:
        title = 'Ensemble Boxpot Metrics'
    mean_across_folds = {'f1':[], 'rec': [], 'pre': [],  'acc': []}
    
    mean_across_folds = {met: [] for met, _ in ens_all_roc_metrics[0].items()}
    if mean:
        for fold in ens_all_roc_metrics:
            for met, values in fold.items():
                values = np.mean(values)
                mean_across_folds[met].append(values)
    else:
        mean_across_folds = ens_all_roc_metrics[0]

    df = pd.DataFrame(mean_across_folds)

    metrics_names = df.columns.to_list()
    metrics_means = df.mean(axis=0).to_list()
    # Reshape da wide a long (necessario per seaborn)
    df_long = df.melt(var_name='Metric', value_name='Score')

    if axis is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        ax = axis

    colors = ['blue', 'orange', 'green', 'red' ]
    full_metrics_names = ['F1-score', 'Recall', 'Precision', 'Accuracy']
    #print(df)
    df.columns = full_metrics_names
    # 2. Plot the boxplot on the first axes (ax1)
    bp = ax.boxplot(df, patch_artist=True, showmeans=True, meanline=True, showfliers=False)
    for i, patch in enumerate(bp['boxes']):
        patch.set_alpha(0.5)
        patch.set_facecolor(colors[i])
        

    for median in bp['medians']:
        median.set_visible(False)

    for i, m in enumerate(bp['means']):
        m.set_color('black')
        m.set_linestyle('--')
        m.set_linewidth(2)
        m.set_alpha(0.8)
        if i == 0:
          round_mean = [round(met, 4) for met in metrics_means]
          m.set_label(f'Metrics Means: {round_mean}')

    #draw dots
    if dots:
        for c, col in enumerate(df.columns):
            sorted_points = np.sort(df[col])

            #ensure the points are not overlapped
            for p, point in enumerate(sorted_points):
                if p == 0:
                    x_pos = c+1
                    counter = 1
                else:
                    if point == sorted_points[p-1]:
                        x_pos = c+1 + random.uniform(-0.2, 0.2)
                        counter = (-1)*counter if counter > 0 else (-1)*counter + 0.5

                    else:
                        x_pos = c+1
                        counter = 1

                label = f'i-th {fold_or_seed} Prediction' if c == 0 and p == 0 else None
                ax.scatter(x_pos, point, color='grey', label=label, s =15)

    ax.set_title(title, fontsize=13, pad=40)
    ax.set_ylabel('Score')
    ax.set_xlabel('Performance Metrics')
    ax.set_ylim(-0.05,1.05)
    ax.set_xticklabels(df.columns, fontsize=10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, legend_pos), 
              frameon=True,
          bbox_transform=ax.transAxes)

    #plt.title("Weight Initialization Sensitivity", pad=40)  # pad aumenta lo spazio tra titolo e legenda
    if axis is None:
        plt.tight_layout()
        if save_dir is not None:
            if file_name is None:
                file_name = '2.1.dot_boxplot'

            plt.savefig(f'{save_dir}/{file_name}.pdf', format = 'pdf', bbox_inches='tight')
            plt.show()
            plt.close()

def show_ensamble_heatmap(ens_all_roc_metrics, axis = None, save_dir = None, mean = True, file_name = None, title = None, left_out_pat_list = None, figsize = (6,5)):
    mean_across_folds = {'f1':[], 'rec': [], 'pre': [],  'acc': []}

    if title is None:
        title = 'Ensemble Boxpot Metrics'
    if mean:
        for fold in ens_all_roc_metrics:
            for met, values in fold.items():
                values = np.mean(values)
                mean_across_folds[met].append(values)
    else:
        mean_across_folds = ens_all_roc_metrics

    df = pd.DataFrame(mean_across_folds)

    if axis is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        ax = axis

    if left_out_pat_list is None:
        heat_yticks = [f'Fold {i+1}' for i in range(folds)]
    else:
        heat_yticks = [f'Pat. {pat}' for pat in left_out_pat_list]
    folds = len(df.index)
    sns.heatmap(df, annot=True, fmt='.3f', cmap='YlGnBu',
            yticklabels=heat_yticks, ax = ax)

    ax.set_title(title, fontsize=13, pad = 40)

    if axis is None:
        plt.tight_layout()
        if save_dir is not None:
            plt.savefig(f'{save_dir}/2.2.predictions_heatmap.pdf', format = 'pdf', bbox_inches='tight')
            plt.show()
            plt.close()


def show_dotbox_heat(samples_info_dict, save_dir = None, mean = True, dots = True, file_name = None,  title_1 = None, title_2 = None, sup_title = None, left_out_pat_list = None, sub_figsize = (6,5),
                     legend_pos = 1.15, fold_or_seed = 'Fold'):
 
  
    w = sub_figsize[0]
    l = sub_figsize[1]
    fig, axs = plt.subplots(1, 2, figsize=(2*w, l+1))

    show_dot_boxplot_metrics(samples_info_dict, axis = axs[0], mean = mean, dots = dots, 
                      title = title_1, figsize = sub_figsize, legend_pos = legend_pos, fold_or_seed = fold_or_seed)
    show_ensamble_heatmap(samples_info_dict, axis = axs[1], mean = mean,  title = title_2, 
                      left_out_pat_list = left_out_pat_list, figsize = sub_figsize )

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    if sup_title is None:
        sup_title = 'Metrics from Ensemble Procedure'
    fig.suptitle(sup_title, fontsize = 15, y=1.05)

    if save_dir:
        if file_name is None:
            file_name = '2.dotbox_heat'

        fig.savefig(f'{save_dir}/{file_name}.pdf', format = 'pdf', bbox_inches='tight')

    plt.show()
    plt.close()

def show_dotbox_dotboxheat(samples_info_dict, all_dir_metrics_across_LOPO, fold_or_seed = 'Fold',
                           save_dir = None, mean = True, dots = True, file_name = None,  
                           title_1 = None, title_2 = None, title_3 = None, sup_title = None, 
                           left_out_pat_list = None, sub_figsize = (6,5), legend_pos = 1.15):
 
  
    w = sub_figsize[0]
    l = sub_figsize[1]
    fig, axs = plt.subplots(1, 3, figsize=(3*w, l+1))

    show_dot_boxplot_metrics(all_dir_metrics_across_LOPO,  axis = axs[0], mean = mean, dots = dots, 
                          title = title_1, figsize = sub_figsize, legend_pos = legend_pos, fold_or_seed = fold_or_seed)
    show_dot_boxplot_metrics(samples_info_dict, axis = axs[1], mean = mean, dots = dots, title = title_2, 
                            figsize = sub_figsize, legend_pos = legend_pos, fold_or_seed = fold_or_seed)

    show_ensamble_heatmap(samples_info_dict, axis = axs[2], mean = mean,  title = title_3, left_out_pat_list = left_out_pat_list, figsize = sub_figsize )

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    if sup_title is None:
        sup_title = 'Metrics from Ensemble Procedure'
    fig.suptitle(sup_title, fontsize = 15, y=1.05)

    if save_dir:
        if file_name is None:
            file_name = '2.dotbox_heat'

        fig.savefig(f'{save_dir}/{file_name}.pdf', format = 'pdf', bbox_inches='tight')

    plt.show()
    plt.close()


def show_dotbox_dotbox(samples_info_dict, all_dir_metrics_across_LOPO, 
                           save_dir = None, mean = True, dots = True, file_name = None,  
                           title_1 = None, title_2 = None, sup_title = None,
                           fold_or_seed = 'Fold', 
                           left_out_pat_list = None, sub_figsize = (6,5), legend_pos = 1.15):
 
  
    w = sub_figsize[0]
    l = sub_figsize[1]
    fig, axs = plt.subplots(1, 2, figsize=(2*w, l+1))

    show_dot_boxplot_metrics(all_dir_metrics_across_LOPO,  axis = axs[0], mean = mean, dots = dots, title = title_1, 
                            figsize = sub_figsize, legend_pos = legend_pos, fold_or_seed = fold_or_seed)
    
    show_dot_boxplot_metrics(samples_info_dict, axis = axs[1], mean = mean, dots = dots, title = title_2, 
                            figsize = sub_figsize, legend_pos = legend_pos, fold_or_seed = fold_or_seed)
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    if sup_title is None:
        sup_title = 'Metrics from Ensemble Procedure'
    fig.suptitle(sup_title, fontsize = 15, y=1.05)

    if save_dir:
        if file_name is None:
            file_name = '2.dotbox_heat'

        fig.savefig(f'{save_dir}/{file_name}.pdf', format = 'pdf', bbox_inches='tight')

    plt.show()
    plt.close()


def retrieve_all_LOPO_ensamble_thresholds(LOPO_folds, cellcnn_path, tuning_exp):
    ens_roc_thr_per_fold = []
    ens_rob_thr_per_fold = []
    for i, _ in enumerate(range(LOPO_folds)):
        tuning_load_dir = f'{cellcnn_path}/experiments/experiment_{tuning_exp}/outer_fold_{i}/ensemble/tuning/resampling/'
        with open(os.path.join(tuning_load_dir, 'ensemble_robust_threshold.pkl'), 'rb') as f:
                    ensemble_robust_threshold = pkl.load(f) # prediction used to tune the threshold

        tuning_load_dir = f'{cellcnn_path}/experiments/experiment_{tuning_exp}/outer_fold_{i}/ensemble/tuning/roc/'
        with open(os.path.join(tuning_load_dir, 'ensemble_roc_threshold.pkl'), 'rb') as f:
                    ensemble_roc_threshold = pkl.load(f) # prediction used to tune the threshold
        
        ens_roc_thr_per_fold.append(ensemble_roc_threshold)
        ens_rob_thr_per_fold.append(ensemble_robust_threshold)

    return ens_roc_thr_per_fold, ens_rob_thr_per_fold

def scores_from_robust_labelling(per_donor_original_test_y_flat, test_total_labels_flat):
    robust_metrics_across_trials= {}
    mod_n = 'f1'
    robust_metrics_across_trials[mod_n] = f1_score(per_donor_original_test_y_flat, test_total_labels_flat, pos_label = 1, zero_division = 0 )
    mod_n = 'rec'
    robust_metrics_across_trials[mod_n] = recall_score(per_donor_original_test_y_flat, test_total_labels_flat, pos_label = 1, zero_division = 0 )
    mod_n = 'pre'
    robust_metrics_across_trials[mod_n] = precision_score(per_donor_original_test_y_flat, test_total_labels_flat, pos_label = 1, zero_division = 0 )
    mod_n = 'acc'
    robust_metrics_across_trials[mod_n] = accuracy_score(per_donor_original_test_y_flat, test_total_labels_flat)


    return robust_metrics_across_trials

def retrieve_all_LOPO_ensamble_metrics(save_ensemble_dir, roc_TUNED_THRESHOLD, pred = False):
        ens_folds = len([fold_name for fold_name in list(os.listdir(save_ensemble_dir)) if 'fold' in fold_name])
        print(ens_folds)

        roc_ens_metrics_dicts = []
        rob_ens_metrics_dicts = []

        for i, _ in enumerate(range(ens_folds)):
            

            save_robust_dir = f'{save_ensemble_dir}/inner_fold_{i}/predictions/robust/'

            with open(os.path.join(save_robust_dir, 'test_total_trial_pred_lists.pkl'), 'rb') as f:
                                test_total_trial_pred_lists = pkl.load(f)

            with open(os.path.join(save_robust_dir, 'per_donor_original_test_y.pkl'), 'rb') as f:
                                per_donor_original_test_y = pkl.load(f)

            #print(flatten(test_total_trial_pred_lists)[:5])
            if pred:
                ens_roc_test_total_labels, ens_roc_test_total_preds = robust_prediction_labelling(test_total_trial_pred_lists, roc_TUNED_THRESHOLD*100, pred = True)
            
            ens_roc_test_total_labels = robust_prediction_labelling(test_total_trial_pred_lists, roc_TUNED_THRESHOLD*100)



            per_donor_original_test_y_flat = flatten(per_donor_original_test_y)
            ens_roc_test_total_labels_flat = flatten(ens_roc_test_total_labels)
            
            if pred:  
                rounded_pred = [round(pr, 6) for pr in  flatten(ens_roc_test_total_preds)] 
                print('Pred Prob:', rounded_pred , 'Pred Labels:', ens_roc_test_total_labels_flat)
            print('Pred Labels:', ens_roc_test_total_labels_flat)
            print('True Labels:', per_donor_original_test_y_flat)

            roc_robust_metrics_across_trials = scores_from_robust_labelling(per_donor_original_test_y_flat, ens_roc_test_total_labels_flat)


            roc_ens_metrics_dicts.append(roc_robust_metrics_across_trials)
 
        df = pd.DataFrame(roc_ens_metrics_dicts)#.to_dict()
        print(df.columns)
        new = {}
        for key in df.columns:
            new[key] = df[key].to_list()
        print(new)
        return new

def elaborate_ens_data_for_box_violin(save_ensemble_dir, thresholds_list, num_samples):
        ens_folds = len([fold_name for fold_name in list(os.listdir(save_ensemble_dir)) if 'fold' in fold_name])

        roc_per_cv_fold_violin = []

        roc_per_cv_fold_box = [[] for _ in range(num_samples)]

        for i, _ in enumerate(range(ens_folds)):

            roc_TUNED_THRESHOLD = thresholds_list[i]
            save_robust_dir = f'{save_ensemble_dir}/inner_fold_{i}/predictions/robust/'

            with open(os.path.join(save_robust_dir, 'test_total_trial_pred_lists.pkl'), 'rb') as f:
                                test_total_trial_pred_lists = pkl.load(f)

            with open(os.path.join(save_robust_dir, 'per_donor_original_test_y.pkl'), 'rb') as f:
                                per_donor_original_test_y = pkl.load(f)

            with open(os.path.join(save_robust_dir, 'per_donor_resampled_test_y.pkl'), 'rb') as f:
                                per_donor_resampled_test_y = pkl.load(f)

            #ROC
            plot_data, boxplot_data, f1_data = final_trials_prediction(test_total_trial_pred_lists,
                            per_donor_original_test_y, per_donor_resampled_test_y, roc_TUNED_THRESHOLD)

            roc_per_cv_fold_violin += plot_data

            for s, sample in enumerate(boxplot_data):
                roc_per_cv_fold_box[s].append(np.mean(sample['Timepoint_trials_scores']))


        roc_cv_box_mean_across_folds = []

        for s, sample in enumerate(boxplot_data):

            dict_across_folds = {}
            dict_across_folds['True_Label'] = sample['True_Label']
            dict_across_folds['Timepoint_trials_scores'] = roc_per_cv_fold_box[s]
            roc_cv_box_mean_across_folds.append(dict_across_folds)
        return roc_per_cv_fold_violin, roc_cv_box_mean_across_folds


def show_all_LOPO_boxplots(boxplot_data, timepoints_blast_perc, thresholds_list, num_samples_cum, 
                        save_dir = None, file_name = None, title = None, fold_or_seed = 'Fold', y_labels = None):
    sns.set_style("ticks")

    round_timepoints_blast_perc = [round(perc, 3) for perc in timepoints_blast_perc]

    timepoints_labels = [timepoint["True_Label"] for i, timepoint in enumerate(boxplot_data)]
    timepoints_trials_scores = [timepoint["Timepoint_trials_scores"] for timepoint in boxplot_data]

    if y_labels:
        x_axis_1_labels = y_labels
    else:
        x_axis_1_labels = timepoints_labels

    print(timepoints_trials_scores)
    n_box = len(timepoints_labels)
    fig, ax = plt.subplots(figsize=(0.5*n_box, 4))

    # 2. Plot the boxplot on the first axes (ax1)
    bp = ax.boxplot(timepoints_trials_scores, labels=x_axis_1_labels, patch_artist=True,
                                                  showmeans=True,      # mostra la media
                                                  meanline=True, showfliers=False) 
    for patch in bp['boxes']:
        patch.set_alpha(0.3)
        patch.set_facecolor('none')

    for whisker in bp['whiskers']:
        whisker.set_alpha(0.3)

    for cap in bp['caps']:
        cap.set_alpha(0.3)

    for median in bp['medians']:
        median.set_visible(False)

    for i, mean in enumerate(bp['means']):
        mean.set_color('red')
        mean.set_linestyle('--')
        mean.set_linewidth(2)
        mean.set_alpha(0.8)
        if i == 0:
            mean.set_label('Mean')

    for s, sample in enumerate(timepoints_trials_scores):
        for p, point in enumerate(sample):
            label = f'i-th {fold_or_seed} Prediction' if s == 0 and p == 0 else None
            ax.scatter(s+1, point, color='grey', label=label, s =10)

    ax.set_xlabel("True Samples Labels")
    ax.set_ylabel("Positivity Score")
    ax.set_xticklabels(x_axis_1_labels, rotation=45, ha='right', fontsize=7)


    ax2 = ax.twiny()

    # Make ax2 have the same limits and ticks as ax1
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(ax.get_xticks()) # Positions [1, 2, 3, 4]

    # Set the new labels for the top axis
    ax2.set_xticklabels(round_timepoints_blast_perc, fontsize=7)
    ax2.set_xlabel("% of Unhealthy cells per Sample")

    for i, threshold in enumerate(thresholds_list):
        if i == 0:
            x_min = 0.5
        else:
            x_min = num_samples_cum[i-1] + 0.5
        x_max = num_samples_cum[i] +0.5

        box_label = f'j-th LOPO Threshold' if i == 0 else None

        ax.hlines(y=threshold, xmin=x_min, xmax=x_max, color='red',
               linestyles='solid', label=box_label, linewidth=1.5, alpha = 0.3)

        ax.vlines(x=x_max, ymin = -0.05, ymax = 1.05, color = 'grey', linestyle='--', alpha = 0.5)

    ax.hlines(y=0.5, xmin=0.5, xmax=x_max, color='black',
               linestyles='dotted', label='Default Threshold = 0.5', linewidth=1.5, alpha = 0.3)

    # dopo il loop dei scatter grigi
    mis_label = True
    for s, sample in enumerate(timepoints_trials_scores):
        thr_index = num_samples_cum.index(next(x for x in num_samples_cum if x > s))
        thr = thresholds_list[thr_index]
        true_label = timepoints_labels[s]        # 0 o 1
        # qui devi avere la predicted label per il campione s
        pred_label = 1 if np.mean(sample) > thr else 0    # 0 o 1

            
        if true_label != pred_label:
            ax.scatter(s + 1, 1.025, color='red', marker='x', s=50, linewidths=1.5, zorder=5,
                      label='Misclassified' if mis_label == True else None)

            if mis_label == True:
                mis_label = False

    plt.ylim(-0.05, 1.05) #
    if title is  None:
        title = "Positivity Score Distribution across LOPOCV Iterations"

    plt.title(title)
    ax.legend()
    if save_dir:
        if file_name is None:
            file_name = '5.2.all_box_straight.pdf'
        fig.savefig(f'{save_dir}/{file_name}', format = 'pdf', bbox_inches='tight')

    plt.show()
    plt.close()


def comulative_num_samples_sum(data):
    num_samples_cum = []
    cum_sum = 0
    for patient in data:#ens_roc_per_LOPO_boxplot_data:
        cum_sum += len(patient)
        num_samples_cum.append(cum_sum)
    print(num_samples_cum)
    return num_samples_cum

def retrieve_blast_perc(per_donor_original_test_datasets):

      patient_blast_perc = []
      for donor in per_donor_original_test_datasets:
          for i, dataset in enumerate(donor):
                    blast_n = (dataset['IsBlast'] == 1).sum()/len(dataset)
                    patient_blast_perc.append(round(blast_n*100,4))
      return patient_blast_perc



def direct_prediction_across_seeds(direct_pred_acorss_seeds, thr = 0.5):
        final_labels_per_patient = []
        mean_prob_per_patient = []
        for p, patient in enumerate(direct_pred_acorss_seeds[0]):
            mean_probs_per_sample = []
            final_labels_per_sample = []
            for  s, sample in enumerate(patient):
                sample_pred = []

                for t, seed in enumerate(direct_pred_acorss_seeds):

                    sample_pred.append(direct_pred_acorss_seeds[t][p][s][1])

                mean_prob = np.mean(sample_pred)
                mean_probs_per_sample.append(mean_prob)
                label = 1 if mean_prob >= thr else 0
                final_labels_per_sample.append(label)
                print(f'patient: {p}, sample: {s}, mean_prob: {mean_prob}, pred_label: {label}')

            final_labels_per_patient.append(final_labels_per_sample)
            mean_prob_per_patient.append(mean_probs_per_sample)
        return final_labels_per_patient, mean_prob_per_patient





"""============================================================================="""

def retrieve_all_LOPO_thresholds(LOPO_folds, cellcnn_path, tuning_exp):

    roc_thr_per_fold = []
    rob_thr_per_fold = []

    for i, _ in enumerate(range(LOPO_folds)):
        tuning_load_dir = f'{cellcnn_path}/experiments/experiment_{tuning_exp}/outer_fold_{i}/tuning/results'


        with open(os.path.join(tuning_load_dir, 'robust_threshold.pkl'), 'rb') as f:
                                robust_threshold = pkl.load(f)

        with open(os.path.join(tuning_load_dir, 'roc_threshold.pkl'), 'rb') as f:
                                roc_threshold = pkl.load(f)
        roc_thr_per_fold.append(roc_threshold)
        rob_thr_per_fold.append(robust_threshold)
    return roc_thr_per_fold, rob_thr_per_fold



def scores_from_robust_labelling(per_donor_original_test_y_flat, test_total_labels_flat):
    robust_metrics_across_trials= {}
    mod_n = 'f1'
    robust_metrics_across_trials[mod_n] = f1_score(per_donor_original_test_y_flat, test_total_labels_flat, pos_label = 1, zero_division = 0 )
    mod_n = 'rec'
    robust_metrics_across_trials[mod_n] = recall_score(per_donor_original_test_y_flat, test_total_labels_flat, pos_label = 1, zero_division = 0 )
    mod_n = 'pre'
    robust_metrics_across_trials[mod_n] = precision_score(per_donor_original_test_y_flat, test_total_labels_flat, pos_label = 1, zero_division = 0 )
    mod_n = 'acc'
    robust_metrics_across_trials[mod_n] = accuracy_score(per_donor_original_test_y_flat, test_total_labels_flat)


    return robust_metrics_across_trials


def from_orig_to_res_structure(original_predictions_list, per_donor_original_test_y):
    patients = []
    samples_trials = []
    num_samples = len(original_predictions_list[0])
    for s in range(num_samples):
        sample_trial = []
        for t, trial in enumerate(original_predictions_list):
            sample_trial.append(original_predictions_list[t][s])
        samples_trials.append(sample_trial)

    orig_pred_for_violin = []
    for sample in samples_trials:
        trials = []
        for neg, pos in sample:
            subsets_pred = [pos]
            #print(neg, pos)
            trials.append(subsets_pred)
        orig_pred_for_violin.append(trials)
    patients.append(orig_pred_for_violin)

    patients_y = []
    for p, pat in enumerate(patients):
        pat_y = []
        for s, sample in enumerate(pat):
            sample_y = []
            sample_label = per_donor_original_test_y[p][s]
            #print(sample_label)

            pat_y.append([sample_label])
        patients_y.append(pat_y)

    return patients, patients_y

def elaborate_data_for_box_violin(save_robust_dir, threshold = 0.5):

    with open(os.path.join(save_robust_dir, 'test_total_trial_pred_lists.pkl'), 'rb') as f:
                            test_total_trial_pred_lists = pkl.load(f)

    with open(os.path.join(save_robust_dir, 'per_donor_resampled_test_y.pkl'), 'rb') as f:
                            per_donor_resampled_test_y = pkl.load(f)

    with open(os.path.join(save_robust_dir, 'per_donor_original_test_y.pkl'), 'rb') as f:
                            per_donor_original_test_y = pkl.load(f)

    plot_data, boxplot_data, _ = final_trials_prediction(test_total_trial_pred_lists,
                            per_donor_original_test_y, per_donor_resampled_test_y, threshold)
    return plot_data, boxplot_data