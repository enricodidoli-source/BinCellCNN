import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from IPython.display import display

# Metriche di classificazione
from sklearn.metrics import (
    f1_score, 
    recall_score, 
    precision_score, 
    accuracy_score,
    precision_recall_curve,
    confusion_matrix
)

# Import dai tuoi moduli interni
from src.cell_level.cell_level_utils import clear_RAM
from src.cell_level.networks_ops import get_filter_prob

def get_best_thr(true_cells_labels, pred_probs, show = False):
    """Calculates the optimal threshold that maximizes the F1-score."""

    print('=========== Finding Best Threshold ==============')

    prec, rec, thr = precision_recall_curve(true_cells_labels, pred_probs)

    # Calculate F1 score for all thresholds
    f1_curve = 2 * prec * rec / (prec + rec + 1e-12)
    best_idx = np.argmax(f1_curve[:-1])
    best_thr = thr[best_idx]
    max_f1 = f1_curve[:-1][best_idx]

    print(f'F1-score: { max_f1*100:.2f}%')

    if show:
        show_f1_curve(thr, f1_curve[:-1])

    clear_RAM()
    return  best_thr, max_f1


def check_detected_cells(comprehensive_df, thr, metrics_list = None, function_list = None):
    """Evaluates the predictions against true labels using given metrics."""

    print('Total number of cells:', len(comprehensive_df))
    print('Total number of cancer cells:', len(comprehensive_df[comprehensive_df['IsBlast'] == 1]))
    print('Percentage of cancer cells:', round(100*len(comprehensive_df[comprehensive_df['IsBlast'] == 1]) / len(comprehensive_df),4), '%' )
    print('Adopted Threshold:', thr)

    true_labels = comprehensive_df['IsBlast'].to_numpy()
    act_cells_values = comprehensive_df.iloc[:,0]


    print('True Labels Count', len(comprehensive_df[comprehensive_df['IsBlast'] == 1]))

    if metrics_list is None:
        metrics_list = ['F1-score', 'Recall', 'Precision', 'Accuracy']

    if function_list is None:
        from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
        function_list = [f1_score, recall_score, precision_score, accuracy_score]

    pred_labels = (act_cells_values > thr).astype(int)
    print('Pred Labels Count', np.sum(pred_labels))

    for met, func in zip(metrics_list, function_list):
        met_score = func(true_labels, pred_labels)
        print(f'{met}: { met_score*100:.2f}%')




def evaluate_single_filter(comprehensive_val_df, true_cells_labels, model_filters, model_scalers):
    """Evaluates individual filter from an initialization seed and returns their max F1 and best threshold."""
    model_max_f1 = []
    model_best_thr = []

    # contains several filters from the same seed (filters from the best 3 nets are have been concatewnated in a single list)
    for f, (filter, scaler) in enumerate(zip(model_filters, model_scalers)):
        print('Analyzing filter:', f)

        pred_probs = get_filter_prob(filter, comprehensive_val_df, scaler=scaler)
        best_thr, max_f1  = get_best_thr(true_cells_labels, pred_probs)
        print(f'F1-score: { max_f1*100:.2f}%')

        model_max_f1.append(max_f1)
        model_best_thr.append(best_thr)

    clear_RAM()
    return model_best_thr, model_max_f1



def get_metrics_df(y_true, y_pred, max_f1, LOPOCV_fold_idx, idx_x):
    """Calculates evaluation metrics and returns them safely formatted as a dictionary."""
    total_cells = len(y_true)
    real_blasts = int(np.sum(y_true))

    #acc = accuracy_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    results_dict = {
        'Sample_ID': f'Pat_{LOPOCV_fold_idx + 1}_{idx_x + 1}',
        'LOPO_Fold': LOPOCV_fold_idx + 1,
        'Total_Cells': total_cells,
        'Real_Blasts': real_blasts,
        'Predicted_Blasts': int(tp + fp),
        'TP (Detected_Blasts)': tp,
        'FN (Not_Detected_Blasts)': fn,
        'FP (False_Positives)': fp,
        'TN (Detected_Healthy)': tn,
        'Filter_F1': np.mean(max_f1)*100  # unica metrica non derivabile dalla cm
    }

    clear_RAM()
    return results_dict





def add_classification_metrics(df):
    df = df.copy()
    tp = df['TP (Detected_Blasts)']
    fp = df['FP (False_Positives)']
    fn = df['FN (Not_Detected_Blasts)']
    tn = df['TN (Detected_Healthy)']
    total = df['Total_Cells']
    real_blasts = df['Real_Blasts']
    predicted_blasts = df['Predicted_Blasts']

    # Changed 'summary_df' to 'df' here so it works for all passed DataFrames
    df['True_Label'] =  (df['Real_Blasts'] > 0).astype(int)

    df['True_%'] = round(100 * real_blasts / total, 4)
    df['Predicted_%'] = round(100 * predicted_blasts / total, 4)
    #df['TP_%'] = np.where(predicted_blasts > 0, round(100 * tp / predicted_blasts, 4), 0.0)

    # replica la logica originale: se non ci sono blasti reali, forzo a 0
    df['Recall'] = np.where(real_blasts > 0, round(100 * tp / (tp + fn), 4), 0.0)

    # FIX: Assign np.where first, then fillna on the Pandas Series
    df['Precision'] = np.where(real_blasts > 0, round(100 * tp / (tp + fp).replace(0, np.nan), 4), 0.0)
    df['Precision'] = df['Precision'].fillna(0.0)

    denom = df['Precision'] + df['Recall']
    df['F1_Score'] = np.where(denom > 0, round(2 * df['Precision'] * df['Recall'] / denom, 4), 0.0)
    df['Accuracy'] = round(100 * (tp + tn) / total, 4)

    clear_RAM()
    return df


# =============================



def show_f1_curve(thresholds: np.ndarray, f1_scores: np.ndarray):
    """Plots the F1 score against varying thresholds (Fixes the previous plotting bug)."""
    fig = plt.figure(figsize=(8, 5))
    plt.plot(thresholds, f1_scores, color='blue', linewidth=2)

    plt.xlabel("Probability Threshold")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs. Probability Threshold")
    plt.grid(True, linestyle='--', alpha=0.7)

    display(fig)
    plt.close(fig)

    clear_RAM()



class PipelineResults:
    def __init__(self, summary_df):
        self.summary_df = summary_df

    def _subset(self, condition=None):
        """ Returns the requested subset"""
        if condition is None:
            return self.summary_df
        elif condition == 'blast':
            return self.summary_df[self.summary_df['True_Label'] == 1]
        elif condition == 'healthy':
            return self.summary_df[self.summary_df['True_Label'] == 0]
        else:
            raise ValueError("condition must be None, 'blast' o 'healthy'")

    def show_metrics_across_filters_f1(self, condition='blast'):

        df = self._subset(condition).copy()

        df['pos_perc_dist'] = 1 - abs(df['Predicted_%'] - df['Precision'])
        df['true_perc_dist'] = 1 - abs(df['Predicted_%'] - df['True_%'])

        df_grouped = df.groupby('Filter_F1')[['Precision', 'pos_perc_dist', 'true_perc_dist']].mean().reset_index()


        cols_to_plot = ['Precision', 'pos_perc_dist', 'true_perc_dist']
        y_labels = [
            'Precision',
            'Distance between Predicted % and True Positive %',
            'Distance between Predicted % and True Cancer Cells %'
        ]

        fig, ax = plt.subplots(1, len(cols_to_plot), figsize=(20, 7))

        for e, col in enumerate(cols_to_plot):

            raw_x = df['Filter_F1'].to_numpy()
            raw_y = df[col].to_numpy()
            samples_ID = df['Sample_ID'].to_numpy()

            ax[e].scatter(raw_x, raw_y, alpha=0.6, color='#3498DB', label='Single Samples')

            for i, txt in enumerate(samples_ID):
                ax[e].annotate(txt, (raw_x[i], raw_y[i]), fontsize=8, alpha=0.7)

            grouped_x = df_grouped['Filter_F1'].to_numpy()
            grouped_y = df_grouped[col].to_numpy()

            z = np.polyfit(grouped_x, grouped_y, 1)
            p = np.poly1d(z)

            ax[e].plot(grouped_x, p(grouped_x), "r--", linewidth=2, label='Trend (Mean)')

            ax[e].set_ylabel(y_labels[e])
            ax[e].set_xlabel('Filter_F1')
            ax[e].legend()

        plt.tight_layout()
        display(fig)
        plt.close(fig)


    def show_metrics_correlation_matrix(self, cols_to_correlate=None, condition='blast'):
        df = self._subset(condition)

        if cols_to_correlate is None:
            cols_to_correlate = ['Filter_F1', 'Recall', 'Precision', 'F1_Score', 'Predicted_%', 'True_%']

        corr_matrix = df[cols_to_correlate].corr(method='pearson')

        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix,
                    annot=True,       # Mostra i numeri dentro i quadrati
                    cmap='coolwarm',  # Colori: rosso (positivo), blu (negativo)
                    fmt=".2f",        # Arrotonda a 2 decimali
                    vmin=-1, vmax=1,  # Fissa la scala da -1 a 1
                    linewidths=0.5)

        title_suffix = {'blast': '(Unhealthy Patients Only)',
                         'healthy': '(Healthy Patients Only)',
                         None: '(All Patients)'}[condition]

        plt.title(f"Metrics Correlation Matrix {title_suffix}",
                   fontsize=15, fontweight='bold', pad=15)

        plt.tight_layout()

        display(fig)
        plt.close(fig)


    def show_corr(self, x='Filter_F1', y='Recall', condition='blast', axis = None):
        df = self._subset(condition).copy()

        # Anche qui separiamo!
        df_grouped = df.groupby(x)[y].mean().reset_index()

        if axis is not None:
            ax = axis
        else:
            fig, ax = plt.subplots(figsize=(9, 6))

        rho, p_value = stats.spearmanr(df_grouped[x], df_grouped[y])

        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        stats_text = f"Spearman Rho: {rho:.2f}\np-value: {p_value:.1e} ({significance})"

        sns.scatterplot(data=df, x=x, y=y,
                        s=80, alpha=0.4, edgecolor='white', color='#2874A6', label='Single Samples', ax = ax)

        sns.regplot(data=df_grouped, x=x, y=y,
                    scatter=False,
                    line_kws={'color': '#E74C3C', 'linewidth': 2}, label='Filter Trend (Mean)', ax = ax)

        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

        ax.set_title(f'Validation vs Test: {x} vs {y}', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel(f"{x} (Validation Set)", fontsize=12)
        ax.set_ylabel(f"{y} (Test Set)", fontsize=12)
        ax.legend(loc='lower right')
        ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
        sns.despine(ax=ax)

        if axis is None:
            display(fig)
            plt.close(fig)


