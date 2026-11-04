
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats



def plot_total_time(comprehensive_time_dict, filters_sel_time, method_list, ax=None, DEFAULT_KEY='Default'):
    """Single stacked bar chart with all methods on the same axis. Error bars (std across folds) are drawn only on the analysis segment."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(13, 5))

    filter_mean = float(np.mean(filters_sel_time))
    colors = {'single': '#4C72B0', 'mean': '#55A868', 'logistic': '#C44E52', 'top3': '#8172B2'}

    labels, xs, group_centers, group_names = [], [], [], []
    pos = 0
    for method in method_list:
        d = dict(comprehensive_time_dict[method])
        cluster_time = float(np.mean(d.pop('filter_sel'))) if 'filter_sel' in d else 0

        # mean and standard deviation across folds, for each threshold
        analysis_mean = {k: float(np.mean(v)) for k, v in d.items()}
        analysis_std = {k: float(np.std(np.asarray(v, dtype=float), ddof=1)) for k, v in d.items()}
        has_filters = method != 'top3'

        start = pos
        first_thr = list(analysis_mean)[0]
        for thr, t in analysis_mean.items():
            base = 0
            if has_filters:
                ax.bar(pos, filter_mean, color='lightgray',
                       label='Filter selection' if pos == 0 else None)
                ax.bar(pos, cluster_time, bottom=filter_mean, color='orange',
                       label='Filter clustering' if pos == 0 else None)
                base = filter_mean + cluster_time

            # only this has error bars
            ax.bar(pos, t, bottom=base, color=colors[method],
                   yerr=analysis_std[thr], capsize=3,
                   error_kw=dict(ecolor='black', elinewidth=1, capthick=1),
                   label=f'{method.capitalize()} analysis' if thr == first_thr else None)

            labels.append('' if thr == DEFAULT_KEY else str(thr))
            xs.append(pos)
            pos += 1
        group_centers.append((start + pos - 1) / 2)
        group_names.append(method.capitalize())
        pos += 1  # add space between method groups

    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    for c, name in zip(group_centers, group_names):
        ax.annotate(name, xy=(c, 0), xycoords=('data', 'axes fraction'),
                    xytext=(0, -28), textcoords='offset points',
                    ha='center', fontweight='bold')
    ax.set_ylabel('Average time (seconds)')
    ax.set_title('Total time: filter extraction + analysis, per method and threshold\n(error bars: ±1 std across folds, analysis only)')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    return ax



def plot_total_boxplots(all_results_dict, score='F1_Score', axis=None, 
                        default_key='Default', title=None):
    """
    Draw a boxplot with overlaid individual points for each method and threshold.
    """
    if axis is None:
        fig, ax = plt.subplots(figsize=(14, 5))
    else:
        ax = axis

    colors = {
        'single': '#4C72B0', 
        'mean': '#55A868',
        'logistic': '#C44E52', 
        'top3': '#8172B2'
    }
    rng = np.random.default_rng(0)

    xs, labels, group_centers, group_names = [], [], [], []
    pos = 0

    for method, d in all_results_dict.items():
        color = colors.get(method, '#8DA0CB')
        start = pos

        for thr, result in d.items():
            values = result[score].dropna()
            if len(values) == 0:
                continue

            bp = ax.boxplot(
                values, positions=[pos], widths=0.5, showfliers=False,
                patch_artist=True, manage_ticks=False,
                medianprops=dict(color='black', linewidth=1.5)
            )
            for patch in bp['boxes']:
                patch.set_facecolor(color)
                patch.set_alpha(0.5)

            jitter = rng.uniform(-0.15, 0.15, size=len(values))
            ax.scatter(pos + jitter, values, color='black', alpha=0.5, s=15, zorder=3)

            labels.append('' if str(thr) == str(default_key) else str(thr))
            xs.append(pos)
            pos += 1

        if pos > start:
            group_centers.append((start + pos - 1) / 2)
            group_names.append(method.capitalize())
            pos += 1  # Gap between methods

    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.8, pos - 1.2)

    # Method names centered below their thresholds
    for c, name in zip(group_centers, group_names):
        ax.annotate(name, xy=(c, 0), xycoords=('data', 'axes fraction'),
                    xytext=(0, -28), textcoords='offset points',
                    ha='center', fontweight='bold')

    # Vertical lines
    div_pos = 0
    for d in list(all_results_dict.values())[:-1]:
        div_pos += len(d)
        ax.axvline(div_pos, color='gray', linestyle=':', alpha=0.5)
        div_pos += 1

    ax.set_ylabel(score)
    ax.set_title(title or f'{score} per Method and Threshold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)

    if axis is None:
        plt.tight_layout()
        plt.show()
    return ax
