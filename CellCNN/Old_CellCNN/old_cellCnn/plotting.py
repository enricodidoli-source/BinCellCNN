""" Copyright 2016-2017 ETH Zurich, Eirini Arvaniti and Manfred Claassen.

This module contains functions for plotting the results of a CellCnn analysis.

"""
#scaler
import os
import sys
import logging
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import ImageGrid
import seaborn as sns
from cellCnn.utils import mkdir_p
import statsmodels.api as sm

logger = logging.getLogger(__name__)
#plt.rcParams["mpl_toolkits.legacy_colorbar"] = False


def plot_results(results, samples, phenotypes, labels, outdir,
                 filter_diff_thres=.2, filter_response_thres=0, response_grad_cutoff=None,
                 stat_test=None, log_yscale=False,
                 group_a='group A', group_b='group B', group_names=None, tsne_ncell=10000,
                 regression=False, show_filters=True):
    """ Plots the results of a CellCnn analysis.

    Args:
        - results :
            Dictionary containing the results of a CellCnn analysis.
        - samples :
            Samples from which to visualize the selected cell populations.
        - phenotypes :
            List of phenotypes corresponding to the provided `samples`.
        - labels :
            Names of measured markers.
        - outdir :
            Output directory where the generated plots will be stored.
        - filter_diff_thres :
            Threshold that defines which filters are most discriminative. Given an array
            ``filter_diff`` of average cell filter response differences between classes,
            sorted in decreasing order, keep a filter ``i, i > 0`` if it holds that
            ``filter_diff[i-1] - filter_diff[i] < filter_diff_thres * filter_diff[i-1]``.
            For regression problems, the array ``filter_diff`` contains Kendall's tau
            values for each filter.
        - filter_response_thres :
            Threshold for choosing a responding cell population. Default is 0.
        - response_grad_cutoff :
            Threshold on the gradient of the cell filter response CDF, might be useful for defining
            the selected cell population.
        - stat_test: None | 'ttest' | 'mannwhitneyu'
            Optionally, perform a statistical test on selected cell population frequencies between
            two groups and report the corresponding p-value on the boxplot figure
            (see plots description below). Default is None. Currently only used for binary
            classification problems.
        - group_a :
            Name of the first class.
        - group_b :
            Name of the second class.
        - group_names :
            List of names for the different phenotype classes.
        - log_yscale :
            If True, display the y-axis of the boxplot figure (see plots description below) in
            logarithmic scale.
        - tsne_ncell :
            Number of cells to include in t-SNE calculations and plots.
        - regression :
            Whether it is a regression problem.
        - show_filters :
            Whether to plot learned filter weights.

    Returns:
        A list with the indices and corresponding cell filter response thresholds of selected
        discriminative filters. \
        This function also produces a collection of plots for model interpretation.
        These plots are stored in `outdir`. They comprise the following:

        - clustered_filter_weights.pdf :
            Filter weight vectors from all trained networks that pass a validation accuracy
            threshold, grouped in clusters via hierarchical clustering. Each row corresponds to
            a filter. The last column(s) indicate the weight(s) connecting each filter to the output
            class(es). Indices on the y-axis indicate the filter cluster memberships, as a
            result of the hierarchical clustering procedure.
        - consensus_filter_weights.pdf :
            One representative filter per cluster is chosen (the filter with minimum distance to all
            other memebers of the cluster). We call these selected filters "consensus filters".
        - best_net_weights.pdf :
            Filter weight vectors of the network that achieved the highest validation accuracy.
        - filter_response_differences.pdf :
            Difference in cell filter response between classes for each consensus filter.
            To compute this difference for a filter, we first choose a filter-specific class, that's
            the class with highest output weight connection to the filter. Then we compute the
            average cell filter response (value after the pooling layer) for validation samples
            belonging to the filter-specific class (``v1``) and the average cell filter response
            for validation samples not belonging to the filter-specific class (``v0``).
            The difference is computed as ``v1 - v0``. For regression problems, we cannot compute
            a difference between classes. Instead we compute Kendall's rank correlation coefficient
            between the predictions of each individual filter (value after the pooling layer) and
            the true response values.
            This plot helps decide on a cutoff (``filter_diff_thres`` parameter) for selecting
            discriminative filters.
        - tsne_all_cells.png :
            Marker distribution overlaid on t-SNE map. 

        In addition, the following plots are produced for each selected filter (e.g. filter ``i``):

        - cdf_filter_i.pdf :
            Cumulative distribution function of cell filter response for filter ``i``. This plot
            helps decide on a cutoff (``filter_response_thres`` parameter) for selecting the
            responding cell population.

        - selected_population_distribution_filter_i.pdf :
            Histograms of univariate marker expression profiles for the cell population selected by
            filter ``i`` vs all cells.

        - selected_population_frequencies_filter_i.pdf :
            Boxplot of selected cell population frequencies in samples of the different classes,
            if running a classification problem. For regression settings, a scatter plot of selected
            cell population frequencies vs response variable is generated.

        - tsne_cell_response_filter_i.png :
            Cell filter response overlaid on t-SNE map.

        - tsne_selected_cells_filter_i.png :
            Marker distribution of selected cell population overlaid on t-SNE map.
    """
    # create the output directory
    mkdir_p(outdir)

    nmark = samples[0].shape[1]   # number of measured markers

    # if results are not present, the execution stops
    if results['selected_filters'] is not None:
        logger.info("Loading the weights of consensus filters.")
        filters = results['selected_filters']
    else:
        sys.exit('Consensus filters were not found.')

    # ======================================================================== #
    if show_filters:
        print('Showing Filters') 
        plot_filters(results, labels, outdir)
    # get discriminative filter indices in consensus matrix
    keep_idx = discriminative_filters(results, outdir, filter_diff_thres,
                                      show_filters=show_filters)


    # ======================================================================== #
    # encode the sample and sample-phenotype for each cell
    sample_sizes = []
    per_cell_ids = []
    for i, x in enumerate(samples):
        sample_sizes.append(x.shape[0]) # sores the dimension of samples
        per_cell_ids.append(i * np.ones(x.shape[0])) #creates a list of arrrays. each array has the same len of the dimension of the sample in its same position and it is composed of only position i values
    
    
    # for each selected filter, plot the selected cell population
    x = np.vstack(samples) # concatenate and show the array of samples in a vertical array
    z = np.hstack(per_cell_ids)
    print(x.shape)
    print(f'x shape: {x.shape[0]}')
    # ======================================================================== #
    # Normalizaition + tSNE visualization

    if results['scaler'] is not None:
        print(f'scaler not present', False)
        x = results['scaler'].transform(x)

    logger.info("Computing t-SNE projection...")
    print("Computing t-SNE projection...")
    print(x.shape)
    print(x)
    tsne_idx = np.random.choice(x.shape[0], tsne_ncell) # samples the chosen number of id's(tsne_cells) from the first dataset of cells
    x_for_tsne = x[tsne_idx].copy() # extract che random sampled subset as a new array

    print(f'x_for_tsne: {tsne_idx}')
    # Calculate appropriate perplexity (this parameter determines how far the tsne looks for neighbours)
    n_samples = x_for_tsne.shape[0] 
    print(n_samples)
    perplexity = min(30, max(1, n_samples - 1)) if n_samples > 1 else 1
    print(f"Using perplexity: {perplexity} for {n_samples} samples")

    x_tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(x_for_tsne) #applyies the tsne algorithm
    #x_tsne = TSNE(n_components=2).fit_transform(x_for_tsne)

    vmin, vmax = np.zeros(x.shape[1]), np.zeros(x.shape[1]) #percentiles

    for seq_index in range(x.shape[1]):
        vmin[seq_index] = np.percentile(x[:, seq_index], 1)
        vmax[seq_index] = np.percentile(x[:, seq_index], 99)

    fig_path = os.path.join(outdir, '12.tsne_all_cells')
    plot_tsne_grid(x_tsne, x_for_tsne, fig_path, labels=labels, fig_size=(20, 20),
                   point_size=5)
    # ======================================================================== #
    # filters analysis 
    return_filters = []
    for i_filter in keep_idx:
        w = filters[i_filter, :nmark] # extract filters' weights 
        b = filters[i_filter, nmark] #extract filters' biases
        g = np.sum(w.reshape(1, -1) * x, axis=1) + b  # RELU activation. computes responses
        g = g * (g > 0)

        # skip a filter if it does not select any cell
        if np.max(g) <= 0:
            continue

        ecdf = sm.distributions.ECDF(g)
        gx = np.linspace(np.min(g), np.max(g))
        gy = ecdf(gx)

        plt.figure()
        sns.set_style('whitegrid')
        a = plt.step(gx, gy)
        t = filter_response_thres # threshold
        # set a threshold to the CDF gradient?
        if response_grad_cutoff is not None:
            by = np.array(a[0].get_ydata())[::-1]
            bx = np.array(a[0].get_xdata())[::-1]
            b_diff_idx = np.where(by[:-1] - by[1:] >= response_grad_cutoff)[0]
            if len(b_diff_idx) > 0:
                t = bx[b_diff_idx[0] + 1]
        plt.plot((t, t), (np.min(gy), 1.), 'r--')
        plt.xlabel('Cell filter response')
        plt.ylabel('Cumulative distribution function (CDF)')
        sns.despine()
        plt.savefig(os.path.join(outdir, '1.cdf_filter_%d.pdf' % i_filter), format='pdf')
        plt.clf()
        plt.close()


        condition = g > t # condition for extract data from the next arrays
        x1 = x[condition] 
        z1 = z[condition]
        g1 = g[condition]

        # skip a filter if it does not select any cell with the new cutoff threshold
        if x1.shape[0] == 0:
            continue

        # else add the filters to selected filters
        return_filters.append((i_filter, t))
        # t-SNE plots for characterizing the selected cell population
        fig_path = os.path.join(outdir, '2.tsne_cell_response_filter_%d.png' % i_filter)

        plot_2D_map(x_tsne, g[tsne_idx], fig_path, s=5)
        # overlay marker values on TSNE map for selected cells
        fig_path = os.path.join(outdir, '3.tsne_selected_cells_filter_%d' % i_filter)

        g_tsne = g[tsne_idx]
        x_pos = x_for_tsne[g_tsne > t]
        x_tsne_pos = x_tsne[g_tsne > t]
        print("DEBUG ncol:", x_pos, type(x_pos))
        plot_tsne_selection_grid(x_tsne_pos, x_pos, x_tsne, vmin, vmax,
                                 fig_path=fig_path, labels=labels, fig_size=(20, 20), s=5,
                                 suffix='png')

        
        suffix = 'filter_%d' % i_filter
        plot_selected_subset(x1, z1, x, labels, sample_sizes, phenotypes,
                             outdir, suffix, stat_test, log_yscale,
                             group_a, group_b, group_names, regression)
    logger.info("Done.")
    return return_filters


def discriminative_filters(results, outdir, filter_diff_thres, show_filters=True):
    """ Checks which filters are the most representative for the phenotypes"""
    
    mkdir_p(outdir)
    keep_idx = np.arange(results['selected_filters'].shape[0]) # keeps all filters and then iterativeli remove those which are not relevant

    # select the discriminative filters based on the validation set
    if 'filter_diff' in results:
        filter_diff = results['filter_diff']
        filter_diff[np.isnan(filter_diff)] = -1 # if filter is NaN, it is substituted with -1

        sorted_idx = np.argsort(filter_diff)[::-1] # decreasing order. Best first
        filter_diff = filter_diff[sorted_idx]
        keep_idx = [sorted_idx[0]] # keeps the best

        # retrives all filters that have a 'small' difference wrt the previous filter (so we compare filter i with filter i - 1)
        for i in range(0, len(filter_diff) - 1):
            if (filter_diff[i] - filter_diff[i + 1]) < filter_diff_thres * filter_diff[i]:
                keep_idx.append(sorted_idx[i + 1])
            else:
                break

        if show_filters:
            plt.figure()
            sns.set_style('whitegrid')
            plt.plot(np.arange(len(filter_diff)), filter_diff, '--')
            plt.xticks(np.arange(len(filter_diff)), ['filter %d' % i for i in sorted_idx],
                       rotation='vertical')
            plt.ylabel('average cell filter response difference between classes')
            sns.despine()
            plt.savefig(os.path.join(outdir, '4.filter_response_differences.pdf'), format='pdf')
            plt.clf()
            plt.close()

    # same code as before but using a different metric
    elif 'filter_tau' in results:
        filter_diff = results['filter_tau']
        filter_diff[np.isnan(filter_diff)] = -1

        sorted_idx = np.argsort(filter_diff)[::-1]
        filter_diff = filter_diff[sorted_idx]
        keep_idx = [sorted_idx[0]]
        for i in range(0, len(filter_diff) - 1):
            if (filter_diff[i] - filter_diff[i + 1]) < filter_diff_thres * filter_diff[i]:
                keep_idx.append(sorted_idx[i + 1])
            else:
                break
        if show_filters:
            plt.figure()
            sns.set_style('whitegrid')
            plt.plot(np.arange(len(filter_diff)), filter_diff, '--')
            plt.xticks(np.arange(len(filter_diff)), ['filter %d' % i for i in sorted_idx],
                       rotation='vertical')
            plt.ylabel('Kendalls tau')
            sns.despine()
            plt.savefig(os.path.join(outdir, '5.filter_response_differences.pdf'), format='pdf')
            plt.clf()
            plt.close()

    return list(keep_idx) # list of filter indexes to keep


def plot_filters(results, labels, outdir):
    mkdir_p(outdir)
    nmark = len(labels)
    # plot the filter weights of the best network
    w_best = results['w_best_net']
    idx_except_bias = np.hstack([np.arange(0, nmark), np.arange(nmark + 1, w_best.shape[1])])
    nc = w_best.shape[1] - (nmark + 1)
    labels_except_bias = labels + [f"out {i}" for i in range(nc)]
    w_best = w_best[:, idx_except_bias]
    fig_path = os.path.join(outdir, '6.best_net_weights.pdf')
    plot_nn_weights(w_best, labels_except_bias, fig_path, fig_size=(10, 10))
    # plot the filter clustering
    cl = results['clustering_result']
    cl_w = cl['w'][:, idx_except_bias]
    fig_path = os.path.join(outdir, '7.clustered_filter_weights.pdf')
    plot_nn_weights(cl_w, labels_except_bias, fig_path, row_linkage=cl['cluster_linkage'],
                    y_labels=cl['cluster_assignments'], fig_size=(10, 10))

    # plot the selected filters
    if results['selected_filters'] is not None:
        w = results['selected_filters'][:, idx_except_bias]
        fig_path = os.path.join(outdir, '8.consensus_filter_weights.pdf')
        plot_nn_weights(w, labels_except_bias, fig_path, fig_size=(10, 10))
        filters = results['selected_filters']
    else:
        sys.exit('Consensus filters were not found.')


def plot_nn_weights(w, x_labels, fig_path, row_linkage=None, y_labels=None, fig_size=(10, 3)):
    if y_labels is None:
        y_labels = np.arange(0, w.shape[0])

    if w.shape[0] > 1:
        plt.figure(figsize=fig_size)
        clmap = sns.clustermap(pd.DataFrame(w, columns=x_labels),
                               method='average', metric='cosine', row_linkage=row_linkage,
                               col_cluster=False, robust=True, yticklabels=y_labels, cmap="RdBu_r")
        plt.setp(clmap.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
        plt.setp(clmap.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
        clmap.cax.set_visible(True)
    else:
        plt.figure(figsize=(10, 1.5))
        sns.heatmap(pd.DataFrame(w, columns=x_labels), robust=True, yticklabels=y_labels)
        plt.tight_layout()
    plt.savefig(fig_path)
    plt.clf()
    plt.close()


def plot_marker_distribution(datalist, namelist, labels, grid_size, fig_path=None, letter_size=16,
                                 figsize=(9, 9), ks_list=None, colors=None, hist=False):
    nmark = len(labels)
    assert len(datalist) == len(namelist)
    g_i, g_j = grid_size
    sns.set_style('white')
    if colors is None:
        colors = sns.color_palette("Set1", n_colors=len(datalist), desat=.5)

    fig = plt.figure(figsize=figsize)
    grid = gridspec.GridSpec(g_i, g_j, wspace=0.1, hspace=.6)
    for i in range(g_i):
        for j in range(g_j):
            seq_index = g_j * i + j
            if seq_index < nmark:
                ax = fig.add_subplot(grid[i, j])
                if ks_list is not None:
                    ax.text(.5, 1.2, labels[seq_index], fontsize=letter_size, ha='center',
                            transform=ax.transAxes)
                    ax.text(.5, 1.02, ks_list[seq_index], fontsize=letter_size - 4, ha='center',
                            transform=ax.transAxes)
                else:
                    ax.text(.5, 1.1, labels[seq_index], fontsize=letter_size, ha='center',
                            transform=ax.transAxes)
                for i_name, (name, x) in enumerate(zip(namelist, datalist)):
                    lower = np.percentile(x[:, seq_index], 0.5)
                    upper = np.percentile(x[:, seq_index], 99.5)
                    if seq_index == nmark - 1:
                        if hist:
                            plt.hist(x[:, seq_index], np.linspace(lower, upper, 10),
                                     color=colors[i_name], label=name, alpha=.5, density=True)
                        else:
                            # FIXED: Use fill=True instead of shade=True
                            sns.kdeplot(x=x[:, seq_index], fill=True, color=colors[i_name], 
                                      label=name, clip=(lower, upper))
                    else:
                        if hist:
                            plt.hist(x[:, seq_index], np.linspace(lower, upper, 10),
                                     color=colors[i_name], alpha=.5, density=True)
                        else:
                            # FIXED: Use fill=True instead of shade=True
                            sns.kdeplot(x=x[:, seq_index], fill=True, color=colors[i_name], 
                                      clip=(lower, upper))
                ax.get_yaxis().set_ticks([])

    plt.legend(bbox_to_anchor=(1.5, 0.9))
    sns.despine()
    if fig_path is not None:
        plt.savefig(fig_path)
        plt.close()
    else:
        plt.show()


def plot_selected_subset(xc, zc, x, labels, sample_sizes, phenotypes, outdir, suffix,
                              stat_test=None, log_yscale=False,
                              group_a='group A', group_b='group B', group_names=None,
                              regression=False):
    ks_values = []
    nmark = x.shape[1]
    for j in range(nmark):
        ks = stats.ks_2samp(xc[:, j], x[:, j])
        ks_values.append(ks[0])

    sorted_idx = np.argsort(np.array(ks_values))[::-1]
    sorted_labels = [labels[i] for i in sorted_idx]
    sorted_ks = [('KS = %.2f' % ks_values[i]) for i in sorted_idx]

    fig_path = os.path.join(outdir, '9.selected_population_distribution_%s.pdf' % suffix)
    plot_marker_distribution([x[:, sorted_idx], xc[:, sorted_idx]], ['all cells', 'selected'],
                                 sorted_labels, grid_size=(4, 9), ks_list=sorted_ks, figsize=(24, 10),
                                 colors=['blue', 'red'], fig_path=fig_path, hist=False)

    if regression:
        frequencies = []
        for i, (n, y_i) in enumerate(zip(sample_sizes, phenotypes)):
            freq = 100. * np.sum(zc == i) / n
            frequencies.append(freq)

        _fig, ax = plt.subplots(figsize=(2.5, 2.5))
        plt.scatter(phenotypes, frequencies)
        if log_yscale:
            ax.set_yscale('log')
        plt.ylim(0, np.max(frequencies) + 1)
        plt.ylabel("selected population frequency (%)")
        plt.xlabel("response variable")
        sns.despine()
        plt.tight_layout()
        fig_path = os.path.join(outdir, 'selected_population_frequencies_%s.pdf' % suffix)
        plt.savefig(fig_path)
        plt.clf()
        plt.close()
    else:
        n_pheno = len(np.unique(phenotypes))
        frequencies = dict()
        for i, (n, y_i) in enumerate(zip(sample_sizes, phenotypes)):
            freq = 100. * np.sum(zc == i) / n
            assert freq <= 100
            if y_i in frequencies:
                frequencies[y_i].append(freq)
            else:
                frequencies[y_i] = [freq]
                
        if (n_pheno == 2) and (stat_test is not None):
            freq_a, freq_b = frequencies[0], frequencies[1]
            if stat_test == 'mannwhitneyu':
                _t, pval = stats.mannwhitneyu(freq_a, freq_b)
            elif stat_test == 'ttest':
                _t, pval = stats.ttest_ind(freq_a, freq_b)
            else:
                _t, pval = stats.ttest_ind(freq_a, freq_b)
        else:
            pval = None

        if group_names is None:
            if n_pheno == 2:
                group_names = [group_a, group_b]
            else:
                group_names = [f"group {y_i + 1}" for y_i in range(n_pheno)]
                
        box_grade = []
        for y_i, group_name in enumerate(group_names):
            box_grade.extend([group_name] * len(frequencies[y_i]))
        box_data = np.hstack([np.array(frequencies[y_i]) for y_i in range(n_pheno)])
        box = pd.DataFrame(columns=['group', 'selected population frequency (%)'])
        box['group'] = box_grade
        box['selected population frequency (%)'] = box_data

        _fig, ax = plt.subplots(figsize=(2.5, 2.5))
        # FIXED: Proper seaborn syntax to avoid palette warning
        ax = sns.boxplot(data=box, x="group", y="selected population frequency (%)", width=.5)
        ax = sns.swarmplot(data=box, x="group", y="selected population frequency (%)", color=".25")
        
        if stat_test is not None:
            ax.text(.45, 1.1, '%s pval = %.2e' % (stat_test, pval), horizontalalignment='center',
                    transform=ax.transAxes, size=8, weight='bold')
        if log_yscale:
            ax.set_yscale('log')
        plt.ylim(0, np.max(box_data) + 1)
        sns.despine()
        plt.tight_layout()
        fig_path = os.path.join(outdir, '10.selected_population_frequencies_%s.pdf' % suffix)
        plt.savefig(fig_path)
        plt.clf()
        plt.close()


def set_dbscan_eps(x, fig_path=None):
    nbrs = NearestNeighbors(n_neighbors=2, metric='l1').fit(x)
    distances, _indices = nbrs.kneighbors(x)
    if fig_path is not None:
        plt.figure()
        plt.hist(distances[:, 1], bins=20)
        plt.savefig(fig_path)
        plt.clf()
        plt.close()
    return np.percentile(distances, 90)


def make_biaxial(train_feat, valid_feat, test_feat, train_y, valid_y, test_y, figpath,
                 xlabel=None, ylabel=None, add_legend=False):
    # make the biaxial figure
    sns.set_style('white')
    palette = np.array(sns.color_palette("Set2", 3))
    plt.figure(figsize=(3, 3))
    ax = plt.subplot(aspect='equal')

    # the training samples
    ax.scatter(train_feat[:, 0], train_feat[:, 1], s=30, alpha=.5,
               c=palette[train_y], marker='>', edgecolors='face')

    # the validation samples
    ax.scatter(valid_feat[:, 0], valid_feat[:, 1], s=30, alpha=.5,
               c=palette[valid_y], marker=(5, 1), edgecolors='face')

    # the test samples
    ax.scatter(test_feat[:, 0], test_feat[:, 1], s=30, alpha=.5,
               c=palette[test_y], marker='o', edgecolors='face')

    # http://stackoverflow.com/questions/13303928/how-to-make-custom-legend-in-matplotlib
    a1 = plt.Line2D((0, 1), (0, 0), color=palette[0])
    a2 = plt.Line2D((0, 1), (0, 0), color=palette[1])
    a3 = plt.Line2D((0, 1), (0, 0), color=palette[2])

    a4 = plt.Line2D((0, 1), (0, 0), color='k', marker='>', linestyle='', markersize=8)
    a5 = plt.Line2D((0, 1), (0, 0), color='k', marker=(5, 1), linestyle='', markersize=8)
    a6 = plt.Line2D((0, 1), (0, 0), color='k', marker='o', linestyle='', markersize=8)

    # Create legend from custom artist/label lists
    if add_legend:
        first_legend = plt.legend([a1, a2, a3], ['healthy', 'CN', 'CBF'], fontsize=16, loc=1,
                                  fancybox=True)
        plt.gca().add_artist(first_legend)
        plt.legend([a4, a5, a6], ['train', 'valid', 'test'], fontsize=16, loc=4, fancybox=True)

    # plt.xlim(-2, 2)
    # plt.ylim(-2, 2)
    ax.set_aspect('equal', 'datalim')
    ax.margins(0.1)
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=12)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=12)

    plt.tight_layout()
    sns.despine()
    plt.savefig(figpath, format='eps')
    plt.clf()
    plt.close()

def plot_tsne_grid(z, x, fig_path, labels=None, fig_size=(9, 9), g_j=7,
                   suffix='png', point_size=.1):
    ncol = int(x.shape[1])
    # --- FIX: conversione sicura a int ---
    g_j = int(round(float(g_j)))  # Assicurati che sia un int Python puro
    g_i = int(np.ceil(float(ncol) / g_j))
    
    # --- DEBUG PRINT (obbligatorio ora) ---
    print("DEBUG g_i:", g_i, type(g_i))
    print("DEBUG g_j:", g_j, type(g_j))
    print("DEBUG ncol:", ncol, type(ncol))
    
    # --- ASSERT PER SICUREZZA ---
    assert isinstance(g_i, int) and g_i > 0, f"g_i must be positive int, got {g_i} ({type(g_i)})"
    assert isinstance(g_j, int) and g_j > 0, f"g_j must be positive int, got {g_j} ({type(g_j)})"
    
    print(np.issubdtype(z.dtype, np.number), f'z deve essere numerico\n')
    print(np.issubdtype(x.dtype, np.number), f'x deve essere numerico\n')  
    if labels is None:
        labels = [str(a) for a in range(ncol)]

    # Ensure we have valid, finite figure size
    fig_size = tuple(float(s) for s in fig_size)
    if not all(np.isfinite(fig_size)):
        fig_size = (9, 9)  # fallback to default
    
    sns.set_style('white')
    fig = plt.figure(figsize=fig_size)
    fig.clf()


    fig, axes = plt.subplots(g_i, g_j, figsize=fig_size)
    if g_i == 1:
        axes = axes.reshape(1, -1)
    grid = axes.flatten()
  
    for seq_index in range(ncol):
        if seq_index >= len(grid):
            break
            
        ax = grid[seq_index]
        ax.text(0, .92, labels[seq_index],
                horizontalalignment='center',
                transform=ax.transAxes, size=20, weight='bold')
        
        # Ensure percentile values are finite
        vmin = np.percentile(x[:, seq_index], 1)
        vmax = np.percentile(x[:, seq_index], 99)
        
        if not np.isfinite(vmin):
            vmin = np.nanmin(x[:, seq_index])
        if not np.isfinite(vmax):
            vmax = np.nanmax(x[:, seq_index])
        if vmin == vmax:
            vmax = vmin + 1  # Avoid identical min/max
            
        # Create scatter plot
        im = ax.scatter(z[:, 0], z[:, 1], s=point_size, marker='o', c=x[:, seq_index],
                        cmap=cm.jet, alpha=0.5, edgecolors='face', vmin=vmin, vmax=vmax)
        
        # Add colorbar if using ImageGrid, otherwise skip
        try:
            if hasattr(ax, 'cax'):
                ax.cax.colorbar(im)
        except:
            pass  # Skip colorbar if it fails
            
        clean_axis(ax)
        ax.grid(False)
    
    # Hide unused subplots if using regular subplots
    if 'axes' in locals():
        for seq_index in range(ncol, len(grid)):
            grid[seq_index].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('.'.join([fig_path, suffix]), format=suffix, bbox_inches='tight', dpi=150)
    plt.clf()
    plt.close()

'''
def plot_tsne_selection_grid(z_pos, x_pos, z_neg, vmin, vmax, fig_path,
                             labels=None, fig_size=(9, 9), g_j=7, s=.5, suffix='png'):
    ncol = x_pos.shape[1]
    ncol = int(x.shape[1])
    g_j = int(g_j)
    g_i = int(ncol / g_j) if (ncol % g_j == 0) else int(ncol / g_j) + 1
    print("DEBUG g_i:", g_i, type(g_i))
    print("DEBUG g_j:", g_j, type(g_j))
    print("DEBUG ncol:", ncol, type(ncol))
    if labels is None:
        labels = [str(a) for a in np.range(ncol)]

    fig = plt.figure(figsize=fig_size)
    fig.clf()
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(g_i, g_j),
                     ngrids=None if ncol % g_j == 0 else ncol,
                     aspect=True,
                     direction="row",
                     axes_pad=(0.15, 0.5),
                     #add_all=True,
                     label_mode="1",
                     share_all=True,
                     cbar_location="top",
                     cbar_mode="each",
                     cbar_size="8%",
                     cbar_pad="5%",
                     )
    for seq_index in range(ncol):
        ax = grid[seq_index]
        ax.text(0, .92, labels[seq_index],
                horizontalalignment='center',
                transform=ax.transAxes, size=20, weight='bold')
        a = x_pos[:, seq_index]
        ax.scatter(z_neg[:, 0], z_neg[:, 1], s=s, marker='o', c='lightgray',
                   alpha=0.5, edgecolors='face')
        im = ax.scatter(z_pos[:, 0], z_pos[:, 1], s=s, marker='o', c=a, cmap=cm.jet,
                        edgecolors='face', vmin=vmin[seq_index], vmax=vmax[seq_index])
        ax.cax.colorbar(im)
        clean_axis(ax)
        ax.grid(False)
    plt.savefig('.'.join([fig_path, suffix]), format=suffix)
    plt.clf()
    plt.close()
'''

def plot_tsne_selection_grid(z_pos, x_pos, z_neg, vmin, vmax, fig_path,
                             labels=None, fig_size=(9, 9), g_j=7, s=.5, suffix='png'):
    ncol = int(x_pos.shape[1])  # Fix: use x_pos instead of undefined x
    g_j = int(g_j)
    g_i = int(np.ceil(ncol / g_j))  # Use np.ceil for proper rounding
    print(np.issubdtype(x_pos.dtype, np.number), f'\n x_pos deve essere numerico\n')
    print("DEBUG g_i:", g_i, type(g_i))
    print("DEBUG g_j:", g_j, type(g_j))
    print("DEBUG ncol:", ncol, type(ncol))
    print(np.isfinite([g_j, g_i]))
    if labels is None:
        labels = [str(a) for a in range(ncol)]  # Fix: use range instead of np.range

    # Ensure we have valid, finite figure size
    fig_size = tuple(float(s) for s in fig_size)
    if not all(np.isfinite(fig_size)):
        fig_size = (9, 9)  # fallback to default

    sns.set_style('white')
    fig = plt.figure(figsize=fig_size)
    fig.clf()
    
    assert np.all(np.isfinite(x_pos)), "x_pos contiene NaN o inf"
    assert np.issubdtype(z_pos.dtype, np.number), "\nz_pos deve essere numerico\n"
    assert np.all(np.isfinite(z_pos)), "z_pos contiene NaN o inf"
    
    try:
        # Create ImageGrid with explicit type conversion and validation
        grid = ImageGrid(fig,
                         rect = (1.0, 1.0, 1.0, 1.0),  # left=1, bottom=1, width=1, height=1
                         nrows_ncols=(int(g_i), int(g_j)),
                         ngrids=int(ncol),  # Always specify exact number needed
                         aspect=True,
                         direction="row",
                         axes_pad=(float(0.15), float(0.5)),
                         label_mode="1",
                         share_all=True,
                         cbar_location="top",
                         cbar_mode="each",
                         cbar_size="8%",
                         cbar_pad="5%",
                         )
    
    except Exception as e:
        print(f"ImageGrid creation failed: {e}")
        # Fallback to regular subplots
        fig, axes = plt.subplots(g_i, g_j, figsize=fig_size)
        if g_i == 1:
            axes = axes.reshape(1, -1)
        grid = axes.flatten()
    
    for seq_index in range(ncol):
        if seq_index >= len(grid):
            break
            
        ax = grid[seq_index]
        ax.text(0, .92, labels[seq_index],
                horizontalalignment='center',
                transform=ax.transAxes, size=20, weight='bold')
        
        a = x_pos[:, seq_index]
        
        # Validate vmin/vmax values
        vmin_val = vmin[seq_index] if np.isfinite(vmin[seq_index]) else np.nanmin(a)
        vmax_val = vmax[seq_index] if np.isfinite(vmax[seq_index]) else np.nanmax(a)
        if vmin_val == vmax_val:
            vmax_val = vmin_val + 1  # Avoid identical min/max
        
        # Plot negative (background) points
        ax.scatter(z_neg[:, 0], z_neg[:, 1], s=s, marker='o', c='lightgray',
                   alpha=0.5, edgecolors='face')
        
        # Plot positive (selected) points
        im = ax.scatter(z_pos[:, 0], z_pos[:, 1], s=s, marker='o', c=a, cmap=cm.jet,
                        edgecolors='face', vmin=vmin_val, vmax=vmax_val)
        
        # Add colorbar if using ImageGrid, otherwise skip
        try:
            if hasattr(ax, 'cax'):
                ax.cax.colorbar(im)
        except:
            pass  # Skip colorbar if it fails
            
        clean_axis(ax)
        ax.grid(False)
    
    # Hide unused subplots if using regular subplots
    if 'axes' in locals():
        for seq_index in range(ncol, len(grid)):
            grid[seq_index].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('.'.join([fig_path, suffix]), format=suffix, bbox_inches='tight', dpi=150)
    plt.clf()
    plt.close()

def plot_2D_map(z, feat, fig_path, s=2, plot_contours=False):
    sns.set_style('white')
    _fig, ax = plt.subplots(figsize=(5, 5))
    if plot_contours:
        sns.kdeplot(z[:, 0], z[:, 1], colors='lightgray', cmap=None, linewidths=0.5)

    if issubclass(feat.dtype.type, np.integer):
        c = np.squeeze(feat)
        colors = sns.color_palette("Set2", len(np.unique(c)))
        for i in np.unique(c):
            plt.scatter(z[c == i, 0], z[c == i, 1], s=s, marker='o', c=colors[i],
                        edgecolors='face', label=str(i))
    else:
        im = ax.scatter(z[:, 0], z[:, 1], s=s, marker='o', c=feat, vmin=np.percentile(feat, 1),
                        cmap=cm.jet, alpha=0.5, edgecolors='face', vmax=np.percentile(feat, 99))
        # magic parameters from
        # http://stackoverflow.com/questions/16702479/matplotlib-colorbar-placement-and-size
        plt.colorbar(im, fraction=0.046, pad=0.04)
    clean_axis(ax)
    ax.grid(False)
    sns.despine()
    if issubclass(feat.dtype.type, np.integer):
        plt.legend(loc="upper left", markerscale=5., scatterpoints=1, fontsize=10)
    plt.xlabel('tSNE dimension 1', fontsize=20)
    plt.ylabel('tSNE dimension 2', fontsize=20)
    plt.savefig(fig_path, format=fig_path.split('.')[-1])
    plt.clf()
    plt.close()


def plot_tsne_per_sample(z_list, data_labels, fig_dir, fig_size=(9, 9),
                         density=True, scatter=True, colors=None, pref=''):
    if colors is None:
        colors = sns.color_palette("husl", len(z_list))
    _fig, ax = plt.subplots(figsize=fig_size)
    for i, z in enumerate(z_list):
        ax.scatter(z[:, 0], z[:, 1], s=1, marker='o', c=colors[i],
                   alpha=0.5, edgecolors='face', label=data_labels[i])
    clean_axis(ax)
    ax.grid(False)

    plt.legend(loc="upper left", markerscale=20., scatterpoints=1, fontsize=10)
    plt.xlabel('t-SNE dimension 1', fontsize=20)
    plt.ylabel('t-SNE dimension 2', fontsize=20)
    plt.savefig(os.path.join(fig_dir, pref + '_tsne_all_samples.png'), format='png')
    plt.clf()
    plt.close()

    # density plots
    if density:
        for i, z in enumerate(z_list):
            _fig = plt.figure(figsize=fig_size)
            sns.kdeplot(z[:, 0], z[:, 1], n_levels=30, shade=True)
            plt.title(data_labels[i])
            plt.savefig(os.path.join(fig_dir, pref + 'tsne_density_%d.png' % i), format='png')
            plt.clf()
            plt.close()

    if scatter:
        for i, z in enumerate(z_list):
            _fig = plt.figure(figsize=fig_size)
            plt.scatter(z[:, 0], z[:, 1], s=1, marker='o', c=colors[i],
                        alpha=0.5, edgecolors='face')
            plt.title(data_labels[i])
            plt.savefig(os.path.join(fig_dir, pref + 'tsne_scatterplot_%d.png' % i), format='png')
            plt.clf()
            plt.close()


def clean_axis(ax):
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
