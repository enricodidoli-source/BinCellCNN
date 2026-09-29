import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

# Moduli per il clustering e le distanze
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

# Import dai tuoi moduli interni
from src.cell_level.cell_level_utils import clear_RAM

from src.cell_level.networks_ops import extract_filters_from_net


def cluster_filters(all_best_filters, show_plot = True):
    """Clusters filters based on cosine similarity and plots a dendrogram."""
    print('=========== Clustering Filters ==============')

    correlations = cosine_similarity(np.array(all_best_filters))
    dissimilarity = 1 - abs(correlations)
    np.fill_diagonal(dissimilarity, 0)

    Z = linkage(squareform(dissimilarity), 'complete')

    if show_plot:
        fig = plt.figure(figsize=(12, 5))
        dendrogram(Z, orientation='top', leaf_rotation=90)
        plt.title("Filter Clustering Dendrogram")
        display(fig)
        plt.close(fig)

    clear_RAM()
    return Z



def sort_clustered_filters(all_best_filters, all_best_scalers, filter_f1_list, best_thr_list, distance_threshold):
    """Selects the best performing filter from each hierarchical cluster."""
    Z = cluster_filters(all_best_filters)

    # Cut the dendrogram at the provided threshold
    cluster_labels = fcluster(Z, t=distance_threshold, criterion='distance')

    sample_filters = []
    unique_clusters = np.unique(cluster_labels)

    for id_cluster in unique_clusters:

        indici_nel_cluster = np.where(cluster_labels == id_cluster)[0]
        f1_del_cluster = [filter_f1_list[i] for i in indici_nel_cluster]
        best_idx = indici_nel_cluster[np.argmax(f1_del_cluster)]

        sample_filters.append(best_idx)

    # Convert lists to arrays for vectorized indexing
    best_f1_from_clusters =  np.array(filter_f1_list)[sample_filters]
    best_filters_from_clusters =  np.array(all_best_filters)[sample_filters]
    best_scalers_from_clusters = np.array(all_best_scalers)[sample_filters]
    best_thr_from_clusters =  np.array(best_thr_list)[sample_filters]

    clear_RAM()
    return best_f1_from_clusters, best_filters_from_clusters, best_scalers_from_clusters, best_thr_from_clusters




def select_relevant_filters(best_f1_from_clusters, best_filters_from_clusters, best_scalers_from_clusters, best_thr_from_clusters, filter_selection_thr = 0.6, one_per_model = False):
    # Filter based on the F1 performance threshold (filter_selection_thr)
    choising_condition = best_f1_from_clusters > filter_selection_thr

    selected_filters = best_filters_from_clusters[choising_condition]
    selected_filters = [np.array(filter) for filter in selected_filters]

    selected_scalers = best_scalers_from_clusters[choising_condition].tolist()

    # Create a backup of the absolute best filter overall, in case none passed the threshold
    best_f1_idx = np.argmax(best_f1_from_clusters)
    best_filter = best_filters_from_clusters[best_f1_idx]
    best_scaler = best_scalers_from_clusters[best_f1_idx]

    filter_backup = [best_f1_from_clusters[best_f1_idx], best_filter, best_scaler]

    # Fallback logic if no filters met the threshold
    if len(selected_filters) == 0 and one_per_model:
        print(f'No filter scored F1 > {filter_selection_thr}. Selected the best one (F1 = {filter_backup[0]:.4f})')
        selected_filters = [best_filter]
        selected_scalers = [best_scaler]

    clear_RAM()
    return selected_filters, selected_scalers, filter_backup




def select_clustered_filters(metadata, DIST_THRESHOLD, FILTER_SEL_THRESHOLD):
    
        model_filters = metadata['filter_weights']
        model_scalers = metadata['scaler']
        model_best_thr = metadata['best_thr']
        model_max_f1 = metadata['max_f1']
        cv = metadata['cv_fold']
        m = metadata['model_ID']
        LOPOCV_fold_idx = metadata['lopo_fold']

        best_f1_from_clusters, best_filters_from_clusters, best_scalers_from_clusters, best_thr_from_clusters = sort_clustered_filters(model_filters, model_scalers, model_max_f1, model_best_thr, DIST_THRESHOLD)


        selected_filters, selected_scalers, filter_backup = select_relevant_filters(best_f1_from_clusters, best_filters_from_clusters, best_scalers_from_clusters,
                                                                                       best_thr_from_clusters, filter_selection_thr = FILTER_SEL_THRESHOLD, one_per_model=False)


        #selected_filters, selected_scalers, filter_backup = select_relevant_filters(model_filters, model_scalers, model_max_f1, model_best_thr,
        #                                                                                DIST_THRESHOLD, filter_selection_thr = FILTER_SEL_THRESHOLD, one_per_model=False)

        selected_metadata = { 'lopo_fold': LOPOCV_fold_idx,
                                          'cv_fold': cv,
                                          'model_ID': m,
                                          'filter_weights': selected_filters,
                                          'scaler': selected_scalers,
                                          'backup': filter_backup
                                      }
        
        return selected_metadata



def rebuild_metadata(all_filter_metadata_df, cv,  DIST_THRESHOLD, FILTER_SEL_THRESHOLD):
    temp_df = all_filter_metadata_df[all_filter_metadata_df['cv_fold'] == cv]
    selected_filters_metadata = []

    for metadata in temp_df.to_dict(orient='records'):
        selected_metadata = select_clustered_filters(metadata, DIST_THRESHOLD, FILTER_SEL_THRESHOLD)
        selected_filters_metadata.append(selected_metadata)
    
    clear_RAM()
    return selected_filters_metadata



# ==================================


def retrieve_filters_scalers(fold_loaded_models_lists):
    """Retrieves all filters and their corresponding scalers from model results."""
    all_best_filters = []
    all_scalers = []
    for m, model in enumerate(fold_loaded_models_lists): # several seeds (our case 1 seed only)

        model_scalers_list = []
        model_filters_list = []

        for net in model.results['best_3_nets']: # enter in the model results

            filters = extract_filters_from_net(net)

            for filt in filters:
                model_filters_list.append(filt.squeeze())  # save each filter
                model_scalers_list.append(model.results['scaler'])

        all_best_filters.append(model_filters_list)
        all_scalers.append(model_scalers_list)

    clear_RAM()
    return all_best_filters, all_scalers



def extract_filter_from_backup(selected_backups_across_folds, selected_filters_across_folds, selected_scalers_across_folds):
    """Finds the best performing filter across all backups if none were selected naturally."""
    max_backup_f1 = 0
    print('no filters selected. Extract the best one in the backup')

    for cv_b, cv_backups in enumerate(selected_backups_across_folds):
        for m_b, model_backups in enumerate(cv_backups):

            f1, filter, scaler = model_backups
            if f1 > max_backup_f1:
                max_backup_f1 = f1
                best_filter = filter
                best_scaler = scaler
                best_cv = cv_b
                best_model = m_b

    selected_filters_across_folds[best_cv][best_model] = [best_filter]
    selected_scalers_across_folds[best_cv][best_model] = [best_scaler]

    clear_RAM()
    return selected_filters_across_folds, selected_scalers_across_folds





def metadata_extract_filter_from_backup(filter_metadata_df):
    """Extracts the best performing filter from backups across all folds."""
    cv_folds = set(filter_metadata_df['cv_fold'].to_list())

    max_backup_f1 = -1
    print('no filters selected. Extract the best one in the backup')

    for cv_b in range(len(cv_folds)):
        cv_backups = filter_metadata_df[filter_metadata_df['cv_fold'] == cv_b]['backup']

        for m_b, model_backups in enumerate(cv_backups):
            f1, filter, scaler = model_backups

            if f1 > max_backup_f1:
                max_backup_f1 = f1

                best_filter = [filter]
                best_scaler = [scaler]
                best_cv = cv_b
                best_model = m_b
                best_model_backup = model_backups

    clear_RAM()
    return best_filter, best_scaler, best_cv, best_model, best_model_backup




def apply_backup_if_needed(filter_metadata_df):
    """Checks if any filters were selected; if not, forcefully applies the best backup."""
    filter_presence = any(len(f) > 0 for f in filter_metadata_df['filter_weights']) # checks the presence of any non empty list

    if not filter_presence:
        best_filter, best_scaler, best_cv, best_model, best_model_backup = metadata_extract_filter_from_backup(filter_metadata_df)

        # set condition we are looking for, so specific cv_fold and specific model_ID
        mask = (filter_metadata_df['cv_fold'] == best_cv) & (filter_metadata_df['model_ID'] == best_model)

        target_row_idx = filter_metadata_df[mask].index[0] #index returns a list of indixes. in our case, given cv eand model id , it is unique

        filter_metadata_df.at[target_row_idx, 'filter_weights'] = best_filter
        filter_metadata_df.at[target_row_idx, 'scaler'] = best_scaler

    clear_RAM()
    return filter_metadata_df