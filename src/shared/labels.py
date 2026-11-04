

'''
remove_labels, retrieve_labels
'''


import pandas as pd
import numpy as np

from src.shared.utils import flatten



def remove_labels(new_test_datasets, label_column = 'IsBlast'):
    """
    Removes the 'IsBlast' column from each dataset, if present.
    """
    new_no_label_test_datasets = []
    for dataset in new_test_datasets:
        if label_column in dataset.columns:
            dataset = dataset.drop(columns = [label_column])

        new_no_label_test_datasets.append(dataset)
    return new_no_label_test_datasets







def retrieve_labels(datasets_extracted, remove = False, flat = False, label_column = 'IsBlast'):
    """
    Extracts binary labels from the 'IsBlast' column of each dataset.
    Optionally removes the label column or flattens the output.
    """
    per_donor_original_datasets = []
    per_donor_original_y = []

    for donor in datasets_extracted:
        donor_datasets = []
        donor_ys = []
        for dataset in donor:
            if (dataset[label_column] == 1).sum() > 0:
                donor_ys.append(1)
            else:
                donor_ys.append(0)

            if remove:
                dataset = dataset.drop(columns = [label_column])

            donor_datasets.append(dataset)

        per_donor_original_datasets.append(donor_datasets)
        per_donor_original_y.append(donor_ys)


    if flat:
        per_donor_original_datasets = flatten(per_donor_original_datasets)
        per_donor_original_y = flatten(per_donor_original_y)

    return per_donor_original_datasets, per_donor_original_y

""" ================================================================="""