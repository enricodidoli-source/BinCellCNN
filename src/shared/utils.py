


'''
show_blast_distribution_perc, save_test_temp, save_element, load_element, flatten, remove_labels
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import sys


def save_element(dir_path, name, element):
            os.makedirs(dir_path, exist_ok=True)
            with open(os.path.join(dir_path, f'{name}.pkl'), 'wb') as f:
                pickle.dump(element, f)
                
def load_element(dir_path, name):
            with open(os.path.join(dir_path, f'{name}.pkl'), 'rb') as f:
                element = pickle.load(f)
            return element




def flatten(nested):
    """
    Recursively flattens a nested list or tuple into a flat list.
    """
    
    if nested is None:
        return []
    if not isinstance(nested, (list, tuple)):
        return [nested]
    nested = list(nested)
    result = []
    for item in nested:
        result.extend(flatten(item))
    return result


    

def remove_from_cache(files: list):
    for file in files:
        if file in sys.modules:
            del sys.modules[file]
            print(f"{file} removed from cache")
        else:
            print(f"{file} not found in cache")
