from src.cell_level.cell_level_utils import clear_RAM
import numpy as np
import pandas as pd

def relu(x):
    return x * (x > 0)


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

# ====================================

def keras_param_vector(params):
    """Extracts weights from a model"""

    W = np.squeeze(params[0])
    b = params[1]
    W_out = params[2]
    # store the (convolutional weights + biases + output weights) per filter

    W_tot = np.hstack([W.T, b.reshape(-1, 1), W_out])

    clear_RAM()
    return W_tot



def extract_filters_from_net(net):
    """Extracts and transposes filters from the network architecture."""
    return net[0].transpose()



# =====================================

def get_full_network_prob(params, data, scaler=None):
    """Calculates network probabilities for given data."""
    if scaler is not None:
        data = scaler.transform(data)

    W = np.squeeze(params[0])
    b = params[1]
    W_out = params[2]
    # store the (convolutional weights + biases + output weights) per filter
    b_out = params[3] if len(params) > 3 else 0.0

    hidden_z = np.dot(data, W) + b
    hidden_a = relu(hidden_z)  # Attivazione ReLU

    logits = np.dot(hidden_a, W_out) + b_out
    probs = 1 / (1 + np.exp(-logits))
    positive_probs = probs[:, 1]

    clear_RAM()
    return np.hstack([positive_probs])



def get_filter_prob(filter_weights, data, scaler=None):

    """Calculates probabilities for a given filter using a sigmoid activation."""
    if scaler is not None:
        data = scaler.transform(data)

    hidden_z = np.dot(data, filter_weights)

    probs = 1 / (1 + np.exp(-hidden_z))

    positive_probs = probs[:, 0] if probs.ndim > 1 else probs.ravel()
    clear_RAM()
    return positive_probs



def get_cells_prob(filter_w, data, scaler=None, filter_response_thres=0):
    """Retrieves continuous ReLU-activated probabilities for cells."""

    nmark = data.shape[1]
    if scaler is not None:
        data = scaler.transform(data)

    w, b = filter_w[:nmark], filter_w[nmark]
    g = np.dot(data, w) + b
    g = relu(g).reshape(-1, 1)

    clear_RAM()
    return np.hstack([g])



def get_selected_cells(filter_w, data, scaler=None, filter_response_thres=0,
                       export_continuous=False):
    """Calculates cell responses based on a specific filter's weights."""

    nmark = data.shape[1]
    if scaler is not None:
        data = scaler.transform(data)

    w, b = filter_w[:nmark], filter_w[nmark]

    g = np.dot(data, w) + b

    if export_continuous:
        g = relu(g).reshape(-1, 1)
        g_thres = (g > filter_response_thres).reshape(-1, 1)

        clear_RAM()
        return np.hstack([g, g_thres])
    else:
        clear_RAM()
        return (g > filter_response_thres).astype(int)



