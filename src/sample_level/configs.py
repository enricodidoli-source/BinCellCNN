from tensorflow import keras

def get_eval_metric(name):
    if name == 'f1_score':
        return keras.metrics.F1Score(average="macro", name="f1_score")
    return name  # es. 'accuracy'

def get_hyper(config):
    return (config['nfilter'], config['maxpool_p'], config['learning_r'])

def get_param_list_dict(config):
    return {
        'ncells': (config['min_ncells'], config['max_ncells'], config['ncells_step']),
        'nsubs': (config['min_nsubs'], config['max_nsubs'], config['nsub_step']),
        'bs': (config['min_bs'], config['max_bs'], config['bs_step']),
    }

def build_tuning_config(config):
    return {
        'hyper': get_hyper(config),
        'grid': config['BO_grid'],
        'tuning_epochs': config['BO_tuning_epochs'],
        'tuning_nruns': config['BO_tuning_nruns'],
        'trials': config['BO_trials'],
        'bayesian_comb': config['bayesian_comb'],
        'eval_metric': get_eval_metric(config['eval_metric']),
    }

def build_training_config(config, ncells, nsubs, bs):
    return {
        'hyper': get_hyper(config),
        'grid': config['grid'],
        'tuning_epochs': config['tuning_epochs'],
        'tuning_nruns': config['tuning_nruns'],
        'trials': config['trials'],
        'ncells': ncells,
        'nsubs': nsubs,
        'bs': int(bs),
        'eval_metric': get_eval_metric(config['eval_metric']),
    }