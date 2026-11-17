# Configuration guide

> The values in the example `config.yml` are set for a quick test run
> (e.g. 1 epoch, 1 trial). Increase them for real experiments.

## Environment
| Parameter | Description |
|---|---|
| `starting_seed` | Base seed for reproducibility |
| `save_memory` | [`MemorySave`](https://github.com/enricodidoli-source/memory-save) for RAM management on limited hardware |
| `colab` | Set to `True` when running on Google Colab |
| `n_patients` | Number of patients to use (`null` = all). For debugging only |
| `label_column` | Column containing the label |
| `remove_columns` | Columns dropped when the dataset is loaded |
| `extension` | File pattern used to find the samples (currently only `*.csv` is supported) |
| `labels_level` | Label granularity (`sample` or `cell`). Only used for plotting: if cell-level labels are available, plots also show the percentages; otherwise only the sample label is shown |

## Model
| Parameter | Description |
|---|---|
| `eval_metric` | Metric used for early stopping (e.g. `f1_score`, `accuracy`) |

## Bayesian Optimization (tuning of augmentation parameters and batch size)
| Parameter | Description |
|---|---|
| `bayesian_comb` | Number of BO iterations, i.e. number of parameter combinations tested |
| `BO_tuning_epochs` | Epochs for each BO trial |
| `BO_tuning_nruns` | Number of random hyperparameter configurations tested per BO trial (ignored if `BO_grid: true`) |
| `BO_grid` | If `True`, uses grid search over the hyperparameters instead of random search |
| `BO_trials` | Number of weight initializations per configuration |
| `min_ncells`, `max_ncells`, `ncells_step` | Search range for cells per subset |
| `min_nsubs`, `max_nsubs`, `nsub_step` | Search range for number of subsets |
| `min_bs`, `max_bs`, `bs_step` | Search range for batch size |

## Training
| Parameter | Description |
|---|---|
| `tuning_epochs` | Epochs for the final training |
| `tuning_nruns` | Number of random hyperparameter configurations tested (ignored if `grid: true`) |
| `grid` | If `True`, uses grid search over the hyperparameters instead of random search |
| `trials` | Number of weight initializations per configuration |
| `perform_tuning` | If `False`, uses the fixed values below instead of the BO results |
| `ncells`, `nsubs`, `bs` | Fixed augmentation and batch-size values |

## Hyperparameters
Candidate values: with `grid: true` all combinations are tested,
otherwise configurations are sampled randomly from the following lists.

| Parameter | Description |
|---|---|
| `nfilter` | Candidate numbers of convolutional filters |
| `maxpool_p` | Candidate percentages of top-activated cells pooled per filter |
| `learning_r` | Candidate learning rates |

## Clinical pipeline

1. Copy the folder `experiments/experiment_clinical/`. It contains two config files: `train_clinical_config.yml` and `test_clinical_config.yml`.

2. Open `train_clinical_config.yml` and set:
   - the path to your training samples, the file extension and the name of the column containing the binary labels;
   - `sequence`: leave the default if you followed the [data guide](../Data/Data_README.md), otherwise adapt the function that match yout sample names;
   - optionally, `perform_tuning: False` to skip Bayesian Optimization, and set `ncells`, `nsubs` and `bs` manually or to previously found values.

3. Open `test_clinical_config.yml` and set the path to the samples to classify. This pipeline is meant for samples that are not manually gated or labelled. To test it on labelled samples, add the label column to `remove_columns`. This must be a list: `remove_columns: ['column_name']`; multiple columns can be added to the list.

4. Open `notebooks/clinical/clinical_training_pipeline.ipynb`, set `whole_exp` to the name of the folder from step 1 (e.g. `whole_exp = 'clinical'`) and `fixed_path` to the path of your clone of this repository. Restart the kernel and run the whole notebook.

5. Once training is finished (or if you already have trained models), open `notebooks/clinical/clinical_classification_pipeline.ipynb`, set `whole_exp` and `fixed_path` as above, restart the kernel and run the whole notebook.