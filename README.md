# Detecting Measurable Residual Disease in B-cell Acute Lymphoblastic Leukaemia using Convolutional Neural Networks

> ⚠️ **Project Status: Work in Progress (WIP)**
> 
> Welcome! This repository contains the code developed for my Master's Thesis, which is currently undergoing a major expansion. 
> 
> To keep the development organized, this repository is split into two main branches:
> * **[`thesis-version` branch](https://github.com/enricodidoli-source/BinCellCNN/tree/thesis-version)**: Contains the exact, finalized code presented for my graduation.
> * **`main` branch (Current)**: Contains the modularized and ongoing experimental work. 
>
> **[Click here to see all code differences between the Thesis version and the Current version](https://github.com/enricodidoli-source/BinCellCNN/compare/thesis-version...main)**.

---

## Overview

This repository contains the code developed for my Master's thesis on **Measurable Residual Disease (MRD) detection** in B-cell Acute Lymphoblastic Leukaemia (B-ALL) using the **BinCellCNN** neural network architecture applied to multiparameter flow cytometry (MFC) data.

**BinCellCNN** is a specialized version of **CellCNN**, model developed by eiriniar. This extension is intended to work as a rare-event detection binary classifier (that's why "Bin" in the model name) rather then subpopolation detector.

Main modifications:
- Mandatory passing of validation set
- Elaboration of data as groups (avoids data leakage in case of correlation within samples doe to same patient)
- Possibility to use different metrics for Early stopping (CellCNN default metric is Accuracy)
- Grid search enabling
- Prediction of an entire sample instead of a subset
- Random reproducibility
- Allows the use of RAM management for limited hardwares

In addition to these modfications, the repository proposes a Research pipeline based on the Leave-One-Group-Out Cross-Validation approach that explores:
- Bayesian Optimization employed to optimize the choice of data augmentation parameters and batch-size.
- Training set up using the tuned parameters and increased number of epochs iterations implementing k-fold CV and several weights initializations for robust results.
- Classification threshold tuning using validation folds.
- Prediction and Classification of the held-out sets showing results in a comprehensive boxplot to analyse model prediciton confidence.

Finally, a pipeline usable in real-world clinical (or not) applications:
- Tuning and Training pipeline notebook with model weights saving
- Prediction and Classification pipeline notebook that restores the training weigths to return a classification label to a never seen sample.

A structured usable guide is going to be released to prepare the set-up for real applications. Stay Tuned!

---

## Roadmap & Current Implementations

- **Code Cleanup:** Ongoing removal of remaining debug, print statements and codebase readability refinement.
- **Uncertainty Quantification using Bayesian Statistics**
- **Cell-Level Analysis:** Implementation and integration of the full cell-level classification pipeline.

---

## Repository Structure
 
```
BinCellCNN/
│
├── notebooks/                # Jupyter notebooks per esperimenti e risultati
│   ├── clinical/
│   ├── research/
│   ├── results_elaboration_and_plotting/
│   ├── training_classification/
│   └── refactoring_single_cell_detection.ipynb
│
├── src/                      # Codice sorgente
│   ├── BinCellCNN/           # Moduli Python (.py) per il modello BinCellCNN (basati su https://github.com/eiriniar/CellCnn)
│   ├── cell_level/           # Moduli Python (.py) per la classificazione a livello di singola cellula
│   ├── sample_level/         # Moduli Python (.py) per la classificazione a livello di campione
│   ├── shared/               # Moduli Python (.py) condivisi tra sample-level e single-cell level
│   ├── anything_else.py
│   ├── ball_dataset.py       # Modulo con le funzioni dipendenti dal dataset
│   └── clinical.py           # Modulo con le funzioni per l'applicazione clinica
│
├── Data/		      # Contains artificial datasets and labels. Necessary for practical implementation guide (see below)
│   ├── art_new_samples/      
│   │	└── samples_labels/    
│   └── art_train_samples/
│   	└── samples_labels/  
│
├── experiments/	      # Contains clinical and research base folders and their respective config.yml file. Necessary for practical implementation guide (see below)
│   ├── experiment_End_to_End_pipeline/ # any results for the Research pipelines are going to be saved here
│   │	└── config.yml
│   └── experiment_clinical/
│   	├── train_clinical_config.yml
│   	└── test_clinical_config.yml
│
├── .gitignore
├── README.md
└── requirements.txt
```

---
 
## Getting Started
 
### 1. Clone the repository
 
```bash
git clone https://github.com/enricodidoli-source/BinCellCNN.git
cd BinCellCNN
```
 
### 2. Install dependencies
 
```bash
pip install -r requirements.txt
```
> **Note:** To run these notebooks you need to preprocess your dataset and divide them into specific folders. A practical guide is going to be released.

### 3. Prepare your data

Follow the [data preparation guide](Data/Data_README.md).

### 4. Run the pipelines

- **Research pipeline** (LOPOCV): see the [research guide](experiments/Exp_README.md#research-pipeline)
- **Clinical pipeline** (training + classification of new samples): see the [clinical guide](experiments/Exp_README.md#clinical-pipeline)

> **Note:** Experiments were run on Google Colab with a CPU runtime due to budget constraints.
> A GPU runtime is recommended to reduce computation time.

---
 
## Dependencies
 
| Library | Version |
|---|---|
| TensorFlow | 2.19.0 |
| Scikit-learn | 1.6.1 |
| NumPy | 2.0.2 |
| Pandas | 2.2.2 |
| Optuna | 4.8.0 |
| FlowIO | 1.4.0 |
| Matplotlib | 3.10.0 |
| Seaborn | 0.13.2 |
| SciPy | 1.16.3 |
 
Install all dependencies with:
 
```bash
pip install -r requirements.txt
```
 
---

📁 **Experiments, results and plots** are available on Google Drive: https://drive.google.com/drive/folders/1JshP_PXYHC2fsba1stgj6rr7_iF3zJu2?usp=sharing  

---












 ### 3. Run the research notebooks
1. Create a folder called `experiments` in BinCellCNN repository.
2. Create a new folder in `experiments/` and name it `experiments_[as you want]` (`experiments_End_to_End_pipeline` for this explanation).
3. Copy the file config.yml in the repository `experiments/Research/config.yml`, open it with a text editor. change any variable you want.
4. Open the `End-to-End_LOPOCV_pipeline.ipynb` in `notebooks/research/`, look for the `whole_exp` variable and change it with the name you chose for the folder at point 2. In this example: `whole_exp = 'End_to_End_pipeline`.
   Also modify the variable `fixed_path` inserting the path for the BinCellCNN repository.
5. Restart the kernel and run entirely the notebook.

#### research: config.yml Configuration

All pipeline settings are controlled through a `config.yml` file.
See the [configuration guide](experiments/README.md) for a description of every parameter.


### 4. Run the clinical notebooks


1. Create a folder called `experiments` in BinCellCNN repository.
2. Create a new folder in `experiments/` and name it `experiments_[as you want]` (`clinical` for this explanation).
3. Copy the file config.yml in the repository `experiments/Clinical/config.yml`, open it with a text editor and change any variable you want.
   Make sure to set correctly you training samples path, the extension, the name o the column that contains the binary labels. If you follow the the instruction in point 4, you mustn't change the sequence parameter, otherwise make sure to modify the function to match your samples name pattern. You can also choose to skip Bayesian Tuning section: to so so, set 'perform_tuning: False' and set ncells and nsubs as you like or as you previously found. 

4. Copy the file in this repository `experiments/Clinical/test_clinical_config.yml`, open it with a text editor and change any variable you want. This notebooks has been created to classfies not manually gated or labelled samples. if you want to use pre-abelled samples for tests, please include the label column name into the remove_columns varable (pay attention to the type of this variable. it must be set as a list of elements, eg remove_columns: ['column_name']. multiple columns might be processed by adding them into the list. ).

5. Open the `Clinical_training_pipeline.ipynb` in `notebooks/clinical/`, look for the `whole_exp` variable and change it with the name you chose for the folder at point 2 (In this example: `whole_exp = 'Clinical_Trial_1'`). Modify also `fixed_path` pasting the path in which you cloned this repository.

6. Restart the kernel and run entirely the notebook.

7. Once training is concluded, or if you already have trained models, open the `Clinical_classification_pipeline.ipynb` in `notebooks/clinical/`, look for the `whole_exp` variable and change it with the name you chose for the folder at point 2 (In this example: `whole_exp = 'clinical'`). Modify also `fixed_path` pasting the path in which you cloned this repository.

8. Restart the kernel and run entirely the notebook.

### 4.1 Data preprocess.
To run these notebooks using your data, you need to follow a few steps before running the notebooks. Additionally, this procedure has been developed to work with patients, splitting them in groups to avoid data leakage. If you have samples that are not correlated, you have to assign deterministically a different patient index each.

For points 1 and 2, the procedire is the same for both research and clinical. the third an fourth points are exclusive for clinical
1. Create a folder named "train_samples", where your samples have to be saved (for illustration purposes, some small artificial sample are present in art_train_samples and in art_new_sample). each sample have to be called "{sequence}{patient_id}_{sample_number}". by default, sequence = 'sample_'. a sample name example is: "sample_1_15". For now, the only extension supported is '.csv'
2. Within the folder  'train_samples', create a new folder called 'samples_labels'. for each sample in train_samples, you need to create a file containing the column 'label' associated to the sample label. each name have to be called {sequence}y_{patient_id}_{sample_number}". by default, sequence = 'sample_'. a sample name example is: "sample_y_1_15".For now, the only extension supported is '.csv'

3. Create a folder named "test_samples", where your samples have to be saved (for illustration purposes, some small artificial sample are present in art_test_samples and in art_new_sample). each sample have to be called "{sequence}{patient_id}_{sample_number}". by default, sequence = 'sample_'. a sample name example is: "sample_1_15". Make sure to include
4. No label needed here, sicne ideally we would use samples that have to be classified


> **Note:** Experiments were run on Google Colab using a CPU-based runtime due to budget constraints. Switching to a GPU runtime is possible and recommended to reduce computation time.
 