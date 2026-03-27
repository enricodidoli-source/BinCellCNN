# Detecting Measurable Residual Disease in B-cell Acute Lymphoblastic Leukaemia using Convolutional Neural Networks
 
---

## Overview
 
This repository contains the code developed for my Master's thesis on **Measurable Residual Disease (MRD) detection** in B-cell Acute Lymphoblastic Leukaemia (B-ALL) using the **CellCNN** neural network architecture applied to multiparameter flow cytometry (MFC) data.
 
---
 
## Repository Structure
 
```
master-thesis/
│
├── CellCNN/
│   ├── CellCNN_modules/  # Python modules (.py) for CellCNN model (modified from the original implementation: https://github.com/eiriniar/CellCnn)
│   ├── modules/          # Python modules (.py) for data processing, training and evaluation
│   └── notebooks/        # Jupyter notebooks for experiments and results
│
├── requirements.txt
└── README.md
```

---
 
## Getting Started
 
### 1. Clone the repository
 
```bash
git clone https://github.com/enricodidoli-source/master-thesis.git
cd master-thesis
```
 
### 2. Install dependencies
 
```bash
pip install -r requirements.txt
```
 
### 3. Run the notebooks
 
Open the Jupyter notebooks in the `CellCNN/notebooks/` folder to reproduce the experiments. The correct execution order is:
 
1. **Tuning:** `CellCNN/notebooks/training_classification/Bayesian_NO_AS_LOPOCV`
2. **Training / Testing:** `CellCNN/notebooks/training_classification/` (any notebook, e.g. `Bayesian_NO_AS_LOPOCV`)
3. **Results and plots:** `CellCNN/notebooks/results_elaboration_and_plotting/`, in this order:
   - `0. Show_patient_info`
   - `1. Show_BO_and_tuned_thresholds`
   - `2. Show_Single_Split` / `Show_Ensemble`
   - `3. Show_experiments_comparison` (requires both Single Split / Ensemble and both AS / NO-AS cases to have been run)
 
> **Note:** The data loading function is data-specific and must be adapted depending on the dataset you are using.
 
> **Note:** Experiments were run on Google Colab using a CPU-based runtime due to budget constraints. Switching to a GPU runtime is possible and recommended to reduce computation time.
 
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


