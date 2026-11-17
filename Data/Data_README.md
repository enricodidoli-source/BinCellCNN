# Data preparation

The pipelines are designed to work with patients: samples from the same patient are kept in the same group to avoid data leakage. If your samples are not correlated, assign a different `patient_id` to each one.

Small artificial datasets are included for illustration in `Data/art_train_samples/` and `Data/art_new_samples/`.

## Training samples (research and clinical)

1. Create a folder `train_samples/` containing one `.csv` file per sample, named:

   ```
   {sequence}{patient_id}_{sample_number}.csv
   ```
   With the default `sequence = 'sample_'`, an example is `sample_1_15.csv` (patient 1, sample 15).

2. Inside `train_samples/`, create a folder `samples_labels/` with one `.csv` file per sample, containing a column `label` with the sample label. Names follow:

   ```
   {sequence}y_{patient_id}_{sample_number}.csv
   ```
   Example: `sample_y_1_15.csv`.

Currently only the `.csv` format is supported.

## New samples to classify (clinical only)

3. Create a folder `test_samples/` with the samples to classify, using the same naming pattern as above.
4. No labels are needed, since these are samples that still have to be classified.

If you want to test the pipeline on already labelled samples, add the label column to `remove_columns` in `test_clinical_config.yml` (see below).