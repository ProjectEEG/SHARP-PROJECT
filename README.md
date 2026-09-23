# Directional Effect Stability and Multimodal Stacking

This repository contains MATLAB code for Directional Effect Stability (DES) feature selection, multimodal stacking, model evaluation, and visualization for conversion prediction in individuals at clinical high risk (CHR) for psychosis.

## Code overview

| File | Purpose |
| --- | --- |
| `feature_selection.m` | Implements regional feature aggregation and two-stage DES feature selection for resting-state and task-related EEG, based on bootstrap effect-direction stability and within-region directional consistency. |
| `boostrap_feature_select.m` | Performs 5,000 bootstrap resamples to estimate effect-direction consistency, the dominant effect direction, and confidence intervals for feature selection. |
| `discovery_.m` | Evaluates the multimodal stacked support vector machine (SVM) model in the discovery set using leave-one-subject-out (LOSO) cross-validation, combining cognitive, symptom, task-related EEG, and resting-state EEG features. |
| `validation_.m` | Trains the multimodal stacking model using the discovery set and evaluates its performance in the held-out validation set, including ROC analysis and threshold-specific classification metrics. |
| `COMPARE_ROCAUC.m` | Visualizes ROC curves for DES-based, LASSO-based, and effect-size-based feature selection, alongside a no-feature-selection baseline, with AUC values displayed in the figure legend. |
| `COMPARE_DCA.m` | Performs decision-curve analysis for the discovery and validation sets, visualizes net benefit across threshold probabilities, and exports the combined figure and numerical results. |
