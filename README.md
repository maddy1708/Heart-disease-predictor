# Heart Attack Risk Prediction Using Machine Learning

## Overview

This project develops a supervised machine-learning system for predicting heart-attack risk using demographic, clinical, lifestyle, and socioeconomic features.

The project uses the Kaggle Heart Attack Risk Prediction Dataset containing 8,763 patient records and 26 columns.

## Problem Statement

Cardiovascular risk is influenced by multiple interacting factors. The objective of this project is to investigate whether machine-learning models can classify patients according to their heart-attack risk and determine which model provides the strongest predictive performance.

## Research Objective

To compare conventional machine-learning classifiers with ensemble and boosting approaches for heart-attack risk prediction.

## Base Paper

**Optimizing Heart Disease Diagnosis with Advanced ML Models**

The project focuses particularly on the performance of advanced ensemble and boosting methods.

## Dataset

Source:

Kaggle — Heart Attack Risk Prediction Dataset

Dataset characteristics:

- 8,763 records
- 26 columns
- 25 predictor variables
- 1 binary target variable
- Target: `Heart Attack Risk`

## Methodology

```text
Raw Dataset
     ↓
Data Cleaning
     ↓
Feature Engineering
     ↓
Train/Test Split
     ↓
Encoding
     ↓
Feature Scaling
     ↓
SMOTE on Training Data
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Cross-Validation
     ↓
Model Selection
     ↓
Feature Importance
     ↓
SHAP Explainability
     ↓
Streamlit Application