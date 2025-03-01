# -*- coding: utf-8 -*-
"""CTGAN-Tabnet.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ZJoaXx3W5GRrRWWzDSv3-fOxKRm5d3-5
"""

# Install necessary libraries
!pip install pytorch-tabnet
!pip install captum
!pip install optuna
!pip install imbalanced-learn
!pip install dask-expr
!pip install scikit-learn-contrib
!pip install lightgbm
!pip install ctgan  # Add this for CTGAN

# Data manipulation and analysis
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing and modeling
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Deep Learning Model
from pytorch_tabnet.tab_model import TabNetClassifier

# Explainable AI
import shap

# Hyperparameter Optimization
import optuna
from optuna import Trial

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# For model saving and loading
import joblib
import pickle  # Add pickle for CTGAN loading

# Import torch for TabNet
import torch

# Additional imports for Permutation Regularization
import torch.nn as nn

# -------------------
# 3. Load and Preprocess the Dataset
# -------------------
# Load the dataset
data = pd.read_csv('/content/fetal_health.csv')  # Update path if needed

# Display the first five rows to verify
print("First five rows of the dataset:")
print(data.head())

# Check the shape of the dataset
print(f"\nDataset Shape: {data.shape}")

# Features to drop based on prior analysis (same as original)
features_to_drop = [
    'fetal_movement', 'histogram_width', 'histogram_max', 'mean_value_of_long_term_variability',
    'histogram_number_of_peaks', 'light_decelerations', 'histogram_tendency',
    'histogram_number_of_zeroes', 'severe_decelerations', 'baseline value', 'histogram_min'
]

# Drop the specified features
data_dropped = data.drop(columns=features_to_drop)

# Verify the remaining features
print("\nFeatures after dropping less important ones:")
print(data_dropped.columns.tolist())

# Check the new shape of the dataset
print(f"\nNew Dataset Shape after dropping features: {data_dropped.shape}")

# Convert 'fetal_health' to integer
data_dropped['fetal_health'] = data_dropped['fetal_health'].astype(int)

# Mapping numerical classes to descriptive labels
health_mapping = {1: 'Normal', 2: 'Suspect', 3: 'Pathological'}
data_dropped['fetal_health_label'] = data_dropped['fetal_health'].map(health_mapping)

# Display the mapping
print("\nDataset with Mapped Labels:")
print(data_dropped[['fetal_health', 'fetal_health_label']].head())

# Features and target
X = data_dropped.drop(['fetal_health', 'fetal_health_label'], axis=1)
y = data_dropped['fetal_health']

# --- Replace ADASYN + Tomek with CTGAN ---
# Load pre-trained CTGAN models (upload ctgan_suspect.pkl and ctgan_pathological.pkl to Colab first)
with open('/content/ctgan_suspect.pkl', 'rb') as f:
    ctgan_suspect = pickle.load(f)
with open('/content/ctgan_pathological.pkl', 'rb') as f:
    ctgan_pathological = pickle.load(f)

# Scale original features (CTGAN models were trained on scaled data)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Number of synthetic samples to balance with Normal (1655)
n_suspect = 1655 - 295  # 1360
n_pathological = 1655 - 176  # 1479

# Generate synthetic samples using CTGAN
synthetic_suspect = ctgan_suspect.sample(n_suspect)
synthetic_pathological = ctgan_pathological.sample(n_pathological)

# Combine synthetic data
synthetic_data = pd.concat([synthetic_suspect, synthetic_pathological], ignore_index=True)
synthetic_df = synthetic_data.drop('fetal_health', axis=1)
synthetic_labels = synthetic_data['fetal_health']

# Combine with original data
X_resampled = pd.concat([X_scaled, synthetic_df], ignore_index=True)
y_resampled = pd.concat([y, synthetic_labels], ignore_index=True)

# Display the shape and class distribution
print(f"\nResampled X shape after CTGAN: {X_resampled.shape}")
print(f"Resampled y distribution after CTGAN:\n{y_resampled.value_counts()}")

# Visualize the resampled class distribution
plt.figure(figsize=(8, 6))
sns.countplot(x=y_resampled, palette='viridis')
plt.title('Class Distribution After CTGAN')
plt.xlabel('Fetal Health')
plt.ylabel('Count')
plt.show()

# Split the resampled data (70% train, 30% test) with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.3, random_state=42, stratify=y_resampled
)

# Display the shapes of the training and testing sets
print(f"\nTraining set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Initialize the MinMaxScaler (already scaled, but reassign for consistency)
X_train_scaled = pd.DataFrame(X_train, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test, columns=X.columns)

# Verify scaling by checking min and max values
print("\nMin of Scaled Training Features (Should be 0):")
print(X_train_scaled.min())
print("\nMax of Scaled Training Features (Should be 1):")
print(X_train_scaled.max())

# Adjust the target values so they start from 0
y_train = y_train - 1
y_test = y_test - 1

# Display the adjusted target distributions
print("\nAdjusted y_train distribution:")
print(pd.Series(y_train).value_counts())
print("\nAdjusted y_test distribution:")
print(pd.Series(y_test).value_counts())

# Further split the training data into training and validation sets (80% train, 20% validation)
X_train_final, X_valid, y_train_final, y_valid = train_test_split(
    X_train_scaled, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# Display the shapes of the final training and validation sets
print(f"\nFinal Training set shape: {X_train_final.shape}")
print(f"Validation set shape: {X_valid.shape}")

# Define the objective function for Optuna
def objective(trial: Trial):
    # Define the hyperparameter space
    n_d = trial.suggest_int('n_d', 32, 128)
    n_a = trial.suggest_int('n_a', 32, 128)
    n_steps = trial.suggest_int('n_steps', 3, 10)
    gamma = trial.suggest_float('gamma', 1.0, 2.0)
    lambda_sparse = trial.suggest_float('lambda_sparse', 1e-4, 1e-2, log=True)
    learning_rate = trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [128, 256, 512])

    # Initialize TabNet with current hyperparameters
    tabnet = TabNetClassifier(
        n_d=n_d,
        n_a=n_a,
        n_steps=n_steps,
        gamma=gamma,
        lambda_sparse=lambda_sparse,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=learning_rate),
        mask_type='sparsemax',
        verbose=0
    )

    # Train the model on the final training set
    tabnet.fit(
        X_train=X_train_final.values,
        y_train=y_train_final.values,
        eval_set=[(X_valid.values, y_valid.values)],
        eval_name=['valid'],
        eval_metric=['accuracy'],
        max_epochs=100,
        patience=20,
        batch_size=batch_size,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False
    )

    # Predict on the validation set
    y_pred = tabnet.predict(X_valid.values)
    accuracy = accuracy_score(y_valid, y_pred)

    return accuracy

# Create and optimize the Optuna study
study = optuna.create_study(direction='maximize', study_name='TabNet Hyperparameter Optimization')
study.optimize(objective, n_trials=50, timeout=3600)  # Adjust n_trials and timeout as needed

# Display the best hyperparameters and validation accuracy
print("Best Hyperparameters: ", study.best_params)
print("Best Validation Accuracy: ", study.best_value)

# -------------------
# Part 1: Data Augmentation Function
# -------------------
def augment_data(X, y, permutation_prob=0.1):
    """
    Augment the dataset by randomly permuting feature orders with a given probability.

    Parameters:
    - X (numpy.ndarray or pandas.DataFrame): Feature matrix.
    - y (numpy.ndarray or pandas.Series): Target vector.
    - permutation_prob (float): Probability of permuting each sample.

    Returns:
    - X_augmented (numpy.ndarray): Augmented feature matrix.
    - y_augmented (numpy.ndarray): Augmented target vector.
    """
    X_augmented = []
    y_augmented = []
    for sample, label in zip(X, y):
        if np.random.rand() < permutation_prob:
            perm = np.random.permutation(sample.shape[0])
            sample = sample[perm]
        X_augmented.append(sample)
        y_augmented.append(label)
    return np.array(X_augmented), np.array(y_augmented)

# -------------------
# Part 2: Apply Data Augmentation
# -------------------
permutation_probability = 0.1
X_train_augmented, y_train_augmented = augment_data(
    X_train_final.values,
    y_train_final.values,
    permutation_prob=permutation_probability
)

print(f"Original Training Set Shape: {X_train_final.shape}")
print(f"Augmented Training Set Shape: {X_train_augmented.shape}")

# -------------------
# Part 3: Initialize and Train TabNet with Augmented Data
# -------------------
perm_reg_tabnet = TabNetClassifier(
    input_dim=X_train_final.shape[1],    # Number of features (10)
    output_dim=3,                        # Number of classes: Normal, Suspect, Pathological
    n_d=study.best_params['n_d'],
    n_a=study.best_params['n_a'],
    n_steps=study.best_params['n_steps'],
    gamma=study.best_params['gamma'],
    lambda_sparse=study.best_params['lambda_sparse'],
    optimizer_fn=torch.optim.Adam,
    optimizer_params={'lr': study.best_params['learning_rate']},
    mask_type='sparsemax',
    verbose=1,
    seed=42
)

perm_reg_tabnet.fit(
    X_train=X_train_augmented,
    y_train=y_train_augmented,
    eval_set=[(X_valid.values, y_valid.values), (X_test_scaled.values, y_test.values)],
    eval_name=['train', 'valid'],
    eval_metric=['accuracy'],
    max_epochs=100,
    patience=20,
    batch_size=study.best_params['batch_size'],
    virtual_batch_size=128,
    num_workers=0,
    drop_last=False
)

# -------------------
# Part 4: Predict and Evaluate on the Test Set
# -------------------
y_pred_perm_reg = perm_reg_tabnet.predict(X_test_scaled.values)

print("\nPermutation Regularized TabNet Classification Report:")
print(classification_report(y_test, y_pred_perm_reg, target_names=['Normal', 'Suspect', 'Pathological']))

conf_matrix = confusion_matrix(y_test, y_pred_perm_reg)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Suspect', 'Pathological'],
            yticklabels=['Normal', 'Suspect', 'Pathological'])
plt.title('Confusion Matrix for Permutation Regularized TabNet')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()













