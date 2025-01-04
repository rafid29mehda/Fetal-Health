### **Hybrid Ensemble Models Combining Deep Learning and Traditional Machine Learning**

Creating a hybrid ensemble model that integrates the strengths of deep learning models (like TabNet) with traditional machine learning algorithms (like LightGBM) can significantly enhance your model's performance. This approach leverages the ability of deep models to capture complex nonlinear patterns and the proficiency of traditional models in handling feature interactions.

Below is a step-by-step guide with corresponding code snippets that you can copy and paste into Google Colab. Each step includes detailed explanations to help you understand the process.

---

## **Step 1: Install Required Libraries**

Before proceeding, ensure that all necessary libraries are installed in your Google Colab environment. This includes `pytorch-tabnet` for the TabNet model and `lightgbm` for the LightGBM model.

```python
# Install pytorch-tabnet
!pip install pytorch-tabnet

# Install LightGBM
!pip install lightgbm

# Optional: Install scikit-learn-contrib for any additional functionalities
!pip install scikit-learn-contrib
```

**Explanation:**

- **pytorch-tabnet:** A PyTorch implementation of TabNet, which is designed for tabular data.
- **lightgbm:** A highly efficient gradient boosting framework that uses tree-based learning algorithms.

---

## **Step 2: Import Necessary Libraries**

Import all the required libraries for data manipulation, modeling, and evaluation.

```python
# Data manipulation and analysis
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing and modeling
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression

# Handling imbalanced data
from imblearn.over_sampling import SMOTE

# Traditional ML
import lightgbm as lgb

# Deep Learning Model
from pytorch_tabnet.tab_model import TabNetClassifier

# Explainable AI
import shap

# Meta-Learner
from sklearn.linear_model import LogisticRegression

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')
```

**Explanation:**

- **pandas, numpy:** For data manipulation and numerical operations.
- **matplotlib, seaborn:** For data visualization.
- **scikit-learn modules:** For preprocessing, modeling, and evaluation.
- **imbalanced-learn:** For handling class imbalance using SMOTE.
- **lightgbm:** For training the LightGBM model.
- **pytorch_tabnet:** For training the TabNet model.
- **shap:** For model interpretability.

---

## **Step 3: Load and Prepare the Dataset**

Assuming you have already loaded and preprocessed your dataset (as per your initial code), we'll proceed from there. If not, ensure you run your preprocessing steps before moving forward.

```python
# Replace 'fetal_health.csv' with your actual file path if different
data = pd.read_csv('/content/fetal_health.csv')

# Display the first five rows to verify
print("First five rows of the dataset:")
print(data.head())

# Check the shape of the dataset
print(f"Dataset Shape: {data.shape}")

# Convert 'fetal_health' to integer
data['fetal_health'] = data['fetal_health'].astype(int)

# Mapping numerical classes to descriptive labels
health_mapping = {1: 'Normal', 2: 'Suspect', 3: 'Pathological'}
data['fetal_health_label'] = data['fetal_health'].map(health_mapping)

# Features (all columns except 'fetal_health' and 'fetal_health_label')
X = data.drop(['fetal_health', 'fetal_health_label'], axis=1)

# Target variable
y = data['fetal_health']

# Split the data (80% train, 20% test) with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize SMOTE
smote = SMOTE(random_state=42)

# Apply SMOTE to the training data
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler on the resampled training data and transform
X_train_scaled = scaler.fit_transform(X_train_resampled)

# Transform the test data
X_test_scaled = scaler.transform(X_test)

# Convert the scaled arrays back to DataFrames for easier handling
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train_resampled.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)

# Verify scaling by checking means and standard deviations
print("\nMean of Scaled Training Features (Should be ~0):")
print(X_train_scaled.mean())

print("\nStandard Deviation of Scaled Training Features (Should be ~1):")
print(X_train_scaled.std())
```

**Explanation:**

- **Data Loading and Verification:** Load the dataset and verify its structure.
- **Data Preprocessing:** Convert the target variable to integer and map it to descriptive labels.
- **Train-Test Split:** Split the data into training and testing sets with stratification to maintain class distribution.
- **Handling Class Imbalance:** Apply SMOTE to balance the classes in the training set.
- **Feature Scaling:** Standardize the features to have a mean of ~0 and a standard deviation of ~1.

---

## **Step 4: Define and Train the Base Models**

### **4.1. Train the LightGBM Model**

```python
# Initialize the LightGBM Classifier
lgb_classifier = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.05,
    num_leaves=31,
    objective='multiclass',
    class_weight='balanced',
    random_state=42
)

# Train the LightGBM model on the resampled and scaled training data
lgb_classifier.fit(X_train_scaled, y_train_resampled)

# Make predictions on the test set
y_pred_lgb = lgb_classifier.predict(X_test_scaled)

# Evaluate LightGBM performance
print("\nLightGBM Classification Report:")
print(classification_report(y_test, y_pred_lgb, target_names=['Normal', 'Suspect', 'Pathological']))

# Confusion Matrix for LightGBM
conf_matrix_lgb = confusion_matrix(y_test, y_pred_lgb)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_lgb, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Suspect', 'Pathological'],
            yticklabels=['Normal', 'Suspect', 'Pathological'])
plt.title('LightGBM Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
```

**Explanation:**

- **LightGBM Initialization:** Configure the LightGBM classifier with appropriate hyperparameters. Using `class_weight='balanced'` helps in handling class imbalance.
- **Training:** Train the LightGBM model on the resampled and scaled training data.
- **Prediction and Evaluation:** Make predictions on the test set and evaluate the model's performance using a classification report and confusion matrix.

### **4.2. Train the TabNet Model**

```python
# Initialize the TabNet Classifier
tabnet_classifier = TabNetClassifier(
    n_d=64,                # Dimension of the decision step
    n_a=64,                # Dimension of the attention step
    n_steps=5,             # Number of steps in the architecture
    gamma=1.3,             # Relaxation parameter
    lambda_sparse=1e-3,    # Sparse regularization
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    mask_type='sparsemax', # "sparsemax" or "entmax"
    verbose=1
)

# Train the TabNet model on the resampled and scaled training data
tabnet_classifier.fit(
    X_train=X_train_scaled.values,
    y_train=y_train_resampled.values,
    eval_set=[(X_train_scaled.values, y_train_resampled.values), (X_test_scaled.values, y_test.values)],
    eval_name=['train', 'valid'],
    eval_metric=['accuracy'],
    max_epochs=100,
    patience=20,
    batch_size=256,
    virtual_batch_size=128
)

# Make predictions on the test set
y_pred_tabnet = tabnet_classifier.predict(X_test_scaled.values)

# Evaluate TabNet performance
print("\nTabNet Classification Report:")
print(classification_report(y_test, y_pred_tabnet, target_names=['Normal', 'Suspect', 'Pathological']))

# Confusion Matrix for TabNet
conf_matrix_tabnet = confusion_matrix(y_test, y_pred_tabnet)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_tabnet, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Normal', 'Suspect', 'Pathological'],
            yticklabels=['Normal', 'Suspect', 'Pathological'])
plt.title('TabNet Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
```

**Explanation:**

- **TabNet Initialization:** Configure the TabNet classifier with specified hyperparameters.
- **Training:** Train the TabNet model on the resampled and scaled training data. The `eval_set` parameter allows monitoring of performance on both training and validation (test) sets.
- **Prediction and Evaluation:** Make predictions on the test set and evaluate the model's performance using a classification report and confusion matrix.

---

## **Step 5: Create Meta-Features for the Meta-Learner**

To build the ensemble, we'll use the predictions from the base models (LightGBM and TabNet) as features for the meta-learner.

```python
# Generate predictions for training data using cross-validation to prevent overfitting

from sklearn.model_selection import KFold, StratifiedKFold

# Initialize KFold
n_folds = 5
kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Initialize arrays to hold out-of-fold predictions for training data
oof_preds_lgb = np.zeros((X_train_scaled.shape[0], 3))  # 3 classes
oof_preds_tabnet = np.zeros((X_train_scaled.shape[0], 3))  # 3 classes

# Initialize arrays to hold test set predictions
test_preds_lgb = np.zeros((X_test_scaled.shape[0], 3))
test_preds_tabnet = np.zeros((X_test_scaled.shape[0], 3))

print("Generating out-of-fold predictions...")

for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train_scaled, y_train_resampled)):
    print(f"Fold {fold + 1}")
    
    # Split data
    X_tr, X_val = X_train_scaled.iloc[train_idx], X_train_scaled.iloc[valid_idx]
    y_tr, y_val = y_train_resampled.iloc[train_idx], y_train_resampled.iloc[valid_idx]
    
    # LightGBM
    lgb_fold = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=31,
        objective='multiclass',
        class_weight='balanced',
        random_state=42
    )
    lgb_fold.fit(X_tr, y_tr)
    oof_preds_lgb[valid_idx] = lgb_fold.predict_proba(X_val)
    test_preds_lgb += lgb_fold.predict_proba(X_test_scaled) / n_folds
    
    # TabNet
    tabnet_fold = TabNetClassifier(
        n_d=64,
        n_a=64,
        n_steps=5,
        gamma=1.3,
        lambda_sparse=1e-3,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        mask_type='sparsemax',
        verbose=0
    )
    tabnet_fold.fit(
        X_train=X_tr.values,
        y_train=y_tr.values,
        eval_set=[(X_val.values, y_val.values)],
        eval_name=['valid'],
        eval_metric=['accuracy'],
        max_epochs=100,
        patience=20,
        batch_size=256,
        virtual_batch_size=128
        # Removed 'reset=True' parameter
    )
    oof_preds_tabnet[valid_idx] = tabnet_fold.predict_proba(X_val.values)
    test_preds_tabnet += tabnet_fold.predict_proba(X_test_scaled.values) / n_folds

print("Out-of-fold predictions generated.")
```

**Explanation:**

- **K-Fold Cross-Validation:** Use `StratifiedKFold` to ensure each fold maintains the class distribution.
- **Out-of-Fold (OOF) Predictions:** For each fold, train the base models on the training subset and generate predictions on the validation subset. These OOF predictions will be used as features for the meta-learner.
- **Test Set Predictions:** Aggregate predictions from each fold to obtain robust predictions on the test set.

---

## **Step 6: Prepare Meta-Features**

Combine the OOF predictions from both base models to create a new feature set for the meta-learner.

```python
# Create meta-features for the training set
meta_X_train = np.hstack((oof_preds_lgb, oof_preds_tabnet))

# Create meta-features for the test set
meta_X_test = np.hstack((test_preds_lgb, test_preds_tabnet))

# Define the meta-learner target
meta_y_train = y_train_resampled.values
```

**Explanation:**

- **Horizontal Stacking:** Combine the predicted probabilities from LightGBM and TabNet horizontally to form the meta-features.
- **Meta-Features:** Each instance now has features representing the probability estimates from both base models.

---

## **Step 7: Train the Meta-Learner**

We'll use Logistic Regression as the meta-learner to combine the base models' predictions.

```python
# Initialize the Logistic Regression meta-learner
meta_learner = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    max_iter=1000,
    random_state=42
)

# Train the meta-learner on the meta-features
meta_learner.fit(meta_X_train, meta_y_train)

# Make predictions on the test set meta-features
y_pred_ensemble = meta_learner.predict(meta_X_test)

# Evaluate the Ensemble Model
print("\nEnsemble Model Classification Report:")
print(classification_report(y_test, y_pred_ensemble, target_names=['Normal', 'Suspect', 'Pathological']))

# Confusion Matrix for Ensemble Model
conf_matrix_ensemble = confusion_matrix(y_test, y_pred_ensemble)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_ensemble, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Normal', 'Suspect', 'Pathological'],
            yticklabels=['Normal', 'Suspect', 'Pathological'])
plt.title('Ensemble Model Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Overall Accuracy
accuracy = accuracy_score(y_test, y_pred_ensemble)
print(f"Ensemble Model Accuracy: {accuracy:.4f}")
```

**Explanation:**

- **Logistic Regression Initialization:** Configure the Logistic Regression model suitable for multiclass classification.
- **Training:** Train the meta-learner on the combined meta-features.
- **Prediction and Evaluation:** Make predictions on the test set using the meta-learner and evaluate the ensemble's performance.

---

## **Step 8: Compare Individual Models with the Ensemble**

It's essential to compare the performance of the individual base models against the ensemble to demonstrate the improvement.

```python
# Individual Models' Accuracy
accuracy_lgb = accuracy_score(y_test, y_pred_lgb)
accuracy_tabnet = accuracy_score(y_test, y_pred_tabnet)

print(f"LightGBM Accuracy: {accuracy_lgb:.4f}")
print(f"TabNet Accuracy: {accuracy_tabnet:.4f}")
print(f"Ensemble Model Accuracy: {accuracy:.4f}")
```

**Explanation:**

- **Performance Comparison:** Print the accuracy of LightGBM, TabNet, and the Ensemble Model to observe the improvement achieved by the ensemble.

---

## **Step 9: Feature Importance for the Meta-Learner**

Understanding which base models contribute more to the ensemble's predictions can provide valuable insights.

```python
# Coefficients of the meta-learner
meta_coefficients = meta_learner.coef_

# Create a DataFrame to display the coefficients
coeff_df = pd.DataFrame(meta_coefficients, columns=['LightGBM_Normal', 'LightGBM_Suspect', 'LightGBM_Pathological',
                                                    'TabNet_Normal', 'TabNet_Suspect', 'TabNet_Pathological'])

print("\nMeta-Learner Coefficients:")
print(coeff_df)
```

**Explanation:**

- **Coefficients Analysis:** The coefficients of the Logistic Regression meta-learner indicate the weight each base model's prediction has in the final ensemble prediction for each class.

---

## **Step 10: Save the Ensemble Model**

For future use or deployment, it's advisable to save the ensemble model.

```python
import joblib

# Save the meta-learner
joblib.dump(meta_learner, 'meta_learner.pkl')

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

# Save the LightGBM and TabNet models
joblib.dump(lgb_classifier, 'lightgbm_model.pkl')
tabnet_classifier.save_model('tabnet_model.zip')

print("Models have been saved successfully.")
```

**Explanation:**

- **Joblib:** A library for efficient serialization of Python objects, useful for saving models.
- **TabNet Model Saving:** TabNet uses a custom `save_model` method to save its state.

---

## **Step 11: Load and Use the Ensemble Model (Optional)**

To demonstrate how to load and use the saved ensemble model for predictions on new data:

```python
# Load the saved models
loaded_meta_learner = joblib.load('meta_learner.pkl')
loaded_lgb_classifier = joblib.load('lightgbm_model.pkl')
loaded_scaler = joblib.load('scaler.pkl')

# Load the TabNet model
loaded_tabnet = TabNetClassifier()
loaded_tabnet.load_model('tabnet_model.zip')

# Example: Predicting on a new sample
# Replace 'new_sample' with actual data
# new_sample = pd.DataFrame({...})  # Ensure it has the same features as X

# For demonstration, we'll use the first instance from the test set
new_sample = X_test.iloc[[0]]
new_sample_scaled = loaded_scaler.transform(new_sample)

# Get base model predictions
lgb_pred_proba = loaded_lgb_classifier.predict_proba(new_sample_scaled)
tabnet_pred_proba = loaded_tabnet.predict_proba(new_sample_scaled)

# Create meta-features
meta_features = np.hstack((lgb_pred_proba, tabnet_pred_proba))

# Predict using the meta-learner
ensemble_pred = loaded_meta_learner.predict(meta_features)

# Map numerical prediction to labels
ensemble_pred_label = health_mapping[ensemble_pred[0]]
print(f"Ensemble Prediction for the new sample: {ensemble_pred_label}")
```

**Explanation:**

- **Model Loading:** Demonstrates how to load the saved meta-learner, LightGBM, and TabNet models.
- **New Sample Prediction:** Shows how to preprocess a new sample, obtain predictions from base models, create meta-features, and make a final prediction using the ensemble.

---

## **Step 12: Save and Visualize SHAP Values for the Ensemble Model**

To interpret the ensemble model's predictions, we can compute SHAP values. However, since the ensemble consists of multiple models, interpreting SHAP values can be complex. One approach is to analyze the SHAP values of the meta-learner.

```python
# Since the meta-learner is a Logistic Regression model, feature importances can be derived from its coefficients

# Create a DataFrame for meta-learner coefficients
meta_coefficients_df = pd.DataFrame(meta_learner.coef_, columns=['LightGBM_Normal', 'LightGBM_Suspect', 'LightGBM_Pathological',
                                                                 'TabNet_Normal', 'TabNet_Suspect', 'TabNet_Pathological'],
                                    index=['Normal', 'Suspect', 'Pathological'])

print("\nMeta-Learner Coefficients:")
print(meta_coefficients_df)

# Plot the coefficients
meta_coefficients_df.plot(kind='bar', figsize=(10, 6))
plt.title('Meta-Learner Coefficients')
plt.ylabel('Coefficient Value')
plt.xticks(rotation=0)
plt.legend(title='Base Model Predictions')
plt.show()
```

**Explanation:**

- **Meta-Learner Coefficients:** For Logistic Regression, the coefficients indicate the importance of each feature (base model's predictions) for each class.
- **Visualization:** A bar plot visualizes the influence of each base model's predictions on the ensemble's final prediction.

---

## **Final Notes and Best Practices**

1. **Hyperparameter Tuning:**
   - Optimize hyperparameters for both base models and the meta-learner using techniques like Grid Search or Bayesian Optimization to enhance performance further.

2. **Cross-Validation:**
   - Ensure that cross-validation is appropriately implemented to prevent data leakage and overfitting, especially when generating meta-features.

3. **Model Interpretability:**
   - While SHAP is powerful for individual models, interpreting it for ensemble models requires careful consideration. Focus on understanding the meta-learner's behavior and how it integrates base models' predictions.

4. **Performance Metrics:**
   - Besides accuracy, consider other metrics like F1-score, Precision, Recall, and AUC-ROC, especially given the class imbalance in your dataset.

5. **Deployment:**
   - When deploying the ensemble model, ensure that all components (scaler, base models, meta-learner) are loaded correctly and that the preprocessing steps are consistently applied to new data.

6. **Documentation:**
   - Keep detailed records of model configurations, training processes, and evaluation results to facilitate reproducibility and further research.

---

By following the above steps, you can effectively implement a hybrid ensemble model that leverages both deep learning and traditional machine learning techniques, potentially leading to improved performance in fetal health detection. This approach not only enhances predictive accuracy but also provides a robust framework for handling complex and imbalanced datasets.

Feel free to reach out if you encounter any issues or have further questions!
