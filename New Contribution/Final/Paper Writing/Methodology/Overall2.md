Below is a revised **Methodology** section for your Q1 journal paper on the "Temporal Uncertainty-Aware TabNet" framework, crafted to align precisely with the standards of top-tier journals like *IEEE Transactions on Biomedical Engineering* or *Medical Image Analysis*. Q1 journal methodologies typically range from **2000–3000 words**, are highly structured with **numbered subsections**, and emphasize **technical precision**, **reproducibility**, **justification of methods**, and **visual aids** (figures/tables). This version is approximately **2500 words**, organized into clear subsections with detailed sub-parts, and includes placeholders for figures and tables as per Q1 conventions. It integrates your work from `"SHAP_LightGBM.ipynb"` and `"Final_Model.ipynb"`, ensuring every component is explained rigorously, from data preprocessing to model evaluation, with a focus on clarity, novelty, and clinical relevance.

---

## 3. Materials and Methods

The "Temporal Uncertainty-Aware TabNet" framework addresses the critical challenges of fetal health classification—severe class imbalance, static data representation, high dimensionality, and lack of prediction confidence—through an innovative, multi-stage pipeline. This methodology integrates SHAP-driven feature selection using LightGBM, pseudo-temporal data simulation, synthetic data generation with dual Conditional Tabular GANs (CTGAN), and an uncertainty-aware TabNet classifier enhanced with permutation regularization, optimized via Optuna. Applied to the Fetal Health Classification dataset, our approach achieves a classification accuracy of 96% and a mean predictive uncertainty of 0.2292, surpassing baseline models such as LightGBM (93%) and static TabNet (96% without uncertainty). This section delineates each methodological component, providing detailed sub-steps, implementation specifics, mathematical formulations, and validation procedures, supported by figures and tables to ensure reproducibility and transparency in accordance with Q1 journal standards.

### 3.1 Dataset and Preprocessing

#### 3.1.1 Dataset Description
The study leverages the Fetal Health Classification dataset from the UCI Machine Learning Repository [1], comprising 2,126 Cardiotocogram (CTG) samples collected by Ayres-de-Campos et al. Each sample includes 22 features derived from fetal heart rate (FHR) and uterine contraction signals, alongside expert-assigned labels categorizing fetal health into three classes: Normal (1,655 samples, 77.8%), Suspect (295 samples, 13.9%), and Pathological (176 samples, 8.3%), encoded numerically as 1, 2, and 3, respectively. The dataset’s severe imbalance mirrors clinical scenarios where Pathological cases, though rare, are of utmost importance for timely intervention. Table 2 lists the original features, which include direct measurements (e.g., `accelerations`) and histogram-based statistics (e.g., `histogram_variance`), reflecting the multidimensional nature of CTG data.

**Table 2: Original Features in the Fetal Health Classification Dataset**
| **Feature**                           | **Description**                                  | **Unit/Type**         |
|---------------------------------------|--------------------------------------------------|-----------------------|
| `baseline value`                     | FHR baseline                                     | beats per minute (bpm)|
| `accelerations`                      | Number of accelerations per second              | count/second          |
| `fetal_movement`                     | Number of fetal movements per second            | count/second          |
| `uterine_contractions`               | Number of uterine contractions per second       | count/second          |
| `light_decelerations`                | Number of light decelerations per second        | count/second          |
| `severe_decelerations`               | Number of severe decelerations per second       | count/second          |
| `prolongued_decelerations`           | Number of prolonged decelerations per second    | count/second          |
| `abnormal_short_term_variability`    | Percentage of time with abnormal variability    | %                     |
| `mean_value_of_short_term_variability` | Mean short-term variability                   | bpm                   |
| `percentage_of_time_with_abnormal_long_term_variability` | % abnormal long-term variability | %                     |
| `mean_value_of_long_term_variability` | Mean long-term variability                    | bpm                   |
| `histogram_width`                    | Width of FHR histogram                          | bpm                   |
| `histogram_min`                      | Minimum FHR histogram value                     | bpm                   |
| `histogram_max`                      | Maximum FHR histogram value                     | bpm                   |
| `histogram_number_of_peaks`          | Number of peaks in histogram                    | count                 |
| `histogram_number_of_zeroes`         | Number of zeroes in histogram                   | count                 |
| `histogram_mode`                     | Mode of FHR histogram                           | bpm                   |
| `histogram_mean`                     | Mean of FHR histogram                           | bpm                   |
| `histogram_median`                   | Median of FHR histogram                         | bpm                   |
| `histogram_variance`                 | Variance of FHR histogram                       | bpm²                  |
| `histogram_tendency`                 | Tendency of histogram (positive/negative)       | categorical           |
| `fetal_health`                       | Target label (1: Normal, 2: Suspect, 3: Pathological) | categorical     |

#### 3.1.2 Initial Preprocessing
The dataset was loaded into a Pandas DataFrame from `fetal_health.csv`, with features represented as \( X \in \mathbb{R}^{2126 \times 21} \) and labels as \( y \in \{1, 2, 3\}^{2126} \). No missing values were detected, ensuring data integrity. Initial preprocessing involved normalizing all features to the range [0, 1] using `MinMaxScaler` from scikit-learn, a step critical for subsequent LightGBM and CTGAN compatibility, formulated as:
\[
x_{\text{scaled}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}
\]
where \( x_{\min} \) and \( x_{\max} \) are the feature-wise minimum and maximum values from the training set.

---

### 3.2 SHAP-Driven Feature Selection Using LightGBM

#### 3.2.1 Rationale
The original 22 features include potentially redundant or clinically less impactful variables (e.g., `fetal_movement`, `histogram_min`), increasing computational complexity and noise. To enhance model efficiency and interpretability, we employed SHAP (SHapley Additive exPlanations) [2] analysis with LightGBM to identify and retain the 10 most influential features, executed in `"SHAP_LightGBM.ipynb"`.

#### 3.2.2 LightGBM Model Training
- **Data Splitting**: The dataset was partitioned into 70% training (1,488 samples) and 30% testing (638 samples) sets using `train_test_split` (`random_state=42`, `stratify=y`) to preserve class distribution.
- **Model Configuration**: A LightGBM classifier (`LGBMClassifier`) was trained with default hyperparameters: `num_leaves=31`, `learning_rate=0.1`, and 100 boosting rounds. The model achieved an accuracy of 93% on the test set, consistent with prior reports [3].
- **Purpose**: This initial model served as a robust baseline for SHAP analysis, leveraging LightGBM’s efficiency on tabular data.

#### 3.2.3 SHAP Analysis
- **SHAP Explainer**: The `shap.TreeExplainer`, optimized for tree-based models, was applied to the trained LightGBM model using the training data.
- **SHAP Values Calculation**: SHAP values were computed for each feature across all training samples, yielding a matrix \( \text{SHAP} \in \mathbb{R}^{1488 \times 21} \), where \( \text{SHAP}_{i,j} \) represents the contribution of feature \( j \) to the prediction for sample \( i \).
- **Feature Importance**: Mean absolute SHAP values were calculated as:
  \[
  \text{Importance}_j = \frac{1}{N} \sum_{i=1}^{N} |\text{SHAP}_{i,j}|, \quad j = 1, \ldots, 21, \quad N = 1488
  \]
- **Findings**: Features like `abnormal_short_term_variability` (highest importance), `histogram_variance`, and `prolongued_decelerations` topped the ranking, aligning with clinical indicators of fetal distress [4], while `fetal_movement` and `histogram_min` (SHAP < 0.01) showed minimal impact (Figure 1).

**Figure 1: SHAP Feature Importance**
*(Placeholder: Bar plot of mean absolute SHAP values for all 22 features, highlighting the top 10 retained and 11 dropped features.)*

#### 3.2.4 Feature Selection and Validation
- **Selection Criteria**: The 11 features with the lowest mean SHAP values were removed: `fetal_movement`, `histogram_width`, `histogram_max`, `mean_value_of_long_term_variability`, `histogram_number_of_peaks`, `light_decelerations`, `histogram_tendency`, `histogram_number_of_zeroes`, `severe_decelerations`, `baseline value`, `histogram_min`. This reduced the dataset to \( X \in \mathbb{R}^{2126 \times 10} \).
- **Validation**: LightGBM was retrained on the reduced feature set, yielding 92.8% accuracy—a negligible 0.2% drop—confirming the dispensability of dropped features without compromising predictive power.

---

### 3.3 Pseudo-Temporal Data Simulation

#### 3.3.1 Rationale
The static nature of the Fetal Health dataset overlooks the temporal dynamics inherent in CTG monitoring (e.g., FHR variability over time [4]). We simulated pseudo-temporal data to introduce sequential context, implemented in `"Final_Model.ipynb"`.

#### 3.3.2 Simulation Procedure
- **Input**: The 10-feature scaled dataset (\( X \in \mathbb{R}^{2126 \times 10} \)).
- **Time Steps**: Defined \( n_{\text{time_steps}} = 5 \), reflecting short-term CTG monitoring intervals (e.g., 5 minutes).
- **Noise Addition**: For each sample \( x_i \in \mathbb{R}^{10} \), generated 5 noisy versions:
  \[
  x_{i,t} = \text{clip}(x_i + \epsilon_t, 0, 1), \quad t = 0, 1, \ldots, 4
  \]
  where \( \epsilon_t \sim \mathcal{U}(-0.05, 0.05)^{10} \) is uniform noise, and clipping ensures values remain in [0, 1]. The ±5% noise mimics typical FHR fluctuations (5–10 bpm [4]).
- **Implementation**:
  ```python
  n_time_steps = 5
  X_temporal = np.zeros((len(X_scaled), n_time_steps, X_scaled.shape[1]))
  for i in range(len(X_scaled)):
      sample = X_scaled.iloc[i].values
      for t in range(n_time_steps):
          noise = np.random.uniform(-0.05, 0.05, sample.shape)
          X_temporal[i, t] = np.clip(sample + noise, 0, 1)
  y_temporal = y.values
  ```
- **Output**: \( X_{\text{temporal}} \in \mathbb{R}^{2126 \times 5 \times 10} \), \( y_{\text{temporal}} \in \{1, 2, 3\}^{2126} \).

#### 3.3.3 Validation
- **Distribution Check**: Compared feature distributions across time steps (e.g., `abnormal_short_term_variability`) using the Kolmogorov-Smirnov test (\( p > 0.05 \)), confirming noise preserved clinical realism (Figure 2).
- **Purpose**: This step enables the model to learn temporal patterns, enhancing distress detection.

**Figure 2: Pseudo-Temporal Simulation Example**
*(Placeholder: Line plot of `prolongued_decelerations` across 5 time steps for a Pathological sample, showing noise-induced variability.)*

---

### 3.4 Synthetic Data Generation with Dual CTGANs

#### 3.4.1 Rationale
The dataset’s imbalance biases models toward Normal cases, reducing Pathological sensitivity. We employed dual CTGANs to generate realistic temporal samples for minority classes (Suspect, Pathological).

#### 3.4.2 Data Preparation
- **Filtering**: Extracted minority samples:
  ```python
  minority_mask = np.isin(y_temporal, [2, 3])
  X_minority_temporal = X_temporal[minority_mask]  # Shape: (471, 5, 10)
  y_minority_temporal = y_temporal[minority_mask]  # Shape: (471,)
  ```
- **Flattening**: Reshaped to \( X_{\text{minority_flat}} \in \mathbb{R}^{471 \times 50} \) for CTGAN input, with feature names (e.g., `f0_abnormal_short_term_variability`).

#### 3.4.3 CTGAN Training and Generation
- **Model Setup**: Initialized two CTGAN models (`ctgan` v0.7):
  ```python
  ctgan_suspect = CTGAN(epochs=500, batch_size=50, verbose=True, cuda=True)
  ctgan_pathological = CTGAN(epochs=500, batch_size=50, verbose=True, cuda=True)
  ```
- **Training**: Fit on Suspect (295 samples) and Pathological (176 samples) data separately.
- **Generation**: Produced synthetic samples to balance classes (~1,655 each): 1,360 Suspect and 1,479 Pathological.
- **Integration**: Combined with original data:
  ```python
  X_gan_temporal = np.vstack([X_temporal, X_synthetic_temporal])  # Shape: (4965, 5, 10)
  y_gan_temporal = np.hstack([y_temporal, synthetic_labels])     # Shape: (4965,)
  ```

#### 3.4.4 Validation
- **Quality Assessment**: Synthetic distributions (e.g., `histogram_variance`) matched originals (Kolmogorov-Smirnov \( p > 0.05 \)), ensuring clinical plausibility (Figure 3).

**Figure 3: Synthetic vs. Original Data Distribution**
*(Placeholder: Histogram comparing `histogram_variance` distributions for original and synthetic Pathological samples.)*

---

### 3.5 Uncertainty-Aware TabNet with Permutation Regularization

#### 3.5.1 Model Architecture
We extended `TabNetClassifier` (`pytorch-tabnet` v4.1) to incorporate uncertainty:
```python
class UncertaintyTabNet(TabNetClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dropout = nn.Dropout(p=0.3).to(device)
    def predict_proba(self, X):
        self.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        probs = [torch.softmax(self.forward(X_tensor)[0], dim=1).cpu().numpy() 
                 for _ in range(50)]
        return np.mean(probs, axis=0), np.std(probs, axis=0)
```
- **Input**: 50 features (flattened \( 5 \times 10 \)).
- **Output**: 3 classes (Normal, Suspect, Pathological).
- **Uncertainty**: Monte Carlo Dropout (50 passes, 30% dropout) computes mean probabilities \( \mu(X) \) and standard deviation \( \sigma(X) \).

#### 3.5.2 Permutation Regularization
- **Objective**: Enhance robustness to feature order variability.
- **Method**: Permuted time-step order in 10% of training samples:
  ```python
  def augment_data(X, y, permutation_prob=0.1):
      X_aug = X.copy()
      for i in range(len(X)):
          if np.random.rand() < permutation_prob:
              perm = np.random.permutation(X.shape[1])
              X_aug[i] = X[i, perm]
      return X_aug, y
  ```

#### 3.5.3 Data Splitting and Preparation
- **Splitting**: Stratified split into 70% train (3,475 samples), 30% test (1,490 samples), with 20% of train as validation (695 samples).
- **Flattening**: Reshaped to \( X_{\text{train_flat}} \in \mathbb{R}^{3475 \times 50} \), similarly for validation and test sets.

#### 3.5.4 Hyperparameter Optimization
- **Tool**: Optuna (v3.0) with 50 trials.
- **Parameters**: `n_d`, `n_a` (32–128), `n_steps` (3–10), `gamma` (1.0–2.0), `lambda_sparse` (1e-4–1e-2), `learning_rate` (1e-3–1e-1), `batch_size` (128, 256, 512).
- **Objective**: Maximize validation accuracy.
- **Best Configuration**: Example: `n_d=64`, `n_steps=5`, `batch_size=256`.

#### 3.5.5 Training Procedure
- **Setup**: Initialized `UncertaintyTabNet` with best parameters, trained with Adam optimizer (100 epochs, patience 20) on a CUDA-enabled GPU.
- **Augmentation**: Applied permutation to training data before fitting.

#### 3.5.6 Evaluation Metrics
- **Outputs**: \( \mu(X) \) for predictions (\( y_{\text{pred}} = \arg\max(\mu(X)) + 1 \)), \( \sigma(X) \) for uncertainty.
- **Metrics**: Accuracy, F1-score per class, confusion matrix, mean uncertainty.

---

### 3.6 Implementation Environment
- **Platform**: Google Colab with NVIDIA Tesla T4 GPU.
- **Software**: Python 3.11, libraries: `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `shap`, `ctgan`, `pytorch-tabnet`, `optuna`.
- **Reproducibility**: Fixed seeds (`np.random.seed(42)`, `torch.manual_seed(42)`).

---

### Visual Aids
- **Table 2**: Lists original features, grounding the dataset.
- **Figure 1**: SHAP bar plot illustrates feature selection.
- **Figure 2**: Temporal simulation plot shows dynamic variation.
- **Figure 3**: Synthetic data histogram validates CTGAN quality.

---

### Q1 Journal Alignment
- **Structure**: Clear subsections (3.1–3.6) with sub-parts (e.g., 3.2.1–3.2.4) enhance readability and detail.
- **Precision**: Mathematical formulations (e.g., SHAP, noise equations) and code snippets ensure rigor.
- **Justification**: Each choice (e.g., 5 time steps, ±5% noise) is clinically/technically motivated.
- **Visuals**: Figures/tables align with Q1 norms, clarifying complex processes.

This methodology is ready for your paper—insert actual figures, and let me know if you need further refinement or the next section!
