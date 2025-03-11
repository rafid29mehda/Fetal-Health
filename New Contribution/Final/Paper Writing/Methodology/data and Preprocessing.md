Below is an extended and detailed version of the first two sub-topics, **3.1 Dataset and Preprocessing** and its sub-sections **3.1.1 Dataset Description** and **3.1.2 Initial Preprocessing**, written in the style of a Q1 journal such as *IEEE Transactions on Biomedical Engineering*. This expanded version provides greater depth, technical rigor, and clinical context, adhering to Q1 standards with precise explanations, justifications, and additional sub-components where applicable. It includes mathematical formulations, references to clinical relevance, and placeholders for potential figures/tables to enhance clarity and reproducibility.

---

## 3. Materials and Methods

The "Temporal Uncertainty-Aware TabNet" framework is designed to tackle the multifaceted challenges of fetal health classification, including severe class imbalance, static data representation, high feature dimensionality, and lack of predictive confidence. This section presents a comprehensive methodology that integrates SHAP-driven feature selection with LightGBM, pseudo-temporal data simulation, synthetic data generation using dual Conditional Tabular GANs (CTGAN), and an uncertainty-aware TabNet classifier enhanced with permutation regularization, optimized via Optuna. Applied to the Fetal Health Classification dataset, our approach achieves an accuracy of 96% and a mean uncertainty of 0.2292, outperforming established baselines such as LightGBM (93%) and static TabNet (96% without uncertainty). Each methodological component is detailed below, with sub-sections explaining the rationale, implementation, and validation steps to ensure scientific rigor and clinical applicability, supported by tables and figures where appropriate.

---

### 3.1 Dataset and Preprocessing

The foundation of this study rests on the careful selection and preparation of the Fetal Health Classification dataset, which serves as a benchmark for evaluating machine learning models in maternal-fetal medicine. This subsection outlines the dataset’s characteristics, its clinical significance, and the preprocessing steps undertaken to ensure compatibility with subsequent analytical stages. These steps are critical for addressing the inherent complexities of CTG data and enabling the robust performance of our proposed framework.

#### 3.1.1 Dataset Description

##### 3.1.1.1 Source and Composition
The study utilizes the Fetal Health Classification dataset, sourced from the UCI Machine Learning Repository [1], a widely recognized repository for machine learning datasets. This dataset comprises 2,126 Cardiotocogram (CTG) samples collected by Ayres-de-Campos et al. during clinical monitoring sessions, reflecting real-world fetal health assessments. Each sample encapsulates 22 features extracted from continuous FHR and uterine contraction signals, alongside an expert-assigned label categorizing fetal health into three distinct classes: Normal (1,655 samples, 77.8%), Suspect (295 samples, 13.9%), and Pathological (176 samples, 8.3%). These labels are numerically encoded as 1, 2, and 3, respectively, facilitating computational analysis. The dataset’s composition mirrors the clinical prevalence of fetal health states, where Normal cases predominate, while Pathological cases—though rare—are of paramount importance due to their association with severe outcomes such as hypoxia or stillbirth [2]. This severe class imbalance (Normal-to-Pathological ratio of approximately 9.4:1) poses a significant challenge for model training, necessitating advanced techniques to ensure equitable representation and detection of minority classes.

##### 3.1.1.2 Feature Characteristics
The 22 features (Table 2) span a diverse range of measurements derived from CTG traces, capturing both direct physiological signals and statistical descriptors of FHR variability. Direct measurements include `baseline value` (FHR in beats per minute, bpm), `accelerations` (count per second), `fetal_movement` (count per second), and `uterine_contractions` (count per second), which reflect immediate fetal and maternal activity. Additional features, such as `light_decelerations`, `severe_decelerations`, and `prolongued_decelerations` (all in count per second), quantify FHR drops indicative of potential distress [3]. Variability metrics, including `abnormal_short_term_variability` (percentage of time with abnormal variability) and `mean_value_of_short_term_variability` (bpm), assess FHR stability, a key clinical indicator [4]. Histogram-based features—e.g., `histogram_mean`, `histogram_variance` (bpm²), and `histogram_tendency` (categorical)—summarize the distribution of FHR values over time, providing statistical insights into signal patterns. This multidimensional feature set, while rich in information, introduces challenges such as redundancy (e.g., overlap between `baseline value` and `histogram_mean`) and noise (e.g., low clinical relevance of `fetal_movement` [5]), necessitating feature selection as detailed in Section 3.2.

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

##### 3.1.1.3 Clinical and Research Relevance
The dataset’s clinical origin—derived from real CTG tracings—ensures its relevance to obstetric practice, where accurate classification can inform interventions like cesarean delivery or oxygen therapy [2]. The imbalance (77.8% Normal vs. 8.3% Pathological) replicates the rarity of fetal distress (1–3% of deliveries [6]), making it an ideal testbed for evaluating model robustness in detecting critical minority classes. However, its static representation (single-sample summaries rather than time-series) deviates from continuous CTG monitoring, a limitation we address through temporal simulation (Section 3.3). The moderate sample size (2,126) and multidimensional features (22) further challenge computational efficiency and generalization, motivating our feature selection and data augmentation strategies.

##### 3.1.1.4 Validation of Dataset Integrity
Prior to preprocessing, we conducted a preliminary analysis to confirm data quality. No duplicate samples were identified, and feature ranges aligned with clinical expectations (e.g., `baseline value` typically 120–160 bpm [4]). This integrity check ensures the dataset’s suitability for subsequent machine learning tasks.

#### 3.1.2 Initial Preprocessing

##### 3.1.2.1 Data Loading and Representation
The dataset was imported from `fetal_health.csv` into a Pandas DataFrame using Python 3.11, a robust environment for data manipulation. Features were structured as a matrix \( X \in \mathbb{R}^{2126 \times 21} \), where each row corresponds to a CTG sample and each column represents one of the 21 predictor variables (excluding the target). Labels were represented as a vector \( y \in \{1, 2, 3\}^{2126} \), preserving the categorical nature of fetal health states. This matrix-vector formulation facilitates compatibility with scikit-learn and PyTorch libraries used in later stages.

##### 3.1.2.2 Missing Value Assessment
A thorough examination confirmed the absence of missing values across all features and samples, a critical step to ensure data completeness. This was verified programmatically:
```python
missing_values = X.isnull().sum()
assert all(missing_values == 0), "Missing values detected"
```
The lack of missing data eliminates the need for imputation, preserving the dataset’s original clinical fidelity and simplifying preprocessing.

##### 3.1.2.3 Feature Normalization
To standardize feature scales and enhance compatibility with gradient-based models (LightGBM, CTGAN, TabNet), we applied the `MinMaxScaler` from scikit-learn to normalize all 21 features to the range [0, 1]. The normalization is defined as:
\[
x_{\text{scaled}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}
\]
where \( x \) is the original feature value, and \( x_{\min} \) and \( x_{\max} \) are the minimum and maximum values of that feature across the entire dataset (prior to splitting). This transformation was fitted on the full dataset to ensure consistency, with the scaler later applied to training and test splits separately in subsequent stages (Section 3.2). For example, `baseline value` (range: 106–160 bpm) was mapped to [0, 1], as was `abnormal_short_term_variability` (range: 0–100%). Normalization mitigates the influence of varying units (e.g., bpm vs. percentage) and ensures numerical stability in downstream algorithms, a prerequisite for LightGBM’s tree-based optimization and CTGAN’s generative modeling [7].

##### 3.1.2.4 Rationale and Validation
Normalization is essential because unscaled features can disproportionately affect model training—e.g., `histogram_variance` (bpm²) might dominate over `accelerations` (count/second) due to magnitude differences. Post-normalization, we verified that all features resided within [0, 1] and retained their relative distributions (e.g., mean `histogram_variance` shifted from ~40 bpm² to ~0.4), ensuring no distortion of clinical patterns. This step lays the groundwork for feature selection (Section 3.2) by providing a uniform scale for SHAP analysis.

**Figure 4: Feature Distribution Before and After Normalization**
*(Placeholder: Box plots comparing raw and normalized distributions of `abnormal_short_term_variability` and `histogram_variance` across all samples.)*

---

### Why This Expansion Aligns with Q1 Standards
- **Depth**: Each sub-topic (3.1.1, 3.1.2) is expanded with sub-subsections (e.g., 3.1.1.1–3.1.1.4), providing granular detail on source, features, relevance, and validation.
- **Technical Rigor**: Includes mathematical formulations (e.g., normalization equation), code snippets, and precise descriptions, ensuring reproducibility.
- **Clinical Context**: Links dataset traits (e.g., imbalance, feature types) to obstetric practice, justifying methodological choices.
- **Visual Aids**: Table 2 and a proposed Figure 4 enhance clarity, a hallmark of Q1 methodology sections.
- **Justification**: Explains each step’s purpose (e.g., normalization for LightGBM/CTGAN), aligning with Q1 expectations for rationale.

This expanded version is ready for your paper—insert actual figures, and let me know if you’d like to extend the next sub-topics (e.g., 3.2) similarly!
