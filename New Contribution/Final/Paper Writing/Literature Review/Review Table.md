Below is an expanded version of "Table 1. Recent Works on CTG Classification," incorporating all 23 papers from your Literature Review. The table follows the same format as the sample (Authors, Method, Contribution, Limitation), ensuring conciseness while capturing the diversity of approaches, their key contributions, and limitations. This comprehensive table aligns with your Q1 journal goal by showcasing the breadth of prior work and highlighting gaps your “Temporal CTGAN + Uncertainty-Aware TabNet” framework addresses—temporal dynamics, advanced imbalance handling, uncertainty quantification, and interpretability. Each entry is succinct, fitting a high-impact journal’s need for clarity and depth.

---

### Table 1. Recent Works on CTG Classification

| **Authors**                        | **Method**                          | **Contribution**                                           | **Limitation**                                                                 |
|------------------------------------|-------------------------------------|-----------------------------------------------------------|--------------------------------------------------------------------------------|
| Rahmayanti et al. (2022)           | LightGBM + Upsampling              | 99% accuracy with outlier removal and grid search.        | Static data, upsampling bias, no temporal dynamics or uncertainty.             |
| Regmi (2022)                       | TabNet + PCA/LDA                   | 94.36% accuracy with attention-based feature reduction.   | Static, no imbalance correction, lacks uncertainty or temporal context.        |
| Gaddam et al. (2022)               | XGBoost + SHAP (RF)                | 94% accuracy, SHAP highlights key predictors.             | Static, no imbalance handling, no uncertainty, SHAP limited to RF.             |
| Dwivedi et al. (2022)              | AutoML (LightGBM) + SMOTE          | 95.61% accuracy with automated optimization.              | Static, noisy SMOTE samples, lacks uncertainty or temporal modeling.           |
| Marvin & Alam (2022)               | LightGBM + Upsampling + ELI5       | 99% accuracy, ELI5 feature importance.                    | Static, upsampling bias, limited interpretability, no uncertainty.             |
| Nasir et al. (2023)                | Federated ML + KNN                 | 99.06% accuracy with blockchain security.                 | Static, possible overfitting, no interpretability or uncertainty.              |
| Innab et al. (2022)                | LightGBM + SMOTE + SHAP            | 99.89% accuracy, SHAP interpretability.                   | Static, SMOTE noise, no uncertainty, risks overfitting.                        |
| Sato et al. (2022)                 | CNN (FHR-only)                     | AUC 0.896 for late deceleration detection.                | Small dataset, static 3-min snippets, no uncertainty, LD focus only.           |
| Arslan (2022)                      | TabNet + Random OverSampling       | 94% avg accuracy with Optuna-tuned attention.             | Static, oversampling noise, inconsistent generalization, no uncertainty.       |
| Ilham et al. (2022)                | CFCM-SMOTE + CART                  | 99.84% accuracy with outlier and imbalance handling.      | Static, complex implementation, SMOTE noise, no uncertainty.                   |
| Abiyev et al. (2022)               | T2-Fuzzy Neural Network            | 96.6% accuracy with fuzzy logic for uncertainty.          | Static, no imbalance correction, all features retained, no explicit uncertainty quantification. |
| Kuzu & Santur (2022)               | Ensemble (XGBoost) + Class Weighting | >99.5% accuracy with boosting and weighting.            | Static, weighting insufficient, no uncertainty, possible overfitting.          |
| Mushtaq & Veningston (2022)        | DNN + SHAP                         | 99% accuracy with batch normalization and SHAP.           | Static, no imbalance correction, lacks uncertainty quantification.             |
| Hussain et al. (2022)              | AlexNet-SVM + Upsampling           | 99.72% accuracy with transfer learning.                   | Static 30-min data, upsampling bias, no interpretability or uncertainty.       |
| Stow (2022)                        | CNN + RandomOverSampling           | 99.99% accuracy with feature selection via RF.            | Static, possible overfitting, oversampling bias, no uncertainty or SHAP.       |
| Georgoulas et al. (2006)           | Wavelet + SVM                      | 88.75% accuracy with wavelet features for pH risk.        | Small dataset, static final segments, no uncertainty or interpretability.      |
| Spilka et al. (2017)               | Sparse SVM                         | 75% effective accuracy with sparse feature selection.     | Static 20-min window, small acidotic sample, no uncertainty, overfitting risk. |
| Das et al. (2020)                  | SVM/RF + SMOTE                     | 97.4–98% (stage 1), 89.3–90.6% (stage 2) accuracy.       | Small dataset, static features, SMOTE noise, no uncertainty or temporal flow.  |
| Piri et al. (2022a)                | MOGA-CD + XGBoost                  | 94% accuracy with genetic feature selection.              | Static, no imbalance correction, no uncertainty, computationally expensive.    |
| Piri & Mohapatra (2022b)          | Associative Classification (CBA)   | 84.02% accuracy with rule-based feature selection.       | Static, no imbalance handling, lower accuracy, lacks uncertainty or SHAP.      |
| Fasihi et al. (2022)               | Shallow 1-D CNN                    | 97.46% accuracy with minimal complexity.                  | Static, no imbalance handling, no uncertainty or advanced interpretability.    |
| Miao & Miao (2022)                 | DNN (Multiclass)                   | 88.02% accuracy for 10 morphologic patterns.              | Static, imbalanced classes, no uncertainty, lower accuracy vs. binary models.  |
| Johnraja Jebadurai et al. (2023)   | Filtering + kNN/SVM                | 92% accuracy with statistical feature selection.         | Static, no imbalance correction, no uncertainty, limited to filtering methods. |

---

### Notes on Table Design
- **Comprehensiveness**: All 23 papers are included, reflecting the full scope of your Literature Review.
- **Grouping Reflection**: While not explicitly grouped in the table, entries align with the review’s categories (traditional ML, ensemble/boosting, DL, specialized methods), maintaining a logical flow.
- **Alignment with Your Work**: Limitations emphasize static data, crude imbalance handling, and lack of uncertainty/interpretability—gaps your temporal simulation, CTGAN, Uncertainty-Aware TabNet (96%, 0.2252 uncertainty), and SHAP address.
- **Conciseness**: Each entry is brief (15–20 words per column), fitting a journal table while retaining critical details.
- **Citation Style**: Authors and years are listed; adjust to your journal’s format (e.g., numbered references [1], [2]) if needed.

### Next Steps
- **Refinement**: If you need fewer entries (e.g., 10–15) for space, I can select the most impactful ones. Let me know your target length or journal guidelines.
- **Formatting**: I can adjust to LaTeX, Word table, or another format if specified.
- **Integration**: Ready to embed this into your paper or refine further based on feedback.

What’s your next move? More tweaks to the table, or on to another section?
