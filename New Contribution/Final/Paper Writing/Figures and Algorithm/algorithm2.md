Below is a polished and impactful "Algorithm" section tailored for your Q1 journal paper. It follows the flow you specified—dataset, SHAP analysis, feature selection, Temporal CTGAN for class balance, Optuna hyperparameter tuning, and the custom TabNet with permutation regularization and attention mask details. The presentation is structured, concise, and mirrors the style of high-impact Q1 journal papers, emphasizing novelty and rigor. I’ve formatted it as a step-by-step algorithm with clear inputs, outputs, and procedural details to underscore its significance.

---

### Algorithm Section: Temporal CTGAN + Uncertainty-Aware TabNet Framework

To address the challenges of class imbalance, temporal dynamics, and clinical interpretability in fetal health classification using Cardiotocogram (CTG) data, we propose a novel **Temporal CTGAN + Uncertainty-Aware TabNet** framework. This algorithm integrates advanced feature selection, synthetic data generation, and an optimized deep learning model with uncertainty quantification. The methodology is outlined in Algorithm 1, detailing the steps from data preprocessing to model deployment.

---

#### Algorithm 1: Temporal CTGAN + Uncertainty-Aware TabNet for Fetal Health Classification

**Input**:  
- \( D \): Raw CTG dataset with 2,126 samples, 21 features, and target \( y \) (fetal health: 1 = Normal, 2 = Suspect, 3 = Pathological).  
- \( T \): Number of simulated time steps (set to 5).  
- \( N_{\text{syn}} \): Desired synthetic samples per minority class (1,360 for Suspect, 1,479 for Pathological).  
- \( H \): Hyperparameter search space for TabNet optimization.

**Output**:  
- Trained model \( M \) with predictions and uncertainty estimates.  
- Selected feature subset \( F_{\text{sel}} \).  
- Balanced temporal dataset \( D_{\text{bal}} \).

**Procedure**:  
1. **Dataset Preprocessing and SHAP Analysis**  
   - Load \( D \) and verify integrity (no missing values, 2,126 × 22 matrix).  
   - Split into features \( X \) (21 columns) and target \( y \).  
   - Train a LightGBM classifier \( M_{\text{LGBM}} \) on \( X \) and \( y \) with SMOTE-balanced training data (for analysis only).  
   - Compute SHAP values \( S = \{s_{i,j,k}\} \) for each feature \( i \), sample \( j \), and class \( k \) using TreeExplainer on \( M_{\text{LGBM}} \).  
   - Aggregate SHAP statistics: mean \( \mu_{i,k} = \text{mean}(s_{i,:,k}) \) and mean absolute \( \mu_{\text{abs},i,k} = \text{mean}(|s_{i,:,k}|) \) per feature and class.  
   - Rank features by \( \mu_{\text{abs},i,k} \) for Pathological class (k = 3) to prioritize clinical relevance.

2. **Feature Selection**  
   - Select top-performing features \( F_{\text{sel}} \) (10 features) based on \( \mu_{\text{abs},i,k} \), dropping 11 low-impact features (e.g., `fetal_movement`, `severe_decelerations`, `histogram_min`).  
   - Update \( X_{\text{sel}} \) as \( D[:, F_{\text{sel}}] \), retaining 2,126 × 10 matrix.

3. **Temporal Data Simulation**  
   - Normalize \( X_{\text{sel}} \) to [0, 1] using MinMaxScaler, yielding \( X_{\text{norm}} \).  
   - For each sample \( x_i \in X_{\text{norm}} \):  
     - Simulate \( T = 5 \) time steps by adding uniform noise \( \epsilon_t \sim U(-0.05, 0.05) \) to \( x_i \), clipped to [0, 1].  
     - Form temporal sequence \( x_i^{\text{temp}} = [x_i + \epsilon_1, \ldots, x_i + \epsilon_5] \).  
   - Output temporal dataset \( X_{\text{temp}} \) (2,126 × 5 × 10) and \( y_{\text{temp}} \) (2,126).

4. **Temporal CTGAN for Class Balance**  
   - Filter minority samples: \( X_{\text{min}} = X_{\text{temp}}[y_{\text{temp}} \in \{2, 3\}] \), \( y_{\text{min}} = y_{\text{temp}}[y_{\text{temp}} \in \{2, 3\}] \) (471 samples).  
   - Flatten \( X_{\text{min}} \) to 2D (471 × 50), naming features \( \{f_{t,col}\} \) for \( t = 1, \ldots, 5 \), \( col \in F_{\text{sel}} \).  
   - Split into Suspect (\( D_{\text{sus}} \), 295 samples) and Pathological (\( D_{\text{path}} \), 176 samples).  
   - Train two CTGAN models:  
     - \( G_{\text{sus}} \) on \( D_{\text{sus}} \) (epochs = 500, batch_size = 50, CUDA-enabled).  
     - \( G_{\text{path}} \) on \( D_{\text{path}} \) (same parameters).  
   - Generate synthetic data:  
     - \( D_{\text{syn,sus}} = G_{\text{sus}}.sample(1,360) \) (balancing to 1,655).  
     - \( D_{\text{syn,path}} = G_{\text{path}}.sample(1,479) \) (balancing to 1,655).  
   - Concatenate \( D_{\text{syn}} = [D_{\text{syn,sus}}, D_{\text{syn,path}}] \), reshape to (2,839 × 5 × 10).  
   - Combine: \( X_{\text{bal}} = [X_{\text{temp}}, X_{\text{syn}}] \) (4,965 × 5 × 10), \( y_{\text{bal}} = [y_{\text{temp}}, y_{\text{syn}}] \) (4,965).

5. **Data Splitting and Preparation**  
   - Split \( X_{\text{bal}} \), \( y_{\text{bal}} \) into train (70%), test (30%) with stratification.  
   - Further split train into train_final (80%) and validation (20%).  
   - Flatten temporal data: \( X_{\text{train,flat}} \) (e.g., 2,780 × 50), \( X_{\text{valid,flat}} \), \( X_{\text{test,flat}} \).

6. **Optuna Hyperparameter Tuning**  
   - Define search space \( H \):  
     - \( n_d, n_a \in [32, 128] \), \( n_{\text{steps}} \in [3, 10] \), \( \gamma \in [1.0, 2.0] \),  
     - \( \lambda_{\text{sparse}} \in [10^{-4}, 10^{-2}] \) (log), \( \text{lr} \in [10^{-3}, 10^{-1}] \) (log),  
     - \( \text{batch_size} \in \{128, 256, 512\} \).  
   - Objective: Maximize accuracy on \( X_{\text{valid,flat}} \) using TabNet with 50 trials.  
   - Output best parameters \( H^* \).

7. **Custom TabNet with Permutation and Attention Mask**  
   - Define \( M = \text{UncertaintyTabNet} \):  
     - Input dim: 50 (5 × 10), output dim: 3.  
     - Parameters: \( n_d, n_a, n_{\text{steps}}, \gamma, \lambda_{\text{sparse}}, \text{lr} \) from \( H^* \).  
     - Mask: `sparsemax` for sparse attention.  
     - Dropout (p = 0.3) for uncertainty via Monte Carlo sampling.  
   - Augment \( X_{\text{train,flat}} \): Randomly permute feature order (10% probability) to regularize attention.  
   - Train \( M \) on augmented \( X_{\text{train,flat}} \), \( y_{\text{train}} \) (max_epochs = 100, patience = 20, batch_size from \( H^* \)).  
   - Predict with uncertainty:  
     - Run 50 forward passes with dropout enabled.  
     - Output mean probabilities \( P_{\text{mean}} \) and std \( P_{\text{std}} \).  
   - Final predictions: \( \hat{y} = \arg\max(P_{\text{mean}}) + 1 \).

8. **Evaluation**  
   - Compute accuracy, precision, recall, F1 on \( X_{\text{test,flat}} \), \( y_{\text{test}} \).  
   - Assess mean uncertainty \( \mu_{\text{unc}} = \text{mean}(\max(P_{\text{std}})) \).  
   - Return \( M \), \( F_{\text{sel}} \), \( D_{\text{bal}} \).

---

### Discussion
This algorithm integrates several novel components:
- **SHAP-Driven Feature Selection**: Reduces dimensionality from 21 to 10 features, enhancing model focus on clinically relevant signals (e.g., variability, decelerations).
- **Temporal CTGAN**: Generates realistic, class-specific temporal data, overcoming limitations of static oversampling methods like SMOTE.
- **Uncertainty-Aware TabNet**: Combines attention-based modeling with permutation regularization and Monte Carlo dropout, offering both high accuracy (96%) and interpretable uncertainty (mean 0.2252).
- **Optimization**: Optuna ensures robust hyperparameter selection, maximizing performance on balanced, temporal data.

The framework outperforms prior approaches (e.g., LightGBM, CNN) by addressing class imbalance, temporal dynamics, and clinical trust, making it a significant advancement in fetal health monitoring.

---

### Formatting Notes
- **Mathematical Notation**: Used to enhance clarity and rigor (e.g., \( \mu_{i,k} \), \( X_{\text{bal}} \)), common in Q1 papers.
- **Structured Steps**: Each phase is numbered and detailed, mirroring algorithmic descriptions in top-tier journals.
- **Impact Emphasis**: Highlights novelty (CTGAN, uncertainty, permutation) and results (96% accuracy, 0.2252 uncertainty) to impress reviewers.

Feel free to adjust specifics (e.g., exact feature names in step 2, or tuning ranges in step 6) based on your final outputs. This should slot seamlessly into your methodology section—let me know if you need tweaks! What’s your next step for the paper?
