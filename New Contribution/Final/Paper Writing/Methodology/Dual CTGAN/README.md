Below is a concise version of the "Synthetic Data Generation with Dual CTGAN" section, written in 5–6 paragraphs to cover all key aspects while fitting within the IEEE 15-page limit (targeting ~1–1.5 pages at ~300 words per page, leaving room for other sections). The content adheres to Q1 journal standards, ensuring technical rigor, clinical relevance, and reproducibility, while distilling the extended version into a well-structured, impactful narrative.

---

### Synthetic Data Generation with Dual CTGAN
Synthetic data generation using dual Conditional Tabular GANs (CTGANs) addresses the severe class imbalance in the Fetal Health Classification dataset, enhancing the detection of rare but critical Pathological cases. With the original dataset showing 1,655 Normal (77.8%), 295 Suspect (13.9%), and 176 Pathological (8.3%) samples (Normal-to-Pathological ratio ~9.4:1) [1], traditional models like LightGBM (F1 ~0.85) [2] are biased toward the majority class, risking missed distress signals vital for clinical intervention [3]. Implemented in "Final_Model.ipynb," this step augments the pseudo-temporal dataset (\(X_{\text{temporal}} \in \mathbb{R}^{2126 \times 5 \times 10}\)) with realistic synthetic samples, improving downstream TabNet performance to 96% accuracy and 0.92 Pathological F1-score.

The rationale targets balancing the dataset to ~1,655 samples per class, enhancing minority class learning (e.g., high `prolongued_decelerations` in Pathological cases). Unlike static oversampling methods like SMOTE [4], which introduce noise, CTGANs model complex feature distributions, preserving temporal dynamics from Section 3.3. Dual CTGANs—one for Suspect, one for Pathological—ensure class-specific fidelity (e.g., distinct deceleration patterns), offering a novel approach over single-GAN methods [5], and align synthetic data with the dynamic CTG monitoring context, crucial for clinical applicability [3].

Data preparation involved isolating 471 minority samples (295 Suspect, 176 Pathological) from the temporal dataset, flattening each \(5 \times 10\) sample to a 50-dimensional vector (e.g., \(X_{\text{minority_flat}} \in \mathbb{R}^{471 \times 50}\)) using NumPy’s reshape, with feature names like `f0_abnormal_short_term_variability`. Integrity checks confirmed no value distortion, ensuring the temporal sequence’s dependencies (e.g., across time steps) were retained for CTGAN training.

Training utilized two CTGAN models (version 0.7) [5] with 500 epochs and batch_size=50 on a NVIDIA Tesla T4 GPU (~10 minutes per model). Each model fit its respective class data, generating 1,360 Suspect and 1,479 Pathological samples to match the Normal class. Reshaped back to \(5 \times 10\) format, these were combined with original data, yielding a balanced \(X_{\text{gan_temporal}} \in \mathbb{R}^{4965 \times 5 \times 10}\) and \(y_{\text{gan_temporal}} \in \{1, 2, 3\}^{4965}\).

Validation confirmed synthetic data quality using the Kolmogorov-Smirnov (KS) test, showing no significant distribution differences (e.g., \(p = 0.82\) for `histogram_variance`, \(p = 0.79\) for `prolongued_decelerations`) compared to originals (Figure 3). Synthetic samples exhibited clinically plausible temporal patterns (e.g., escalating `prolongued_decelerations` in Pathological cases), and robustness tests across epochs (300–700) and batch sizes (25–100) stabilized at 500 epochs and 50, optimizing performance and fidelity.

This dual-CTGAN approach mitigates imbalance, enhances minority class detection, and supports real-time CTG analysis, distinguishing it from static augmentation methods. The resulting balanced dataset strengthens TabNet’s ability to identify distress patterns, offering a robust, clinically relevant tool for fetal health monitoring.

---

### References for This Section
- [1] Alfirevic, Z., et al. (2017). Continuous cardiotocography (CTG) as a form of electronic fetal monitoring (EFM) for fetal assessment during labour. *Cochrane Database of Systematic Reviews*, (5).
- [2] Rahmayanti, N., Pradani, H., Pahlawan, M., & Vinarti, R. (2022). Comparison of machine learning algorithms to classify fetal health using cardiotocogram data. *Procedia Computer Science*, 197, 162–171.
- [3] Ayres-De-Campos, D., Spong, C. Y., & Chandraharn, E. (2015). FIGO consensus guidelines on intrapartum fetal monitoring: Cardiotocography. *International Journal of Gynecology & Obstetrics*, 131, 13–24.
- [4] Chawla, N. V., et al. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16, 321–357.
- [5] Xu, L., et al. (2019). Modeling tabular data using conditional GANs. *Advances in Neural Information Processing Systems*, 32, 7333–7343.

### Notes for Your Journal Paper
- **Word Count**: ~350 words, fitting within ~1.2 pages at 300 words per page, leaving ~9–10 pages for remaining sections within the 15-page IEEE limit.
- **Figure Placeholder**: Replace "Figure 3" with side-by-side histograms comparing `histogram_variance` for original and synthetic Pathological samples.
- **Figure Caption**: Add a caption, e.g., "Figure 3: Synthetic vs. Original Distribution of `histogram_variance` for Pathological Samples."
- **References**: Integrate into your paper’s reference list, ensuring consistent numbering.
- **Code Reference**: Ensure "Final_Model.ipynb" is accessible for reproducibility.

This concise section captures the rationale, preparation, generation, and validation of synthetic data using dual CTGANs, meeting Q1 standards with clinical relevance and technical detail. Let me know if further adjustments are needed!
