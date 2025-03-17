Below is a concise version of the "Dataset and Preprocessing" section, written in 3–4 paragraphs to cover all key aspects while fitting within the IEEE 15-page limit (targeting ~1 page at ~300 words per page, leaving room for the remaining sections). The content adheres to Q1 journal standards, ensuring technical rigor, clinical relevance, and reproducibility, while distilling the extended version into a well-written narrative.

---

### Dataset and Preprocessing
The Fetal Health Classification dataset, sourced from the UCI Machine Learning Repository [1], forms the foundation of this study, comprising 2,126 Cardiotocogram (CTG) samples from clinical monitoring sessions by Ayres-de-Campos et al. It includes 22 features capturing fetal heart rate (FHR) and uterine contraction signals, labeled into three classes: Normal (1,655 samples, 77.8%), Suspect (295 samples, 13.9%), and Pathological (176 samples, 8.3%), encoded as 1, 2, and 3. Features range from direct measurements (e.g., baseline value in bpm, accelerations in count/second) to variability metrics (e.g., abnormal_short_term_variability in %) and histogram-based statistics (e.g., histogram_mean, histogram_variance in bpm²), reflecting clinical indicators of fetal distress [2]. The dataset’s severe imbalance (Normal-to-Pathological ratio ~9.4:1) mirrors real-world prevalence (fetal distress ~1–3% [6]), challenging model training, while its static nature (single-snapshot) deviates from continuous CTG monitoring, motivating our temporal simulation approach (Section 3.3).

Preprocessing ensures data quality and compatibility with downstream models like LightGBM, CTGAN, and TabNet. Loaded into a Pandas DataFrame using Python 3.11, the dataset was structured as \(X \in \mathbb{R}^{2126 \times 21}\) (features) and \(y \in \{1, 2, 3\}^{2126}\) (labels). Integrity checks confirmed no duplicates or missing values, preserving clinical fidelity. MinMaxScaler normalized all 21 features to [0, 1] using \(x_{\text{scaled}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}\), fitted on the full dataset for consistency, mapping ranges like baseline value (106–160 bpm) and abnormal_short_term_variability (0–100%) to a uniform scale [7]. Post-normalization, feature distributions (e.g., histogram_variance) were verified to remain undistorted, ensuring numerical stability for gradient-based algorithms.

This preprocessing addresses challenges like class imbalance, feature redundancy (e.g., baseline value vs. histogram_mean), and scale disparities (e.g., bpm vs. percentage), laying a robust foundation for SHAP-driven feature selection (Section 3.2). The dataset’s clinical origin ensures relevance to obstetric practice, where accurate classification can guide critical interventions [2], while preprocessing enhances model performance and generalizability in detecting rare Pathological cases.

---

### References for This Section
- [1] Dua, D., & Graff, C. (2019). UCI Machine Learning Repository. *University of California, Irvine, School of Information and Computer Sciences*.
- [2] Ayres-De-Campos, D., Spong, C. Y., & Chandraharn, E. (2015). FIGO consensus guidelines on intrapartum fetal monitoring: Cardiotocography. *International Journal of Gynecology & Obstetrics*, 131, 13–24.
- [6] Alfirevic, Z., et al. (2017). Continuous cardiotocography (CTG) as a form of electronic fetal monitoring (EFM) for fetal assessment during labour. *Cochrane Database of Systematic Reviews*, (5).
- [7] Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

### Notes for Your Journal Paper
- **Word Count**: ~300 words, fitting within ~1 page at 300 words per page, leaving ~11–12 pages for remaining sections (Introduction, Related Work, Results, Conclusion) within the 15-page IEEE limit.
- **References**: Integrate into your paper’s reference list, ensuring consistent numbering.
- **Figures/Tables**: The extended version’s Table 2 (feature descriptions) and Figure 4 (normalization distributions) were omitted for brevity but can be added if space permits.

This concise section covers the dataset’s composition, clinical relevance, preprocessing steps, and their rationale, ensuring a strong foundation for the methodology while meeting Q1 journal standards. Let me know if you need further adjustments!
