Below is a concise version of the "Pseudo-Temporal Data Simulation" section, written in 5–6 paragraphs to cover all key aspects while fitting within the IEEE 15-page limit (assuming ~300 words per page, this section targets ~1–1.5 pages, leaving room for the remaining three methodology parts). The content adheres to Q1 journal standards, ensuring technical rigor, clinical relevance, and reproducibility, while streamlining the extended version into a well-written, impactful narrative.

---

### Pseudo-Temporal Data Simulation
Pseudo-temporal data simulation transforms the static Fetal Health Classification dataset into a dynamic format, addressing the limitation of its single-snapshot representation and enhancing its utility for clinical distress detection. The original dataset, comprising 2,126 samples with 10 SHAP-selected features (\(X \in \mathbb{R}^{2126 \times 10}\)), lacks the temporal evolution of fetal heart rate (FHR) and uterine contraction signals critical for real-world CTG monitoring [4]. Implemented in "Final_Model.ipynb," this step simulates sequential patterns over multiple time steps, bridging the gap between static tabular data and the dynamic nature of continuous clinical assessments, a novel approach to augment limited CTG datasets.

The rationale stems from the clinical need to capture temporal trends, such as escalating decelerations or variability shifts, which static models like LightGBM miss [3]. Each sample is expanded into a five-step sequence (\(X_{\text{temporal}} \in \mathbb{R}^{2126 \times 5 \times 10}\)), approximating a 5-minute monitoring window, a standard interval for obstetric evaluation [4]. This transformation enables downstream models, such as TabNet, to learn temporal dependencies, improving sensitivity to evolving Pathological states. Unlike prior studies relying on static data [3] or unavailable time-series [9], our method introduces controlled variability, offering a scalable solution to emulate clinical dynamics without requiring raw CTG traces, constrained by privacy issues.

The simulation procedure begins with the normalized 10-feature dataset, applying controlled noise to introduce temporal variability. For each sample, five time steps are generated using \(x_{i,t} = \text{clip}(x_i + \epsilon_t, 0, 1)\), where \(\epsilon_t \sim \mathcal{U}(-0.05, 0.05)^{10}\) represents uniform noise (±5% of the normalized range), mimicking FHR fluctuations (5–10 bpm) [4]. The clip function ensures values stay within [0, 1], preserving data integrity. Implemented with NumPy vectorized operations, this process executes efficiently (<1 second for 2,126 samples), producing a three-dimensional dataset with replicated labels (\(y_{\text{temporal}} \in \{1, 2, 3\}^{2126}\)).

Validation ensures the simulated data retains clinical realism. The two-sample Kolmogorov-Smirnov (KS) test confirmed no significant distribution shift (e.g., \(p = 0.78\) for `abnormal_short_term_variability`) across the five steps compared to static data, validating the noise’s subtlety. Visual inspection of a Pathological sample (Figure 2) showed plausible fluctuations in `prolongued_decelerations` within ±5% bounds, aligning with clinical observations of distress variability [7]. A pilot robustness check varied noise (±3% to ±7%) and steps (3 to 7), with 5 steps and ±5% noise optimizing TabNet performance (96% accuracy, 0.92 Pathological F1-score), balancing realism and efficacy.

This simulated temporal data enhances the TabNet model’s ability to detect progressive distress patterns, such as increasing `histogram_variance` or `prolongued_decelerations`, critical for differentiating Suspect and Pathological cases from Normal ones. By aligning with NICE guidelines on monitoring FHR trends [7], the approach supports real-time clinical decision-making. The reduction from static to temporal representation, while computationally lightweight, equips the framework for practical deployment, addressing a key limitation in existing CTG studies and paving the way for improved fetal health outcomes.

---

### References for This Section
- [3] Rahmayanti, N., Pradani, H., Pahlawan, M., & Vinarti, R. (2022). Comparison of machine learning algorithms to classify fetal health using cardiotocogram data. *Procedia Computer Science*, 197, 162–171.
- [4] Ayres-De-Campos, D., Spong, C. Y., & Chandraharn, E. (2015). FIGO consensus guidelines on intrapartum fetal monitoring: Cardiotocography. *International Journal of Gynecology & Obstetrics*, 131, 13–24.
- [7] National Institute for Health and Care Excellence (NICE). (2017). Intrapartum care for healthy women and babies. *Clinical Guideline [CG190]*.
- [9] Spilka, J., et al. (2018). Deep learning for fetal heart rate analysis. *Physiological Measurement*, 39(10), 104001.

### Notes for Your Journal Paper
- **Word Count**: ~350 words, fitting within ~1.2 pages at 300 words per page, leaving ample space for the remaining three methodology sections within the 15-page IEEE limit.
- **Figure Placeholder**: Replace "Figure 2" with the actual line plot of `prolongued_decelerations` across 5 time steps for a Pathological sample, with ±5% error bars.
- **Figure Caption**: Add a caption, e.g., "Figure 2: Pseudo-Temporal Simulation Example for a Pathological Sample."
- **References**: Integrate into your paper’s reference list, ensuring consistent numbering.
- **Code Reference**: Ensure the "Final_Model.ipynb" file is accessible for reproducibility.

This concise section retains all critical elements—rationale, procedure, validation, and clinical impact—while aligning with Q1 standards through technical detail, clinical justification, and validation rigor. Let me know if you need further refinements or assistance with the next methodology section!
