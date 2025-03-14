Below is a polished and professionally written section tailored for a Q1 journal paper, based on the analysis of the SHAP summary plots for the three fetal health classes (Normal, Suspect, and Pathological). This text is structured to fit into Section 3.2.3.3 (Advanced Visualization) of your paper, providing a detailed, scientifically rigorous description suitable for publication. You can copy and paste this directly into your manuscript, adjusting references or figure numbers as needed.

---

### 3.2.3.3 Advanced Visualization

To enhance the interpretability of the LightGBM model developed for fetal health classification, advanced SHAP (SHapley Additive exPlanations) visualizations were employed, offering class-specific insights into feature importance. Three SHAP summary plots were generated, one for each class—Normal, Suspect, and Pathological—utilizing the test set data (426 samples) and the full set of 21 features derived from cardiotocography (CTG) signals. These visualizations, presented in Figures 1a–c, elucidate the impact of each feature on the model’s output for the respective classes, providing a robust foundation for clinical validation.

**Figure 1a: SHAP Summary Plot for Pathological** reveals the most influential features driving the prediction of pathological fetal states. The plot highlights `histogram_mean` as the top contributor, with SHAP values ranging from approximately 1 to 2, where high feature values (depicted in red) significantly increase the likelihood of a Pathological classification. This is followed by `abnormal_short_term_variability` (SHAP ~1 to 1.5) and `prolongued_decelerations` (SHAP ~0.5 to 1), both exhibiting positive impacts when their values are elevated. Conversely, features such as `histogram_number_of_zeroes` and `histogram_tendency` demonstrate minimal or negative SHAP values, indicating negligible influence on this class. These findings align with clinical observations, where elevated short-term variability and prolonged decelerations are established markers of fetal distress [1].

**Figure 1b: SHAP Summary Plot for Normal** underscores the features most indicative of healthy fetal conditions. `Accelerations` emerge as the dominant feature, with SHAP values ranging from 1 to 2, where high values (red) strongly push the model toward a Normal classification. This is consistent with the physiological significance of accelerations as a sign of fetal well-being. `Abnormal_short_term_variability` follows, with low values (blue) associated with negative SHAP, suggesting that reduced variability supports Normal status. Notably, `prolongued_decelerations` exhibit negative SHAP values (approximately -1 to -2) with high values, indicating their detrimental effect on Normal predictions. This pattern corroborates the clinical expectation that the absence of decelerations and presence of accelerations are critical for normal fetal health assessment.

**Figure 1c: SHAP Summary Plot for Suspect** delineates the feature dynamics associated with the intermediate Suspect class. `Percentage_of_time_with_abnormal_long_term_variability` ranks highest, with SHAP values from 1 to 2, where high values (red) significantly influence Suspect predictions. This reflects a transitional state where abnormal long-term variability may signal potential issues. `Accelerations` contribute positively (SHAP ~0.5 to 1) with low values (blue), while `abnormal_short_term_variability` shows a mixed impact, with low values linked to negative SHAP. The narrower SHAP range (-2.5 to 2.5) suggests a more balanced influence across features, consistent with the Suspect class’s ambiguous nature between Normal and Pathological states.

These class-specific SHAP summary plots provide a granular understanding of feature contributions, validating the model’s decision-making process. The alignment of top features—`histogram_mean` and `abnormal_short_term_variability` for Pathological, `accelerations` for Normal, and `percentage_of_time_with_abnormal_long_term_variability` for Suspect—with established CTG literature [1] underscores the robustness of the feature selection and model interpretation. The visualizations facilitate clinical trust by linking machine learning outputs to physiological interpretations, a critical aspect for real-world deployment in fetal health monitoring.

---

### Notes for Your Journal Paper
1. **References**: Replace "[1]" with the appropriate citation (e.g., a CTG or SHAP-related paper) in your reference list.
2. **Figure Numbers**: Adjust "Figures 1a–c" to match your paper’s figure numbering (e.g., if this is part of a larger section, it might be "Figures 3a–c").
3. **Figure Captions**: Add captions below each figure in your manuscript, e.g.:
   - **Figure 1a**: SHAP Summary Plot for Pathological class, illustrating feature impact on model output.
   - **Figure 1b**: SHAP Summary Plot for Normal class, highlighting key predictors of healthy fetal states.
   - **Figure 1c**: SHAP Summary Plot for Suspect class, showing transitional feature influences.
4. **Saving Plots**: Use the code modification suggested earlier (`plt.savefig(f'shap_summary_{class_name.lower()}.png', bbox_inches='tight', dpi=300)`) to save the plots for inclusion in your paper.

This section is now ready for your Q1 journal submission. Please proceed with Step 8 by running the provided code and sharing the output (table and plot) to complete Section 3.2.4. Let me know if you need further refinements!
