# figures/

Python scripts to reproduce every figure in the manuscript.

| Script | Output | Used In |
|--------|--------|---------|
| fig1_class_distribution.py | fig1_class_distribution.png | Figure 1, SI S4 |
| fig2_confusion_matrix.py | fig2_confusion_matrix.png | Figure 2, SI S4 |
| fig3_roc_curve.py | fig3_roc_curve.png | Figure 3, SI S4 |
| fig4_feature_importance.py | fig4_feature_importance.png | Figure 4, SI S4 |
| fig_graphical_abstract.py | fig_graphical_abstract.png | Graphical Abstract |

## Run all figures

    pip install -r requirements.txt
    python figures/fig1_class_distribution.py
    python figures/fig2_confusion_matrix.py
    python figures/fig3_roc_curve.py
    python figures/fig4_feature_importance.py
    python figures/fig_graphical_abstract.py

Output saved to output_figures/ at 300 DPI.

## Reproduce from trained model

Figures 3 and 4 include an Option B block (commented out).
Uncomment it after running svm_pipeline.py to get svm_grid_model.pkl.

## Citation

Soham, P. (2026). SVM-Based Smart Grid Stability Prediction (v1.0).
Zenodo. https://doi.org/10.5281/zenodo.19516705
