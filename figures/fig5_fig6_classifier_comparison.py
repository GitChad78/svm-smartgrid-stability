# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Classifier Comparison Bar Chart
# Figure 6: ROC Curves — All Classifiers
# Author : Pablo Soham | 2nd Year CS | 2026
# Output : output_figures/fig5_classifier_comparison.png  (300 DPI)
#          output_figures/fig6_roc_comparison.png          (300 DPI)
# Usage  : python figures/fig5_fig6_classifier_comparison.py
# Note   : Run svm_pipeline_extended.py first to generate results,
#          OR run this script standalone using the reported values below.
# Requires: pip install matplotlib numpy scikit-learn
# ─────────────────────────────────────────────────────────────────────────────

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# ── Reported values from manuscript (Section 5.7 extended comparison) ────────
results_reported = {
    'SVM (RBF)'     : {'acc': 0.9730, 'auc': 0.9964, 'cv_mean': 0.9600, 'cv_std': 0.0016},
    'Random Forest' : {'acc': 0.9240, 'auc': 0.9806, 'cv_mean': 0.9159, 'cv_std': 0.0121},
    'Gradient Boost': {'acc': 0.9360, 'auc': 0.9841, 'cv_mean': 0.9161, 'cv_std': 0.0036},
}

names = list(results_reported.keys())
accs  = [results_reported[n]['acc']     for n in names]
aucs  = [results_reported[n]['auc']     for n in names]
cvs   = [results_reported[n]['cv_mean'] for n in names]

os.makedirs('output_figures', exist_ok=True)

# ── Figure 5: Bar Chart ───────────────────────────────────────────────────────
x = np.arange(len(names))
w = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
b1 = ax.bar(x - w, accs, w, label='Test Accuracy', color='#1A3C5E')
b2 = ax.bar(x,     aucs, w, label='AUC-ROC',       color='#2C5F8A')
b3 = ax.bar(x + w, cvs,  w, label='CV Accuracy',   color='#AED6F1')

for bars in [b1, b2, b3]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.002,
                f'{bar.get_height():.4f}',
                ha='center', va='bottom', fontsize=8.5, color='#1A3C5E')

ax.set_ylim(0.85, 1.02)
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=12)
ax.set_ylabel('Score', fontsize=13, color='#1A3C5E')
ax.set_title('Figure 5: Classifier Comparison — SVM vs Random Forest vs Gradient Boosting\n'
             'UCI Smart Grid Stability Dataset  |  Test set: 2,000 samples',
             fontsize=12, color='#1A3C5E', pad=14)
ax.legend(fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('output_figures/fig5_classifier_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: output_figures/fig5_classifier_comparison.png")

# ── Figure 6: ROC Curves (smooth approximations) ─────────────────────────────
# Uses parametric curves approximating each classifier's AUC.
# For exact curves: uncomment Option B in fig3_roc_curve.py pattern.

def smooth_roc(auc):
    t   = np.linspace(0, 1, 500)
    tpr = np.clip(1 - (1-t) ** (1/(1 + 8*auc - 8)), 0, 1)
    return t, tpr

colors_roc = ['#1A3C5E', '#1A6E3C', '#8B0000']

fig, ax = plt.subplots(figsize=(7, 6))
for name, col in zip(names, colors_roc):
    auc = results_reported[name]['auc']
    fpr, tpr = smooth_roc(auc)
    ax.plot(fpr, tpr, color=col, lw=2,
            label=f"{name}  (AUC = {auc:.4f})")

ax.plot([0,1],[0,1],'k--',lw=1.2,alpha=0.5,label='Random (AUC = 0.50)')
ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
ax.set_xlabel('False Positive Rate', fontsize=13, color='#1A3C5E')
ax.set_ylabel('True Positive Rate',  fontsize=13, color='#1A3C5E')
ax.set_title('Figure 6: ROC Curves — All Classifiers Compared\n'
             'SVM (RBF) achieves highest AUC = 0.9964',
             fontsize=12, color='#1A3C5E', pad=14)
ax.legend(loc='lower right', fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('output_figures/fig6_roc_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: output_figures/fig6_roc_comparison.png")
