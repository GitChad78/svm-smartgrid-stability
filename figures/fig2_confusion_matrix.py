# Figure 2: Confusion Matrix — SVM (RBF Kernel)
# Author : Pablo Soham | 2nd Year CS | 2026
# Output : output_figures/fig2_confusion_matrix.png  (300 DPI)
# Usage  : python figures/fig2_confusion_matrix.py
# Requires: pip install matplotlib numpy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# Exact values from manuscript Section 5.2
# TP=1248  FN=28  FP=26  TN=698
cell_colors = [['#1A6E3C','#8B0000'],['#F39C12','#1A3C5E']]
cell_labels = [
    ['TP = 1,248\n(Correctly flagged\nunstable)', 'FN = 28\n(Missed unstable\nCritical)'],
    ['FP = 26\n(False alarm)',                     'TN = 698\n(Correctly\nidentified stable)'],
]

fig, ax = plt.subplots(figsize=(6, 5.2))
ax.set_facecolor('white')
fig.patch.set_facecolor('white')
ax.grid(False)

for i in range(2):
    for j in range(2):
        ax.add_patch(plt.Rectangle((j-0.5,i-0.5),1,1,
                     fill=True,color=cell_colors[i][j],alpha=0.88,zorder=2))
        ax.text(j,i,cell_labels[i][j],ha='center',va='center',
                fontsize=11,fontweight='bold',color='white',zorder=3)

ax.set_xlim(-0.5,1.5); ax.set_ylim(-0.5,1.5)
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(['Predicted\nUnstable','Predicted\nStable'],fontsize=12)
ax.set_yticklabels(['Actual\nUnstable','Actual\nStable'],fontsize=12)
ax.set_xlabel('Predicted Label',fontsize=13,color='#1A3C5E',labelpad=10)
ax.set_ylabel('True Label',fontsize=13,color='#1A3C5E',labelpad=10)
ax.set_title('Figure 2: Confusion Matrix — SVM (RBF Kernel)\n'
             'Test Set: 2,000 samples  |  Total Errors: 54  |  Accuracy: 97.30%',
             fontsize=11,color='#1A3C5E',pad=14)

os.makedirs('output_figures', exist_ok=True)
plt.tight_layout()
plt.savefig('output_figures/fig2_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: output_figures/fig2_confusion_matrix.png")
