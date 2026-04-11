# Graphical Abstract: Research Pipeline Overview
# Author : Pablo Soham | 2nd Year CS | 2026
# Output : output_figures/fig_graphical_abstract.png  (300 DPI)
# Usage  : python figures/fig_graphical_abstract.py
# Requires: pip install matplotlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import os

stages = [
    {'title':'UCI Smart Grid\nDataset',  'detail':'10,000 samples\n12 features\nBinary label\n(stable/unstable)', 'color':'#E3F2FD','border':'#2C5F8A','icon':'[DATA]'},
    {'title':'Preprocessing',            'detail':'StandardScaler\nzero mean, unit var\nStratified 80/20\nseed = 42',        'color':'#E3F2FD','border':'#2C5F8A','icon':'[PREP]'},
    {'title':'SVM Classifier',           'detail':'RBF Kernel\nC = 10\ng = scale\nprobability = True',                      'color':'#E3F2FD','border':'#1A3C5E','icon':'[SVM]'},
    {'title':'Evaluation',               'detail':'5-Fold CV\nROC Curve\nConfusion Matrix\nPerm. Importance',                'color':'#E3F2FD','border':'#2C5F8A','icon':'[EVAL]'},
    {'title':'Results',                  'detail':'Accuracy: 97.30%\nAUC: 0.9964\nCV: 96.04%+/-0.33%\nModel: 147 KB',       'color':'#E8F5E9','border':'#1A6E3C','icon':'[DONE]'},
]

fig, ax = plt.subplots(figsize=(16, 5))
ax.set_xlim(0,16); ax.set_ylim(0,5); ax.axis('off')
fig.patch.set_facecolor('white')

box_w=2.6; box_h=3.8; gap=0.3; start_x=0.4; y_bot=0.6

for i, s in enumerate(stages):
    x = start_x + i*(box_w+gap)
    ax.add_patch(FancyBboxPatch((x,y_bot),box_w,box_h,
                 boxstyle="round,pad=0.08",linewidth=2,
                 edgecolor=s['border'],facecolor=s['color'],zorder=2))
    ax.text(x+box_w/2, y_bot+box_h-0.38, s['icon'],
            ha='center',va='center',fontsize=11,fontweight='bold',
            color=s['border'],zorder=3)
    ax.text(x+box_w/2, y_bot+box_h-0.88, s['title'],
            ha='center',va='center',fontsize=11,fontweight='bold',
            color='#1A3C5E',zorder=3)
    ax.plot([x+0.15,x+box_w-0.15],[y_bot+box_h-1.15]*2,
            color=s['border'],lw=1.0,zorder=3)
    ax.text(x+box_w/2, y_bot+box_h/2-0.30, s['detail'],
            ha='center',va='center',fontsize=9.5,color='#333333',
            linespacing=1.55,zorder=3)
    if i < len(stages)-1:
        ax.add_patch(FancyArrowPatch((x+box_w+0.01, y_bot+box_h/2),
                     (x+box_w+gap-0.02, y_bot+box_h/2),
                     arrowstyle='->',color='#1A3C5E',
                     linewidth=2.0,mutation_scale=18,zorder=4))

ax.text(8.0,4.80,'SVM-Based Smart Grid Stability Prediction - Research Pipeline',
        ha='center',va='center',fontsize=14,fontweight='bold',color='#1A3C5E')
ax.text(8.0,4.50,
        'Pablo Soham  |  Dept. of Computer Science and Engineering  |  2026  |  DOI: 10.5281/zenodo.19516705',
        ha='center',va='center',fontsize=9,color='#555555',style='italic')

os.makedirs('output_figures', exist_ok=True)
plt.tight_layout()
plt.savefig('output_figures/fig_graphical_abstract.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: output_figures/fig_graphical_abstract.png")
