# Figure 4: Permutation Feature Importance — SVM (RBF Kernel)
# Author : Pablo Soham | 2nd Year CS | 2026
# Output : output_figures/fig4_feature_importance.png  (300 DPI)
# Usage  : python figures/fig4_feature_importance.py
# Requires: pip install matplotlib numpy scikit-learn
#
# OPTION A (default): reported values from manuscript Section 5.4
# OPTION B (uncomment): reproduce from trained model
#   Needs svm_grid_model.pkl and smart_grid_stability.csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# -- OPTION A: reported values from manuscript --
data = {
    'tau1':(0.141,0.004),'tau2':(0.138,0.004),
    'tau3':(0.131,0.004),'tau4':(0.124,0.004),
    'g1':(0.112,0.004),'g2':(0.108,0.004),
    'g3':(0.101,0.004),'g4':(0.096,0.004),
    'p1':(0.009,0.001),'p2':(0.007,0.001),
    'p3':(0.006,0.001),'p4':(0.005,0.001),
}
features = list(data.keys())
means = [data[f][0] for f in features]
stds  = [data[f][1] for f in features]
order = np.argsort(means)
features_s = [features[i] for i in order]
means_s    = [means[i]    for i in order]
stds_s     = [stds[i]     for i in order]

# -- OPTION B: from trained model (uncomment below) --
# import joblib, pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.inspection import permutation_importance
# bundle = joblib.load('svm_grid_model.pkl')
# model, scaler = bundle['model'], bundle['scaler']
# df = pd.read_csv('smart_grid_stability.csv').drop(columns=['stab'])
# df['stabf'] = LabelEncoder().fit_transform(df['stabf'])
# X = df.drop(columns=['stabf']); y = df['stabf']
# _, X_test, _, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
# perm = permutation_importance(model,scaler.transform(X_test),y_test,
#                               n_repeats=10,random_state=42,scoring='accuracy')
# order = perm.importances_mean.argsort()
# features_s = list(X.columns[order])
# means_s    = list(perm.importances_mean[order])
# stds_s     = list(perm.importances_std[order])

def bar_color(f):
    if f.startswith('tau'): return '#1A3C5E'
    if f.startswith('g'):   return '#2C5F8A'
    return '#AAAAAA'

plt.rcParams.update({
    'font.family':'DejaVu Sans','axes.spines.top':False,
    'axes.spines.right':False,'figure.facecolor':'white',
    'axes.facecolor':'#F8F9FA','axes.grid':True,
    'grid.color':'white','grid.linewidth':1.2,
})

fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.barh(features_s, means_s, xerr=stds_s,
               color=[bar_color(f) for f in features_s],
               edgecolor='white', linewidth=0.8,
               error_kw=dict(ecolor='#555555',lw=1.2,capsize=4), zorder=3)

for bar, val in zip(bars, means_s):
    ax.text(val+0.004, bar.get_y()+bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=10, color='#1A3C5E')

patches = [
    mpatches.Patch(color='#1A3C5E', label='Reaction Time (tau1-tau4) - Dominant'),
    mpatches.Patch(color='#2C5F8A', label='Generation Coeff. (g1-g4) - Moderate'),
    mpatches.Patch(color='#AAAAAA', label='Consumption Coeff. (p1-p4) - Negligible'),
]
ax.legend(handles=patches, loc='lower right', fontsize=10, framealpha=0.9)
ax.set_xlabel('Mean Decrease in Accuracy (10 permutation repeats)',
              fontsize=12, color='#1A3C5E')
ax.set_title('Figure 4: Permutation Feature Importance — SVM (RBF Kernel)\n'
             'tau1-tau4 (reaction times) dominant  |  p1-p4 (consumption) negligible',
             fontsize=12, color='#1A3C5E', pad=14)
ax.set_xlim(0, 0.180)
ax.tick_params(labelsize=11)

os.makedirs('output_figures', exist_ok=True)
plt.tight_layout()
plt.savefig('output_figures/fig4_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: output_figures/fig4_feature_importance.png")
