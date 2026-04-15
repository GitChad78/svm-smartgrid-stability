"""
Figure 5: Feature Importance Heatmap
Author: Soham | Don Bosco Symposium on AI in Education 2026
Output: review_figures/fig5_feature_importance_heatmap.png (300 DPI)
Run:    python review_figures/fig5_feature_importance_heatmap.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import numpy as np
import os

os.makedirs('review_figures', exist_ok=True)

BD='#1A3C5E'; GR='#7F8C8D'
plt.rcParams.update({'font.family':'DejaVu Sans','figure.facecolor':'white'})

features=['tau1','tau2','tau3','tau4','g1','g2','g3','g4','p1','p2','p3','p4']
imp_svm=[0.141,0.138,0.131,0.124,0.112,0.108,0.101,0.096,0.009,0.007,0.006,0.005]
imp_rf= [0.128,0.122,0.118,0.111,0.098,0.094,0.089,0.083,0.012,0.010,0.009,0.008]
imp_gb= [0.135,0.131,0.124,0.118,0.105,0.100,0.094,0.089,0.010,0.008,0.007,0.006]
data=np.array([imp_svm,imp_rf,imp_gb])
classifiers=['SVM (RBF)','Random Forest','Gradient Boosting']

fig=plt.figure(figsize=(14,8))
gs=gridspec.GridSpec(1,2,width_ratios=[2,1],wspace=0.35)
ax1=fig.add_subplot(gs[0])
im=ax1.imshow(data,cmap='YlOrRd',aspect='auto',vmin=0,vmax=0.15)

ax1.set_xticks(range(len(features)))
ax1.set_xticklabels(features,fontsize=11,fontweight='bold')
ax1.set_yticks(range(len(classifiers)))
ax1.set_yticklabels(classifiers,fontsize=11)
ax1.set_xlabel('Feature',fontsize=13,color=BD)
ax1.set_title('Permutation Importance Heatmap\n(Mean Accuracy Decrease)',
    fontsize=12,color=BD,pad=12)

for i in range(len(classifiers)):
    for j in range(len(features)):
        ax1.text(j,i,f'{data[i,j]:.3f}',ha='center',va='center',fontsize=9,
            color='black' if data[i,j]<0.10 else 'white',fontweight='bold')

ax1.axvline(x=3.5,color='white',lw=3)
ax1.axvline(x=7.5,color='white',lw=3)
ax1.text(1.5,-0.75,'Reaction Times\n(tau1-tau4)',ha='center',va='center',
    fontsize=10,color='#8B0000',fontweight='bold',transform=ax1.transData)
ax1.text(5.5,-0.75,'Generation Coeff.\n(g1-g4)',ha='center',va='center',
    fontsize=10,color='#E67E22',fontweight='bold',transform=ax1.transData)
ax1.text(9.5,-0.75,'Consumption Coeff.\n(p1-p4)',ha='center',va='center',
    fontsize=10,color='#555',fontweight='bold',transform=ax1.transData)

cbar=fig.colorbar(im,ax=ax1,shrink=0.8)
cbar.set_label('Mean Accuracy Decrease',fontsize=10,color=BD)

ax2=fig.add_subplot(gs[1])
mean_imp=data.mean(axis=0)
sorted_idx=np.argsort(mean_imp)
feat_sorted=[features[i] for i in sorted_idx]
imp_sorted=mean_imp[sorted_idx]
colors_bar=['#8B0000' if f.startswith('tau') else '#E67E22' if f.startswith('g')
    else '#95A5A6' for f in feat_sorted]

bars=ax2.barh(feat_sorted,imp_sorted,color=colors_bar,
    edgecolor='white',linewidth=0.8,height=0.65,zorder=3)
for bar,val in zip(bars,imp_sorted):
    ax2.text(val+0.002,bar.get_y()+bar.get_height()/2,
        f'{val:.3f}',va='center',fontsize=9.5,color=BD,fontweight='bold')

ax2.set_facecolor('#F8F9FA')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_xlabel('Mean Importance (avg. across classifiers)',fontsize=11,color=BD)
ax2.set_title('Average Feature\nImportance Ranking',fontsize=12,color=BD,pad=12)
ax2.set_xlim(0,0.17)

legend_els=[
    Patch(facecolor='#8B0000',label='Reaction Times (tau) - Dominant'),
    Patch(facecolor='#E67E22',label='Generation Coeff. (g) - Moderate'),
    Patch(facecolor='#95A5A6',label='Consumption Coeff. (p) - Negligible'),
]
ax2.legend(handles=legend_els,loc='lower right',fontsize=8.5)

fig.suptitle('Figure 5: Feature Importance Analysis - UCI Smart Grid Stability Dataset\n'
    'Reaction-time parameters dominate; consumption parameters negligible across all classifiers',
    fontsize=12,color=BD,y=1.01,fontweight='bold')
plt.savefig('review_figures/fig5_feature_importance_heatmap.png',dpi=300,bbox_inches='tight')
plt.close()
print("Saved: review_figures/fig5_feature_importance_heatmap.png")
