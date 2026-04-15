






"""
Figure 4: Accuracy vs Inference Latency Trade-off (Bubble Chart)
Author: Soham | Don Bosco Symposium on AI in Education 2026
Output: review_figures/fig4_accuracy_vs_latency.png (300 DPI)
Run:    python review_figures/fig4_accuracy_vs_latency.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('review_figures', exist_ok=True)

BD='#1A3C5E'; GR='#7F8C8D'
plt.rcParams.update({'font.family':'DejaVu Sans','figure.facecolor':'white',
    'axes.facecolor':'#F8F9FA','axes.spines.top':False,'axes.spines.right':False,
    'axes.grid':True,'grid.color':'white','grid.linewidth':1.4})

models=[
    (97.30, 0.074,  147,  'SVM (RBF)\n[Soham 2026]',       '#1A3C5E'),
    (93.60, 0.074,  850,  'Gradient\nBoosting',             '#E67E22'),
    (92.40,28.683,45000,  'Random\nForest',                 '#8B0000'),
    (99.50, 8.500, 3500,  'Deep Neural\nNetwork [Lahon]',   '#1A6E3C'),
    (98.92, 1.200,  800,  'ANN+SVM\nFusion [Sreelakshmi]', '#148F77'),
    (98.94, 3.500, 4200,  'StarNet\nEnsemble [Ahmed]',      '#8E44AD'),
    (96.00, 0.120,  200,  'BO-SVM\n[Danach]',               '#2C5F8A'),
    (95.80, 0.400,  300,  'SVM/NN\n[Bashir]',               '#7F8C8D'),
]

fig,ax=plt.subplots(figsize=(12,8))

for acc,lat,size,label,color in models:
    bsize=np.log10(size)*300
    ax.scatter(lat,acc,s=bsize,color=color,alpha=0.82,
        edgecolors='white',linewidths=2,zorder=3)
    ox=0.4 if lat<5 else -1.5
    oy=-0.25
    if label.startswith('Random'): ox=1.5
    if label.startswith('SVM (RBF)'): oy=0.25
    ax.annotate(label,xy=(lat,acc),xytext=(lat+ox,acc+oy),
        fontsize=9,color=color,fontweight='bold',
        arrowprops=dict(arrowstyle='-',color=color,lw=0.8))

ax.axvspan(0,1,alpha=0.08,color='#1A6E3C',label='Ideal deployment zone (<1ms)')
ax.axhline(y=97,color='#555',lw=1.5,linestyle='--',alpha=0.5,label='97% accuracy threshold')
ax.axvline(x=1, color='#555',lw=1.5,linestyle='--',alpha=0.5,label='1ms latency threshold')
ax.text(0.05,99.7,'IDEAL ZONE\n(High accuracy + Fast inference)',
    fontsize=9,color='#1A6E3C',fontweight='bold',style='italic')

for kb,lbl in [(150,'~150 KB'),(1000,'~1 MB'),(10000,'~10 MB'),(45000,'~45 MB')]:
    ax.scatter([],[],s=np.log10(kb)*300,color='grey',alpha=0.5,label=f'Model size: {lbl}')

ax.set_xscale('log')
ax.set_xlim(0.03,80)
ax.set_ylim(91.5,100.5)
ax.set_xlabel('Inference Latency per Sample (ms, log scale)',fontsize=13,color=BD)
ax.set_ylabel('Test Accuracy (%)',fontsize=13,color=BD)
ax.set_title('Figure 4: Accuracy vs Inference Latency Trade-off\n'
    'Bubble size = model storage size  |  Ideal: top-left corner',
    fontsize=12,color=BD,pad=14)
ax.legend(loc='lower left',fontsize=8.5,framealpha=0.92,ncol=2)
ax.annotate('SVM offers the best\naccuracy-latency-size\ncombination',
    xy=(0.074,97.30),xytext=(0.5,94.5),fontsize=9.5,color=BD,fontweight='bold',
    arrowprops=dict(arrowstyle='->',color=BD,lw=2),
    bbox=dict(boxstyle='round,pad=0.4',facecolor='#E8F4FD',edgecolor=BD))

plt.tight_layout()
plt.savefig('review_figures/fig4_accuracy_vs_latency.png',dpi=300,bbox_inches='tight')
plt.close()
print("Saved: review_figures/fig4_accuracy_vs_latency.png")
