"""
Figure 3: Accuracy Comparison Across Studies (2021-2026)
Author: Pablo Soham | Don Bosco Symposium on AI in Education 2026
Output: review_figures/fig3_accuracy_comparison.png (300 DPI)
Run:    python review_figures/fig3_accuracy_comparison.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('review_figures', exist_ok=True)

BD='#1A3C5E'; BM='#2C5F8A'; GN='#1A6E3C'
OR='#E67E22'; RD='#8B0000'; TL='#148F77'; PU='#8E44AD'

plt.rcParams.update({'font.family':'DejaVu Sans','figure.facecolor':'white',
    'axes.facecolor':'#F8F9FA','axes.spines.top':False,'axes.spines.right':False,
    'axes.grid':True,'grid.color':'white','grid.linewidth':1.4})

studies=['Bashir et al.\n2021','Hinz & Drossel\n2023','Zhang et al.\n2023',
    'Lahon et al.\n2024','Sreelakshmi\net al. 2024','Hassan et al.\n2024',
    'Danach et al.\n2025','Ahmed et al.\n2025','Soham\n2026']
best_acc=[95.8,96.5,97.1,99.5,98.92,97.5,96.0,98.94,97.30]
svm_acc=[95.8,96.5,None,98.9,98.92,None,96.0,96.52,97.30]
methods=['SVM/NN','MLP/XGB/SVM','ML+SMOTE','DNN','ANN+SVM',
    'Ensemble','BO-SVM','StarNet','SVM+RF+GB']
colors=[BM,TL,GN,RD,OR,PU,BD,'#C0392B',BM]

x=np.arange(len(studies))
fig,ax=plt.subplots(figsize=(14,7))

ax.axhspan(99,100,alpha=0.07,color=GN)
ax.axhspan(97,99,alpha=0.07,color=BM)
ax.axhspan(95,97,alpha=0.07,color=OR)

bars=ax.bar(x,best_acc,color=colors,edgecolor='white',linewidth=1.5,width=0.6,zorder=3)

for i,(sv,m) in enumerate(zip(svm_acc,[v is not None for v in svm_acc])):
    if m and sv != best_acc[i]:
        ax.plot(i,sv,'D',color='white',markersize=10,
            markeredgecolor=BD,markeredgewidth=2,zorder=5)
        ax.text(i,sv-0.35,f'{sv:.1f}%',ha='center',va='top',
            fontsize=7.5,color=BD,fontweight='bold')

for bar,val in zip(bars,best_acc):
    ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.1,
        f'{val:.2f}%',ha='center',va='bottom',fontsize=9,fontweight='bold',color=BD)

for bar,method in zip(bars,methods):
    ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()/2+44,
        method,ha='center',va='center',fontsize=7.5,color='white',
        fontweight='bold',rotation=90)

ax.axhline(y=97.30,color=BD,lw=2,linestyle='--',alpha=0.8,
    label='SVM baseline (Soham 2026): 97.30%')
ax.axhline(y=95.0,color='#AAA',lw=1.5,linestyle=':',alpha=0.6,
    label='Minimum competitive threshold: 95%')

ax.set_xticks(x)
ax.set_xticklabels(studies,fontsize=9.5)
ax.set_ylim(88,100.5)
ax.set_ylabel('Classification Accuracy (%)',fontsize=13,color=BD)
ax.set_xlabel('Study (Chronological Order)',fontsize=13,color=BD)
ax.set_title('Figure 3: Classification Accuracy Across Studies - UCI Smart Grid Stability Benchmark\n'
    'White diamonds show SVM accuracy where SVM is not the best method in that study',
    fontsize=12,color=BD,pad=14)
ax.legend(loc='lower right',fontsize=10,framealpha=0.92)
ax.text(8.6,99.4,'Excellent (>99%)',fontsize=8,color=GN,style='italic')
ax.text(8.6,97.6,'Very Good (97-99%)',fontsize=8,color=BM,style='italic')
ax.text(8.6,95.2,'Competitive (95-97%)',fontsize=8,color=OR,style='italic')

plt.tight_layout()
plt.savefig('review_figures/fig3_accuracy_comparison.png',dpi=300,bbox_inches='tight')
plt.close()
print("Saved: review_figures/fig3_accuracy_comparison.png")
