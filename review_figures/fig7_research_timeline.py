"""
Figure 7: Research Timeline - Key Milestones (2018-2026)
Author: Soham | Don Bosco Symposium on AI in Education 2026
Output: review_figures/fig7_research_timeline.png (300 DPI)
Run:    python review_figures/fig7_research_timeline.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import os

os.makedirs('review_figures', exist_ok=True)

BD='#1A3C5E'; BM='#2C5F8A'; GN='#1A6E3C'; OR='#D35400'
RD='#8B0000'; TL='#148F77'; PU='#6C3483'; GR='#7F8C8D'; GL='#ECF0F1'

plt.rcParams.update({'font.family':'DejaVu Sans','figure.facecolor':'white'})
fig,ax=plt.subplots(figsize=(16,9))
ax.set_xlim(2017.5,2026.8); ax.set_ylim(-4.5,4.5); ax.axis('off')
fig.patch.set_facecolor('white')

ax.axhline(y=0,xmin=0.02,xmax=0.98,color=BD,lw=3,zorder=1)

for yr in range(2018,2027):
    ax.axvline(x=yr,ymin=0.47,ymax=0.53,color=BD,lw=1.5,zorder=2)
    ax.text(yr,-0.35,str(yr),ha='center',va='top',fontsize=10,
        color=BD,fontweight='bold')

events=[
    (2018,-1.5,"UCI Dataset\nIntroduced",
     "Arzamasov et al.\n4-node star network\n10K instances",GR),
    (2021,-1.8,"First Systematic\nML Comparison",
     "Bashir et al.\nSVM vs KNN vs LR\n~95.8% accuracy",BM),
    (2023,-2.1,"Data Balancing\n& Hybrid Methods",
     "Zhang et al.; Hinz & Drossel\nSMOTE + ML fusion\n~97% accuracy",TL),
    (2024,-1.6,"DNN Surpasses\n99% Accuracy",
     "Lahon et al.\nDNN vs SVM\n99.5% achieved",GN),
    (2020,1.6,"Explainable AI\nfor Grid Stability",
     "SHAP frameworks\napplied to SVM\nGrid fault detection",PU),
    (2024,2.2,"ANN+SVM Fusion\nArchitecture",
     "Sreelakshmi et al.\n98.92% accuracy\nHybrid approach",OR),
    (2025,1.8,"StarNet Ensemble\n& BO-SVM",
     "Ahmed et al.; Danach et al.\n98.94% / 96.00%\nCross-dataset validation",RD),
    (2026,2.5,"Inference Latency\nBenchmarking",
     "Soham (this review)\nSVM: 0.074ms\n387x faster than RF",BD),
]

for yr,yo,label,sublabel,color in events:
    ax.plot([yr,yr],[0,yo*0.72],color=color,lw=1.8,linestyle='--',alpha=0.7,zorder=2)
    ax.scatter([yr],[0],s=120,color=color,edgecolors='white',linewidths=2,zorder=5)
    bw=2.0; bh=1.1; bx=yr-bw/2; by=yo-bh/2
    ax.add_patch(FancyBboxPatch((bx,by),bw,bh,boxstyle="round,pad=0.1",
        linewidth=2,edgecolor=color,facecolor='white',zorder=3))
    ax.text(yr,by+bh-0.3,label,ha='center',va='center',
        fontsize=8.5,fontweight='bold',color=color,zorder=4)
    ax.text(yr,by+0.3,sublabel,ha='center',va='center',
        fontsize=7,color=GR,zorder=4)

ax.axvspan(2026.2,2026.7,alpha=0.08,color='gold')
ax.text(2026.45,3.5,'Future\nDirections',ha='center',va='center',
    fontsize=9,color='#B7950B',fontweight='bold',style='italic')
for i,fut in enumerate(['Real PMU data','Temporal LSTM','XAI Standards']):
    ax.text(2026.45,2.8-i*0.55,f'- {fut}',ha='center',va='center',
        fontsize=8,color='#B7950B')

patches=[
    mpatches.Patch(facecolor=GR,label='Dataset / Benchmark'),
    mpatches.Patch(facecolor=BM,label='Classical ML advances'),
    mpatches.Patch(facecolor=GN,label='Deep learning advances'),
    mpatches.Patch(facecolor=OR,label='Hybrid architectures'),
    mpatches.Patch(facecolor=PU,label='Explainability advances'),
    mpatches.Patch(facecolor=RD,label='Ensemble / optimisation'),
    mpatches.Patch(facecolor=BD,label='Deployment analysis'),
]
ax.legend(handles=patches,loc='lower left',fontsize=8.5,
    framealpha=0.9,ncol=4,bbox_to_anchor=(0.01,-0.05))

ax.text(2017.7,2.0,'Methodology\nAdvances',ha='left',va='center',fontsize=10,
    color=BD,fontweight='bold',
    bbox=dict(boxstyle='round',facecolor=GL,edgecolor=BD))
ax.text(2017.7,-1.5,'Dataset &\nBenchmark',ha='left',va='center',fontsize=10,
    color=BD,fontweight='bold',
    bbox=dict(boxstyle='round',facecolor=GL,edgecolor=BD))

ax.text(8.0/16*9.3+2017.5,4.2,
    'Figure 7: Research Timeline - Key Milestones in ML for Smart Grid Stability (2018-2026)',
    ha='center',va='center',fontsize=13,fontweight='bold',color=BD,
    transform=ax.transData)

plt.tight_layout()
plt.savefig('review_figures/fig7_research_timeline.png',dpi=300,bbox_inches='tight')
plt.close()
print("Saved: review_figures/fig7_research_timeline.png")
