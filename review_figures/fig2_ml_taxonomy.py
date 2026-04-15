"""
Figure 2: ML Algorithm Taxonomy for Smart Grid Stability Prediction
Author: Soham | Don Bosco Symposium on AI in Education 2026
Output: review_figures/fig2_ml_taxonomy.png (300 DPI)
Run:    python review_figures/fig2_ml_taxonomy.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

os.makedirs('review_figures', exist_ok=True)

BD='#1A3C5E'; BM='#2C5F8A'; GN='#1A6E3C'; OR='#E67E22'
TL='#148F77'; RD='#8B0000'; PU='#6C3483'; GR='#7F8C8D'
BL='#AED6F1'; GL='#A9DFBF'; OL='#FAD7A0'; TLl='#A2D9CE'
GRL='#D5D8DC'

fig,ax=plt.subplots(figsize=(16,10))
ax.set_xlim(0,16); ax.set_ylim(0,10); ax.axis('off')
fig.patch.set_facecolor('white')

def box(x,y,w,h,label,sub,fill,border,fs=10):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.08",
        linewidth=2,edgecolor=border,facecolor=fill,zorder=2))
    ax.text(x+w/2,y+h/2+(0.15 if sub else 0),label,ha='center',va='center',
        fontsize=fs,fontweight='bold',color=border,zorder=3)
    if sub:
        ax.text(x+w/2,y+h/2-0.22,sub,ha='center',va='center',
            fontsize=8,color=GR,zorder=3,style='italic')

def arrow(x1,y1,x2,y2,col=BD):
    ax.annotate("",xy=(x2,y2),xytext=(x1,y1),
        arrowprops=dict(arrowstyle='->',color=col,lw=1.8))

box(5.5,8.8,5.0,0.9,"Machine Learning for Smart Grid Stability",
    "Binary Classification: Stable / Unstable",'#E8F4FD',BD,11)

branches=[(0.3,6.8,3.0,0.85,"Classical\nClassifiers","",GRL,GR),
          (4.0,6.8,3.0,0.85,"Ensemble\nMethods","",OL,OR),
          (7.6,6.8,3.0,0.85,"Deep Learning\nApproaches","",GL,GN),
          (11.2,6.8,3.0,0.85,"Hybrid &\nExplainable AI","",TLl,TL)]
for bx,by,bw,bh,label,sub,fill,border in branches:
    box(bx,by,bw,bh,label,sub,fill,border,10)
    arrow(8.0,8.8,bx+bw/2,by+bh)

classics=[(0.1,5.2,1.3,0.7,"SVM\n(RBF kernel)",'#D6EAF8',BM),
          (1.55,5.2,1.3,0.7,"Logistic\nRegression",GRL,GR),
          (3.0,5.2,1.3,0.7,"KNN /\nNaive Bayes",GRL,GR)]
for cx,cy,cw,ch,label,fill,border in classics:
    box(cx,cy,cw,ch,label,"",fill,border,8)
    arrow(1.8,6.8,cx+cw/2,cy+ch,border)

ax.add_patch(FancyBboxPatch((0.05,4.55),1.4,0.55,boxstyle="round,pad=0.05",
    linewidth=1.5,edgecolor='#E74C3C',facecolor='#FADBD8',zorder=4))
ax.text(0.75,4.83,"Best for edge\ndeployment",ha='center',va='center',
    fontsize=7.5,color='#E74C3C',fontweight='bold',zorder=5)

ensembles=[(3.9,5.2,1.5,0.7,"Random\nForest",OL,OR),
           (5.55,5.2,1.5,0.7,"Gradient\nBoosting/XGB",OL,OR),
           (7.1,5.2,1.3,0.7,"StarNet\nStacking",'#FAD7A0','#C0392B')]
for ex,ey,ew,eh,label,fill,border in ensembles:
    box(ex,ey,ew,eh,label,"",fill,border,8)
    arrow(5.5,6.8,ex+ew/2,ey+eh,border)

dls=[(7.5,5.2,1.5,0.7,"DNN /\nANN",GL,GN),
     (9.1,5.2,1.5,0.7,"LSTM /\nCNN",GL,GN),
     (10.7,5.2,1.5,0.7,"Transformer\nAttn.",'#A9DFBF','#1E8449')]
for dx,dy,dw,dh,label,fill,border in dls:
    box(dx,dy,dw,dh,label,"",fill,border,8)
    arrow(9.1,6.8,dx+dw/2,dy+dh,border)

hybrids=[(11.1,5.2,1.5,0.7,"ANN+SVM\nFusion",TLl,TL),
         (12.7,5.2,1.5,0.7,"SHAP /\nExplain.",TLl,TL),
         (14.3,5.2,1.3,0.7,"Federated\nLearn.",'#A2D9CE','#117A65')]
for hx,hy,hw,hh,label,fill,border in hybrids:
    box(hx,hy,hw,hh,label,"",fill,border,8)
    arrow(12.7,6.8,hx+hw/2,hy+hh,border)

box(5.5,3.6,5.0,0.85,"UCI Smart Grid Stability Dataset",
    "10,000 instances | 12 features | Binary label | ID: 471",'#FEF9E7','#F39C12',10)

ax.add_patch(FancyBboxPatch((0.2,2.5),15.6,0.85,boxstyle="round,pad=0.08",
    linewidth=1.5,edgecolor=BD,facecolor='#EBF5FB',zorder=2))
ax.text(8.0,2.93,"Reported Accuracy Range on UCI Benchmark (2021-2026)",
    ha='center',va='center',fontsize=10.5,color=BD,fontweight='bold',zorder=3)

segments=[(0.3,"SVM: 96-99%",'#AED6F1',BM),(3.3,"RF: 92-95%",OL,OR),
          (6.3,"GB: 93-97%",'#FAD7A0','#C0392B'),(9.3,"DNN: 97-99.5%",GL,GN),
          (12.3,"Ensemble: up to 98.94%",TLl,TL)]
for sx,slabel,sfill,sborder in segments:
    ax.add_patch(FancyBboxPatch((sx,1.55),2.8,0.7,boxstyle="round,pad=0.05",
        linewidth=1.5,edgecolor=sborder,facecolor=sfill,zorder=3))
    ax.text(sx+1.4,1.90,slabel,ha='center',va='center',
        fontsize=8.5,color=sborder,fontweight='bold',zorder=4)

ax.text(8.0,9.85,'Figure 2: Machine Learning Algorithm Taxonomy for Smart Grid Stability Prediction',
    ha='center',va='center',fontsize=13,fontweight='bold',color=BD)
ax.text(8.0,9.55,'Don Bosco Symposium on AI in Education 2026  |  Soham Ambudkar',
    ha='center',va='center',fontsize=9.5,color=GR,style='italic')

plt.tight_layout()
plt.savefig('review_figures/fig2_ml_taxonomy.png',dpi=300,bbox_inches='tight')
plt.close()
print("Saved: review_figures/fig2_ml_taxonomy.png")
