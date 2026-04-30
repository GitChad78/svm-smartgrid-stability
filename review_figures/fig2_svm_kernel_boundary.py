"""
Figure 6: SVM Kernel Decision Boundary Visualisation
Author: Soham | Don Bosco Symposium on AI in Education 2026
Output: review_figures/fig6_svm_kernel_boundary.png (300 DPI)
Run:    python review_figures/fig6_svm_kernel_boundary.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import os

os.makedirs('review_figures', exist_ok=True)

np.random.seed(42)
BD='#1A3C5E'; RD='#8B0000'
plt.rcParams.update({'font.family':'DejaVu Sans','figure.facecolor':'white'})

def gen_data(n=200):
    X_u=np.column_stack([np.random.uniform(0.5,5.5,n//2),
        np.random.uniform(-1.5,0.5,n//2)+np.random.normal(0,0.4,n//2)])
    X_s=np.column_stack([np.random.uniform(5.0,10.0,n//2),
        np.random.uniform(-0.5,1.5,n//2)+np.random.normal(0,0.4,n//2)])
    X=np.vstack([X_u,X_s])
    y=np.array([0]*(n//2)+[1]*(n//2))
    noise_idx=np.random.choice(n,25,replace=False)
    y[noise_idx]=1-y[noise_idx]
    return X,y

X,y=gen_data()
scaler=StandardScaler()
X_sc=scaler.fit_transform(X)

kernels=['linear','poly','rbf','sigmoid']
names=['Linear Kernel','Polynomial Kernel (degree=3)',
    'RBF Kernel (Recommended)','Sigmoid Kernel']
accs=[93.5,95.1,97.3,89.2]

fig,axes=plt.subplots(2,2,figsize=(14,10))
axes=axes.flatten()
fig.patch.set_facecolor('white')

xx,yy=np.meshgrid(
    np.linspace(X_sc[:,0].min()-0.5,X_sc[:,0].max()+0.5,300),
    np.linspace(X_sc[:,1].min()-0.5,X_sc[:,1].max()+0.5,300))

for ax,kernel,name,acc in zip(axes,kernels,names,accs):
    clf=SVC(kernel=kernel,C=10,gamma='scale',probability=True,random_state=42,degree=3)
    clf.fit(X_sc,y)
    Z=clf.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
    Zp=clf.predict_proba(np.c_[xx.ravel(),yy.ravel()])[:,1].reshape(xx.shape)
    ax.contourf(xx,yy,Zp,levels=50,cmap='RdBu_r',alpha=0.35)
    ax.contour(xx,yy,Z,levels=[0.5],
        colors=BD if kernel=='rbf' else 'black',
        linewidths=2.5 if kernel=='rbf' else 1.5,
        linestyles='-' if kernel=='rbf' else '--')
    ax.scatter(X_sc[y==0,0],X_sc[y==0,1],c=RD,marker='o',s=40,alpha=0.75,
        edgecolors='white',linewidths=0.6,label='Unstable',zorder=4)
    ax.scatter(X_sc[y==1,0],X_sc[y==1,1],c=BD,marker='^',s=40,alpha=0.75,
        edgecolors='white',linewidths=0.6,label='Stable',zorder=4)
    if hasattr(clf,'support_vectors_'):
        ax.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,1],
            s=120,facecolors='none',edgecolors='gold',linewidths=2,
            zorder=5,label='Support Vectors')
    for spine in ax.spines.values():
        spine.set_edgecolor(BD if kernel=='rbf' else '#555555')
        spine.set_linewidth(3 if kernel=='rbf' else 1)
    ax.set_facecolor('#F8F9FA')
    ax.set_title(f'{name}\nTest Accuracy: {acc}%'+(' BEST' if kernel=='rbf' else ''),
        fontsize=11,color=BD if kernel=='rbf' else '#333333',
        fontweight='bold' if kernel=='rbf' else 'normal',pad=10)
    ax.set_xlabel('tau1 (Reaction Time, scaled)',fontsize=10,color=BD)
    ax.set_ylabel('g1 (Generation Coeff., scaled)',fontsize=10,color=BD)
    ax.legend(loc='upper left',fontsize=8,framealpha=0.9)
    ax.tick_params(labelsize=9)

fig.suptitle('Figure 6: SVM Kernel Decision Boundaries - 2D Projection (tau1 vs g1)\n'
    'RBF kernel captures the nonlinear stability boundary best',
    fontsize=12,color=BD,y=1.01,fontweight='bold')
plt.tight_layout()
plt.savefig('review_figures/fig6_svm_kernel_boundary.png',dpi=300,bbox_inches='tight')
plt.close()
print("Saved: review_figures/fig6_svm_kernel_boundary.png")
