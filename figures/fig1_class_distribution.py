# Figure 1: Class Distribution in the UCI Smart Grid Stability Dataset
# Author : Pablo Soham | 2nd Year CS | 2026
# Output : output_figures/fig1_class_distribution.png  (300 DPI)
# Usage  : python figures/fig1_class_distribution.py
# Requires: pip install matplotlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

classes  = ['Unstable', 'Stable']
counts   = [6380, 3620]
percents = [63.8, 36.2]

plt.rcParams.update({
    'font.family':'DejaVu Sans','axes.spines.top':False,
    'axes.spines.right':False,'figure.facecolor':'white',
    'axes.facecolor':'#F8F9FA','axes.grid':True,
    'grid.color':'white','grid.linewidth':1.2,
})

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(classes, counts, color=['#8B0000','#1A3C5E'],
              width=0.5, edgecolor='white', linewidth=1.5, zorder=3)

for bar, cnt, pct in zip(bars, counts, percents):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+80,
            f'{cnt:,}\n({pct}%)', ha='center', va='bottom',
            fontsize=13, fontweight='bold', color='#1A3C5E')

ax.set_ylim(0, 7800)
ax.set_ylabel('Number of Instances', fontsize=13, color='#1A3C5E')
ax.set_xlabel('Grid State Label', fontsize=13, color='#1A3C5E')
ax.set_title('Figure 1: Class Distribution — UCI Smart Grid Stability Dataset\n'
             'Total: 10,000 instances  |  Unstable: 6,380 (63.8%)  |  Stable: 3,620 (36.2%)',
             fontsize=11, color='#1A3C5E', pad=14)
ax.tick_params(labelsize=12)

os.makedirs('output_figures', exist_ok=True)
plt.tight_layout()
plt.savefig('output_figures/fig1_class_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: output_figures/fig1_class_distribution.png")
