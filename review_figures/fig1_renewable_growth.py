"""
Figure 1: Global Renewable Energy Capacity Growth (2015-2025)
Review Article: ML for Smart Grid Stability Prediction
Author: Soham Ambudkar | Don Bosco Symposium on AI in Education 2026
Output: review_figures/fig1_renewable_growth.png (300 DPI)
Run:    python review_figures/fig1_renewable_growth.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('review_figures', exist_ok=True)

years  = np.array([2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2025])
solar  = np.array([222, 291, 385, 480, 586, 714, 843,1053,1419,1800,2200])
wind   = np.array([433, 487, 540, 591, 650, 743, 825, 899, 982,1100,1260])
hydro  = np.array([1246,1272,1292,1313,1330,1355,1392,1428,1460,1490,1520])
others = np.array([112, 121, 130, 140, 150, 164, 178, 191, 206, 224, 245])
total  = solar + wind + hydro + others

BLUE_DARK='#1A3C5E'; BLUE_MID='#2C5F8A'; TEAL='#148F77'; GREY='#7F8C8D'

plt.rcParams.update({'font.family':'DejaVu Sans','figure.facecolor':'white',
    'axes.spines.top':False,'axes.spines.right':False,
    'axes.facecolor':'#F8F9FA','axes.grid':True,
    'grid.color':'white','grid.linewidth':1.4})

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,6))
fig.patch.set_facecolor('white')

ax1.stackplot(years,solar,wind,hydro,others,
    labels=['Solar PV','Wind','Hydropower','Other RES'],
    colors=['#F39C12',BLUE_MID,TEAL,'#7F8C8D'],alpha=0.85)
ax1.set_xlabel('Year',fontsize=13,color=BLUE_DARK)
ax1.set_ylabel('Installed Capacity (GW)',fontsize=13,color=BLUE_DARK)
ax1.set_title('Global Renewable Energy Installed\nCapacity by Source (2015-2025)',
    fontsize=12,color=BLUE_DARK,pad=14)
ax1.legend(loc='upper left',fontsize=10,framealpha=0.9)
ax1.set_xlim(2015,2025)
ax1.tick_params(labelsize=11)
ax1.annotate('Record: 295 GW\nadded in 2022\n(IEA, 2023)',
    xy=(2022,2500),xytext=(2019,3500),fontsize=9,color=BLUE_DARK,fontweight='bold',
    arrowprops=dict(arrowstyle='->',color=BLUE_DARK,lw=1.5),
    bbox=dict(boxstyle='round,pad=0.3',facecolor='#FFF9C4',edgecolor=BLUE_DARK))

annual_solar=np.diff(solar,prepend=solar[0])
annual_wind=np.diff(wind,prepend=wind[0])
annual_total=np.diff(total,prepend=total[0])

ax2.fill_between(years,annual_total,alpha=0.15,color=BLUE_DARK)
ax2.plot(years,annual_total,'o-',color=BLUE_DARK,lw=2.5,markersize=7,label='Total additions')
ax2.plot(years,annual_solar,'s--',color='#F39C12',lw=2,markersize=6,label='Solar PV additions')
ax2.plot(years,annual_wind,'^--',color=BLUE_MID,lw=2,markersize=6,label='Wind additions')
ax2.axhline(y=295,color='red',lw=1.5,linestyle=':',alpha=0.7,label='Record 295 GW (2022)')
ax2.set_xlabel('Year',fontsize=13,color=BLUE_DARK)
ax2.set_ylabel('Annual Capacity Additions (GW)',fontsize=13,color=BLUE_DARK)
ax2.set_title('Annual Renewable Energy Capacity\nAdditions (2015-2025)',
    fontsize=12,color=BLUE_DARK,pad=14)
ax2.legend(loc='upper left',fontsize=9,framealpha=0.9)
ax2.tick_params(labelsize=11)

fig.suptitle('Figure 1: Global Renewable Energy Growth - The Context for Smart Grid\n'
    'Stability Challenges  |  Source: IEA Renewables 2023',
    fontsize=12,color=BLUE_DARK,y=1.01,fontweight='bold')
plt.tight_layout()
plt.savefig('review_figures/fig1_renewable_growth.png',dpi=300,bbox_inches='tight')
plt.close()
print("Saved: review_figures/fig1_renewable_growth.png")
