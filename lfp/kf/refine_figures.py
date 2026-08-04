#!/usr/bin/env python3
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
temp=pd.read_csv(ROOT/'results/temperature_metrics.csv'); comp=pd.read_csv(ROOT/'results/complexity.csv')
oracle=temp[temp.initial_condition=='oracle']; best='hysteresis_2rc_ekf'
heat=oracle[oracle.model==best].pivot(index='profile',columns='temperature_C',values='MAE_pct')
fig,ax=plt.subplots(figsize=(10,3.5)); im=ax.imshow(heat,aspect='auto',cmap='magma'); ax.set_xticks(range(len(heat.columns)),[f'{x:g}' for x in heat.columns]); ax.set_yticks(range(len(heat.index)),heat.index); ax.set_xlabel('Temperature (°C)')
for i in range(len(heat.index)):
    for j in range(len(heat.columns)):
        value=heat.iloc[i,j]; ax.text(j,i,f'{value:.1f}',ha='center',va='center',color='white' if value<20 else 'black',fontsize=9)
fig.colorbar(im,ax=ax,label='SOC MAE (%)'); fig.tight_layout(); fig.savefig(ROOT/'figures/profile_temperature_heatmap.png',dpi=180); plt.close(fig)

acc=oracle.groupby('model').MAE_pct.mean().rename('MAE_pct').to_frame().join(comp.groupby('model').mean_step_latency_us.mean())
labels={'coulomb_count':'CC','plain_2rc_ekf':'plain 2RC-EKF','hysteresis_2rc_ekf':'2RC+H EKF','adaptive_hysteresis_2rc_ekf':'adaptive 2RC+H EKF','hysteresis_2rc_ukf':'2RC+H UKF'}
offsets={'coulomb_count':(6,5),'plain_2rc_ekf':(-75,10),'hysteresis_2rc_ekf':(8,10),'adaptive_hysteresis_2rc_ekf':(8,-18),'hysteresis_2rc_ukf':(-85,5)}
colors={'coulomb_count':'#1b9e77','plain_2rc_ekf':'#7570b3','hysteresis_2rc_ekf':'#d95f02','adaptive_hysteresis_2rc_ekf':'#66a61e','hysteresis_2rc_ukf':'#e7298a'}
fig,ax=plt.subplots(figsize=(8,5))
for model,row in acc.iterrows():
    ax.scatter(row.mean_step_latency_us,row.MAE_pct,s=65,color=colors[model],zorder=3)
    ax.annotate(labels[model],(row.mean_step_latency_us,row.MAE_pct),xytext=offsets[model],textcoords='offset points',fontsize=9,arrowprops={'arrowstyle':'-','lw':.5,'color':'0.5'})
ax.set_xscale('log'); ax.set_xlabel('PC mean step latency (µs)'); ax.set_ylabel('SOC MAE (%)'); ax.grid(True,which='both',alpha=.2); fig.tight_layout(); fig.savefig(ROOT/'figures/accuracy_latency_tradeoff.png',dpi=180); plt.close(fig)
print('refined figures')
