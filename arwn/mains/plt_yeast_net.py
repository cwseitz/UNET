import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm
from _format_ax import *

df = pd.read_csv('data/yeast/Yeast-1_dream4_timeseries.tsv', sep='\t')
df2 = pd.read_csv('data/yeast/Yeast-1_proteins_dream4_timeseries.tsv', sep='\t')
arr = df.to_numpy()
names = list(df.columns)[1:]
arr = arr[:,1:]
nsteps, ngenes = arr.shape

arr2 = df2.to_numpy()
arr2 = arr2[:,1:]

jet = cm.get_cmap('rainbow', 12)
colors = jet(np.linspace(0,1,ngenes))
fig, ax = plt.subplots(2,3, figsize=(12,6))

ax[0,0].set_axis_off()
for n in range(ngenes):
    ax[0,1].plot(arr[:,n],color=colors[n], label=names[n])
    ax[1,1].plot(arr2[:,n],color=colors[n], label=names[n])

ax[0,1].legend()
ax[0,2].set_axis_off()
format_ax(ax[0,1], xlabel='Time', ylabel='[RNA]', ax_is_box=False, legend_bbox_to_anchor=(1,1))
format_ax(ax[1,1], xlabel='Time', ylabel='[PR]', ax_is_box=False, show_legend=False)
ax[1,0].set_axis_off()
ax[1,2].set_axis_off()

plt.tight_layout()
plt.show()
