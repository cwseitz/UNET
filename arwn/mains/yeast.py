import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
from matplotlib import cm
from _format_ax import *
from arwn import models

T, Nt, trials = 100,1000,1
yeast = models.LinearYeastExample(T, Nt, trials, plot=False, cmap='viridis')
X,Y = yeast.run_dynamics()

fig, ax = plt.subplot_mosaic([['left', 'upper right'],['left', 'lower right']],
                              figsize=(5.5, 3.5), constrained_layout=True)

yeast._add_graph_to_axis(ax=ax['left'])
yeast._add_dyn_to_axis(ax['upper right'],ax['lower right'])

format_ax(ax['upper right'], xlabel='Time', ylabel='[RNA]', ax_is_box=False, legend_bbox_to_anchor=(1,1))
format_ax(ax['lower right'], xlabel='Time', ylabel='[Protein]', ax_is_box=False, show_legend=False)


plt.tight_layout()
plt.show()
