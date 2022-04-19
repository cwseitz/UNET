import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sentinel.models import *
from sentinel.util import *
from _format_ax import *

##################################################
## Run a Monte Carlo simulation on a Yeast network
##################################################
## Author: Clayton Seitz
## Copyright: 2021, The Hebb Project
## Email: cwseitz@uchicago.edu
##################################################

save_dir = '/home/cwseitz/Desktop/data/'

########################
## Load Params
########################

with open(save_dir + 'params.json', 'r') as fh:
    params = json.load(fh)

x0 = np.load(save_dir + 'mc_grn_x0.npz')['arr_0']
W = np.load(save_dir + 'mc_grn_w.npz')['arr_0']
mat = np.load(save_dir + 'mc_grn_mat.npz')['arr_0']

########################
## Run Monte-Carlo sim
########################

grn = LinearGRN(params['N'],
                params['trials'],
                params['Nrecord'],
                params['T'],
                params['Nt'])

grn.call(x0,W,mat)
nbatch, nsteps, nmol = grn.X.shape
ngenes = nmol // 2

jet = cm.get_cmap('rainbow')
colors = jet(np.linspace(0,1,nmol))
fig, ax = plt.subplots(2,3, figsize=(12,6))
trial_idx = 0

def add_spring_graph(ax, net, alpha=0.05, sparse=False, arrows=False):

    """
    Draw a graph in spring format
    Parameters
    ----------
    ax : object,
        matplotlib axis object
    net : object,
        network object
    alpha : float, optional
        transparency param
    arrows : bool, optional
        whether or not to draw the direction of an edge via arrows
    """

    if arrows:
        arrows = True
    if sparse:
        G = nx.convert_matrix.from_scipy_sparse_matrix(net.C, create_using=nx.DiGraph)
    else:
        G = nx.convert_matrix.from_numpy_array(net.C, create_using=nx.DiGraph)

    pos = nx.spring_layout(G)
    colors = []
    for n in G.nodes():
        colors.append('dodgerblue')
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=20, node_shape='x')
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='black', alpha=alpha, arrows=arrows, arrowsize=10)

add_spring_graph(ax[0,0],
for n in range(ngenes):
    r = grn.X[trial_idx,:,n]
    p = grn.X[trial_idx,:,2*n]
    ax[0,1].plot(r,color=colors[n])
    ax[1,1].plot(p,color=colors[2*n])

ax[0,1].legend()
ax[0,2].set_axis_off()
format_ax(ax[0,1], xlabel='Time', ylabel='[RNA]', ax_is_box=False, legend_bbox_to_anchor=(1,1))
format_ax(ax[1,1], xlabel='Time', ylabel='[PR]', ax_is_box=False, show_legend=False)
ax[1,0].set_axis_off()
ax[1,2].set_axis_off()

plt.tight_layout()
plt.show()

