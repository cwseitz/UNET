import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import json
import argparse
from matplotlib import cm
from arwn.utils import format_ax
from arwn import dynamics

#a - translation rate
#b - protein degradation rate
#c - RNA degradation rate

def main(config):

    yeast = dynamics.HillYeastExample(config['T'], 
                                      config['Nt'], 
                                      config['trials'],
                                      config['Nrecord'],
                                      config['a'],
                                      config['b'],
                                      config['c'],
                                      config['x0_min'],
                                      config['x0_max'],
                                      config['y0_min'],
                                      config['y0_max'],
                                      config['mu_nx'],
                                      config['sg_nx'],
                                      config['mu_ny'],
                                      config['sg_ny'],
                                      config['h'],
                                      config['bias'],
                                      config['q'],
                                      config['n'],
                                      plot=False, 
                                      cmap='viridis')
    X,Y = yeast.run_dynamics()
    fig, ax = plt.subplot_mosaic([['left', 'upper right'],['left', 'lower right']],
                                  figsize=(5.5, 3.5), constrained_layout=True)

    yeast._add_graph_to_axis(ax=ax['left'])
    yeast._add_dyn_to_axis(ax['upper right'],ax['lower right'])

    format_ax(ax['upper right'], xlabel='Time', ylabel='[RNA]', ax_is_box=False, legend_bbox_to_anchor=(1,1))
    format_ax(ax['lower right'], xlabel='Time', ylabel='[Protein]', ax_is_box=False, show_legend=False)

    plt.tight_layout()
    fig, ax2 = plt.subplots(1,2)
    ax2[0].imshow(yeast.mat)
    ax2[1].imshow(yeast.adj)
    plt.show()
    plt.show()


if __name__ == '__main__':

    args = argparse.ArgumentParser(description='Hill Gene Network')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args = args.parse_args()
    with open(args.config, 'r') as f:
        config = json.load(f)
        main(config)
