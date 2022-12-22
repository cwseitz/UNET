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

    yeast = dynamics.LinearYeast25(config['T'], 
                                      config['Nt'], 
                                      config['trials'],
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
                                      plot=False, 
                                      cmap='coolwarm')
                                      
    X,Y = yeast.run_dynamics()
    fig, ax = plt.subplots()
    yeast._add_graph_to_axis(ax=ax)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    args = argparse.ArgumentParser(description='Linear Gene Network')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args = args.parse_args()
    with open(args.config, 'r') as f:
        config = json.load(f)
        main(config)
