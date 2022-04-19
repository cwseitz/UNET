import json
import numpy as np
import matplotlib.pyplot as plt
from sentinel.models import *
from sentinel.util import *
from sentinel.data import *

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
w1 = np.load(save_dir + 'mc_grn_w1.npz')['arr_0']
w2 = np.load(save_dir + 'mc_grn_w2.npz')['arr_0']
h = np.load(save_dir + 'mc_grn_h.npz')['arr_0']
K = np.load(save_dir + 'mc_grn_K.npz')['arr_0']
b = np.load(save_dir + 'mc_grn_b.npz')['arr_0']
lam = np.load(save_dir + 'mc_grn_lam.npz')['arr_0']
q = np.load(save_dir + 'mc_grn_q.npz')['arr_0']
n = np.load(save_dir + 'mc_grn_n.npz')['arr_0']

#######################################
## Parameters
#######################################

#######################################
## Build graph from DOT and visualize
#######################################

graph, adj = load_yeast_example()
mat = build_linear_system(adj)

########################
## Run Monte-Carlo sim
########################

grn = HillGRN(params['N'],
              params['trials'],
              params['Nrecord'],
              params['T'],
              params['Nt'])

grn.call(x0,w1,w2,mat)

#colors = ['red','blue','purple']
#mu_x = np.mean(grn.X,axis=0) #ensemble average
# mu_t = np.mean(grn.X[:,params['Nt']//2:,:],axis=1) #time average
# for i in range(params['N']):
#     plt.plot(mu_x[:,i],color=colors[i])
#     plt.plot(mu_t[:,i],color=colors[i],linestyle='dotted')
# plt.show()
