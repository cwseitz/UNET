import json
import numpy as np
import matplotlib.pyplot as plt
from sentinel.data import *

##################################################
## Set params for a Monte Carlo simulation for
## a test gene regulatory network
##################################################
## Author: Clayton Seitz
## Copyright: 2021, The Sentinel Project
## Email: cwseitz@iu.edu
##################################################

save_dir = '/home/cwseitz/Desktop/data/'

########################
## Define network params
########################

trials = 1
T = 10
Nt = 1000
dt = T/Nt

########################
## Params
########################

graph, adj = load_yeast_example()
mat = build_linear_system(adj)
vals, vecs = np.linalg.eig(mat)
N = mat.shape[0]//2; Nrecord = N
x0 = np.concatenate([0*np.ones((N,)),np.zeros((N,))])
W = 10*np.sqrt(dt)*np.random.normal(0,1,size=(trials,2*N,Nt))

########################
## Dump params to disk
########################

params = {
'trials':trials,
'N':N,
'Nrecord':Nrecord,
'T':T,
'Nt':Nt,
}

np.savez_compressed(save_dir + 'mc_grn_x0', x0)
np.savez_compressed(save_dir + 'mc_grn_w', W)
np.savez_compressed(save_dir + 'mc_grn_mat', mat)

with open(save_dir + 'params.json', 'w') as fp:
    json.dump(params, fp)
