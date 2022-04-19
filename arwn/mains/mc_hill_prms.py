import json
import numpy as np

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

trials = 1000
N = 3
Nrecord = 10
T = 1000
Nt = 1000
dt = T/Nt

########################
## Params
########################

x0 = 100*np.ones((N,))
w1 = np.sqrt(dt)*np.random.normal(0,1,size=(trials,N,Nt))
w2 = np.sqrt(dt)*np.random.normal(0,1,size=(trials,N,Nt))
h = 0.05*np.ones((N,N))
K = np.ones((N,N))
b = np.zeros((N,))
lam = 0.1*np.ones((N,))
q = np.ones((N,))
n = np.ones((N,N))

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
np.savez_compressed(save_dir + 'mc_grn_w1', w1)
np.savez_compressed(save_dir + 'mc_grn_w2', w2)
np.savez_compressed(save_dir + 'mc_grn_h', h)
np.savez_compressed(save_dir + 'mc_grn_K', K)
np.savez_compressed(save_dir + 'mc_grn_b', b)
np.savez_compressed(save_dir + 'mc_grn_lam', lam)
np.savez_compressed(save_dir + 'mc_grn_q', q)
np.savez_compressed(save_dir + 'mc_grn_n', n)

with open(save_dir + 'params.json', 'w') as fp:
    json.dump(params, fp)
