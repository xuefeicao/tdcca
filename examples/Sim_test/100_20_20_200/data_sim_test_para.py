import sys
import os
import numpy as np 
sys.path.append('../../..')
from tdcca import *

import matplotlib
matplotlib.pyplot.switch_backend('agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale


n, d_1, d_2, T = 100, 20, 20, 200
ext = str(n) + '_' + str(d_1) + '_' + str(d_2) + '_' + str(T) + '/'
folder_name = 'data/' + ext 

num_sim = 1

u_1 = np.zeros((d_1,))
v_1 = np.zeros((d_2,))
u_1[0:d_1/4] = 1
u_1[d_1/4:d_1/2] = -1
v_1[d_2/2:d_2/4*3] = 1
v_1[d_2/4*3:] = -1

u_2 = np.zeros((d_1,))
v_2 = np.zeros((d_2,))
u_2[d_1/4:d_1/2] = -1
u_2[d_1/2:d_1/4*3] = 1
v_2[0:d_2/4] = 1
v_2[d_2/4*3:] = -1 
data = [0] * num_sim
folder_name_all = [0] * num_sim

if not os.path.exists(folder_name):
    os.makedirs(folder_name)

for i in range(num_sim):
    # generate data X_t \in R^{n,d_1}, Y_t \in R^{n,d_2}
    X = np.zeros((n, d_1, T))
    Y = np.zeros((n, d_2, T))
    folder_name_i = folder_name + 'sim_' + str(i) + '/'
    if not os.path.exists(folder_name_i):
            os.makedirs(folder_name_i)
    folder_name_all[i] = folder_name_i
    sigma = 0.1
    sigma_1 = 0.1
    for t in range(T/2):
        Z = np.random.normal(size=n)

        for j in range(n):
            u_1_1 = u_1 + sigma*np.random.normal(0, np.ones((d_1,)))
            v_1_1 = v_1 + sigma*np.random.normal(0, np.ones((d_2,)))
            X[j,:,t] = Z[j]*u_1_1 + sigma_1 * np.random.normal(0, np.ones((d_1,)))
            Y[j,:,t] = Z[j]*v_1_1 + sigma_1 * np.random.normal(0, np.ones((d_2,)))

    for t in range(T/2,T):
        Z = np.random.normal(size=n)
       
        for j in range(n):
            u_2_1 = u_2 + sigma*np.random.normal(0, np.ones((d_1,)))
            v_2_1 = v_2 + sigma*np.random.normal(0, np.ones((d_2,)))
            X[j,:,t] = Z[j]*u_2_1 + sigma_1 * np.random.normal(0, np.ones((d_1,)))
            Y[j,:,t] = Z[j]*v_2_1 + sigma_1 * np.random.normal(0, np.ones((d_2,)))

        

    data[i] = {0: X, 1:Y}

nu = [10]
lam = [0.0001, 0.001]
mu = [0.0001, 0.0002, 0.0005, 0.0008, 0.001, 0.002, 0.005]

        

N = len(data[0])
l = 0
real_W = {}
real_W[(0,l)] = np.concatenate((np.repeat(u_1.reshape((-1,1)),T/2,axis=1), np.repeat(u_2.reshape((-1,1)),T/2,axis=1)),axis=1)
real_W[(1,l)] = np.concatenate((np.repeat(v_1.reshape((-1,1)),T/2,axis=1), np.repeat(v_2.reshape((-1,1)),T/2,axis=1)),axis=1)
# see help(multi_sim) for function details
multi_sim(data, lam, mu, nu, 1, folder_name_all, real_W, num_cores=1, admm_method='admm_2',max_iter=5000, tol_admm=1e-4, T_dif = [T/2-1], folds=5, num_val=2, out_put=False)










