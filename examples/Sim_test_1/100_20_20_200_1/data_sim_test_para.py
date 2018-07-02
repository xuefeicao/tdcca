import sys
import os
import numpy as np 
sys.path.append('../../..')
from tdcca import *



n, d_1, d_2, T = 100, 20, 20, 200
ext = str(n) + '_' + str(d_1) + '_' + str(d_2) + '_' + str(T) + '_1/'
# write a folder name where you want to save the analysis
folder_name = 'data/' + ext 

# number of independent trials 
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
rho = 0.3
A = np.zeros((d_1, d_1))
B = np.zeros((d_2, d_2))
for i in range(d_1):
    for j in range(d_1):
        A[i,j] = rho**(abs(i-j))
for i in range(d_2):
    for j in range(d_2):
        B[i,j] = rho**(abs(i-j))

re_use = False 
cor = 0.9
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

for i in range(num_sim):
    X = np.zeros((n, d_1, T))
    Y = np.zeros((n, d_2, T))
    folder_name_i = folder_name + 'sim_' + str(i) + '/'
    if not os.path.exists(folder_name_i):
        os.makedirs(folder_name_i)
    folder_name_all[i] = folder_name_i
    if not re_use:
        sigma = 0
        sigma_1 = 0.1
        Cov = np.zeros((d_1+d_2, d_1+d_2))

        
        for t in range(T/2):
            scl_1 = 1/ np.dot(u_1,np.dot(A, u_1))
            scl_2 = 1/ np.dot(v_1, np.dot(B, v_1))
            Cov[0:d_1,0:d_1] = A * scl_1
            Cov[d_1:(d_1+d_2),d_1:(d_1+d_2)] = B * scl_2
            for j in range(n):
                u_1_1 = u_1 + sigma*np.random.normal(0, np.ones((d_1,)))
                v_1_1 = v_1 + sigma*np.random.normal(0, np.ones((d_2,)))
                Cov[0:d_1,d_1:(d_1+d_2)] = cor * np.dot(np.dot(A*scl_1,u_1_1).reshape((-1,1)), np.dot(v_1_1, B*scl_2).reshape((1,-1)))
                Cov[d_1:(d_1+d_2), 0:d_1] = Cov[0:d_1,d_1:(d_1+d_2)].T
                Z = np.random.multivariate_normal(np.zeros((d_1+d_2,)), Cov)
                X[j,:,t] = Z[0:d_1] + scl_1*sigma_1 * np.random.normal(0, np.ones((d_1,)))
                Y[j,:,t] = Z[d_1:(d_1+d_2)] + scl_2*sigma_1 * np.random.normal(0, np.ones((d_2,)))
             
        for t in range(T/2,T):
            scl_1 = 1/ np.dot(u_2,np.dot(A, u_2))
            scl_2 = 1/ np.dot(v_2, np.dot(B, v_2))
            Cov[0:d_1,0:d_1] = A * scl_1
            Cov[d_1:(d_1+d_2),d_1:(d_1+d_2)] = B * scl_2
            for j in range(n):
                u_2_1 = u_2 + sigma*np.random.normal(0, np.ones((d_1,)))
                v_2_1 = v_2 + sigma*np.random.normal(0, np.ones((d_2,)))
                Cov[0:d_1,d_1:(d_1+d_2)] = cor * np.dot(np.dot(A*scl_1,u_2_1).reshape((-1,1)), np.dot(v_2_1, B*scl_2).reshape((1,-1)))
                Cov[d_1:(d_1+d_2), 0:d_1] = Cov[0:d_1,d_1:(d_1+d_2)].T
                Z = np.random.multivariate_normal(np.zeros((d_1+d_2,)), Cov)
                
                X[j,:,t] = Z[0:d_1] + scl_1*sigma_1 * np.random.normal(0, np.ones((d_1,)))
                Y[j,:,t] = Z[d_1:(d_1+d_2)] + scl_2*sigma_1 * np.random.normal(0, np.ones((d_2,)))
    data[i] = {0: X, 1:Y}

    







l = 0
real_W = {}
real_W[(0,l)] = np.concatenate((np.repeat(u_1.reshape((-1,1)),T/2,axis=1), np.repeat(u_2.reshape((-1,1)),T/2,axis=1)),axis=1)
real_W[(1,l)] = np.concatenate((np.repeat(v_1.reshape((-1,1)),T/2,axis=1), np.repeat(v_2.reshape((-1,1)),T/2,axis=1)),axis=1)




nu = [10**i for i in range(1, 2)]
lam = [(10**i) for i in range(-2,2)] + [5*(10**i) for i in range(-2,2)]
mu = [10**i for i in range(-2,2)] + [5*(10**i) for i in range(-2,2)] 


multi_sim(data, lam, mu, nu, 1, folder_name_all, real_W, num_cores=1, admm_method='admm_2',max_iter=5000, tol_admm=1e-4, T_dif = [T/2-1], folds=5, num_val=2, scaling=True, pre_sign=True, calculate_init=True)











