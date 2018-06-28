import sys
import os
import numpy as np 
#sys.path.append('/gpfs_home/xcao1/scratch/research_Rossi_1/tcca/tcca_1/src')
sys.path.append('/gpfs/data/xl6/xuefei/research_Rossi_1/tcca/tcca_1/src')
#sys.path.append('/home/chen604/caoxuefei/Research/tcca/tcca_1/src')
from multi_sim_val import multi_sim 
import matplotlib
matplotlib.pyplot.switch_backend('agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale


n, d_1, d_2, T = 100, 50, 50, 200
ext = str(n) + '_' + str(d_1) + '_' + str(d_2) + '_' + str(T) + '_1/'
folder_name = '/gpfs/data/xl6/xuefei/research_Rossi_1/data/tcca/test_with_val/test_para_1/' + ext 
#folder_name = '/home/chen604/caoxuefei/Research/data/tcca/' + ext 

num_sim = 50

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

re_use = True 
cor = 0.9
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
                #if np.amin(np.linalg.eig(Cov)[0]) < 0:
                #    raise Exception('bad eig')
                #print np.amin(np.linalg.eig(Cov)[0])
                Z = np.random.multivariate_normal(np.zeros((d_1+d_2,)), Cov)
                #print Z
                X[j,:,t] = Z[0:d_1] + scl_1*sigma_1 * np.random.normal(0, np.ones((d_1,)))
                Y[j,:,t] = Z[d_1:(d_1+d_2)] + scl_2*sigma_1 * np.random.normal(0, np.ones((d_2,)))
             
            #print np.sum(np.dot(u_1, (X_ans).T)**2), np.sum(np.dot(v_1, (Y_ans).T)**2)
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
                #if np.amin(np.linalg.eig(Cov)[0]) < 0:
                #    raise Exception('bad eig')
                Z = np.random.multivariate_normal(np.zeros((d_1+d_2,)), Cov)
                
                X[j,:,t] = Z[0:d_1] + scl_1*sigma_1 * np.random.normal(0, np.ones((d_1,)))
                Y[j,:,t] = Z[d_1:(d_1+d_2)] + scl_2*sigma_1 * np.random.normal(0, np.ones((d_2,)))
            #print np.sum(np.dot(u_2, (X_ans/(n**0.5)).T)**2), np.sum(np.dot(v_2, (Y_ans/(n**0.5)).T)**2)


    #for t in range(T):
    #    X[:,:,t] = scale(X[:,:,t], with_std=False)
    #    X[:,:,t] = X[:,:,t]/(np.amax(np.sum(X[:,:,t]**2,axis=0)**0.5)*d_1)
    #    Y[:,:,t] = scale(Y[:,:,t], with_std=False)
    #    Y[:,:,t] = Y[:,:,t]/(np.amax(np.sum(Y[:,:,t]**2,axis=0)**0.5)*d_2)
    data[i] = {0: X, 1:Y}

    




#lam = [10**i for i in range(-4,0)]


l = 0
real_W = {}
real_W[(0,l)] = np.concatenate((np.repeat(u_1.reshape((-1,1)),T/2,axis=1), np.repeat(u_2.reshape((-1,1)),T/2,axis=1)),axis=1)
real_W[(1,l)] = np.concatenate((np.repeat(v_1.reshape((-1,1)),T/2,axis=1), np.repeat(v_2.reshape((-1,1)),T/2,axis=1)),axis=1)

#nu = [10**i for i in range(-5,-3)]
#lam = [10**i for i in range(3,5)] + [5*(10**i) for i in range(3,5)]
#mu = [10**i for i in range(3,5)] +  [5*(10**i) for i in range(3,5)] + [1e5, 5e5, 1e6]
nu = [10**i for i in range(1, 2)]
lam = [(10**i) for i in range(-2,2)] + [5*(10**i) for i in range(-2,2)] 
mu = [10**i for i in range(-2,2)] + [5*(10**i) for i in range(-2,2)] 

# scaling, by_pass, pre_sign, with_sign, with_init 
multi_sim(data, lam, mu, nu, folder_name_all, real_W, num_cores=10, admm_method='admm_2',max_iter=5000, tol_admm=1e-4, T_dif = [T/2-1], folds=5, num_val=2, scaling=False, pre_sign=True, with_sign=True, by_pass=False, calculate_init=True)








# f, axarr= plt.subplots(2, 1)
# axarr[0].plot(range(d_1),X[0,:,2],color='r')
# axarr[0].set_title('X')
# axarr[1].plot(range(d_2),Y[0,:,2],color='r')
# axarr[1].set_title('Y')
# f.savefig(folder_name+'XY.png')





