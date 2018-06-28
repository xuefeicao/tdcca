from six.moves import cPickle as pkl
from scipy.io import loadmat 
import numpy as np
import sys
sys.path.append('../../src/')
from admm_computation_2 import eval_zero

ff = ['100_20_20_200', '100_50_50_200', '20_20_20_200', '50_100_100_200']


folder = '/gpfs/data/xl6/xuefei/research_Rossi_1/data/tcca/test_with_val/test_para/'
sim_n = 50

for f in ff:
    folder_name = folder + f 
    data = loadmat(folder_name + '/RSCCA.mat')
    print data.keys()
    W_R_0 = data['W_1']
    W_R_1 = data['W_2']
    cc = data['cc']
    T = W_R_0.shape[1]
    d_1, d_2 = W_R_0.shape[0], W_R_1.shape[1]

    auc = [0] * sim_n
    f1 = [0] * sim_n
    dif = [0] * sim_n 
    angle = [0] * sim_n 


    auc_td = [0] * sim_n
    f1_td = [0] * sim_n
    dif_td = [0] * sim_n 
    angle_td = [0] * sim_n 
    cor_td = [0] * sim_n 

    D = np.zeros((T,T-1))
    for t in range(T-1):
        D[t,t] = 1
        D[t+1,t] = -1

    for i in range(sim_n):
        f_name = folder_name + '/sim_' + str(i) + '/full_data/0/W.pkl'
        dd = pkl.load(open(f_name))
        real_W = dd['real_W']
        W_t = dd['init_tcca'][0]['W_t']
        auc[i], f1[i] = eval_zero({(0,0): W_R_0[:,:,i], (1,0): W_R_1[:,:,i]}, 2, T, 1, real_W) 
        auc_td[i], f1_td[i], cor_td[i], dif_td[i] = np.mean(dd['auc_score']), np.mean(dd['F1_score']), np.mean(dd['cor_score']), np.mean(dd['T_dif_score'])
        tmp = np.mean(abs(np.dot(W_R_0[:,:,i], D)), axis=0) + np.mean(abs(np.dot(W_R_1[:,:,i], D)), axis=0)
        dif[i] = tmp[T/2-1]/np.mean(tmp)
        for t in range(T):
            #print abs(np.dot(W_t[(0,0)][:,t], real_W[(0,0)][:,t])),np.sum(W_t[(0,0)][:,t]**2)**0.5, np.sum(real_W[(0,0)]**2)**0.5
            angle[i] += abs(np.dot(W_R_0[:,t, i], real_W[(0,0)][:,t]))/(np.sum(W_R_0[:,t,i]**2)**0.5*np.sum(real_W[(0,0)][:,t]**2)**0.5)
            angle[i] += abs(np.dot(W_R_1[:,t, i], real_W[(1,0)][:,t]))/(np.sum(W_R_1[:,t,i]**2)**0.5*np.sum(real_W[(1,0)][:,t]**2)**0.5)
            angle_td[i] += abs(np.dot(W_t[(0,0)][:,t], real_W[(0,0)][:,t]))/(np.sum(W_t[(0,0)][:,t]**2)**0.5*np.sum(real_W[(0,0)][:,t]**2)**0.5)
            angle_td[i] += abs(np.dot(W_t[(1,0)][:,t], real_W[(1,0)][:,t]))/(np.sum(W_t[(1,0)][:,t]**2)**0.5*np.sum(real_W[(1,0)][:,t]**2)**0.5)

        angle[i] = angle[i]/(2*T)
        angle_td[i] = angle_td[i]/(2*T)
    print 'SCCA/CAPIT: sim:{0}, auc:{1}, f1:{2}, cor:{3}, dif:{4}, ang:{5}'.format(f, np.mean(auc), np.mean(f1), np.mean(cc), np.mean(dif),
np.mean(angle))

    print 'TD: sim:{0}, auc:{1}, f1:{2}, cor:{3}, dif:{4}, ang:{5}'.format(f, np.mean(auc_td), np.mean(f1_td), np.mean(cor_td), np.mean(dif_td), np.mean(angle_td))





