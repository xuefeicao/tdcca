import numpy as np 
import os
import h5py
import prox_tv as ptv 
from pathos.multiprocessing import ProcessingPool as Pool
from six.moves import cPickle as pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.cluster import KMeans
import copy
import time
from cvxpy import Variable, norm, sum_squares, Problem, Minimize, trace, CVXOPT
from shutil import copyfile
from collections import defaultdict
from collections import OrderedDict
import math
from real_data_val import task_fmri_ev



def admm_comp(data_X, val_config, method, k_fold, with_init={}, test=False):
    """
    main computation algorithm
    @param data_X(class), val_config(class), method('admm_1,admm_2,admm_3')
    @return 
    """
    l = data_X.l 
    N = data_X.N
    T = data_X.T 
    di = data_X.di
    folder_name = data_X.folder_name
    ind1, ind2, ind3 = val_config.ijk
    big_data = False
    # load preprocessed data
    with_init = copy.deepcopy(with_init)
    A_inv = None
    cvx_val = None
    if hasattr(val_config,'real_W'):
        val_config.dif_lam_mu = [[1] for i in range(N)]
    lam, mu, nu = val_config.lam, val_config.mu, val_config.nu
    if test:
        val_config.time = 0
        val_config.cvx_time = 0
        val_config.cvx_time_admm = 0 
    if os.path.exists(folder_name+'UTU.hdf5'):
        UTU = load_dict_from_hdf5(folder_name+'UTU.hdf5')
        US = load_dict_from_hdf5(folder_name + 'US.hdf5')
        P = load_dict_from_hdf5(folder_name+'P.hdf5')
        SVD_ijt = load_dict_from_hdf5(folder_name + 'SVD_ijt.hdf5')
        SVD_x = load_dict_from_hdf5(folder_name+'SVD_x.hdf5')

    elif os.path.exists(folder_name+'svd_all.pkl'):
        with open(folder_name+'svd_all.pkl') as f:
            save = pickle.load(f)
            UTU = save['UTU']
            US = save['US']
            P = save['P']
            SVD_x = save['SVD_x']
            SVD_ijt = save['SVD_ijt']
            s_time = time.time()
            A_inv = dict()
            for i in range(N):
                for t in range(T):
                    ind = np.ravel_multi_index((i,t),(N,T))
                    A_inv[ind] = np.linalg.inv((N-1) * UTU[ind] + 2*nu*np.eye(di[i]))
            if test:
                val_config.time = time.time() - s_time 
                print val_config.time, 'time_0'

    elif os.path.exists(folder_name + 'SVD_x.pkl'):
        with open(folder_name + 'US.pkl') as f:
            US = pickle.load(f)
        with open(folder_name + 'P.pkl') as f:
            P = pickle.load(f)
        with open(folder_name + 'SVD_x.pkl') as f:
            SVD_x = pickle.load(f)
        with open(folder_name + 'SVD_ijt.pkl') as f:
            SVD_ijt = pickle.load(f)
        A_f = h5py.File(folder_name + 'UTU_' + str(ind3) + '.hdf5')
        if os.path.getsize(folder_name + 'UTU_' + str(ind3) + '.hdf5') < 60 * 1024* 1e6:
            A_inv = dict()
            s_time = time.time()
            for i in range(N):
                for t in range(T):
                    ind = np.ravel_multi_index((i,t),(N,T))
                    A_inv[ind] = A_f[str(i) + '/' + str(t)].value
                   
            print 'use processed data:', time.time() - s_time
        else:
            big_data = True
            A_inv = A_f
    else:
        raise Exception('no preprocessed data available')

    

        

    

  

    
    max_iter = data_X.max_iter
    out_put = data_X.out_put
    tol_dif = 1e-4
    #real_W = val_config.real_W
    T_dif = val_config.T_dif
    # cross validation tuning parameters
    
    
    tol_admm = data_X.tol_admm
    W = dict()
    W_h = dict()
    W_t = dict()
    Theta_all = dict()
    Phi_all = dict()
    SVD_ijt_v = dict()
    print di
    for i in range(N):
        for ll in range(l):
            W[(i,ll)] = np.random.uniform(size=(di[i],T))
            W_h[(i,ll)] = np.random.uniform(size=(di[i],T))
            W_t[(i,ll)] = np.random.uniform(size=(di[i],T))
            Theta_all[(i,ll)] = np.random.uniform(size=(di[i],T))
            Phi_all[(i,ll)] = np.random.uniform(size=(di[i],T))
            

            W[(i,ll)] = np.zeros((di[i],T))
            W_h[(i,ll)] = np.zeros((di[i],T))
            W_t[(i,ll)] = np.zeros((di[i],T))
            Theta_all[(i,ll)] = np.zeros((di[i],T))
            Phi_all[(i,ll)] = np.zeros((di[i],T))
            
    for i in range(N):
        for j in range(N):
            for t in range(T):
                SVD_ijt_v[(i,j,t)] = False
    # compare with cvx and cvx_admm
    if test: 
        W_cvx_admm = copy.deepcopy(W)
        W_h_cvx_admm = copy.deepcopy(W_h)
        W_t_cvx_admm = copy.deepcopy(W_t)
        Theta_all_cvx_admm = copy.deepcopy(Theta_all)
        Phi_all_cvx_admm = copy.deepcopy(Phi_all)


    
    # main computation 

    def obj_value_1(W_i, W_i_t, W_i_h, Phi, Theta, i, ll, out_put=False):
        """
        compute the value of objective funciton (ADMM)
        """
        loss = 0
        D = np.zeros((T,T-1))
        for t in range(T-1):
            D[t,t] = 1
            D[t+1,t] = -1

        loss += np.sum(np.absolute(np.dot(W_i, D)))

        loss0 = np.sum(np.absolute(W_i_t))
        loss_admm_2 = (np.linalg.norm(W_i-W_i_h,'fro')**2 + np.linalg.norm(W_i-W_i_t,'fro')**2)
        loss_admm_1 = np.sum((W_i-W_i_t)*Theta) + np.sum((W_i-W_i_h)*Phi)

        loss1 = 0
        
        for t in range(T):
            Sigma_iti, U_1it= SVD_x[np.ravel_multi_index((i,t),(N,T))][1:3]
            for j in range(i+1,N):
                loss1 += np.linalg.norm((np.dot(np.transpose(U_1it), W_i_h[:,t]) - np.dot(Sigma_iti, SVD_ijt[np.ravel_multi_index((i,j,t),(N,N,T))][0][:,ll])).reshape((-1,1)),'fro')**2

            for w in range(i):
                loss1 += np.linalg.norm((np.dot(np.transpose(U_1it), W_i_h[:,t]) - np.dot(Sigma_iti, SVD_ijt[np.ravel_multi_index((w,i,t),(N,N,T))][1][:,ll])).reshape((-1,1)),'fro')**2
        if out_put:
            print 'without tuning para:', loss, loss0, loss1, loss_admm_1, loss_admm_2
            print 'with tuning para:', loss, loss0*lam, loss1*mu/2.0, loss_admm_1, nu/2.0*loss_admm_2
        loss = loss + loss0*lam + loss1*mu/2.0 + loss_admm_1 + nu/2.0*loss_admm_2
        return loss

    def admm_sep_1(i_ll, tol=1e-2, max_iter=1000):
        """
        computation for one view of data
        """
        obj_value = obj_value_1
        i, ll = i_ll 

        W_i = W[(i,ll)]
        W_i_h = np.random.uniform(size=W_i.shape)
        W_i_t = np.random.uniform(size=W_i.shape)
        Theta = np.random.uniform(size=W_i.shape)
        Phi = np.random.uniform(size=W_i.shape)
        d_i = W_i.shape[0]
         
        loss_0 = obj_value(W_i, W_i_t, W_i_h, Phi, Theta, i, ll)
        loss_1 = loss_0 + 1
        l5 = 1e32
        iter = 0
        while (np.sum(np.absolute(W_i-W_i_h)+np.absolute(W_i-W_i_t)) > tol or abs(loss_0 -loss_1) > tol) and iter <= max_iter:

            #step 1, parallel 
            tmp = (W_i_h + W_i_t)/2 - (Theta + Phi)/(2.0*nu)
            for j in range(d_i):
                W_i[j,:] = ptv.tv1_1d(tmp[j,:], 1.0/(2*nu))
            l1 = obj_value(W_i, W_i_t, W_i_h, Phi, Theta, i, ll)
            if l1 >= l5+tol_dif:
                test_admm(lam, mu, nu, 1, l1, l5)
                break 
            #step 2
            W_i_t = np.sign(W_i + Theta/nu)*np.maximum(np.absolute(W_i + Theta/nu)-1.0*lam/nu,0)
            l2 = obj_value(W_i, W_i_t, W_i_h, Phi, Theta, i, ll)
            if l2 >= l1+tol_dif:
                test_admm(lam, mu, nu, 2, l2, l1)
                break

            #step 3, parallel
            for t in range(T):
                ind = np.ravel_multi_index((i,t),(N,T))
                A = mu * (N-1) * UTU[ind] + nu*np.eye(d_i)
                b = Phi[:,t] + mu * np.dot(US[ind], P[ind][:,ll]) + nu * W_i[:,t]
                W_i_h[:,t] = np.linalg.solve(A, b)
            l3 = obj_value(W_i, W_i_t, W_i_h, Phi, Theta, i, ll)
            if l3 >= l2+tol_dif:
                test_admm(lam, mu, nu, 3, l3, l2)
                break
                

            #step 4
            Theta = Theta + nu*(W_i - W_i_t)
            l4 = obj_value(W_i, W_i_t, W_i_h, Phi, Theta, i, ll)
            if l4 <= l3-tol_dif:
                test_admm(lam, mu, nu, 4, l4, l3)
            #step 5
            Phi = Phi + nu*(W_i - W_i_h)
            l5 = obj_value(W_i, W_i_t, W_i_h, Phi, Theta, i, ll)
            if l5 <= l4-tol_dif:
                test_admm(lam, mu, nu, 5, l5, l4)

            iter += 1
            if iter%10 == 0:
                loss_1 = loss_0
                loss_0 = obj_value(W_i, W_i_t, W_i_h, Phi, Theta, i, ll)


            if iter > max_iter:
                warnings.warn(str(lam)+' '+ str(mu)+ ' '+ str(nu)+'warning: does not converge!')
           

        return W_i_t
             
           
        

    def obj_value_2(W_i, W_i_t, W_i_h, Phi, Theta, i, ll, SVD_ijt, lam=lam, mu=mu, nu=nu, out_put=False):
        """
        compute the value of objective funciton (ADMM)
        """
        loss = 0
        for t in range(T):
            Sigma_iti, U_1it= SVD_x[np.ravel_multi_index((i,t),(N,T))][1:3]
            for j in range(i+1,N):
                loss += np.linalg.norm((np.dot(np.transpose(U_1it), W_i[:,t]) - np.dot(Sigma_iti, SVD_ijt[np.ravel_multi_index((i,j,t),(N,N,T))][0][:,ll])).reshape((-1,1)),'fro')**2

            for w in range(i):
                loss += np.linalg.norm((np.dot(np.transpose(U_1it), W_i[:,t]) - np.dot(Sigma_iti, SVD_ijt[np.ravel_multi_index((w,i,t),(N,N,T))][1][:,ll])).reshape((-1,1)),'fro')**2
        
        loss1 = np.sum(np.absolute(W_i_t))

        D = np.zeros((T,T-1))
        for t in range(T-1):
            D[t,t] = 1
            D[t+1,t] = -1

        loss2 = np.sum(np.absolute(np.dot(W_i_h, D)))
        loss_admm_1 = np.sum((W_i-W_i_t)*Theta) + np.sum((W_i-W_i_h)*Phi)
        loss_admm_2 = (np.linalg.norm(W_i-W_i_h,'fro')**2 + np.linalg.norm(W_i-W_i_t,'fro')**2)
        if out_put:
            print 'without tuning para:', loss, loss1, loss2, loss_admm_1, loss_admm_2 
            print 'with tuning para:', 1.0/2*loss, loss1*lam, loss2*mu, loss_admm_1, nu/2.0*loss_admm_2, lam, mu, nu, ind1, ind2, ind3
            print np.sum(np.absolute(np.dot(W_i_h, D)), axis=0)
            #print W_i_h[0,:]
            #print W_i_h, W_i_t
            return np.sum(np.absolute(np.dot(W_i_h, D)), axis=0)
            #print 'check_norm:'+str(np.sum(np.dot(X[i][:,:,0], W_i_t[:,0].reshape((-1,1)))**2))
        loss = 1.0/2*loss + loss1*lam + loss2*mu+ loss_admm_1 + nu/2.0*loss_admm_2
        return loss 
    def org_f(W_i, i, ll, SVD_ijt):
        loss = 0
        for t in range(T):
            Sigma_iti, U_1it= SVD_x[np.ravel_multi_index((i,t),(N,T))][1:3]
            for j in range(i+1,N):
                loss += np.linalg.norm((np.dot(np.transpose(U_1it), np.array(W_i[:,t]).reshape((-1,))) - np.dot(Sigma_iti, SVD_ijt[np.ravel_multi_index((i,j,t),(N,N,T))][0][:,ll])).reshape((-1,1)),'fro')**2

            for w in range(i):
                loss += np.linalg.norm((np.dot(np.transpose(U_1it), np.array(W_i[:,t]).reshape((-1,))) - np.dot(Sigma_iti, SVD_ijt[np.ravel_multi_index((w,i,t),(N,N,T))][1][:,ll])).reshape((-1,1)),'fro')**2
        
        loss1 = np.sum(np.absolute(W_i))

        D = np.zeros((T,T-1))
        for t in range(T-1):
            D[t,t] = 1
            D[t+1,t] = -1

        loss2 = np.sum(np.absolute(np.dot(W_i, D)))
        return 0.5*loss + loss1*lam + loss2*mu
    
    def dif_value(W_i_h):
        D = np.zeros((T,T-1))
        for t in range(T-1):
            D[t,t] = 1
            D[t+1,t] = -1

        return np.sum(np.absolute(np.dot(W_i_h, D)), axis=0)

    def sign_vary(W_i, i, t, tmp):
        Sigma_iti, U_1it= SVD_x[np.ravel_multi_index((i,t),(N,T))][1:3]
        return np.linalg.norm((np.dot(np.transpose(U_1it), W_i[:,t]) - np.dot(Sigma_iti, tmp)).reshape((-1,1)),'fro')**2
      

    #@profile
    def admm_sep_2(i_ll,P, SVD_ijt, USP, tol=1e-2, max_iter=1000, with_init=with_init, lam=lam, mu=mu, nu=nu, A_inv=A_inv, cvx_val = None):
        """
        computation for one view of data
        """ 
        print 'para:', lam, mu, nu
        _cvx_conv = False
        
        obj_value = obj_value_2 
        i, ll = i_ll 
        #if len(with_init) >= 5 and not test:
        #    W_i = with_init['W'][(i,ll)].copy()
        #    W_i_h = with_init['W_h'][(i,ll)].copy()
        #    W_i_t = with_init['W_t'][(i,ll)].copy()
        #    Theta = with_init['Theta'][(i,ll)].copy()
        #    Phi = with_init['Phi'][(i,ll)].copy()
        #else:
        W_i = W[(i,ll)].copy()
        W_i_h = W_h[(i,ll)].copy()
        W_i_t = W_t[(i,ll)].copy()
        Theta = Theta_all[(i,ll)].copy()
        Phi = Phi_all[(i,ll)].copy()

        d_i = W_i.shape[0]
         
        loss_0 = obj_value(W_i, W_i_t, W_i_h, Phi, Theta, i, ll, SVD_ijt, lam, mu, nu)
        loss_1 = loss_0 + 1
        l5 = 1e32
        _conv = True
        iter = 0
        s_time = time.time()
        while iter <= max_iter:
            #step 1, parallel 
            b_0 = -Phi - Theta + nu * (W_i_h + W_i_t)
            #tmp_time_0 = time.time()
            for t in range(T):
                ind = np.ravel_multi_index((i,t),(N,T))
                #A = (N-1) * UTU[ind] + 2*nu*np.eye(d_i)
                #b = -Phi[:,t] - Theta[:,t] + np.dot(US[ind], P[ind][:,ll]) + nu * (W_i_h[:,t] + W_i_t[:,t]) 
                b = USP[(i,t,ll)] + b_0[:,t] 
                #W_i[:,t] = np.linalg.solve(A, b)
                if not big_data:
                    W_i[:,t] = np.dot(A_inv[ind], b)
                else:
                    tmp = A_inv[str(i) + '/' + str(t)]
                    W_i[:,t] = np.dot(tmp, b)
                #if np.isnan(W_i).any():
                #    print 'error:1'+ str(lam) + ' ' + str(mu) + ' ' + str(nu)
                #    print  np.isnan(b).any(), np.isnan(Phi).any(), np.isnan(Theta).any(), np.isnan(A_inv[ind]).any(), np.amax(abs(A_inv[ind])), np.amax(abs(b))
                  
            
            #l1 = obj_value(W_i, W_i_t, W_i_h, Phi, Theta, i, ll, P, SVD_ijt)
            #if l1 >= l5+tol_dif:
            #    test_admm(lam, mu, nu, 1, l1, l5)
            #    break
            


            #step 2
            W_i_t = np.sign(W_i + Theta/nu)*np.maximum(np.absolute(W_i + Theta/nu)-1.0*lam/nu,0)
            #if iter<100 and len(with_init) >= 5 and i == 0:
            #    W_i_t = W_i_t * (abs(with_init['W_t'][(i,ll)])>0)
            #    W_i = W_i * (abs(with_init['W_t'][(i,ll)])>0)
            #l2 = obj_value(W_i, W_i_t, W_i_h, Phi, Theta, i, ll, P, SVD_ijt)
            #if l2 >= l1+tol_dif:
            #    test_admm(lam, mu, nu, 2, l2, l1)
            #    break

            #step 3, parallel
            tmp = W_i + 1.0*Phi/nu 

            
            for j in range(d_i):
                #print j
                W_i_h[j,:] = ptv.tv1_1d(tmp[j,:], 1.0*mu/nu)
                
                #W_i_h[j,:] = ptv.tv1_1d(tmp[j,:], 1.0*mu/nu)
           

            
            #l3 = obj_value(W_i, W_i_t, W_i_h, Phi, Theta, i, ll, P, SVD_ijt)
            #if l3 >= l2+tol_dif:
            #    test_admm(lam, mu, nu, 3, l3, l2)
            #    break
            #step 4
            Theta = Theta + nu*(W_i - W_i_t)
            #l4 = obj_value(W_i, W_i_t, W_i_h, Phi, Theta, i, ll, P, SVD_ijt)
            #if l4 <=l3 - tol_dif:
            #    test_admm(lam, mu, nu, 4, l4, l3)
            #    break

            #step 5
            Phi = Phi + nu*(W_i - W_i_h)
            #l5 = obj_value(W_i, W_i_t, W_i_h, Phi, Theta, i, ll, P, SVD_ijt)
            #if l5 <= l4 - tol_dif:
            #    test_admm(lam, mu, nu, 5, l5, l4)
            #    break
            iter += 1
            if iter%10 == 0:
                loss_1 = loss_0
                loss_0 = obj_value(W_i, W_i_t, W_i_h, Phi, Theta, i, ll, SVD_ijt, lam, mu, nu)
                #print np.sum(np.dot(X[i][:,:,0], W_i[:,0].reshape((-1,1)))**2), obj_value(W_i, W_i_t, W_i_h, Phi, Theta, i, ll, SVD_ijt, out_put=True)
                if (np.sum(np.absolute(W_i-W_i_h)+np.absolute(W_i-W_i_t))/(np.sum(abs(W_i)) + 1e-2) < tol) and abs(loss_0 -loss_1)/loss_0 < tol and iter > 500:
                    if (test and _cvx_conv):
                        tmp = val_config.time
                        val_config.time = tmp + time.time() - s_time
                        break
                    if not test: 
                        break
            if test and not _cvx_conv:
                print abs(org_f(W_i, i, ll, SVD_ijt) - cvx_val)/cvx_val
                if abs(org_f(W_i, i, ll, SVD_ijt) - cvx_val)/cvx_val < 0.1 or org_f(W_i, i, ll, SVD_ijt) <= cvx_val:
                    #tmp = val_config.time
                    #val_config.time = tmp + time.time() - s_time
                    _cvx_conv = True
                    
            if iter > max_iter:
                #warnings.warn(str(lam)+' '+ str(mu)+ ' '+ str(nu)+'warning: does not converge!')
                _conv = False
            if d_i > 1000 and iter%100 == 0:
                print time.time() - s_time
            #print obj_value(W_i, W_i_t, W_i_h, Phi, Theta, i, ll, SVD_ijt)
            #print time.time() - s_time
        print iter, 'final iter'
        T_dif_i = obj_value(W_i, W_i_t, W_i_h, Phi, Theta, i, ll, SVD_ijt, lam, mu, nu, out_put=True)
        #T_dif_i = obj_value(W_i/1.4, W_i_t/1.4, W_i_h/1.4, Phi, Theta, i, ll, SVD_ijt, out_put=True)
        #obj_value(-W_i, -W_i_t, -W_i_h, Phi, Theta, i, ll, SVD_ijt, out_put=True)
        return W_i, _conv, T_dif_i, W_i_t, W_i_h, Theta, Phi 

    def cvx_1(i_ll, SVD_ijt, tol=1e-2, max_iter=1000, lam=lam, mu=mu, nu=nu):
        i, ll = i_ll 
        tmp = val_config.cvx_time
        s_time = time.time()
        W_i = Variable(di[i],T)
        W_i_t = Variable(di[i],T)
        W_i_h = Variable(di[i],T)
        funcs = []
        for t in range(T):
            Sigma_iti, U_1it= SVD_x[np.ravel_multi_index((i,t),(N,T))][1:3]

            for j in range(i+1, N):
                funcs.append(0.5*sum_squares(np.transpose(U_1it) * W_i[:,t] - np.dot(Sigma_iti, SVD_ijt[np.ravel_multi_index((i,j,t),(N,N,T))][0][:,ll]).reshape((-1,1))))
                
            for w in range(i):
                funcs.append(0.5*sum_squares(np.transpose(U_1it) * W_i[:,t] - np.dot(Sigma_iti, SVD_ijt[np.ravel_multi_index((w,i,t),(N,N,T))][1][:,ll]).reshape((-1,1))))
                
        

        D = np.zeros((T,T-1))
        for t in range(T-1):
            D[t,t] = 1
            D[t+1,t] = -1

        funcs.append(norm(W_i_h*D,1)*mu)
        funcs.append(norm(W_i_t,1)*lam)
        constraints = [W_i_h == W_i, W_i_t == W_i]
        #funcs.append(sum_squares(W_i)*1e-6)
        prob = Problem(Minimize(sum(funcs)), constraints)
        result = prob.solve(max_iters=int(max_iter*(T**3)*(di[i]**3)), reltol=1e-6, abstol=1e32)
        #result = prob.solve(max_iters=int(max_iter*(T**3)*(di[i]**3)))
        val_config.cvx_time = time.time() - s_time + tmp 
        print W_i.value[:,0], prob.value
        print W_i.value[:,1]
        return W_i.value, prob.value 


    def cvx_admm(i_ll, SVD_ijt, tol=1e-2, max_iter=1000, lam=lam, mu=mu, nu=nu, cvx_val = None):
        i, ll = i_ll 
        s_time = time.time()
        W_i = Variable(di[i], T) 
        W_i_h = Variable(di[i], T) 
        W_i_t = Variable(di[i], T)
        Theta = Variable(di[i], T)
        Phi = Variable(di[i], T)
        W_i.value = W_cvx_admm[(i,ll)]
        W_i_h.value = W_h_cvx_admm[(i,ll)]
        W_i_t.value = W_t_cvx_admm[(i,ll)]
        Theta.value = Theta_all_cvx_admm[(i,ll)]
        Phi.value = Phi_all_cvx_admm[(i,ll)]

        def opt_admm_1(W_i, W_i_t, W_i_h, Theta, Phi):
            funcs = []
            for t in range(T):
                Sigma_iti, U_1it= SVD_x[np.ravel_multi_index((i,t),(N,T))][1:3]

                for j in range(i+1, N):
                    funcs.append(0.5*sum_squares(np.transpose(U_1it) * W_i[:,t] - np.dot(Sigma_iti, SVD_ijt[np.ravel_multi_index((i,j,t),(N,N,T))][0][:,ll]).reshape((-1,1))))

                for w in range(i):
                    funcs.append(0.5*sum_squares(np.transpose(U_1it) * W_i[:,t] - np.dot(Sigma_iti, SVD_ijt[np.ravel_multi_index((w,i,t),(N,N,T))][1][:,ll]).reshape((-1,1))))
            funcs.append(sum_squares(W_i - W_i_h.value)*nu/2.0)
            funcs.append(sum_squares(W_i - W_i_t.value)*nu/2.0)
            funcs.append(trace(np.transpose(Theta.value)*(W_i - W_i_t.value)))
            funcs.append(trace(np.transpose(Phi.value)*(W_i-W_i_h.value)))
            prob = Problem(Minimize(sum(funcs)))
            result = prob.solve(max_iters=int(max_iter*(T**3)*(di[i]**3)), abstol=1e32, reltol=tol, solver=CVXOPT)

        def opt_admm_2(W_i, W_i_t, W_i_h, Theta, Phi):
            funcs = []
            funcs.append(norm(W_i_t,1)*lam)
            funcs.append(trace(np.transpose(Theta.value)*(W_i.value - W_i_t)))
            funcs.append(sum_squares(W_i.value - W_i_t)*nu/2.0)
            prob = Problem(Minimize(sum(funcs)))
            result = prob.solve(max_iters=int(max_iter*(T**3)*(di[i]**3)), abstol=1e32, reltol=tol)

        def opt_admm_3(W_i, W_i_t, W_i_h, Theta, Phi):
            funcs = []
            D = np.zeros((T,T-1))
            for t in range(T-1):
                D[t,t] = 1
                D[t+1,t] = -1
            funcs.append(norm(W_i_h*D,1)*mu)
            funcs.append(trace(np.transpose(Theta.value)*(W_i.value - W_i_h)))
            funcs.append(sum_squares(W_i.value - W_i_h)*nu/2.0)
            prob = Problem(Minimize(sum(funcs)))
            result = prob.solve(max_iters=int(max_iter*(T**3)*(di[i]**3)), abstol=1e32, reltol=tol)


        iter = 0
        while iter <= max_iter:
            #step 1, parallel 
            opt_admm_1(W_i, W_i_t, W_i_h, Theta, Phi)
            #l1 = obj_value(W_i, W_i_t, W_i_h, Phi, Theta, i, ll, P, SVD_ijt)
            #if l1 >= l5+tol_dif:
            #    test_admm(lam, mu, nu, 1, l1, l5)
            #    break
           

            #step 2
            opt_admm_2(W_i, W_i_t, W_i_h, Theta, Phi)
            #l2 = obj_value(W_i, W_i_t, W_i_h, Phi, Theta, i, ll, P, SVD_ijt)
            #if l2 >= l1+tol_dif:
            #    test_admm(lam, mu, nu, 2, l2, l1)
            #    break

            #step 3, parallel

            opt_admm_3(W_i, W_i_t, W_i_h, Theta, Phi)
            #l3 = obj_value(W_i, W_i_t, W_i_h, Phi, Theta, i, ll, P, SVD_ijt)
            #if l3 >= l2+tol_dif:
            #    test_admm(lam, mu, nu, 3, l3, l2)
            #    break

            #step 4
            Theta.value = Theta.value + nu*(W_i.value - W_i_t.value)
            #l4 = obj_value(W_i, W_i_t, W_i_h, Phi, Theta, i, ll, P, SVD_ijt)
            #if l4 <=l3 - tol_dif:
            #    test_admm(lam, mu, nu, 4, l4, l3)
            #    break

            #step 5
            Phi.value = Phi.value + nu*(W_i.value - W_i_h.value)
            #l5 = obj_value(W_i, W_i_t, W_i_h, Phi, Theta, i, ll, P, SVD_ijt)
            #if l5 <= l4 - tol_dif:
            #    test_admm(lam, mu, nu, 5, l5, l4)
            #    break
            iter += 1
            if test:
                if abs(org_f(W_i.value, i, ll, SVD_ijt) - cvx_val)/cvx_val < tol:
                    tmp = val_config.cvx_time_admm
                    val_config.time = tmp + time.time() - s_time
                    break

            
    def obj_value_3(W_i, W_i_t, W_i_h, Phi, Theta, Psi, i, ll):
        """
        compute the value of objective funciton (ADMM)
        """
        loss = 0
        D = np.zeros((T,T-1))
        for t in range(T-1):
            D[t,t] = 1
            D[t+1,t] = -1

        loss += np.sum(np.absolute(np.dot(W_i_h, D)))

        loss0 = np.sum(np.absolute(W_i_t))
        loss_admm_1 = (np.linalg.norm(W_i-W_i_h,'fro')**2 + np.linalg.norm(W_i-W_i_t,'fro')**2)
        loss_admm_3 = np.sum((W_i-W_i_t)*Theta) + np.sum((W_i-W_i_h)*Phi)

        loss_admm_2 = 0
        loss_admm_4 = 0
        
        for t in range(T):
            Sigma_iti, U_1it= SVD_x[np.ravel_multi_index((i,t),(N,T))][1:3]
            for j in range(i+1,N):
                j_mat = (np.dot(np.transpose(U_1it), W_i[:,t]) - np.dot(Sigma_iti, SVD_ijt[np.ravel_multi_index((i,j,t),(N,N,T))][0][:,ll])).reshape((-1,1))
                loss_admm_2 += np.linalg.norm(j_mat,'fro')**2
                loss_admm_4 += np.dot(Psi[t][:,j], j_mat)[0]
            for w in range(i):
                w_mat = (np.dot(np.transpose(U_1it), W_i[:,t]) - np.dot(Sigma_iti, SVD_ijt[np.ravel_multi_index((w,i,t),(N,N,T))][1][:,ll])).reshape((-1,1))
                loss_admm_2 += np.linalg.norm(w_mat,'fro')**2
                loss_admm_4 += np.dot(Psi[t][:,w], w_mat)[0]
        if out_put:
            print 'without tuning para:', loss, loss0, loss_admm_1, loss_admm_2, loss_admm_3, loss_admm_4
            print 'with tuning para:', loss, loss0*lam, nu/2.0*loss_admm_1, mu/2.0*loss_admm_2, loss_admm_3, loss_admm_4
        loss = loss + loss0*lam + nu/2.0*loss_admm_1 + mu/2.0*loss_admm_2 + loss_admm_3 + loss_admm_4
        return loss



    def admm_sep_3(i_ll, tol=1e-2, max_iter=1000):
        """
        computation for one view of data
        """
        i, ll = i_ll 
        obj_value = obj_value_3
        W_i = W[(i,ll)]
        W_i_h = np.random.uniform(size=W_i.shape)
        W_i_t = np.random.uniform(size=W_i.shape)
        Theta = np.random.uniform(size=W_i.shape)
        Phi = np.random.uniform(size=W_i.shape)
        #r,N,T
        Psi = dict()
        for t in range(T):
            r = SVD_x[np.ravel_multi_index((i,t),(N,T))][2].shape[1]
            Psi[t] = np.random.uniform(size=(r, N))
            Psi[t][:,i] = 0
        
        d_i = W_i.shape[0]
        l6 = 1e32
        loss_0 = obj_value(W_i, W_i_t, W_i_h, Phi, Theta, Psi, i, ll)
        loss_1 = loss_0 + 1

        iter = 0
        _conv = True


        while (np.sum(np.absolute(W_i-W_i_h)+np.absolute(W_i-W_i_t)) > tol or abs(loss_0 -loss_1) > tol) and iter <= max_iter:

            #step 1, parallel 
            for t in range(T):
                U_1it= SVD_x[np.ravel_multi_index((i,t),(N,T))][2]
                ind = np.ravel_multi_index((i,t),(N,T))
                A = mu*(N-1) * UTU[ind] + 2*nu*np.eye(d_i)
                b = -Phi[:,t] - Theta[:,t] + mu*np.dot(US[ind], P[ind][:,ll]) + nu * (W_i_h[:,t] + W_i_t[:,t]) - np.dot(U_1it,np.sum(Psi[t],axis=1).reshape((-1,1))).reshape((-1,))
                W_i[:,t] = np.linalg.solve(A, b)
            
            l1 = obj_value(W_i, W_i_t, W_i_h, Phi, Theta, Psi, i, ll)
            if l1 >= l6+tol_dif:
                test_admm(lam, mu, nu, 1, l1, l6)
                break

            #step 2
            W_i_t = np.sign(W_i + Theta/nu)*np.maximum(np.absolute(W_i + Theta/nu)-1.0*lam/nu,0)
            l2 = obj_value(W_i, W_i_t, W_i_h, Phi, Theta, Psi, i, ll)
            if l2 >= l1+tol_dif:
                test_admm(lam, mu, nu, 2, l2, l1)
                break
            
            #step 3, parallel
            tmp = W_i + Phi/nu 
            for j in range(d_i):
                W_i_h[j,:] = ptv.tv1_1d(tmp[j,:], 1.0/nu)

            l3 = obj_value(W_i, W_i_t, W_i_h, Phi, Theta, Psi, i, ll)
            if l3 >= l2+tol_dif:
                test_admm(lam, mu, nu, 3, l3, l2)
                break

            #step 4, parallel
            for t in range(T):
                Sigma_iti, U_1it= SVD_x[np.ravel_multi_index((i,t),(N,T))][1:3]
                tmp = np.dot(np.transpose(U_1it), W_i[:,t])
                for j in range(i+1,N):
                    Psi[t][:,j] = Psi[t][:,j] + mu * (tmp - np.dot(Sigma_iti, SVD_ijt[np.ravel_multi_index((i,j,t),(N,N,T))][0][:,ll]))
                for w in range(i):
                    Psi[t][:,w] = Psi[t][:,w] + mu * (tmp - np.dot(Sigma_iti, SVD_ijt[np.ravel_multi_index((w,i,t),(N,N,T))][1][:,ll]))
            
            l4 = obj_value(W_i, W_i_t, W_i_h, Phi, Theta, Psi, i, ll)
            if l4 <= l3-tol_dif:
                test_admm(lam, mu, nu, 4, l4, l3)
                break


            #step 5
            Theta = Theta + nu*(W_i - W_i_t)
            l5 = obj_value(W_i, W_i_t, W_i_h, Phi, Theta, Psi, i, ll)
            if l5 <= l4-tol_dif:
                test_admm(lam, mu, nu, 5, l5, l4)

            #step 6
            Phi = Phi + nu*(W_i - W_i_h)
            l6 = obj_value(W_i, W_i_t, W_i_h, Phi, Theta, Psi, i, ll)
            if l6 <= l5-tol_dif:
                test_admm(lam, mu, nu, 6, l6, l5)
            iter += 1
            if iter%10 == 0:
                loss_1 = loss_0
                loss_0 = obj_value(W_i, W_i_t, W_i_h, Phi, Theta, Psi, i, ll)


            if iter > max_iter:
                #warnings.warn(str(lam)+' '+ str(mu)+ ' '+ str(nu)+'warning: does not converge!')
                _conv = False


        return W_i_t, _conv


    if method == 'admm_1':
        admm_sep = admm_sep_1
    elif method == 'admm_2':
        admm_sep = admm_sep_2
    elif method == 'admm_3':
        admm_sep = admm_sep_3
    else:
        raise Exception('method has not been implemented, check again!')

    # init changing 
    if len(with_init) >= 5:
        for i in range(N):
            for j in range(i+1, N):
                for ll in range(l):
                    W_i = with_init['W'][(i,ll)]
                    W_i_h = with_init['W_h'][(i,ll)]
                    W_i_t = with_init['W_t'][(i,ll)]
                    W_j = with_init['W'][(j,ll)]
                    W_j_h = with_init['W_h'][(j,ll)]
                    W_j_t = with_init['W_t'][(j,ll)]
                    t_list = []
                    s_list = []
                    for t in range(1, T):
                        #print np.sum(abs(W_i_h[:,t]+W_i_h[:,t-1])), np.sum(abs(W_j_h[:,t]+W_j_h[:,t-1])), np.sum(abs(W_i_h[:,t]-W_i_h[:,t-1])), np.sum(abs(W_j_h[:,t]-W_j_h[:,t-1])), np.sum(abs(W_i_h[:,t]+W_i_h[:,t-1])) + np.sum(abs(W_j_h[:,t]+W_j_h[:,t-1])) < (np.sum(abs(W_i_h[:,t]-W_i_h[:,t-1]))+np.sum(abs(W_j_h[:,t]-W_j_h[:,t-1])))
                        if np.sum(abs(W_i_h[:,t]+W_i_h[:,t-1])) + np.sum(abs(W_j_h[:,t]+W_j_h[:,t-1])) < (np.sum(abs(W_i_h[:,t]-W_i_h[:,t-1]))+np.sum(abs(W_j_h[:,t]-W_j_h[:,t-1]))):
                            W_i[:,t], W_i_t[:,t], W_i_h[:,t] = -W_i[:,t], -W_i_t[:,t], -W_i_h[:,t]
                            W_j[:,t], W_j_t[:,t], W_j_h[:,t] = -W_j[:,t], -W_j_t[:,t], -W_j_h[:,t]
                            t_list.append(t)
                            #print t, '1'

                    for t in range(T):
                        tmp_1 = SVD_ijt[np.ravel_multi_index((i,j,t),(N,N,T))][0][:,ll]
                        tmp_2 = SVD_ijt[np.ravel_multi_index((i,j,t),(N,N,T))][1][:,ll]
                        if (sign_vary(W_i, i, t, tmp_1)+sign_vary(W_j, j, t, tmp_2)) > (sign_vary(W_i, i, t, -tmp_1)+sign_vary(W_j, j, t, -tmp_2)):                
                        
                            SVD_ijt[np.ravel_multi_index((i,j,t),(N,N,T))][0][:,ll] = -tmp_1
                            SVD_ijt[np.ravel_multi_index((i,j,t),(N,N,T))][1][:,ll] = -tmp_2
                            s_list.append(t)
                            #print t, '2'
                    #print t_list, s_list

        for i in range(N):
            for t in range(T):
                tmp = 0 
                for j in range(i+1,N):
                    tmp = tmp + SVD_ijt[np.ravel_multi_index((i,j,t),(N,N,T))][0]
                for w in range(i):
                    tmp = tmp + SVD_ijt[np.ravel_multi_index((w,i,t),(N,N,T))][1]
                P[np.ravel_multi_index((i,t),(N,T))] = tmp 



    USP = dict()
    for i in range(N):
        for t in range(T):
            ind = np.ravel_multi_index((i,t),(N,T))
            for ll in range(l):
                USP[(i,t,ll)] = np.dot(US[ind], P[ind][:,ll])




    _norm = []
    # change this to I - W diff if higher l
    _conv_all = True
    T_dif_all = []

    W_tmp = defaultdict(dict)
    _conv_all_tmp = defaultdict(dict)
    T_dif_i_tmp = defaultdict(dict)
    W_t_tmp = defaultdict(dict)
    W_h_tmp = defaultdict(dict)
    Theta_all_tmp = defaultdict(dict)
    Phi_all_tmp = defaultdict(dict)
    
    _conv_all = 0

    for i in range(N):
        for ll in range(l):
            for x, x_val in enumerate(val_config.dif_lam_mu[i]):
                if test:
                    _, cvx_val = cvx_1((i,ll), SVD_ijt=SVD_ijt, tol=tol_admm, max_iter=max_iter, lam=lam*x_val, mu=mu*x_val, nu=nu)
                    #cvx_admm((i,ll), SVD_ijt, tol=tol_admm, max_iter=max_iter, lam=lam*1.0, mu=mu*1.0, nu=nu*1.0, cvx_val=cvx_val)

                #W[(i,ll)], _conv, T_dif_i, W_t[(i,ll)], W_h[(i,ll)], Theta_all[(i,ll)], Phi_all[(i,ll)] = admm_sep((i,ll), P=P, SVD_ijt=SVD_ijt, tol=tol_admm, max_iter=max_iter, with_init=with_init, lam = 1.0*lam*W[(0,0)].shape[0]/W[(i,ll)].shape[0], mu = 1.0*mu*W[(0,0)].shape[0]/W[(i,ll)].shape[0], nu = nu* (1.0*W[(0,0)].shape[0]/W[(i,ll)].shape[0])**2)
                W_tmp[(i,ll)][x], _conv, T_dif_i_tmp[(i,ll)][x], W_t_tmp[(i,ll)][x], W_h_tmp[(i,ll)][x], Theta_all_tmp[(i,ll)][x], Phi_all_tmp[(i,ll)][x] = admm_sep((i,ll), P=P, SVD_ijt=SVD_ijt, USP=USP, tol=tol_admm, max_iter=max_iter, with_init=with_init, lam = 1.0*x_val*lam, mu = 1.0*mu*x_val, nu = 1.0*nu, cvx_val=cvx_val)
                
                if test:
                    print val_config.time, val_config.cvx_time, 'time'
                
                _conv_all_tmp[(i,ll)][x] = _conv 
    try:
        X = data_X.X
        X_test = data_X.X_test
    except:
        with open(folder_name + 'data.pkl') as f:
           save = pickle.load(f)
           X = save['data'][0]
           X_test = save['data'][1]
   
    list_dif = []
    for i in range(N):
        list_dif.append(range(len(val_config.dif_lam_mu[i])))

    index = np.meshgrid(*list_dif)
    for i in range(len(index)):
        index[i] = list(index[i].flat)

    #index_tmp = {}
    cor_tmp = {}
    for k in range(len(index[0])):
        #index_tmp[k] = []
        W_k = {}
        for kk in range(len(index)):
            #index_tmp[k].append(index[kk][k])
            for ll in range(l):
                W_k[(kk,ll)] = W_tmp[(kk,ll)][index[kk][k]]
        if X_test:
            cor_tmp[k] = np.mean(val_cor(X_test, W_k, l, T)[0])
        else:
            cor_tmp[k] = np.mean(val_cor(X, W_k, l, T)[0])
        print k, val_config.ijk, np.sum(abs(W_t_tmp[(1,0)][index[kk][k]])>0)
    ind = sorted(range(len(index[0])), key=lambda x: cor_tmp[x])[-1]

    for i in range(N):
        spar = []
        for ll in range(l):
            W[(i,ll)] = W_tmp[(i,ll)][index[i][ind]]
            W_t[(i,ll)] = W_t_tmp[(i,ll)][index[i][ind]]
            W_h[(i,ll)] = W_h_tmp[(i,ll)][index[i][ind]]
            Theta_all[(i,ll)] = Theta_all[(i,ll)][index[i][ind]]
            Phi_all[(i,ll)] = Phi_all[(i,ll)][index[i][ind]]
            _conv_all += _conv_all_tmp[(i,ll)][index[i][ind]]
            T_dif_all.append(T_dif_i_tmp[(i,ll)][index[i][ind]])
            spar.append(np.sum(np.absolute(W_t[(i,ll)])>0)*1.0/(di[i]*T))
        val_config.spar.append(spar)
    dif_lam_mu = val_config.dif_lam_mu
    val_config.dif_lam_mu = [[dif_lam_mu[i][index[i][ind]]] for i in range(N)]
    
    print val_config.dif_lam_mu, 'lam_mu', cor_tmp, 'cor_tmp', lam, mu, nu, val_config.ijk, val_config.spar
    

        


    for i in range(N):
        for ll in range(l):
            tmp_norm = []
            for t in range(T):
                tmp_norm.append(np.sum(np.dot(X[i][:,:,t], W[(i,ll)][:,t].reshape((-1,1)))**2))
            _norm.append(tmp_norm)
    
    #if not X_test and len(with_init) == 0:
    if True:
        for i in range(N):
            for j in range(i+1, N):
                for ll in range(l):
                    W_i = W[(i,ll)]
                    W_i_h = W_h[(i,ll)]
                    W_i_t = W_t[(i,ll)]
                    W_j = W[(j,ll)]
                    W_j_h = W_h[(j,ll)]
                    W_j_t = W_t[(j,ll)]
                    for t in range(1, T):
                        if np.sum(abs(W_i_h[:,t]+W_i_h[:,t-1])) + np.sum(abs(W_j_h[:,t]+W_j_h[:,t-1])) < (np.sum(abs(W_i_h[:,t]-W_i_h[:,t-1]))+np.sum(abs(W_j_h[:,t]-W_j_h[:,t-1]))):
                            W_i[:,t], W_i_t[:,t], W_i_h[:,t] = -W_i[:,t], -W_i_t[:,t], -W_i_h[:,t]
                            W_j[:,t], W_j_t[:,t], W_j_h[:,t] = -W_j[:,t], -W_j_t[:,t], -W_j_h[:,t]
    
    val_config.init_tcca.append({'W':W, 'W_t': W_t, 'W_h':W_h, 'Theta':Theta_all, 'Phi':Phi_all})
    if X_test:
        #val_config.cor_score.append(val_cor(X_test, W, l, T)[0])
        val_config.cor_score.append(val_cor(X_test, W_t, l, T)[0])
        #print '###########', lam, mu, nu, np.mean(val_config.cor_score), 'w_h'
        #print '###########', lam, mu, nu, val_cor(X_test, W_t, l, T)[0], 'cor_test'
        #print '###########', lam, mu, nu, val_cor(X, W_t, l, T)[0], 'cor_training'
        #val_config.cor_score.append(np.absolute(val_cor(X_test, W_t, l, T)[0]))
        #print '###########', lam, mu, nu, np.mean(val_config.cor_score), 'w_t'
    else:
        #val_config.cor_score.append(val_cor(X, W, l, T)[0])
        #val_config.cor_score.append(val_cor(X, W_h, l, T)[0])
        val_config.cor_score.append(val_cor(X, W_t, l, T)[0])
    folder_name_0 = data_X.folder_name_0
    with open(folder_name_0 + 'full_data/0/data.pkl') as f:
        full_X = pickle.load(f)['data'][0]
        val_config.cor_score_full_data.append(val_cor(full_X, W_t, l, T)[0])
    if T_dif:
        val_config.T_dif_score.append(val_t_dif(T_dif_all, T_dif))
    else:
        val_config.T_dif_score.append(0)
    
    val_config.T_dif_av.append(av_t_dif(T_dif_all))
    
    
    if hasattr(val_config,'real_W'):
        auc_score, F1_score = eval_zero(W_t, N, T, l, val_config.real_W)
        val_config.auc_score.append(auc_score)
        val_config.F1_score.append(F1_score)
    else:
        val_config.auc_score.append(0)
        val_config.F1_score.append(0)

    
    val_config.check_norm.extend(_norm)
    val_config.check_conv.append(_conv_all)
    val_config.W.append(W) 

    
    f_2 = val_config.folder_name 
    #if len(with_init) >= 5 and not X_test:
    #    with open(f_2 + '/init.pkl','rb') as f:
    #        save = pickle.load(f)
    #        cor_score = np.mean(save['cor_score'])
    #    if cor_score > np.mean(val_config.cor_score):
    #        copyfile(f_2 + '/init.pkl', f_2 + '/W.pkl')
    #        return False
   

    if not os.path.exists(f_2):
        os.mkdir(f_2)
    if len(val_config.W) <= 1:
        for t in range(3):
            f, axarr= plt.subplots(2, 1)
            axarr[0].plot(range(di[0]),W_t[(0,0)][:,t],color='r')
            axarr[0].set_title('X')
            axarr[1].plot(range(di[1]),W_t[(1,0)][:,t],color='r')
            axarr[1].set_title('Y')
            f.savefig(f_2 + ('/k_fold_W_ind.png'.replace('k_fold', str(k_fold))).replace('ind', str(t)))
            plt.close()
        if not hasattr(val_config,'real_W') and 'task' in f_2:
            if not hasattr(val_config, 'other'):
                val_config.other = defaultdict(list)
            
            val_config.other['cor_tcca'], val_config.other['cor_avg'], res0, res1 = task_fmri_ev(W, W_h, X[0], X[1], f_2 + '/', k_fold, folder_name_0)
            val_config.other['cluster_score'].append([max(res0),max(res1)])

            
    return True

   
    




def test_admm(lam, mu, nu, i, a, b):
    #raise Exception(' '.join([lam, mu, nu, i, a, b]))
    warnings.warn(' '.join([str(x) for x in [lam, mu, nu, i, a, b]])+' exception')
    

def load_dict_from_hdf5(filename):
    with h5py.File(filename, 'r') as h5file:
        return recursively_load(h5file, '/')

def recursively_load(h5file, path):
    ans = {}
    for key, item in h5file[path].items():
        i = int(key)
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[i] = item.value
        elif isinstance(item, h5py._hl.group.Group):
            ans[i] = []
            for subgroup in item.keys():
                ans[i] = ans[i] + [h5file['/' + key + '/' + subgroup].value]
    return ans

def val_cor(X, W, l, T):
    N = len(X)
    l_score = dict()
    V_all = dict()
    for ll in range(l):
        for i in range(N):
            W_tmp = W[(i,ll)]
            for t in range(T):
                V_all[(i,t,ll)] = (np.dot(X[i][:,:,t], W_tmp[:,t]))
    cor_score = []
    for t in range(T):
        for ll in range(l):
             for i in range(N):
                 for j in range(i+1, N):
                     if np.sum(abs(V_all[(i,t,ll)])) < 1e-32 or np.sum(abs(V_all[(j,t,ll)])) < 1e-32:
                         l_score[(i,j,t,ll)] = -1
                     else:
                         l_score[(i,j,t,ll)] = stats.pearsonr(V_all[(i,t,ll)], V_all[(j,t,ll)])[0]
                     # if np.isnan(l_score[(i,j,t,ll)]):
                     #     print X[i][:,:,t], W[(i,ll)][:,t]
                     #     print V_all[(i,t,ll)], V_all[(j,t,ll)]
                     cor_score.append(l_score[(i,j,t,ll)])
    return cor_score, l_score



def val_t_dif(T_dif_all, T_dif):
    T_dif_all = np.array(T_dif_all)
    ans = np.zeros((len(T_dif),))
    for i in range(len(T_dif)):
        ind = T_dif[i]
        ans[i] += np.mean(T_dif_all[:,ind])
        #T_dif_all[:,ind] = 0
    o_dif = np.sum(T_dif_all)/(T_dif_all.shape[0]*(T_dif_all.shape[1])) 
    
    if o_dif < 1e-32:
       if np.sum(ans) < 1e-32:
          return ans*0
       else:
          return ans*1e32
    return ans/o_dif
def av_t_dif(T_dif_all):

    T_dif_all = np.array(T_dif_all)
    ss = np.sum(T_dif_all, axis=1)
    
    for i in range(len(ss)):
        if ss[i] > 1e-32:
            T_dif_all[i,:] = T_dif_all[i,:]/ss[i]
            
    s = 0
    for i in range(T_dif_all.shape[0]):
        if ss[i] > 1e-32:
            for j in range(T_dif_all.shape[1]):
                if abs(1-T_dif_all[i,j]) > 1e-32:
                    s += T_dif_all[i,j]/(1 - T_dif_all[i,j])
                else:
                    s += T_dif_all[i,j]*1e32
    return s
        


def eval_zero(W, N, T, l, real_W):
    
    s = 0
    s1 = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for ll in range(l):
            for i in range(N):
                tmp = W[(i,ll)]
                if np.isnan(tmp).any():
                    return 0, 0
                for t in range(T):
                    s += fd_1(real_W[(i,ll)][:,t], tmp[:,t]) 
                    s1 += f1_score(abs(real_W[(i,ll)][:,t])>0, abs(tmp[:,t])>0)
    return s/(N*T*l), s1/(N*T*l)
        
    
def fd_1(A1,A):
    if np.sum(abs(A1))<1e-32:
        return -1
    tmp = abs(A)
    if np.max(tmp)==0:
        tmp=tmp
    else:
        tmp=tmp/np.max(tmp)
    A1=(abs(A1)>0)
    fpr,tpr,thresholds=metrics.roc_curve(A1.reshape((-1)),tmp.reshape((-1)))
    sr=roc_auc_score(A1.reshape((-1)),tmp.reshape((-1)))
    #return (fpr,tpr, thresholds, sr, tmp)
    return sr


        



    
            
        



