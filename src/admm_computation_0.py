import numpy as np 
import os
import h5py
import prox_tv as ptv 
from pathos.multiprocessing import ProcessingPool as Pool
from six.moves import cPickle as pickle
import matplotlib.pyplot as plt
import warnings
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn import metrics
import copy

def admm_comp(data_X, val_config, method, k_fold, with_sign=False, with_init={}):
    """
    main computation algorithm
    @param data_X(class), val_config(class), method('admm_1,admm_2,admm_3')
    @return 
    """
    folder_name = data_X.folder_name
    # load preprocessed data
    with_init = copy.deepcopy(with_init)
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
    else:
        raise Exception('no preprocessed data available')

    

        

    


    l = data_X.l 
    N = data_X.N
    T = data_X.T 
    di = data_X.di
    
    max_iter = data_X.max_iter
    out_put = data_X.out_put
    tol_dif = 1e-4
    #real_W = val_config.real_W
    T_dif = val_config.T_dif
    # cross validation tuning parameters
    lam, mu, nu = val_config.lam, val_config.mu, val_config.nu
    ind1, ind2, ind3 = val_config.ijk
    tol_admm = data_X.tol_admm*lam
    W = dict()
    W_h = dict()
    W_t = dict()
    Theta_all = dict()
    Phi_all = dict()
    SVD_ijt_v = dict()
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
             
           
        

    def obj_value_2(W_i, W_i_t, W_i_h, Phi, Theta, i, ll, P, SVD_ijt, out_put=False):
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
    
    def dif_value(W_i_h):
        D = np.zeros((T,T-1))
        for t in range(T-1):
            D[t,t] = 1
            D[t+1,t] = -1

        return np.sum(np.absolute(np.dot(W_i_h, D)))

    def sign_vary(W_i, i, t, tmp):
        Sigma_iti, U_1it= SVD_x[np.ravel_multi_index((i,t),(N,T))][1:3]
        return np.linalg.norm((np.dot(np.transpose(U_1it), W_i[:,t]) - np.dot(Sigma_iti, tmp)).reshape((-1,1)),'fro')**2
      

    #@profile
    def admm_sep_2(i_ll,P, SVD_ijt, USP, tol=1e-2, max_iter=1000, with_init=with_init, lam = lam, mu = mu, nu = nu):
        """
        computation for one view of data
        """
        print 'para:', lam, mu, nu
        obj_value = obj_value_2 
        i, ll = i_ll 
        if len(with_init) >= 5:
            W_i = with_init['W'][(i,ll)].copy()
            W_i_h = with_init['W_h'][(i,ll)].copy()
            W_i_t = with_init['W_t'][(i,ll)].copy()
            Theta = with_init['Theta'][(i,ll)].copy()
            Phi = with_init['Phi'][(i,ll)].copy()
        else:
            W_i = W[(i,ll)]
            W_i_h = W_h[(i,ll)]
            W_i_t = W_t[(i,ll)]
            Theta = Theta_all[(i,ll)]
            Phi = Phi_all[(i,ll)]
            #W_i_h = np.zeros(W_i.shape)
            #W_i_t = np.zeros(W_i.shape)

        d_i = W_i.shape[0]
         
        loss_0 = obj_value(W_i, W_i_t, W_i_h, Phi, Theta, i, ll, P, SVD_ijt)
        loss_1 = loss_0 + 1
        l5 = 1e32
        _conv = True
        iter = 0
        A_inv = dict()
        for t in range(T):
            ind = np.ravel_multi_index((i,t),(N,T))
            A_inv[ind] = np.linalg.inv((N-1) * UTU[ind] + 2*nu*np.eye(d_i))

        while iter <= max_iter:
            if with_sign and len(with_init) < 5:
                if iter <= 100:
                    for t in range(T):
                        p_tmp = P[np.ravel_multi_index((i,t),(N,T))][:,ll]
                        pp_tmp = P[np.ravel_multi_index((i,t),(N,T))][:,ll].copy()
                        for j in range(i+1,N):
                            tmp = SVD_ijt[np.ravel_multi_index((i,j,t),(N,N,T))][0][:,ll]
                            if 0.5*sign_vary(W_i, i, t, tmp) > sign_vary(W_i, i, t, -tmp):
                                SVD_ijt[np.ravel_multi_index((i,j,t),(N,N,T))][0][:,ll] = -tmp
                                SVD_ijt[np.ravel_multi_index((i,j,t),(N,N,T))][1][:,ll] = -SVD_ijt[np.ravel_multi_index((i,j,t),(N,N,T))][1][:,ll]
                                P[np.ravel_multi_index((j,t),(N,T))][:,ll] += 2*SVD_ijt[np.ravel_multi_index((i,j,t),(N,N,T))][1][:,ll]
                                #SVD_ijt_v[(i,j,t)] = True
                                p_tmp += 2*tmp 
                                print t, iter, True, '1'
                        for w in range(i):
                            tmp = SVD_ijt[np.ravel_multi_index((w,i,t),(N,N,T))][1][:,ll]
                            if 0.5*sign_vary(W_i, i, t, tmp) > sign_vary(W_i, i, t, -tmp):
                                SVD_ijt[np.ravel_multi_index((w,i,t),(N,N,T))][1][:,ll] = -tmp
                                SVD_ijt[np.ravel_multi_index((w,i,t),(N,N,T))][0][:,ll] = -SVD_ijt[np.ravel_multi_index((w,i,t),(N,N,T))][0][:,ll]                          
                                P[np.ravel_multi_index((w,t),(N,T))][:,ll] += 2*SVD_ijt[np.ravel_multi_index((w,i,t),(N,N,T))][0][:,ll]
                                #SVD_ijt_v[(w,i,t)] = True
                                p_tmp += 2*tmp 
                                print t, iter, True, '2'
                        

            #step 1, parallel 
            b_0 = -Phi - Theta + nu * (W_i_h + W_i_t)
            for t in range(T):
                ind = np.ravel_multi_index((i,t),(N,T))
                #A = (N-1) * UTU[ind] + 2*nu*np.eye(d_i)
                #b = -Phi[:,t] - Theta[:,t] + np.dot(US[ind], P[ind][:,ll]) + nu * (W_i_h[:,t] + W_i_t[:,t]) 
                b = USP[(i,t,ll)] + b_0[:,t] 
                #W_i[:,t] = np.linalg.solve(A, b)
                W_i[:,t] = np.dot(A_inv[ind], b)
            #l1 = obj_value(W_i, W_i_t, W_i_h, Phi, Theta, i, ll, P, SVD_ijt)
            #if l1 >= l5+tol_dif:
            #    test_admm(lam, mu, nu, 1, l1, l5)
            #    break
            if np.isnan(W_i).any():
                print 'error:1'+ str(lam) + ' ' + str(mu) + ' ' + str(nu)
                print A, sorted(np.linalg.eig(A)[0]), np.isnan(b).any(), np.isnan(Phi).any(), np.isnan(Theta).any()


            #step 2
            W_i_t = np.sign(W_i + Theta/nu)*np.maximum(np.absolute(W_i + Theta/nu)-1.0*lam/nu,0)
            #l2 = obj_value(W_i, W_i_t, W_i_h, Phi, Theta, i, ll, P, SVD_ijt)
            #if l2 >= l1+tol_dif:
            #    test_admm(lam, mu, nu, 2, l2, l1)
            #    break

            #step 3, parallel
            tmp = W_i + 1.0*Phi/nu 
            
            for j in range(d_i):
                W_i_h[j,:] = ptv.tv1_1d(tmp[j,:], 1.0*mu/nu, method = 'dp')
            if np.isnan(W_i_h).any():
                print 'error:3'+  str(lam) + ' ' + str(mu) + ' ' + str(nu)
            
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
                loss_0 = obj_value(W_i, W_i_t, W_i_h, Phi, Theta, i, ll, P, SVD_ijt)
                if np.sum(np.absolute(W_i-W_i_h)+np.absolute(W_i-W_i_t)) < tol and abs(loss_0 -loss_1) < tol:
                    break
            if iter > max_iter:
                #warnings.warn(str(lam)+' '+ str(mu)+ ' '+ str(nu)+'warning: does not converge!')
                _conv = False
        T_dif_i = obj_value(W_i, W_i_t, W_i_h, Phi, Theta, i, ll, P, SVD_ijt, out_put=True)
        obj_value(-W_i, -W_i_t, -W_i_h, Phi, Theta, i, ll, P, SVD_ijt, out_put=True)
        return W_i, _conv, T_dif_i, W_i_t, W_i_h, Theta, Phi 




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
                    for t in range(T):
                        if np.sum(abs(W_i_h[:,t]+W_i_h[:,t-1])) + np.sum(abs(W_j_h[:,t]+W_j_h[:,t-1])) < 0.8*(np.sum(abs(W_i_h[:,t]-W_i_h[:,t-1]))+np.sum(abs(W_j_h[:,t]-W_j_h[:,t-1]))):
                            W_i[:,t], W_i_t[:,t], W_i_h[:,t] = -W_i[:,t], -W_i_t[:,t], -W_i_h[:,t]
                            W_j[:,t], W_j_t[:,t], W_j_h[:,t] = -W_j[:,t], -W_j_t[:,t], -W_j_h[:,t]
                            t_list.append(t)
                            print t, '1'

                    for t in range(T):
                        tmp_1 = SVD_ijt[np.ravel_multi_index((i,j,t),(N,N,T))][0][:,ll]
                        tmp_2 = SVD_ijt[np.ravel_multi_index((i,j,t),(N,N,T))][1][:,ll]
                        #if 0.5*(sign_vary(W_i, i, t, tmp_1)+sign_vary(W_j, j, t, tmp_2)) > (sign_vary(W_i, i, t, -tmp_1)+sign_vary(W_j, j, t, -tmp_2)):                
                        if t in t_list:
                            SVD_ijt[np.ravel_multi_index((i,j,t),(N,N,T))][0][:,ll] = -tmp_1
                            SVD_ijt[np.ravel_multi_index((i,j,t),(N,N,T))][1][:,ll] = -tmp_2
                            print t, '2'

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



    spar = []
    norm = []
    # change this to I - W diff if higher l
    _conv_all = True
    T_dif_all = []
    for i in range(N):
        for ll in range(l):
            #W[(i,ll)], _conv, T_dif_i, W_t[(i,ll)], W_h[(i,ll)], Theta_all[(i,ll)], Phi_all[(i,ll)] = admm_sep((i,ll), P=P, SVD_ijt=SVD_ijt, tol=tol_admm, max_iter=max_iter, with_init=with_init, lam = 1.0*lam*W[(0,0)].shape[0]/W[(i,ll)].shape[0], mu = 1.0*mu*W[(0,0)].shape[0]/W[(i,ll)].shape[0], nu = nu* (1.0*W[(0,0)].shape[0]/W[(i,ll)].shape[0])**2)
            W[(i,ll)], _conv, T_dif_i, W_t[(i,ll)], W_h[(i,ll)], Theta_all[(i,ll)], Phi_all[(i,ll)] = admm_sep((i,ll), P=P, SVD_ijt=SVD_ijt, USP=USP, tol=tol_admm, max_iter=max_iter, with_init=with_init, lam = 1.0*lam, mu = 1.0*mu, nu = 1.0*nu)
            T_dif_all.append(T_dif_i)
            spar.append(np.sum(np.absolute(W_t[(i,ll)])>0)*1.0/(di[i]*T))
            for t in range(T):
                norm.append(np.sum(np.dot(X[i][:,:,t], W[(i,ll)][:,t].reshape((-1,1)))**2))
            if not _conv:
                _conv_all = False
    try:
        X = data_X.X
        X_test = data_X.X_test
    except:
        with open(folder_name + 'data.pkl') as f:
           save = pickle.load(f)
           X = save['data'][0]
           X_test = save['data'][1]

    val_config.init_tcca.append({'W':W, 'W_t': W_t, 'W_h':W_h, 'Theta':Theta_all, 'Phi':Phi_all})
    if X_test:
        val_config.cor_score.append(val_cor(X_test, W, l, T)[0])
    else:
        val_config.cor_score.append(val_cor(X, W, l, T)[0])
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

    val_config.spar.append(spar)
    val_config.check_norm.append(norm)
    val_config.check_conv.append(_conv_all)
    val_config.W.append(W) 

    print '###########', lam, mu, nu, spar

    f, axarr= plt.subplots(2, 1)


    axarr[0].plot(range(di[0]),W_t[(0,0)][:,0],color='r')
    axarr[0].set_title('X')


    axarr[1].plot(range(di[1]),W_t[(1,0)][:,0],color='r')
    axarr[1].set_title('Y')

    f_2 = val_config.folder_name 
    if not os.path.exists(f_2):
        os.mkdir(f_2)
    f.savefig(f_2 + '/k_fold_W_1.png'.replace('k_fold', str(k_fold)))
    plt.close()

    f, axarr= plt.subplots(2, 1)


    axarr[0].plot(range(di[0]),W_t[(0,0)][:,1],color='r')
    axarr[0].set_title('X')


    axarr[1].plot(range(di[1]),W_t[(1,0)][:,1],color='r')
    axarr[1].set_title('Y')

    f_2 = val_config.folder_name 
    if not os.path.exists(f_2):
        os.mkdir(f_2)
    f.savefig(f_2 + '/k_fold_W_2.png'.replace('k_fold', str(k_fold)))
    plt.close()
    




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
                     l_score[(i,j,t,ll)] = stats.pearsonr(V_all[(i,t,ll)], V_all[(j,t,ll)])[0]
                     if np.isnan(l_score[(i,j,t,ll)]):
                         print X[i][:,:,t], W[(i,ll)][:,t]
                         print V_all[(i,t,ll)], V_all[(j,t,ll)]
                     cor_score.append(l_score[(i,j,t,ll)])
    return cor_score, l_score

def val_t_dif(T_dif_all, T_dif):
    T_dif_all = np.array(T_dif_all)
    ans = np.zeros((len(T_dif),))
    for i in range(len(T_dif)):
        ind = T_dif[i]
        ans[i] += np.mean(T_dif_all[:,ind])
        T_dif_all[:,ind] = 0
    o_dif = np.sum(T_dif_all)/(T_dif_all.shape[0]*(T_dif_all.shape[1]-len(T_dif))) 
    
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
        if ss[i] < 1e-32:
            return 0
    T_dif_all = T_dif_all / ss.reshape((-1,1))
    s = 0
    for i in range(T_dif_all.shape[0]):
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


        



    
            
        



