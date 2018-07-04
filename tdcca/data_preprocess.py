import numpy as np 
import os 
import h5py
import sys
from six.moves import cPickle as pickle
from scipy.io import savemat 

#to be done: how to select the tol 
def data_prepare(data_X, folder_name, nu, pre_sign=True, same_cut=True, add_class=True):
    """
    preprocess the data X and save the processed data ('data/folder_name/*')

    Parameters
    ----------------------
    data_X: a class of tcca_config
    folder_name: path for saving the data
    pre_sign: boolean, default True.  
             solving sign issue, first try 
    same_cut: whether select the same number of svd vectors for all time points. In other words, 
              whether the dimension of \Sigma should be same along time dimension 
    add_class: whether add to class attr

    Returns
    ----------------------
    None 

    Saved: SVD_x[(i,t)] = (Q_{1it}, \Sigma_it^{-1}, U_{1it}); SVD_ijt[(i,j,t)] = (P_it, P_jt);
    UTU[(i,t)] = U_{1it}U_{1it}^T ; US[(i,t)] = U_{1it}\Sigma_{it}^{-1}; P[(i,t)] = \sum_{i<j} P_it^{(i,j)} + \sum_{w<i} P_{it}^{(w,i)}
    
    """

    X = data_X.X
    X_test = data_X.X_test 
    N = data_X.N
    T = data_X.T
    l = data_X.l
    tol = data_X.tol_eig
    folder_name = data_X.folder_name
    SVD_x = dict()
    UTU = dict()
    US = dict()
    P = dict()
    SVD_ijt = dict()
    #real data

    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    # changing this to parallel
    for i in range(N):
        cc = []
        if same_cut:
            for t in range(T):
                tmp = X[i][:,:,t]
                Q_it, s_it, U_itT = np.linalg.svd(tmp, full_matrices=0)
                cut_ind = search_ind(s_it, tol)
                cc.append(cut_ind)
            cut_ind = int(np.mean(cc))

        for t in range(T):
            tmp = X[i][:,:,t]
            Q_it, s_it, U_itT = np.linalg.svd(tmp, full_matrices=0)
            if not same_cut:
                cut_ind = search_ind(s_it, tol)
            Q_1it = Q_it[:,:cut_ind]
            S_it = np.diag(1.0/s_it[:cut_ind]) # store the inversed matrix 
            U_1it = np.transpose(U_itT)[:,:cut_ind]
            if t <= 5:
                print t, i, Q_1it.shape, U_1it.shape, s_it

            ind = np.ravel_multi_index((i,t),(N,T))
            if pre_sign:

                if t >= 1:                
                    if np.sum(abs(U_1it[:,0] - SVD_x[ind-1][2][:,0])) > np.sum(abs(-U_1it[:,0]-SVD_x[ind-1][2][:,0])):
                        #print U_1it, SVD_x[ind-1][2]
                        U_1it = - U_1it
                        Q_1it = - Q_1it

            SVD_x[ind] = [Q_1it, S_it, U_1it]
            UTU[ind] = np.dot(U_1it, np.transpose(U_1it))
            US[ind] = np.dot(U_1it, S_it)
        

    
    for i in range(N):
        for j in range(i+1, N):
            for t in range(T):
                Q_it = SVD_x[np.ravel_multi_index((i,t),(N,T))][0]
                Q_jt = SVD_x[np.ravel_multi_index((j,t),(N,T))][0]
                tmp = np.dot(np.transpose(Q_it), Q_jt)
                P_i, _, P_jT = np.linalg.svd(tmp, full_matrices=1)
                P_j = P_jT.T
                SVD_ijt[np.ravel_multi_index((i,j,t),(N,N,T))] = (P_i[:,0:l].reshape((-1,l)), P_j[:,0:l].reshape((-1,l)))
    

    for t in range(T):
        for i in range(N):
            tmp = 0 
            for j in range(i+1,N):
                tmp = tmp + SVD_ijt[np.ravel_multi_index((i,j,t),(N,N,T))][0]


            for w in range(i):
                tmp = tmp + SVD_ijt[np.ravel_multi_index((w,i,t),(N,N,T))][1]

            SVD_x[np.ravel_multi_index((i,t),(N,T))][0] = 0
            P[np.ravel_multi_index((i,t),(N,T))] = tmp 
    if not os.path.exists(folder_name+'RSCCA/') and False:
        os.mkdir(folder_name+'RSCCA/')
        for t in range(T):
            np.save(folder_name + 'RSCCA/0_' + str(t) + '.npy', X[0][:,:,t])
            np.save(folder_name + 'RSCCA/1_' + str(t) + '.npy', X[1][:,:,t])
        if X_test:
            for t in range(T):
                np.save(folder_name + 'RSCCA/0_' + str(t) + '_t.npy', X_test[0][:,:,t])
                np.save(folder_name + 'RSCCA/1_' + str(t) + '_t.npy', X_test[1][:,:,t])
    if not os.path.exists(folder_name+'RSCCA_1/'):
        os.mkdir(folder_name+'RSCCA_1/')
        for t in range(T):
            savemat(folder_name + 'RSCCA_1/0_' + str(t) + '.mat', {'data':X[0][:,:,t]})
            savemat(folder_name + 'RSCCA_1/1_' + str(t) + '.mat', {'data':X[1][:,:,t]})
        if X_test:
            for t in range(T):
                savemat(folder_name + 'RSCCA_1/0_' + str(t) + '_t.mat', {'data':X_test[0][:,:,t]})
                savemat(folder_name + 'RSCCA_1/1_' + str(t) + '_t.mat', {'data':X_test[1][:,:,t]})



    if 'data.pkl' not in ''.join(os.listdir(folder_name)):
        with open(folder_name + 'data.pkl', 'wb') as f:
            save = {
                'data': [X, X_test],
                }
            pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
 
                       

    # decide to save this to h, remove this afterwards
    if sys.getsizeof(UTU[0]) < 0.5 * 1e6 and False:
        with open(folder_name + 'svd_all.pkl', 'wb') as f:
            save = {
            'SVD_x': SVD_x,
            'SVD_ijt': SVD_ijt,
            'UTU': UTU,
            'US': US,
            'P': P,
            }
            pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    elif sys.getsizeof(UTU[0]) < 500 * 1e6:
        for k in range(len(nu)):
            if str(k) + '.hdf5' not in ''.join(os.listdir(folder_name)):
                with h5py.File(folder_name+'UTU_' + str(k) + '.hdf5', 'w') as f:
                    for i in range(N):
                        d_i = data_X.di[i]
                        for t in range(T):
                            ind = np.ravel_multi_index((i,t),(N,T))
                          
                            tmp= np.linalg.inv((N-1) * UTU[ind] + 2 * nu[k] * np.eye(d_i))
                 
                            f[str(i)+'/' + str(t)] = tmp
        UTU = [] 

        with open(folder_name + 'SVD_x.pkl', 'wb') as f:
            pickle.dump(SVD_x, f, pickle.HIGHEST_PROTOCOL)

        with open(folder_name + 'SVD_ijt.pkl', 'wb') as f:
            pickle.dump(SVD_ijt, f, pickle.HIGHEST_PROTOCOL)

        with open(folder_name + 'US.pkl', 'wb') as f:
            pickle.dump(US, f, pickle.HIGHEST_PROTOCOL)

        with open(folder_name + 'P.pkl', 'wb') as f:
            pickle.dump(P, f, pickle.HIGHEST_PROTOCOL)


        
    else:
        save_dict_to_hdf5(SVD_x, folder_name+'UTU.hdf5')
        save_dict_to_hdf5(SVD_ijt, folder_name+'SVD_ijt.hdf5')
        save_dict_to_hdf5(UTU, folder_name + 'UTU.hdf5') 
        save_dict_to_hdf5(US, folder_name + 'US.hdf5') 
        save_dict_to_hdf5(P, folder_name +'P.hdf5')
        

    del data_X.X
    del data_X.X_test




def search_ind(l, tol = 0.95):
    """
    search a descending order list and return the first index whose value is less than or equal to tol
    """
    l = (l**2).tolist()
    ss = np.sum(l)
    s = 0
    #if l[-1] > 0 and l[0]/l[-1] <= 10000:
    #    return len(l)
    for i, num in enumerate(l):
        s += num
        if s/ss >= tol or (i > 1 and l[i-1]/l[i] > 1e4):
            return i + 1 
        
    return len(l)


# helper function 

def save_dict_to_hdf5(dic, filename):
    with h5py.File(filename, 'w') as h5file:
        recursively_save(h5file, '/', dic)

def recursively_save(h5file, path, dic):
    for key, item in dic.items():
        if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes)):
            h5file[path + str(key)] = item
        elif isinstance(item, tuple):
            for i in range(len(item)):
                h5file[path + str(key) + '/' + str(i)] = item[i]
        else:
            raise ValueError('Cannot save %s type'%type(item))

    


