import numpy as np 
import os
from pathos.multiprocessing import ProcessingPool as Pool
from six.moves import cPickle as pickle
from tcca_config import Tcca_config, Validation_config
from tdcca.admm_computation import admm_comp 
from data_preprocess import data_prepare
from functools import partial
from sklearn.model_selection import KFold
from sklearn.preprocessing import scale
from shutil import copyfile
from shutil import rmtree
import time
from scipy import stats
from copy import deepcopy
import math


def scale_val(X, scaling, X_1=[]):
    """
    scale the data
    """
    T = X.shape[2]
    d = X.shape[1]

    X_ans = np.copy(X)
    
    if len(X_1) > 1:
        X_1 = np.copy(X_1)
    if not scaling:
        if len(X_1) > 1:
            return X_ans, X_1 
        else:
            return X_ans 
    for t in range(T):
        tmp = np.mean(X_ans[:,:,t], axis=0)
        X_ans[:,:,t] = X_ans[:,:,t] - tmp
        
        if len(X_1) > 1: 
            X_1[:,:,t] = X_1[:,:,t] - np.mean(X_1[:,:,t], axis=0)
    tmp = min([max(np.sum(X_ans[:,:,t]**2,axis=0)) for t in range(T)])**0.5

    for t in range(T):
        X_ans[:,:,t] = X_ans[:,:,t]/np.amax(np.sum(X_ans[:,:,t]**2,axis=0)**0.5 * d)

    if len(X_1) > 1:
        return X_ans, X_1 
    else:
        return X_ans
def scale_val_d(data, scaling):
    """
    scale the data 
    """
    data = deepcopy(data)
    if not scaling:
        return data     
    for i in range(len(data)):
        data[i] = scale_val(data[i], scaling)
    
    return data

#TBD fold == 0, remove?
def admm_val(data_org, lam, mu, nu, num_l, folder_name, real_W=None, T_dif=[], num_cores=8, admm_method='admm_2',max_iter=1000, tol_admm=1e-2, folds=2, with_one=True, out_put=True, tol_eig=0.8, shuffle=False, scaling=True, pre_sign=True, with_init = {}, with_init_part=[{}]*5, ratio_y=[1], test=False, rel1='nprop', p_cor=0.1):
    """ 
    use multiple lam, mu, nu for ADMM and select tuning parameters

    Parameters
    ----------
    data: dict {0:X, 1:Y}, X \in R^{n * d_1 * T}, Y \in R^{n * d_2 * T}
    lam, mu, nu :training tuning parameters, list of scalar
    num_l: number of canonical vectors
    folder_name: data and estimation saving directory, dir for saving entire analysis 
    real_W: sysnthetic data corresponding truth, default None
    num_cores: parallel computing for multi tuning parameters
    admm_method: default 'admm_2' used in our paper
    max_iter: max iteration
    tol_admm: tol of convergence
    T_dif: list of change points if known, used for evaluation in simulations
    folds: k-fold cross-validations
    with_one: boolean variable, if true, we only do one validation, used to save time.
              For example, if folds=5, we partition into five datasets 1, 2, 3, 4, 5 but only use dataset 1 to do validation. 
    output: verbose details of computation
    tol_eig: the cut off for svd of X and Y used in preprocessing data
    shuffle: whether to shuffle data to produce k-folds datasets for validation. 
    scaling: boolean variable
    with_init: dict of init values for W, W_h, W_t, Theta, Phi for the whole dataset.
    with_init_part: dict of init values for W, W_h, W_t, Theta, Phi for each fold of datset.
    pre_sign: True, do not modify
    ratio_y: list of scalar ,e.g [1, 2] which indicates the penalty for y can be [lambda, mu] or [2*lambda, 2*mu].
             use this when you need different penalty on x and y 
    test: whether to compare with cvxpy result
    rel1: keep this for future algorithm dev. do not modify
    p_cor: we add some weight on the sparsity of the solutions when choosing the tuning parameters
           the score for each group of tuning parameters is cor + p_cor*(1-sparsity)*max_cor
           This is not necessary when you have good tuning parameters candiates. 
           Otherwise, you may miss the sparse sols and in this case, adding some score for those with high
           sparsity might be a good idea. 



    Returns
    -------
    Selected tuning para
    """
    

    folder_name_full = folder_name + 'full_data/'
    if len(with_init) == 0:
        folder_name_val = folder_name + 'val_init/'
    else: 
        folder_name_val = folder_name + 'val/'

    if not os.path.exists(folder_name_full):
        os.mkdir(folder_name_full)

    if not os.path.exists(folder_name + 'full_data.pkl'):
        with open(folder_name + 'full_data.pkl', 'wb') as f:
            save = {'data': data_org}
            pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
                
    if not os.path.exists(folder_name_val):
        os.mkdir(folder_name_val)

    data = [0] * (folds+1)
    file_tmp = os.listdir(folder_name_full)
    data = []
    for i in range(1+folds):
        data.append([dict(), dict()])
    if os.path.exists(folder_name_full + '0/data.pkl'):
        with open(folder_name_full + '0/data.pkl') as f:
                print 'used 0 data'
                save = pickle.load(f)
                data[0] = save['data']
    else:
        data[0] = [scale_val_d(data_org,scaling), []]
    if folds >= 2:

        kf = KFold(folds, shuffle=shuffle)
        fold = 1
        for train_index, test_index in kf.split(range(len(data_org[0]))):
            if os.path.exists(folder_name_full + str(fold) + '/data.pkl'):
                with open(folder_name_full + str(fold) + '/data.pkl') as f:
                        print 'used 1 data'
                        save = pickle.load(f)
                        data[fold] = save['data']
            else:
                for i in range(len(data_org)):
                
                    data[fold][0][i], data[fold][1][i] = scale_val(data_org[i][train_index], scaling, data_org[i][test_index])
                
            fold += 1
    elif folds == 1:
        data[1] = data[0]

    if with_one == True:
        folds = 1

    pr_t = [0] * (folds +1)
    for i in range(folds + 1):
        folder_name_i = folder_name_full + str(i) + '/'
        if not os.path.exists(folder_name_i):
            os.makedirs(folder_name_i)

        pr_t[i] = Tcca_config(data[i][0], data[i][1], folder_name_i, max_iter=max_iter, out_put=out_put, tol_eig=tol_eig, tol_admm=tol_admm, rel1 = rel1, l=num_l)
        pr_t[i].get_di()
        pr_t[i].folder_name_0 = folder_name
        file_tmp = ''.join(os.listdir(folder_name_i))
        if ('svd_all.pkl' not in  file_tmp and '.hdf5' not in file_tmp) or ('SVD_x.pkl' in file_tmp and len([f for f in os.listdir(folder_name_i) if '.hdf5' in f]) < len(nu)):
            data_prepare(pr_t[i],folder_name_i, pre_sign=pre_sign, nu = nu)


    def val_all(par, method, with_init=with_init):
        folder_name_1, val_config = par
        for k in range(1, folds+1):
            admm_comp(pr_t[k], val_config, method=method, k_fold=k, with_init=with_init_part[k-1])
        print val_config
        with open(folder_name_1 + '/W.pkl', 'wb') as f:
            save = { 'W':val_config.W,
            'para':(val_config.lam, val_config.mu, val_config.nu),
            'auc_score' : val_config.auc_score,
            'F1_score': val_config.F1_score,
            'cor_score' : val_config.cor_score,
            'spar': val_config.spar,
            'check_norm' : val_config.check_norm,
            'check_conv': val_config.check_conv,
            'T_dif_score': val_config.T_dif_score,
            'ijk': val_config.ijk,
            'init_tcca': val_config.init_tcca,
            'T_dif_av': val_config.T_dif_av,
            'dif_lam_mu': val_config.dif_lam_mu,
            }
            if hasattr(val_config, 'other'):
                save['other'] = val_config.other
                 
            pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        

        return val_c

    if folds > 0:

        files_done = [folder_name_val + f for f in os.listdir(folder_name_val) if os.path.isdir(folder_name_val+f)]
        para_done = []
        for ff in files_done:
            if os.path.exists(ff+'/W.pkl'):
                with open(ff+'/W.pkl','rb') as f:
                    save = pickle.load(f)
                    lam_0, mu_0, nu_0 = save['para']
                    para_done.append([lam_0, mu_0, nu_0])
        para = []
        if pr_t[1].nu_mu == 'eq':
            for i in range(len(lam)):
                for j in range(len(mu)):
                    if len(para_done) == 0 or np.amin(np.sum(abs(np.array([lam[i],mu[j],mu[j]]) - np.array(para_done)),axis=1)) > min(lam+mu+nu):
                        folder_name_1 = folder_name_val + 'val_sim_'+str(i) +'_' + str(j) + '_' + str(j)
                        val_c = Validation_config(*[lam[i],mu[j],nu[k]], folder_name=folder_name_1, real_W=real_W, T_dif=T_dif, ijk=[i,j,k], ratio_y=ratio_y)
                        para.append([folder_name_1, val_c])
        elif pr_t[1].mu_lam == 'prop':
            for i in range(len(lam)):
                for j in range(i, i+1):
                    for k in range(len(nu)):
                        if len(para_done) == 0 or np.amin(np.sum(abs(np.array([lam[i],mu[j],nu[k]]) - np.array(para_done)),axis=1)) > min(lam+mu+nu):
                            folder_name_1 =  folder_name_val + 'val_sim_'+str(i) +'_' + str(j) + '_' + str(k)
                            val_c = Validation_config(*[lam[i],mu[j],nu[k]], folder_name=folder_name_1, real_W=real_W, T_dif=T_dif, ijk=[i,j,k], ratio_y=ratio_y)
                            para.append([folder_name_1, val_c])


            
        else: 
            for i in range(len(lam)):
                for j in range(len(mu)):
                    for k in range(len(nu)):
                        if True:
                            if len(para_done) == 0 or np.amin(np.sum(abs(np.array([lam[i]*nu[k],mu[j]*nu[k],nu[k]]) - np.array(para_done)),axis=1)) > min(lam+mu+nu)*min(nu):
                                folder_name_1 = folder_name_val + 'val_sim_'+str(i) +'_' + str(j) + '_' + str(k)
                                val_c = Validation_config(*[lam[i]*nu[k],mu[j]*nu[k],nu[k]], folder_name=folder_name_1, real_W=real_W, T_dif=T_dif, ijk=[i,j,k], ratio_y=ratio_y)
                                para.append([folder_name_1, val_c])

        val_all_method = partial(val_all, method = admm_method)

        if num_cores > 1 and len(para) > 0:
            print 'multi-cores'
            print num_cores 
            pool = Pool(min(num_cores, len(para))+(len(with_init)==5))
            pool.map(val_all_method, para)
            #time.sleep(10)
            pool.close()
            pool.join()
            pool.clear()
            
        else:
            for i in range(len(para)):
                val_all_method(para[i])

    

        files_done = [folder_name_val+f for f in os.listdir(folder_name_val) if os.path.isdir(folder_name_val+f)]
        files_done = sorted(files_done)

        res = []
        ijk = []
        dif_lam_mu = []


       
        for ff in files_done:
            with open(ff+'/W.pkl','rb') as f:
                save = pickle.load(f)
                para = save['para']
                auc_score = np.mean(save['auc_score'])
                spar = np.mean(save['spar'])
                cor_score = np.mean(save['cor_score'])
                F1_score = np.mean(save['F1_score'])
                check_norm = save['check_norm']
                T_dif_score = np.mean(save['T_dif_score'])
                check_conv = save['check_conv']
                ijk.append(save['ijk'])
                dif_lam_mu.append(save['dif_lam_mu'])
                T_dif_av = np.mean(save['T_dif_av'])
                res.append([para, cor_score, auc_score, F1_score, T_dif_score, T_dif_av, spar, check_norm, check_conv])
                print res[-1][0:7], (min(check_norm[0]), max(check_norm[0])), (min(check_norm[1]), max(check_norm[1])), save['ijk'], check_conv, dif_lam_mu[-1]
                if 'other' in save:
                    print 'other:', save['other']['cluster_score']
                
        def norm_cmp(item):
            tmp_norm = item[-2]
            s = []
            for i in range(len(tmp_norm)):
                s.append(abs(min(tmp_norm[i]) - max(tmp_norm[i]))/(max(tmp_norm[i]) + min(tmp_norm[i])))
            return True
            #return np.mean(s) < 0.3
        
        cor_pre = np.nanmax([item[1] for item in res])
        tmp = [(norm_cmp(item), np.nan_to_num(item[1])*norm_cmp(item) + p_cor*(1-item[6])*cor_pre, 1-item[6], T_dif_av) for item in res]
        max_cor_ind = sorted(range(len(res)), key=lambda x: tmp[x])[-1]

        tmp = [item[2] for item in res]
        max_auc_ind = np.nanargmax(tmp)


        tmp = [item[3] for item in res]
        max_F1_ind = np.nanargmax(tmp)
        tmp = [item[4] for item in res]
        max_T_dif_compare_ind = np.nanargmax(tmp)
       

        max_cor = res[max_cor_ind][1]
        min_spar = 1
        ind_chosen = 0
        
        for i in range(len(res)):
            if norm_cmp(res[i]) and abs(res[i][1]-max_cor) < p_cor*max_cor and res[i][6] < min_spar:
                min_spar = res[i][6]
                ind_chosen = i
        
        print_admm(max_cor_ind, res, 'cor')
        print_admm(max_auc_ind, res, 'auc')
        print_admm(max_F1_ind, res, 'F1')
        print_admm(max_T_dif_compare_ind, res, 'T_dif')
        print_admm(ind_chosen, res, 'other chosen')
 
        ind1, ind2, ind3 = ijk[max_cor_ind]
        dif_lam_mu = dif_lam_mu[max_cor_ind]
        para = res[max_cor_ind][0]
    
    else:

        para = [lam, mu, nu]

    folder_name_1 =  folder_name_full + '/0'
    def pro(f_name, pt=True):
        if (folds == 1 and not with_one):
            if not os.path.exists(folder_name_1 + f_name):
                ff = 'val_sim_'+str(ind1) +'_' + str(ind2) + '_' + str(ind3)
                copyfile(folder_name_val + ff + f_name, folder_name_1 + f_name)
                print 'data copied'
        else:
            if not os.path.exists(folder_name_1 + f_name) or f_name == '/W.pkl':
                val_config = Validation_config(*para, folder_name=folder_name_1, T_dif=T_dif, real_W=real_W, ijk=[ind1, ind2, ind3], ratio_y=dif_lam_mu[1])
                k_fold = 0 if f_name == '/init.pkl' else 1
                _save = True
                if f_name == '/init.pkl':
                    k_fold = 0
                    ff = 'val_sim_'+str(ind1) +'_' + str(ind2) + '_' + str(ind3)
                    copyfile(folder_name_val + ff + '/W.pkl', folder_name_1 + '/init_part.pkl')
                else:
                    k_fold = 1
                    with open(folder_name_1 + '/init_part.pkl','rb') as f:
                        save = pickle.load(f)
                        cor_score = np.mean(save['cor_score'])
                    if cor_score > max_cor and False:
                        copyfile(folder_name_1 + '/init.pkl', folder_name_1 + '/W.pkl')
                        _save = False
                        print 'not improved: cor_score:{0:.4f} and max_cor:{1:.4f}'.format(cor_score, max_cor)
                    
                if _save:
                    admm_comp(pr_t[0], val_config, method=admm_method, k_fold=k_fold, with_init=with_init, test=test*k_fold)
                    with open(folder_name_1 + f_name, 'wb') as f:
                        save = { 'W':val_config.W,
                    'para':(val_config.lam, val_config.mu, val_config.nu),
                    'auc_score' : val_config.auc_score,
                    'F1_score': val_config.F1_score,
                    'cor_score' : val_config.cor_score,
                    'spar': val_config.spar,
                    'check_norm' : val_config.check_norm,
                    'check_conv': val_config.check_conv,
                    'T_dif_score': val_config.T_dif_score,
                    'init_tcca': val_config.init_tcca,
                    'T_dif_av': val_config.T_dif_av,
                    'ijk': val_config.ijk,
                    'dif_lam_mu': val_config.dif_lam_mu,
                        }
                        save['real_W'] = val_config.real_W
                        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
                    if pt:
                        print val_config
                        if test*k_fold: 
                            print val_config.time, val_config.cvx_time, val_config.cvx_time_admm
    if len(with_init) == 0:
        pro('/init.pkl',pt=True) 
        with open(folder_name_1 + '/init.pkl', 'rb') as f:
            save = pickle.load(f)
            with_init_result = save['init_tcca'][0]
            para = save['para']
        with open(folder_name_1 + '/init_part.pkl', 'rb') as f:
            save = pickle.load(f)
            with_init_result_part = save['init_tcca']
            para = save['para']
        return para, with_init_result, with_init_result_part
    else:
        pro('/W.pkl',pt=True)  
        return para         


def print_admm(ind, res, which_criterion):
    check_norm = res[ind][-2]
    print which_criterion +' para, cor_score, auc_score, f1_score, T_dif, T_dif_av, spar, norm_check, conv_check: ', res[ind][0:7], (min(check_norm[0]), max(check_norm[0])), (min(check_norm[1]), max(check_norm[1])), res[ind][-1]

    

        






