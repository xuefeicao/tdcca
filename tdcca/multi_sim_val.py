from tdcca.val_computation import admm_val
from six.moves import cPickle as pickle
import numpy as np


def multi_sim(data, lam, mu, nu, num_l, folder_name, real_W=None, T_dif=[], num_cores=1, admm_method='admm_2',max_iter=1000, tol_admm=1e-2, folds=2, with_one=True, out_put=True, tol_eig=0.8, shuffle=False, scaling=True, pre_sign=True, ratio_y=[1], test=False,  mu_init=1e-10, calculate_init=True, num_val=2, p_cor=0.1):
    """
    test for multi simulations
    
    Parameters
    --------------------------
    data: list of data ,e.g [{0:X_sim_1, 1:Y_sim_1}, {0:X_sim_1, 1:Y_sim_2}]
    lam, mu, nu: tuning para. When input is (lam, mu, nu), the tuning para used is actually (lam*nu, mu*nu, nu).
    num_l: number of canonical vectors
    folder_name: list of folder names for each dataset
    real_W: sysnthetic data corresponding truth, default None
    num_cores: parallel computing for multi tuning parameters
    admm_method: default 'admm_2' used in our paper
    max_iter: max iteration
    tol_admm: tol of convergence
    T_dif: list of change points if known, used for evaluation in simulations
    folds: k-fold cross-validations
    with_one: boolean variable, if true, we only do one validation, used to save time.
    For example, if folds=5, we partition into five datasets 1, 2, 3, 4, 5 but only use dataset 1 to do validation.
    output: verbose details of the computation
    tol_eig: the cut off for svd of X and Y used in preprocessing data
    shuffle: whether to shuffle data to produce k-folds datasets for validation.
    scaling: boolean variable
    pre_sign: True, do not modify
    ratio_y: list of scalar ,e.g [1, 2] which indicates the penalty for y can be [lambda, mu] or [2*lambda, 2*mu].
    use this when you need different penalty on x and y
    test: whether to compare with cvxpy result
    boolean variable, whether to test our data in cvxpy, not used in our paper, users can try.
    In this case, you need to uncomment the related functions.
    mu_init: use very low mu for the first step, the mu used in the first step will be min(mu)*mu_init
    calculate_init: whether to use two step method
    num_val: integer, 1 when you only do validation for one of your datasets and use that for the remaining
    otherwise, pass 2 into the function
    p_cor: we add some weight on the sparsity of the solutions when choosing the tuning parameters
    the score for each group of tuning parameters is cor + p_cor*(1-sparsity)*max_cor
    This is not necessary when you have good tuning parameters candidates.
    Otherwise, you may miss the sparse sols and in this case, adding some score for those with high
    sparsity might be a good idea.
    
    Returns
    ---------------------------
    None
    
    """

    lam_init = lam
    nu_init = nu 
    mu_init = [mu_init*min(mu)]
    print 'max_iter:{0} and tol:{1}'.format(max_iter, tol_admm)
        
    print 'para:', lam, mu, nu
    if calculate_init:
        ans = admm_val(data[0], lam_init, mu_init, nu_init, num_l, folder_name[0], real_W, T_dif, num_cores, admm_method, max_iter, tol_admm, folds, with_one, out_put, tol_eig, shuffle, scaling, pre_sign, ratio_y=[1], test=False, p_cor=p_cor)
        print 'init finished'
        with_init = ans[1]
        with_init_part = ans[2]
        print with_init.keys()
    else:
        with_init = {'W':None}
        with_init_part = [{'W':None}]


    lam_0, mu_0, nu_0 = admm_val(data[0], lam, mu, nu, num_l, folder_name[0], real_W, T_dif, num_cores, admm_method,max_iter, tol_admm, folds, with_one, out_put, tol_eig, scaling, pre_sign, with_init = with_init, test=test, with_init_part = with_init_part, ratio_y=ratio_y, p_cor=p_cor)


    def p_ans(nums):
        AUC = []
        F1 = []
        T_DIF = []
        NORM = []
        COR = []

        for i in range(nums):
            folder = folder_name[i]
            file_name = folder + 'full_data/0/W.pkl'
            with open(file_name) as f:
                save = pickle.load(f)
                para = save['para']
                auc_score = np.mean(save['auc_score'])
                #spar = np.mean(save['spar'])
                cor_score = np.mean(save['cor_score'])
                F1_score = np.mean(save['F1_score'])
                check_norm = np.mean(save['check_norm'])
                T_dif_score = np.mean(save['T_dif_score'])
                AUC.append(auc_score)
                F1.append(F1_score)
                T_DIF.append(T_dif_score)
                NORM.append(check_norm)
                COR.append(cor_score)
                #check_conv = save['check_conv']
        print '##################################################################'
        print 'result(COR: %.4f, AUC: %.4f, F1: %.4f, T_DIF: %.4f, NORM: %.4f):'%(np.mean(COR), np.mean(AUC), np.mean(F1), np.mean(T_DIF), np.mean(NORM)), para
        print 'std result(COR: %.4f, AUC: %.4f, F1: %.4f, T_DIF: %.4f, NORM: %.4f):'%(np.std(COR), np.std(AUC), np.std(F1), np.std(T_DIF), np.std(NORM)), para
    if num_val == 1:
        for i in range(1, len(data)):
            admm_val(data[i], [lam_0], [mu_0], [nu_0], num_l, folder_name[i], real_W, T_dif, num_cores, admm_method, max_iter, tol_admm, 1, False, out_put, tol_eig, shuffle, scaling, pre_sign, ratio_y=[1], test=False, with_init=with_init, with_init_part=with_init_part)
            p_ans(i+1)
    else:
        for i in range(1, len(data)):
            print 'begin:' + str(i)
            if calculate_init:
                ans = admm_val(data[i], lam, mu_init, nu, num_l, folder_name[i], real_W, T_dif, num_cores, admm_method, max_iter, tol_admm, folds, with_one, out_put, tol_eig, shuffle, scaling, pre_sign, ratio_y=[1], test=False, p_cor=p_cor)
                with_init = ans[1]
            else:
                with_init = {'W':None}
            admm_val(data[i], lam, mu, nu, num_l, folder_name[i], real_W, T_dif, num_cores, admm_method,max_iter, tol_admm, folds, with_one, out_put, tol_eig, scaling, pre_sign, with_init = with_init, test=test, with_init_part = with_init_part, ratio_y=ratio_y, p_cor=p_cor)
            p_ans(i+1)

        
         











