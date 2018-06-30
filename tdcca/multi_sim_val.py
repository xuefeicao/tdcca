from val_computation import admm_val
from six.moves import cPickle as pickle
import numpy as np


def multi_sim(data, lam, mu, nu, folder_name, real_W, num_cores, admm_method, max_iter=5000, tol_admm=1e-4, folds=5, T_dif=[], num_val=2, scaling=True, with_sign=True, pre_sign=True, by_pass=True, mu_init=[], calculate_init=True, with_init={}, test=False, with_one=True):
    """
    use multi simulations 
    @para same as admm_val
    data: list of data
    folder_name: list of folder names 
    """
    
    
    
    if by_pass:
        nu = [10]
        lam = [0.0001, 0.001]
        mu = [0.0001, 0.0002, 0.0005, 0.0008, 0.001, 0.002, 0.005]

        

    if not real_W:
        
        if 'task' in folder_name[0]:
            tol_admm = 1e-4
            max_iter = 500
            if data[0][0].shape[1] < 500:
                for key in data[0]:
                    data[0][key] = data[0][key][:,:,:130]
           
                
                nu = [10]
            
                if '223_1/' in folder_name[0]:
                    #lam = [0.0001, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.007, 0.01]
                
                    lam = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
                    #lam = [0.0001, 0.0005, 0.001]
                    #mu = [0.001, 0.01, 0.02, 0.05, 0.1, 0.5, 1, 5]
                    mu = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]
                    #mu = [0.001, 0.01, 0.02]
                elif '223_2/' in folder_name[0]:
                    lam = [0.007, 0.07, 0.7, 7]
                    mu = [0.001, 0.01, 0.05, 0.5, 1]
                else: 
                    lam = [0.0007, 0.007, 0.07, 0.7, 7]
                    mu = [0.002, 0.02, 0.2, 2, 10]
                
                nu_init = nu
                print data[0][0].shape[2]
        else:
            nu_init = [1]
    else:
        nu_init = nu 
    lam_init = lam
    mu_init = [1e-10*min(mu)]
    print 'max_iter:{0} and tol:{1}'.format(max_iter, tol_admm)
        
    print 'para:', lam, mu, nu
    if calculate_init:
        ans = admm_val(data[0], lam_init, mu_init, nu_init, folder_name[0], real_W, num_cores, admm_method,max_iter, tol_admm, T_dif, folds, scaling=scaling, with_sign=with_sign, pre_sign=pre_sign, with_one=with_one)
        print 'init finished'
        with_init = ans[1]
        with_init_part = ans[2]
        print with_init.keys()
    else:
        with_init = {'W':None}
        with_init_part = [{'W':None}]


    lam_0, mu_0, nu_0 = admm_val(data[0], lam, mu, nu, folder_name[0], real_W, num_cores, admm_method,max_iter, tol_admm, T_dif, folds, scaling=scaling, with_sign=with_sign, pre_sign=pre_sign, with_init = with_init, test=test, with_init_part = with_init_part, with_one=with_one)


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
                spar = np.mean(save['spar'])
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
        print T_DIF
        print '##################################################################'
        print 'result(COR: %.4f, AUC: %.4f, F1: %.4f, T_DIF: %.4f, NORM: %.4f):'%(np.mean(COR), np.mean(AUC), np.mean(F1), np.mean(T_DIF), np.mean(NORM)), para
        print 'result(COR: %.4f, AUC: %.4f, F1: %.4f, T_DIF: %.4f, NORM: %.4f):'%(np.std(COR), np.std(AUC), np.std(F1), np.std(T_DIF), np.std(NORM)), para
    if num_val == 1:
        for i in range(1, len(data)):
            if calculate_init:
                ans = admm_val(data[i], lam, mu_init, nu, folder_name[i], real_W, num_cores, admm_method,max_iter, tol_admm, T_dif, folds, scaling=scaling, with_sign=with_sign, pre_sign=pre_sign, with_one=with_one)
                with_init = ans[1]
            else:
                with_init = {'W':None}
            admm_val(data[i], lam_0, mu_0, nu_0, folder_name[i], real_W, num_cores, admm_method, max_iter, tol_admm, T_dif, 0, scaling=scaling, with_sign=with_sign, pre_sign=pre_sign, with_init = with_init, test=test, with_init_part = with_init_part, with_one=with_one)
            p_ans(i+1)
    else:
        for i in range(1, len(data)):
            print 'begin:' + str(i)
            if calculate_init:
                ans = admm_val(data[i], lam, mu_init, nu, folder_name[i], real_W, num_cores, admm_method,max_iter, tol_admm, T_dif, folds, scaling=scaling, with_sign=with_sign, pre_sign=pre_sign, with_one=with_one)
                with_init = ans[1]
            else:
                with_init = {'W':None}
            admm_val(data[i], lam, mu, nu, folder_name[i], real_W, num_cores, admm_method, max_iter, tol_admm, T_dif, folds, scaling=scaling, with_sign=with_sign, pre_sign=pre_sign, with_init = with_init, test=test, with_init_part = with_init_part, with_one=with_one)
            p_ans(i+1)

        
         











