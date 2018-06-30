# problem configuration 
import numpy as np 
import warnings 

class Tcca_config:
    """
    initialized in seperate folder such as simulation folder
    """
    def __init__(self, data, data_test, folder_name, tol_admm=1e-2, max_iter=1e3, tol_eig=1e-2, out_put=False, rel='neq', rel1 = 'nprop', l=1):
    # data: dict, (N, num_views, num_parameter, T)
        self.X = data 
        self.X_test = data_test 
        self.l = l 
        self.N = len(data)
        self.T = data[0].shape[-1]
        self.folder_name = folder_name
        self.tol_admm = tol_admm
        self.max_iter = max_iter
        self.tol_eig = tol_eig
        self.nu_mu = rel 
        self.mu_lam = rel1
        self.out_put = out_put


    def get_di(self):
        """
        return the d_i, number of features for each view
        """
        self.di = []
        for i in range(len(self.X)):
            self.di.append(self.X[i].shape[1])


class Validation_config:
    def __init__(self, lam, mu, nu, folder_name):
        self.lam = lam
        self.mu = mu
        self.nu = nu 
        self.folder_name = folder_name
        self.cor_score = []
        self.auc_score = []
        self.F1_score = []
        self.T_dif_score = []
        self.spar = []
        self.check_norm = []
        self.check_conv = []
        self.W = []
        self.init_tcca = []
        self.T_dif_av = []
        self.cor_score_full_data = []
        #self.dif_lam_mu = [[1], [0.1, 1, 2]]
        if 'test' in folder_name:
            self.dif_lam_mu = [[1], [1]]
        else:
            self.dif_lam_mu = [[1], [0.1, 2]]
#two 
    def __str__(self):
        if hasattr(self, 'real_W'):
            return 'lam:{0} , mu:{1}, nu:{2}, cor_score:{3}, auc_score:{4}, F1_score:{5}, dif_lam_mu:{6}, spar:{7}, ijk:{8}, norm:{9}'.format(self.lam, self.mu, self.nu, np.mean(self.cor_score), np.mean(self.auc_score), np.mean(self.F1_score), self.dif_lam_mu, (np.mean(self.spar[0::2]),np.mean(self.spar[1::2])), self.ijk, [(np.amin(self.check_norm[0::2]), np.amax(self.check_norm[0::2])), (np.amin(self.check_norm[1::2]), np.amax(self.check_norm[1::2]))])
        else:
            return 'lam:{0} , mu:{1}, nu:{2}, cor_score:{3}, other:{4}, dif_lam_mu:{5}, spar:{6}, ijk:{7}, norm:{8}, cor_full:{9}'.format(self.lam, self.mu, self.nu, np.mean(self.cor_score), self.other['cluster_score'], self.dif_lam_mu, (np.mean(self.spar[0::2]),np.mean(self.spar[1::2])), self.ijk, [(np.amin(self.check_norm[0::2]), np.amax(self.check_norm[0::2])), (np.amin(self.check_norm[1::2]), np.amax(self.check_norm[1::2]))], np.mean(self.cor_score_full_data))




