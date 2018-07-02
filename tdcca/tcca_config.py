# problem configuration 
import numpy as np 
import warnings 

class Tcca_config:
    """
    initialized in seperate folder such as simulation folder
    """
    def __init__(self, data, data_test, folder_name, tol_admm=1e-4, max_iter=5e3, tol_eig=0.8, out_put=False, rel='neq', rel1 = 'nprop', l=1):
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
    def __init__(self, lam, mu, nu, folder_name, real_W, T_dif, ijk, ratio_y=[1]):
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
        self.dif_lam_mu = [[1], ratio_y]
        self.ijk = ijk 
        self.real_W = real_W
        self.T_dif = T_dif 
       
#two 
    def __str__(self):
        try:
            if self.real_W:
                return 'lam:{0} , mu:{1}, nu:{2}, cor_score:{3}, auc_score:{4}, F1_score:{5}, dif_lam_mu:{6}, spar:{7}, ijk:{8}, norm:{9}'.format(self.lam, self.mu, self.nu, np.mean(self.cor_score), np.mean(self.auc_score), np.mean(self.F1_score), self.dif_lam_mu, (np.mean(self.spar[0::2]),np.mean(self.spar[1::2])), self.ijk, [(np.amin(self.check_norm[0::2]), np.amax(self.check_norm[0::2])), (np.amin(self.check_norm[1::2]), np.amax(self.check_norm[1::2]))])
            else:
                return 'lam:{0} , mu:{1}, nu:{2}, cor_socre:{3}, dif_lam_mu:{4}, spar:{5}, ijk:{6}, norm:{7}, cor_full:{8}'.format(self.lam, self.mu, self.nu, np.mean(self.cor_score), self.dif_lam_mu, (np.mean(self.spar[0::2]),np.mean(self.spar[1::2])), self.ijk, [(np.amin(self.check_norm[0::2]), np.amax(self.check_norm[0::2])), (np.amin(self.check_norm[1::2]), np.amax(self.check_norm[1::2]))], np.mean(self.cor_score_full_data))
        except:
            raise(Exception('you need to run the computation first'))



