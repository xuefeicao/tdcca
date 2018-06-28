import numpy as np 
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn import linear_model
from sklearn import model_selection
from six.moves import cPickle as pickle
from sklearn.ensemble import RandomForestClassifier as RandomForest
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import OrderedDict

#file_folder = '/gpfs_home/xcao1/scratch/research_Rossi_1/data/tcca/real_data/task/events/'
#  T * n_features
def task_fmri_ev(W, W_h, X1, X2, folder_name, k_fold, folder_name_0):
    segt = [0, 60, 120, 201, 282, 402]
    N_t = [29, 45, 64, 89, 109]
    #N_t = [1, 2, 3, 4, 5]
    #name_t = ['right hand', 'left foot', 'tongue', 'left hand', 'right foot']
    name_t = ['right hand', 'left foot', 'tongue', 'right foot', 'left hand']
    #name_tick = ['right foot', 'right hand', 'left foot', 'left hand', 'tongue', 'Thalamus']
    name_tick = ['left hand', 'right hand', 'left foot', 'right foot', 'tongue', 'Thalamus']
    cl = ['b','g', 'r', 'c', 'm', 'y', 'k']
    N = 2
    vox = range(W[(0,0)].shape[0])
    T = W[(0,0)].shape[1]
    l = len(W)/2 
    file_folder = '/gpfs_home/xcao1/scratch/research_Rossi_1/data/tcca/real_data/task/events/'
    J = 6
    dt = 0.72
    labl = np.zeros((T,))
    #move = ['rest', 'cue', 'left foot', 'right foot', 'left hand', 'right hand', 'tongue']
    move = ['rest', 'cue', 'left foot', 'left hand', 'right foot', 'right hand', 'tongue']
    cl = ['b','g', 'r', 'c', 'm', 'y', 'k']

    def dif_value(W_i_h):
        D = np.zeros((T,T-1))
        for t in range(T-1):
            D[t,t] = 1
            D[t+1,t] = -1
        return np.sum(np.absolute(np.dot(W_i_h, D)), axis=0)

    dif_sum = {}
    for i in range(N):
        for ll in range(l):
            dif_sum[(i,ll)] = dif_value(W_h[(i,ll)])

    W_cluster = []
    for i in range(1):
        for ll in range(l):
            if len(W_cluster) == 0:
                W_cluster = W[(i,ll)]
            else:
                W_cluster = np.concatenate((W_cluster, W[(i,ll)]), axis=0)
    
    cut_time = 6
    fig = plt.figure()
    axarr = fig.add_subplot(1, 1, 1)   
    axarr.plot(range(cut_time,T), dif_sum[(0,0)][cut_time-1:], color = 'blue')
    for j in range(J-1, -1, -1):
        evj = np.loadtxt(file_folder + 'ev' + str(j+1) + '.txt')
        for k in range(evj.shape[0]):
            start = int(math.ceil(evj[k,0]/dt))
            end = int(round((evj[k,1]+evj[k,0])/dt + 1))
            labl[start:end] = j + 1
            if start < T:
                axarr.plot(range(start,end+1), [0.5*max(np.absolute(dif_sum[(0,0)]))]*(end+1-start), color = cl[j+1], label=move[j+1])

    handles, labels = axarr.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    axarr.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0, 1.02, 1.02, 0.502), loc=3,
               ncol=6, mode="expand", borderaxespad=0., fontsize=8)  
    plt.savefig(folder_name+'k_fold_W_0_dis.png'.replace('k_fold', str(k_fold)))
    plt.close()

    fig = plt.figure()
    axarr = fig.add_subplot(1, 1, 1)   
    axarr.plot(range(cut_time,T), dif_sum[(1,0)][cut_time-1:], color = 'blue')
    for j in range(J-1, -1, -1):
        evj = np.loadtxt(file_folder + 'ev' + str(j+1) + '.txt')
        for k in range(evj.shape[0]):
            start = int(math.ceil(evj[k,0]/dt))
            end = int(round((evj[k,1]+evj[k,0])/dt + 1))
            labl[start:end] = j + 1
            if start < T:
                axarr.plot(range(start,end+1), [0.5*max(np.absolute(dif_sum[(1,0)]))]*(end+1-start), color = cl[j+1], label=move[j+1])

    handles, labels = axarr.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    axarr.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0, 1.02, 1.02, 0.502), loc=3,
               ncol=6, mode="expand", borderaxespad=0., fontsize=8)  
    plt.savefig(folder_name+'k_fold_W_1_dis.png'.replace('k_fold', str(k_fold)))
    plt.close()
            
                
    with open(folder_name_0 + 'full_data.pkl') as f:
        save = pickle.load(f)
        X_org = save['data'][0]
    dif_sum_1 = dif_value(X_org.reshape((-1,T)))
    fig = plt.figure()
    axarr = fig.add_subplot(1, 1, 1)   
    axarr.plot(range(cut_time,T), dif_sum_1[cut_time-1:], color = 'blue')
    for j in range(J-1, -1, -1):
        evj = np.loadtxt(file_folder + 'ev' + str(j+1) + '.txt')
        for k in range(evj.shape[0]):
            start = int(math.ceil(evj[k,0]/dt))
            end = int(round((evj[k,1]+evj[k,0])/dt + 1))
            labl[start:end] = j + 1
            if start < T:
                axarr.plot(range(start,end+1), [0.5*max(np.absolute(dif_sum_1))]*(end+1-start), color = cl[j+1], label=move[j+1])

    handles, labels = axarr.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    axarr.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0, 1.02, 1.02, 0.502), loc=3,
               ncol=6, mode="expand", borderaxespad=0., fontsize=8)  
    plt.savefig(folder_name+'k_fold_X_dis.png'.replace('k_fold', str(k_fold)))
    plt.close()


    K = [5, 6, 7, 8]
    res0 = []
    res1 = []

    for i in range(len(K)):
        est = KMeans(n_clusters=K[i])
        tcca = est.fit(W[(0,0)].copy().T)
        ans1 = metrics.adjusted_rand_score(labl, tcca.labels_)
        res0.append(ans1)
    for i in range(len(K)):
        est = KMeans(n_clusters=K[i])
        tcca = est.fit(W[(1,0)].copy().T)
        ans1 = metrics.adjusted_rand_score(labl, tcca.labels_)
        res1.append(ans1)

    f, axarr = plt.subplots(3,2,figsize=(10,10))
    #ll = ['A', 'B', 'C', 'D', 'E']
    #plt.setp(axarr, xticks=[0, 100, 200, 300, 400], xticklabels=['0', '100', '200', '300', '400'], yticks = [-600, -300, 0, 300], yticklabels=['-600', '-300', '0', '300'])
    for i in range(3):
        for j in range(2):
            n = i*2 + j
            axarr[i,j].spines['right'].set_visible(False)
            axarr[i,j].spines['top'].set_visible(False)

            # Only show ticks on the left and bottom spines
            axarr[i,j].yaxis.set_ticks_position('left')
            axarr[i,j].xaxis.set_ticks_position('bottom')
            if n <= 4:
                axarr[i,j].plot(vox,W[(0,0)][:,N_t[n]],color='black') 
                axarr[i,j].set_title(name_t[n], fontsize=10, fontweight='bold')
                for v in range(len(segt)-1):
                    axarr[i,j].plot(range(segt[v], segt[v+1]), [0]*(segt[v+1]-segt[v]), label=name_tick[v], color=cl[v+1])

            if n == 5:
                axarr[i,j].set_xticks([])
                axarr[i,j].set_yticks([])  
                axarr[i,j].spines['left'].set_visible(False)
                axarr[i,j].spines['bottom'].set_visible(False)  
            handles, labels = axarr[i, j].get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            print len(by_label)
            if i == 0 and j == 0:
                axarr[i, j].legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0.2, 1.19, 1.6, 0.602), loc=3,
            ncol=len(by_label), mode="expand", borderaxespad=0., fontsize=10)  
    f.subplots_adjust(top=0.85, wspace = .25, hspace = .35)  
     
    plt.savefig(folder_name + 'dif_w_real_data.svg')
    plt.close()




    n = len(segt)
    n_t = len(N_t)
    cor_tcca = np.zeros((n, n, n_t))
    cor_avg = np.zeros((n, n, n_t))
    X1_segt = {}
    W1_segt = {}

    for i in range(n-1):
        X1_segt[i] = X1[:,segt[i]:segt[i+1],:]
        W1_segt[i] = W[(0,0)][segt[i]:segt[i+1],:]
    W1_segt[n-1] = W[(1,0)]
    X1_segt[n-1] = X2

    for i in range(n):
        for j in range(n):
            for k, t in enumerate(N_t):
                cor_tcca[i,j,k] = np.nan_to_num(stats.pearsonr(np.dot(X1_segt[i][:,:,t], W1_segt[i][:,t]), np.dot(X1_segt[j][:,:,t], W1_segt[j][:,t]))[0])
                cor_avg[i,j,k] = np.nan_to_num(stats.pearsonr(np.sum(X1_segt[i][:,:,t], axis=1), np.sum(X1_segt[j][:,:,t], axis=1))[0])
    for t in range(n_t):
        fig, ax = plt.subplots(1,2, figsize=(15,7))
        f1 = sns.heatmap(cor_tcca[:,:,t], xticklabels=name_tick, yticklabels=name_tick, ax=ax[0])
        f2 = sns.heatmap(cor_avg[:,:,t], xticklabels=name_tick, yticklabels=name_tick, ax=ax[1])
        fig.subplots_adjust(wspace = .35, hspace = .25 )
        fig.savefig(folder_name + name_t[t] + '.png')
        plt.close()
    seg = segt
    W = abs(W_h[(0,0)])
    name = ['ROI_1(lh)', 'ROI_2(rh)', 'ROI_3(lf)', 'ROI_4(rf)', 'ROI_5(t)', 'ROI_6(t)']
    fig, axarr = plt.subplots(3,2, figsize=(15,20))
    for m in range(3):
        for n in range(2):
            kk = m*2 + n
            if kk < 5:
                tmp = np.sum(W[seg[kk]:seg[kk+1],:], axis=0)
                axarr[m, n].plot(range(T), (tmp - np.amin(tmp))/(np.amax(tmp)-np.amin(tmp)), color = 'xkcd:brick red',  label = 'our method')
                
                for j in range(J):
                    evj = np.loadtxt(file_folder + 'ev' + str(j+1) + '.txt')
                    for k in range(evj.shape[0]):
                        start = int(math.ceil(evj[k,0]/dt))
                        end = int(math.ceil((evj[k,1]+evj[k,0])/dt))
                        labl[start:end] = j + 1
                        print start, end, j
                        if start < T:
                            tt = np.arange(start,end,0.05)
                            axarr[m, n].plot(tt, [1]*len(tt), color = cl[j+1], label=move[j+1], linewidth=4)
                axarr[m, n].set_title(name[kk], fontsize=12)
                axarr[m,n].set_xlabel('Frames(1 frame = 0.72s)')
                axarr[m, n].set_yticks([])
                handles, labels = axarr[m, n].get_legend_handles_labels()
                by_label = OrderedDict(zip(labels, handles))
                        #axarr[m,n].legend(by_label.values(), by_label.keys())

                if m == 0 and n == 0:
                    axarr[m, n].legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0, 1.12, 1.9, 0.502), loc=3,
               ncol=4, mode="expand", borderaxespad=0., fontsize=8)  
            if kk == 5:
                axarr[m,n].set_xticks([])
                axarr[m,n].set_yticks([])  
                axarr[m,n].spines['left'].set_visible(False)
                axarr[m,n].spines['bottom'].set_visible(False) 
                axarr[m,n].spines['right'].set_visible(False)
                axarr[m,n].spines['top'].set_visible(False) 
    fig.subplots_adjust(top=0.9, wspace = .26, hspace = .26 )        
    plt.savefig(folder_name+'W_ROI.png')
    plt.close()
    return cor_tcca, cor_avg, res0, res1




def val_evt(labl, W, X):
    

    # cluster labl from X vs cluster labl from W 
    K = [6, 7, 8]
    res = []

    for i in range(len(K)):
        est = KMeans(n_clusters=K[i])
        tcca = est.fit(W)
        est = KMeans(n_clusters=K[i])
        baseline = est.fit(X)
        ans1 = metrics.adjusted_rand_score(labl, tcca.labels_)
        ans2 = metrics.adjusted_rand_score(labl, baseline.labels_)
        res.append([ans1, ans2])
    print res


    kf = model_selection.KFold(n_splits=5, shuffle=True)
    train_ind , test_ind = kf.split(X).next()
    W_train, W_test = W[train_ind,:], W[test_ind,:]
    X_train, X_test, Y_train, Y_test = X[train_ind,:], X[test_ind,:], labl[train_ind], labl[test_ind]
    


    print W_train.shape, W_test.shape, X_train.shape, X_test.shape, Y_train.shape, Y_test.shape
    #predict labl from X vs predict labl from W, logistic regression 
    cv = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=None)
    logreg = linear_model.LogisticRegressionCV(cv=cv, max_iter = 1e4, solver='saga', multi_class='multinomial')
    tcca = logreg.fit(W_train, Y_train)

    cv = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=None)
    logreg = linear_model.LogisticRegressionCV(cv=cv, max_iter = 1e4, solver='saga', multi_class='multinomial')
    baseline = logreg.fit(X_train, Y_train)

    l1 = tcca.score(W_test, Y_test)
    #l2 = l1
    l2 = baseline.score(X_test, Y_test)

    # random forest 
    param_grid = {
                 'n_estimators': [10, 50, 100, 500, 1000],
                 'max_depth': [2, 5, 7, 9]
                 }
    clf_tcca = RandomForest()
    grid_tcca = GridSearchCV(clf_tcca, param_grid, cv=5)
    grid_tcca.fit(W_train, Y_train)
    s1 = grid_tcca.score(W_test, Y_test)

 
    clf_baseline = RandomForest()
    grid_baseline = GridSearchCV(clf_baseline, param_grid, cv=5)
    grid_baseline.fit(X_train, Y_train)
    s2 = grid_baseline.score(X_test, Y_test)
    print Y_test, grid_baseline.predict(X_test), grid_tcca.predict(W_test)




    return res, [s1, s2], [l1, l2]







