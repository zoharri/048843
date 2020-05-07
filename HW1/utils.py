import numpy as np
from libsvm.svmutil import *
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
def grid_search_params(data_train,data_val,linear=True,min_g=-12,max_g=1,min_c=-5,max_c=3):
    Xt,Yt=data_train
    Xv,Yv=data_val
    best_g=None; best_c=None; acc_best=-1; best_model=None
    resolution_g=20
    resolution_c=5
    range_g=[r/resolution_g for r in range(min_g*resolution_g,max_g*resolution_g)]
    range_c=[r/resolution_c for r in range(min_c*resolution_c,max_c*resolution_c)]
    classifier_type = 'linear' if linear else 'rbf'
    for idx_g,g in enumerate(range_g):
        for c in range_c:
            if linear:
                m = svm_train(Yt, Xt, '-c '+ str(2**c)+' -t 0 -q')
            else:
                m = svm_train(Yt, Xt, '-c '+ str(2**c)+' -g '+str(2**g)+' -t 2 -q')
            p_label, p_acc, p_val = svm_predict(Yv, Xv, m,'-q')
            if p_acc[0] > acc_best:
                acc_best = p_acc[0]
                best_g = 2**g
                best_c = 2**c
                best_model = m
                print('best so far '+classifier_type,acc_best)
        if linear:
            break        
        print('------- Grid search in progress ',idx_g/len(range_g))
    return best_c,  best_g, best_model            

def get_weight_bias(model,d):
    #import pdb; pdb.set_trace()
    svs = np.array([k for sv in model.get_SV() for a,k in sv.items() if int(a)>0])
    svs = svs.reshape(-1,d)
    #svs = np.array([a[key] for key in a for a in model.get_SV() if key>0])
    svs_coef = np.array(model.get_sv_coef())
    weight = np.matmul(svs_coef.transpose(),svs)
    bias=-model.rho[0]
    return weight,bias

def suffle_data(data,sample_size=None):
    if sample_size is None:
       sample_size = data[0].shape[0]

    indices = np.random.permutation(data[0].shape[0])
    data_features = data[0][indices[:sample_size]]
    data_targets = data[1][indices[:sample_size]]
    return [data_features,data_targets]

def sort_close_to_margin(X,Y,m):
    p_label, p_acc, p_val = svm_predict(Y, X, m)
    idx = np.argsort(np.absolute(np.array(p_val).flatten()))
    return X[idx],Y[idx]

def sort_close_to_margin_skl(X,Y,Xts,Yts,best_params,strat_set,linear,loo):
    num_exp = strat_set if loo else 1
    dist=0; result_init=0
    for lo in range(num_exp):
        if linear:                
            svm = SVC(kernel='linear', random_state=0, C=best_params['C'], cache_size=4000,max_iter=1e6)
        else:
            svm = SVC(kernel='rbf', random_state=0, C=best_params['C'],gamma=best_params['gamma'], cache_size=4000,max_iter=1e6)
        Xlo=np.concatenate((X[:lo],X[(lo+1):strat_set]),0) if loo else X[:strat_set]
        Ylo=np.concatenate((Y[:lo],Y[(lo+1):strat_set]),0) if loo else Y[:strat_set]
        Xts=X[lo].reshape(1,-1) if loo else Xts
        Yts=np.array([Y[lo]]) if loo else Yts  
        svm.fit(Xlo,Ylo)
        Xta,Yta=X[strat_set:],Y[strat_set:]    
        y = svm.decision_function(Xta)
        if linear:
            w_norm = np.linalg.norm(svm.coef_)
        dist += y / w_norm if linear else y      
        result_init += svm.score(Xts,Yts)/num_exp

    idx = np.argsort(np.absolute(dist))

    return Xta[idx],Yta[idx],result_init

def simple_alg(Xtr,Ytr,Xpool,Ypool,best_params,strat_set,linear,pool_size=20):
    for i in range(pool_size):
        if linear:                
            svm = SVC(kernel='linear', random_state=0, C=best_params['C'], cache_size=4000,max_iter=1e6)
        else:
            svm = SVC(kernel='rbf', random_state=0, C=best_params['C'],gamma=best_params['gamma'], cache_size=4000,max_iter=1e6) 
        svm.fit(Xtr,Ytr)
        y = svm.decision_function(Xpool)
        if linear:
            w_norm = np.linalg.norm(svm.coef_)
        dist = y / w_norm if linear else y      
        idx = np.argmin(np.absolute(dist))
        Xtr=np.concatenate((Xtr,Xpool[idx].reshape(1,-1)),0)
        Ytr=np.concatenate((Ytr,np.array([Ypool[idx]])),0)
        Xpool = np.delete(Xpool,idx,0)
        Ypool = np.delete(Ypool,idx,0)
    return Xtr,Ytr,Xpool,Ypool,svm

def suffle_and_balance(Xtrp,Ytrp,cv_size):
    s,d=Xtrp.shape
    indices = np.random.permutation(s)
    balanced = False
    while not balanced:
        Xtrp = Xtrp[indices]
        Ytrp = Ytrp[indices]
        CVYTR = Ytrp.reshape(cv_size,-1).sum(0)
        if  not (any(CVYTR==5) or any(CVYTR==-5)):
            balanced = True
            import pdb; pdb.set_trace()
    return Xtrp,Ytrp

def print_subplot_data(data,xpoints,title,loo):
    legend=[]
    for class_key in data.keys():
        for alg_key in data[class_key].keys():
            plt.plot(xpoints,data[class_key][alg_key])
            legend.append(alg_key+'-'+class_key)       
    plt.legend(legend)    
    if loo: title += ' LOO'    
    plt.title(title)
    plt.xlabel('number of labeled samples')
    plt.ylabel('accuracy')
    plt.savefig(title)
    plt.show()
