import numpy as np
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from utils import * #grid_search_params,get_weight_bias,suffle_data,sort_close_to_margin_skl,print_subplot_data
from data.synthetic import create_synt_data, create_synt_data_full
from data.prepare_data import suffle_and_split
import matplotlib.pyplot as plt
import math
import multiprocessing
from joblib import Parallel, delayed
import copy
import os
import sys
from  warnings import simplefilter

if not sys.warnoptions:
    simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses

#METAVARS
run_all = False
run_eq1 = True
run_eq2 = False
run_eq3 = False
loo = False
max_iter=1e6

#Eq1c
if run_all or run_eq1:
    dims = [1,20,50,200]
    for d in dims:
        acc=[]
        for i in range(5):  
            mu1 = 1*np.ones(d)/math.sqrt(d)
            mu2 = 1*(np.ones(d)*-1)/math.sqrt(d)
            sig1 = sig2 = np.eye(d) 
            data = create_synt_data(mu1,mu2,sig1,sig2,d)
            w = mu1-mu2
            b=-0.5*(np.dot(mu1.transpose(),mu1)-np.dot(mu2.transpose(),mu2))

            Xtr, Ytr = data['train_data']
            Xv, Yv = data['val_data']
            Xts, Yts = data['test_data']
            range_c=[0.001,0.01,0.1,1,10,100]
            y=np.sign(np.matmul(Xts,w)+b)
            acc.append((y.reshape(-1)==Yts).mean())
        print('--------- for input of dimention %d the accuracy is %f'%(d,sum(acc)/len(acc)) )   
        import pdb; pdb.set_trace()
    print('Done Q1')    

#Eq2
def paralle_svm_exp(i,data,classifier_types=['linear','rbf']):
    range_g=[0.001,0.01,0.1,1,10]
    range_c=[0.001,0.01,0.1,1,10,100]
    sheffled_data = suffle_and_split(data)
    Xtr, Ytr  = sheffled_data['train_data']
    Xv, Yv = sheffled_data['val_data']
    Xts, Yts = sheffled_data['test_data']
    accuracy_rand={}; accuracy_simple={}
    
    for classifier_type in classifier_types:
        if classifier_type=='linear':
            tuned_parameters = [{'kernel': ['linear'],'C': range_c}]               
        else:
            tuned_parameters = [{'kernel': ['rbf'], 'gamma': range_g,'C': range_c}]
        
        print('running grid search params for %s classifier...'%classifier_type)   
        clf = GridSearchCV(SVC(max_iter=max_iter), tuned_parameters)
        clf.fit(Xv,Yv)
        if classifier_type=='linear':
            best_params_linear = clf.best_params_
            print('Best params found: ',best_params_linear)
        else:
            best_params_rbf = clf.best_params_
            print('Best params found: ',best_params_rbf)

        accuracy_rand[classifier_type] = np.zeros([len(num_addtional_points)])
        accuracy_simple[classifier_type] = np.zeros([len(num_addtional_points)])

    for classifier_type in classifier_types:    
        #RAND Alg:
        linear = True if classifier_type=='linear' else False
        
        for idx_n,n in enumerate(num_addtional_points):
            if linear:                
                svm = SVC(kernel='linear', C=best_params_linear['C'], max_iter=max_iter)
            else:
                svm = SVC(kernel='rbf', C=best_params_rbf['C'],gamma=best_params_rbf['gamma'], max_iter=max_iter)     
            if loo:
                cvr = 2 if n==5 else 5
                scores = cross_val_score(svm, Xtr[:n], Ytr[:n], cv=cvr) 
                acc=scores.mean()
            else:              
                svm.fit(Xtr[:n],Ytr[:n])       
                acc = svm.score(Xts,Yts)  
            accuracy_rand[classifier_type][idx_n] = acc
            
        #SIMPLE Alg:
        best_params = best_params_linear if linear else best_params_rbf
        Xpool=Xtr[strat_set:]; Ypool=Ytr[strat_set:]
        Xtr=Xtr[:strat_set]; Ytr=Ytr[:strat_set]
        accuracy_simple[classifier_type][0] = accuracy_rand[classifier_type][0]
        for idx_n,n in enumerate(num_addtional_points[1:]):  
            Xtr,Ytr,Xpool,Ypool,svm=simple_alg(Xtr,Ytr,Xpool,Ypool,best_params,strat_set,linear,pool_size=20) 
            if loo:
                cvr = 2 if n==5 else 5
                scores = cross_val_score(svm, Xtr, Ytr, cv=cvr) 
                acc=scores.mean()
            else:                 
                acc = svm.score(Xts,Yts)  
            accuracy_simple[classifier_type][idx_n+1] = acc
    return [accuracy_rand,accuracy_simple]

if run_all or run_eq2:
    num_repeat = 30
    strat_set = 5
    num_addtional_points = [strat_set]+[n for n in range(20,280,20)]  

    data1 = np.load('./data/breast_cancer_scaled_full.npy')
    data1=data1.tolist()
    data2 = np.load('./data/diabetes_full.npy')
    data2=data2.tolist()
    
    dataS=[]
    for d in [2,50,100,200]:
        mu1 = np.ones(d)/math.sqrt(d)
        mu2 = (np.ones(d)*-1)/math.sqrt(d)
        sig1 = sig2 = np.eye(d) 
        dataS.append(create_synt_data_full(mu1,mu2,sig1,sig2,d))
    
    results = {}
    classifier_types = ['linear','rbf']
    datanames=['breast_cancer','diabetes','sythetic2','sythetic50','sythetic100','sythetic200']
    
    print('loo',loo,'num_repeat',num_repeat)
    for ID,data in enumerate([data1,data2]+dataS):
        print('Running on datset ',ID)
        results[ID]={}
        accuracy_rand = {'linear': np.zeros([num_repeat,len(num_addtional_points)]),'rbf': np.zeros([num_repeat,len(num_addtional_points)])}
        accuracy_simple = {'linear': np.zeros([num_repeat,len(num_addtional_points)]),'rbf': np.zeros([num_repeat,len(num_addtional_points)])}
        processed_list = Parallel(n_jobs=num_repeat)(delayed(paralle_svm_exp)(i,data) for i in range(num_repeat))        
        sum_acc_simple={'linear': None,'rbf': None}; sum_acc_rand={'linear': None,'rbf': None}
        
        for clf_type in classifier_types:
            for i in range(num_repeat):
                sum_acc_rand[clf_type] = processed_list[i][0][clf_type] if sum_acc_rand[clf_type] is None else sum_acc_rand[clf_type]+processed_list[i][0][clf_type]
                sum_acc_simple[clf_type] = processed_list[i][1][clf_type] if sum_acc_simple[clf_type] is None else sum_acc_simple[clf_type]+processed_list[i][1][clf_type]
            sum_acc_rand[clf_type] /= num_repeat
            sum_acc_simple[clf_type] /= num_repeat
                        
        results[ID]['RAND'] = copy.deepcopy(sum_acc_rand)
        results[ID]['SIMPLE'] = copy.deepcopy(sum_acc_simple)

    filename='results_q2a_test_data' if loo else 'results_q2a_loo'
    np.save(filename,results)
    for ID,data in enumerate([data1,data2]+dataS):
        print_subplot_data(results[ID],num_addtional_points,datanames[ID],loo)
        classifier_types = ['linear','rbf']
    print('Done Q2')

#Eq3
if run_all or run_eq3:
    ##Dataset which confuses SIMPLE for linear clasifier 
    num_repeat = 30
    strat_set = 20
    loo=False
    num_addtional_points = [strat_set]+[n for n in range(20,280,20)]  
    X1 = np.random.uniform(5,10,270)
    Y1 = np.ones(270)
    X2 = np.random.uniform(-10,-5,270)
    Y2 = -np.ones(270)
    X3 = np.random.uniform(0,1,15)
    Y3 = np.ones(15)
    X4 = np.random.uniform(-1,0,15)
    Y4 = -np.ones(15)
    X5 = np.random.uniform(-4,-2,30)
    Y5 = np.ones(30)
    X = np.concatenate((X1,X2,X3,X4,X5),0).reshape(-1,1)
    Y = np.concatenate((Y1,Y2,Y3,Y4,Y5),0)
    data_simple_linear={'data':X,'targets':Y}
    balanced_split(data_simple_linear)
    import pdb; pdb.set_trace()
    aa=paralle_svm_exp(0,data_simple_linear)
    results={'RAND':{'linear': aa[0]['linear']},'SIMPLE':{'linear': aa[1]['linear']}}
    print_subplot_data(results,num_addtional_points,'SIMPLE failue on linear classifier',loo)
    
    import pdb; pdb.set_trace()
    
    samples=None; targets=None
    a=1
    for i in range(0,10,2):
        num_samples=270 if (i==0 or i==8) else 15
        if i==4: num_samples=30
        theta = np.random.uniform(0, 2*np.pi, num_samples)
        x,y=np.random.uniform(i*2,(i+1)*2,num_samples) * np.cos(theta), np.random.uniform(i*2,(i+1)*2,num_samples) * np.sin(theta)
        plt.scatter(x,y)
        print(x.shape)
        targets= a*np.ones(num_samples) if targets is None else np.concatenate((targets,a*np.ones(num_samples)),0)
        a*=-1
        samples= np.array([x,y]) if samples is None else np.concatenate((samples,np.array([x,y])),1)
        
    targets = np.array(targets)    
    samples=samples.transpose()
    targets=targets.reshape(-1)
    data_simple_rbf={'data':samples/samples.max(),'targets':targets}
    aa=paralle_svm_exp(0,data_simple_rbf,classifier_types=['rbf'])
    results={'RAND':{'rbf': aa[0]['rbf']},'SIMPLE':{'rbf': aa[1]['rbf']}}
    print_subplot_data(results,num_addtional_points,'SIMPLE failue on rbf classifier',loo)
    print('Done Q3')
    




    
