import numpy as np

def create_synt_data(mu1,mu2,sig1,sig2,d,sample_size=300):
    features1 = np.random.multivariate_normal(mu1, sig1, sample_size)
    features2 = np.random.multivariate_normal(mu2, sig2, sample_size)    
    data_synt={}
    data_synt['train_data']= [np.concatenate((features1[:150],features2[:150]),0),np.concatenate((np.ones(150),np.ones(150)*-1),0)]
    data_synt['val_data'] = [np.concatenate((features1[150:225],features2[150:225]),0),np.concatenate((np.ones(75),np.ones(75)*-1),0)]
    data_synt['test_data'] = [np.concatenate((features1[225:],features2[225:]),0),np.concatenate((np.ones(75),np.ones(75)*-1),0)]
    
    for key in data_synt.keys():
        indices = np.random.permutation(data_synt[key][0].shape[0])
        data_synt[key][0] = data_synt[key][0][indices]
        data_synt[key][1] = data_synt[key][1][indices]
    return data_synt
def create_synt_data_full(mu1,mu2,sig1,sig2,d,sample_size=300):
    features1 = np.random.multivariate_normal(mu1, sig1, sample_size)
    features2 = np.random.multivariate_normal(mu2, sig2, sample_size)
    data_synt={}
    data_synt['data'] = np.concatenate((features1[:300],features2[:300]),0)
    data_synt['targets'] = np.concatenate((np.ones(300),np.ones(300)*-1),0)
    indices = np.random.permutation(data_synt['data'].shape[0])
    data_synt['data'] = data_synt['data'][indices]
    data_synt['targets'] = data_synt['targets'][indices]
    return data_synt
#np.save('data_synt',data_synt)




