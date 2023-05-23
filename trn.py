#Training DL via RMSProp+Pinv

import pandas     as pd
import numpy      as np
import utility    as ut


# Training miniBatch for softmax
def train_sft_batch(x, y, W, V, param):
    costo = []    
    for i in range(numBatch):   
        pass

    return(W, V, costo)


# Softmax's training via SGD with Momentum
def train_softmax(x, y, par1, par2):
    W        = ut.iniW(y.shape[0], x.shape[0])
    V        = np.zeros(W.shape) 

    for Iter in range(1, par1[0]):
        idx     = np.random.permutation(x.shape[1])
        xe, ye  = x[:,idx], y[:,idx]         
        W, V, c = train_sft_batch(xe, ye, W, V, param)

    return(W, Costo)
    

# AE's Training with miniBatch
def train_ae_batch(x, w1, v, w2, param):
    numBatch = np.int16(np.floor(x.shape[1]/param[0]))
    cost = [] 
    
    for i in range(numBatch):
        pass

    return(w1, v, cost)


# AE's Training by use miniBatch RMSprop+Pinv
def train_ae(X, param):
    w1 = ut.iniW(X, )

    for Iter in range(1, param):
        Xe       = X[:, np.random.permutation(X.shape[1])]
        w1, v, c = train_ae_batch(Xe, w1, v, w2, param)

    return(w2.T)


#SAE's Training
def train_sae(X, params):
    W = {}
    
    for encoder_nodes in list(params.values())[5:]:
        w1 = train_ae(X, encoder_nodes, params)
        X  = ut.act_functs(w1, X, params)
        Apilar(W, w1)

    return(W, X)


#load Data for Training
def load_data_trn():
    xe = np.loadfromtxt('X_train.csv', delimiter=',')
    ye = np.loadfromtxt('Y_train.csv', delimiter=',')

    return xe, ye


# Beginning ...
def main():
    p_sae, p_sft = ut.load_config()
    xe, ye       = load_data_trn()
    W, Xr        = train_sae(xe, p_sae)
    #Ws, cost     = train_softmax(Xr, ye, p_sft, p_sae)
    #ut.save_w_dl(W, Ws, cost)
            

if __name__ == '__main__':   
	 main()
