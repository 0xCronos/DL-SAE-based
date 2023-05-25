#Training DL via RMSProp+Pinv

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
    W = ut.iniW(y.shape[0], x.shape[0])
    V = np.zeros(W.shape) 

    for Iter in range(1, par1[0]):
        idx = np.random.permutation(x.shape[1])
        xe, ye  = x[:,idx], y[:,idx]
        W, V, c = train_sft_batch(xe, ye, W, V, param)

    return(W, Costo)
    

# AE's Training with miniBatch
def train_ae_batch(X, w1, v, w2, param):
    amount_of_batches = np.int16(np.floor(X.shape[1]/param[0]))
    costs = []
    
    for i in range(amount_of_batches):
        pass

    return(w1, v, costs)


# AE's Training by use miniBatch RMSprop+Pinv
def train_ae(X, amount_of_nodes, param):
    w1 = ut.iniW(amount_of_nodes, X.shape[0])

    for i in range(param['sae_max_it']):
        rand_X = X[:, np.random.permutation(X.shape[1])]
        w2 = np.linalg.pinv(rand_X) # TODO: usar ut.pinv_ae
        print(w2.shape)
        exit()
        w1, v, c = train_ae_batch(rand_X, w1, v, w2, param)

    return(w2.T)



#SAE's Training
def train_dl(X, params):
    encoders_weights = []
    
    encoders_nodes = list(params.values())[5:]
    for amount_of_nodes in encoders_nodes:
        W_enc = train_ae(X, amount_of_nodes, params)
        X_enc  = ut.act_functs(W_enc, X, params)
        encoders_weights.append(X_enc)
        
    return(encoders_weights, X_enc)


#load Data for Training
def load_data_trn():
    X = np.loadtxt('X_train.csv', delimiter=',')
    Y = np.loadtxt('Y_train.csv', delimiter=',')
    return X.T, Y.T


# Beginning ...
def main():
    params = ut.load_config()
    xe, ye = load_data_trn()
    W, Xr = train_dl(xe, params)
    #Ws, cost = train_softmax(Xr, ye, p_sft, p_sae)
    #ut.save_w_dl(W, Ws, cost)
            

if __name__ == '__main__':   
	 main()
