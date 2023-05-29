#Training DL via RMSProp+Pinv

import numpy      as np
import utility    as ut


# Training miniBatch for softmax
def train_sft_batch(ann, X, Y, params):
    minibatch_size = params['sft_minibatch_size']
    amount_of_batches = np.int16(np.floor(X.shape[1] / minibatch_size))
    costs = []

    for n in range(amount_of_batches):   
        idx = get_batch_indexes(minibatch_size, n)
        Xb, Yb = X[:, idx], Y[:, idx]
        ann['A'][0] = Xb
        
        Yb_pred = ut.sft_forward(ann, Xb, Yb, params)
        W, V = ut.sft_backward(ann, Yb, params)

        # TODO: check if cost function is correct (notice inf swap to zero value)
        cost = ut.calculate_cost(Yb, Yb_pred, params)
        # if np.isnan(cost):
        #     print(Yb.shape)
        #     print(Yb_pred.shape)
        #     print(np.log(Yb_pred.T))
        #     exit()
        costs.append(cost)

    return costs


def create_ann(X, Y, W):    
    WL = ut.iniW(Y.shape[0], W[-1].shape[0])
    V = np.zeros_like(WL)
    W.append(WL)
    
    A = [None] * (len(W) + 1)
    Z = [None] * (len(W) + 1)
    
    return {'W': W, 'V': V, 'A': A, 'Z': Z, 'layers': len(W)+1}


# Softmax's training via SGD with Momentum
def train_softmax(X, Y, W, params):
    ann = create_ann(X, Y, W)
    mse = []
    for i in range(params['sft_max_it']):
        idx = np.random.permutation(X.shape[1])
        Xr, Yr = X[:,idx], Y[:,idx]
        costs = train_sft_batch(ann, Xr, Yr, params)
        mse.append(np.mean(costs))
    
    return(W, np.array(mse))
    

def get_batch_indexes(minibatch_size, n):
    start_index = n * minibatch_size
    end_index = start_index + minibatch_size
    return np.arange(start_index, end_index).astype(int)


# AE's Training with miniBatch
def train_ae_batch(ae, X, params):
    minibatch_size = params['sae_minibatch_size']
    amount_of_batches = np.int16(np.floor(X.shape[1] / minibatch_size))
    
    costs = []
    for n in range(amount_of_batches):
        idx = get_batch_indexes(minibatch_size, n)
        Xb = X[:, idx]

        X_prime = ut.ae_forward(ae, Xb, params)
        We = ut.ae_backward(ae, params) # We, Wd and V updated at this
        cost = (np.sum((X_prime - Xb) ** 2)) / (2 * minibatch_size)
        costs.append(cost)
        
    print(f'Finished {amount_of_batches} batches with mean cost: {np.mean(costs)}')
    return We, costs


# Create single-layer autoencoder
def create_ae(hidden_layer_nodes, features):
    W = [
        ut.iniW(hidden_layer_nodes, features), #W encoder
        ut.iniW(features, hidden_layer_nodes), #W decoder
    ]
    V = np.zeros_like(W[0]) # Setup momentum for only W encoder
    return { 'W': W, 'V': V, 'A': [], 'Z': []}


# AE's Training by use miniBatch RMSprop+Pinv
def train_ae(X, amount_of_nodes, params):
    ae = create_ae(amount_of_nodes, X.shape[0])

    mse = []
    for i in range(params['sae_max_it']):
        Xr = X[:, np.random.permutation(X.shape[1])]
        We, costs = train_ae_batch(ae, Xr, params)
        mse.append(np.mean(costs))

        if i % 10 == 0 and i != 0:
            print(f'Iteration: {i}', mse[i])

    return We, np.array(mse)


#SAE's Training
def train_dl(X, params):
    W = []
    Xr = X
    aes_mses = []

    encoders_nodes = list(params.values())[8:]
    for n, amount_of_nodes in enumerate(encoders_nodes):
        print(f'Training autoencoder {n+1}...')
        We, ae_mse = train_ae(Xr, amount_of_nodes, params) # Encoded Weights
        Xr = ut.act_function((We @ Xr), params['encoder_act_func']) # New Data

        W.append(We)
        aes_mses.append(ae_mse)
        print(f'Training autoencoder {n+1}...: Done')
    
    #ut.plot_mses(aes_mses)

    return W, X


#load Data for Training
def load_data_trn():
    X = np.loadtxt('X_train.csv', delimiter=',')
    Y = np.loadtxt('Y_train.csv', delimiter=',')
    return X.T, Y.T


# Beginning ...
def main():
    params = ut.load_config()
    Xe, Ye = load_data_trn()
    W, _ = train_dl(Xe, params)
    Ws, costs = train_softmax(Xe, Ye, W, params)
    print(costs)
    ut.plot_mses([costs])
    #ut.save_w_dl(W, Ws, cost)
            

if __name__ == '__main__':
	main()
