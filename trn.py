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
        Xe = X[:, idx]
        
        X_prime = ut.ae_forward(ae, Xe, params)

        cost = (np.sum((X_prime - Xe) ** 2)) / (2 * minibatch_size)

        We = ut.ae_backward(ae, params) # We, Wd and V updated at this

        costs.append(cost) 
        
    print(f'Finished {amount_of_batches} batches with mean cost: {np.mean(costs)}')
        
    return We, costs


# Create single-layer autoencoder
def create_ae(hidden_layer_nodes, features):
    W = [
        ut.iniW(hidden_layer_nodes, features), #W encoder
        ut.iniW(features, hidden_layer_nodes), #W decoder
    ]
    V = [
        np.zeros_like(W[0]) # Setup momentum for only W encoder
    ]
    return { 'W': W, 'V': V, 'A': [], 'Z': []}


# AE's Training by use miniBatch RMSprop+Pinv
def train_ae(X, amount_of_nodes, params):
    ae = create_ae(amount_of_nodes, X.shape[0])

    mse = []
    # train the autoencoder n iterations
    for i in range(params['sae_max_it']):
        X_permuted = X[:, np.random.permutation(X.shape[1])]
        We, costs = train_ae_batch(ae, X_permuted, params)
        
        mse.append(np.mean(costs))
        
        if i % 10 == 0 and i != 0:
            print(f'Iteration: {i}', mse[i])

    return We, np.array(mse)


#SAE's Training
def train_dl(X, params):
    W = []
    A = [X]

    aes_mse = []
    encoders_nodes = list(params.values())[8:]
    for n, amount_of_nodes in enumerate(encoders_nodes):
        print(f'Training autoencoder {n+1}...')

        We, ae_mse = train_ae(X, amount_of_nodes, params) # Encoded Weights
        X = ut.act_function((We @ X), params['encoder_act_func']) # New Data

        W.append(We) # Store encoded weights for later
        A.append(X)
        aes_mse.append(ae_mse)
        print(f'Training autoencoder {n+1}...: Done')
        break
    
    ut.plot_mses(aes_mse)

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
    W, Xr = train_dl(Xe, params)
    #Ws, cost = train_softmax(Xr, Ye, p_sft, p_sae)
    #ut.save_w_dl(W, Ws, cost)
            

if __name__ == '__main__':
	 main()
