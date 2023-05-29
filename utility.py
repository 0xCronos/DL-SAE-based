# My Utility : auxiliars functions
import numpy  as np


# Configurations of autoencoders and softmax as dict 
def load_config():
    cnf_sae = np.genfromtxt('cnf_sae.csv', delimiter=',')    
    cnf_sft = np.genfromtxt('cnf_softmax.csv', delimiter=',')    
    params = {
        'p_inverse_param' : cnf_sae[0],
        'encoder_act_func' : int(cnf_sae[1]),
        'sae_max_it' : int(cnf_sae[2]),
        'sae_minibatch_size' : cnf_sae[3],
        'sae_learning_rate' : float(cnf_sae[4]),
        'sft_max_it' : int(cnf_sft[0]),
        'sft_learning_rate' : cnf_sft[1],
        'sft_minibatch_size' : cnf_sft[2],
        'encoder1_nodes' : int(cnf_sae[5]),
        'encoder2_nodes' : int(cnf_sae[6]),
    }
    return params


# Activation function
def act_function(Z, activation_function):
    if activation_function == 1:
        return np.maximum(0, Z)
    if activation_function == 2:
        return np.maximum(0.01 * Z, Z)
    if activation_function == 3:
        return np.where(Z > 0, Z, 1 * (np.exp(Z) - 1))
    if activation_function == 4:
        lam=1.0507; alpha=1.6732
        return np.where(Z > 0, Z, alpha*(np.exp(Z)-1)) * lam
    if activation_function == 5:
        return 1 / (1 + np.exp(-Z))
    if activation_function == -1: # Special case: use for f(x) = x
        return Z


# Derivatives of the activation funciton
def deriva_act(Z, activation_function):
    if activation_function == 1:
        return np.where(Z >= 0, 1, 0)
    if activation_function == 2:
        return np.where(Z >= 0, 1, 0.01)
    if activation_function == 3:
        return np.where(Z >= 0, 1, 0.01 * np.exp(Z))
    if activation_function == 4:
        lam=1.0507; alpha=1.6732
        return np.where(Z > 0, 1, alpha*np.exp(Z)) * lam
    if activation_function == 5:
        s = act_function(Z, activation_function)
        return s * (1 - s)


# Initialize one-wieght
def iniW(next, prev):
    r = np.sqrt(6/(next+ prev))
    w = np.random.rand(next, prev)
    w = w*2*r-r
    return(w)
    

# STEP 1: Feed-forward of AE
def ae_forward(ae, X, params):
    Z1 = ae['W'][0] @ X
    H = act_function(Z1, params['encoder_act_func'])
    Z2 = ae['W'][1] @ H
    X_prime = Z2 # act_function(Z2, -1) # f(x) = x
    ae['Z'] = [Z1, Z2]
    ae['A'] = [X, H, X_prime]
    return X_prime


def ae_backward(ae, X, params):
    gW1 = gradW1(ae, params)
    We, _ = updW1_rmsprop(ae, gW1, params)
    return We
    
   
# Calculate Pseudo-inverse
def calculate_pinv(X, H, C):
    A = (H @ H.T) + (np.identity(H.T.shape[1]) / C)
    U, S, V = np.linalg.svd(A, full_matrices=False)
    A_inv = V.T @ np.linalg.inv(np.diag(S)) @ U.T
    Wd = X @ H.T @ A_inv
    return Wd


 
# STEP 2: Feed-Backward for AE
def gradW1(ae, params):
    encoder_act_func = params['encoder_act_func']
    E = (ae['A'][2] - ae['A'][0]) # (X' - X) -> also delta
    gW1 = (ae['W'][1].T @ E) * deriva_act(ae['Z'][0], encoder_act_func)
    gW1 = gW1 @ ae['A'][0].T
    assert gW1.shape == ae['W'][0].shape, "TEST FAILED: Dimensions of encoder weight and gradient must be equal."
    return gW1


def sft_forward(ann, params):
    L = ann['layers']
    for l in range(1, L):
        ann['Z'][l] = (ann['W'][l-1] @ ann['A'][l-1])
        if l != L: # hidden layers
            ann['A'][l] = act_function(ann['Z'][l], params['encoder_act_func'])
        else: # output layer
            ann['A'][l] = softmax(ann['Z'][l])
    return ann['A'][-1]


def sft_backward(ann, Y, params):
    gW = gradW_softmax(ann, Y, params)
    W, V = updW_sft_rmsprop(ann, gW, params)
    return W, V


def calculate_cost(Y, Y_pred, params):
    minibatch_size = params['sft_minibatch_size']
    log_Y_pred = np.log(Y_pred)
    log_Y_pred[log_Y_pred == -np.inf] = 0
    cost = -np.sum(np.sum(Y * log_Y_pred, axis=0) / Y.shape[0]) / minibatch_size
    return cost

# Update AE's weight via RMSprop
def updW1_rmsprop(ae, gW1, params):
    beta, epsilon = 0.9, 1e-8
    ae['V'] = (beta * ae['V']) + (1 - beta) * np.square(gW1)
    grms = (1 / np.sqrt(ae['V'] + epsilon)) * gW1
    ae['W'][0] = ae['W'][0] - params['sae_learning_rate'] * grms
    return ae['W'][0], ae['V']


# Update Softmax's weight via RMSprop
def updW_sft_rmsprop(ann, gW, params):
    beta, epsilon = 0.9, 1e-8
    ann['V'] = beta * ann['V'] + (1 - beta) * np.square(gW)
    grms = (1 / np.sqrt(ann['V'] + epsilon)) * gW
    ann['W'][-1] = ann['W'][-1] - (params['sft_learning_rate'] * grms)
    return ann['W'][-1], ann['V']


# Softmax's gradient
def gradW_softmax(ann, Y, params):
    minibatch_size = params['sft_minibatch_size']
    gW = -(1/minibatch_size) * ((Y - ann['A'][-1]) @ ann['A'][0].T)
    return gW

# Calculate Softmax
def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z))
    return(exp_Z / exp_Z.sum(axis=0, keepdims=True))


# save weights SAE and cost of Softmax
def save_w_dl(W, costs):
    np.savez("W_snn.npz", *W)
    np.savetxt("costo.csv", costs, fmt="%.10f")
    
    
def plot_mses(X):
    import matplotlib.pyplot as plt

    for i, x in enumerate(X):
        plt.plot(range(0, len(x)), x, label=f'Iteration {i+1}')

        # Set plot labels and title
        plt.title('MSE minimización')
        plt.xlabel('Iteraciones')
        plt.ylabel('MSE')

        # Save the plot to a file
        plt.savefig('trn-graph.png')

    # Display the plot
    plt.show()
   