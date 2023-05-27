# My Utility : auxiliars functions
import numpy  as np


# Configurations of Saes and softmax as dict 
def load_config():      
    cnf_sae = np.genfromtxt('cnf_sae.csv', delimiter=',')    
    cnf_sft = np.genfromtxt('cnf_softmax.csv', delimiter=',')    
    params = {
        'p_inverse_param' : cnf_sae[0],
        'encoder_act_func' : int(cnf_sae[1]),
        'sae_max_it' : int(cnf_sae[2]),
        'sae_minibatch_size' : cnf_sae[3],
        'sae_learning_rate' : float(cnf_sae[4]),
        'sft_max_it' : cnf_sft[0],
        'sft_learning_rate' : cnf_sft[1],
        'sft_minibatch_size' : cnf_sft[2],
        'encoder1_nodes' : int(cnf_sae[5]),
        'encoder2_nodes' : int(cnf_sae[6]),
    }
    return params


#Activation function
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


def ae_backward(ae, params):
    Xe = ae['A'][0]

    Wd = calculate_pinv(Xe, ae['W'][0], params['p_inverse_param'])
    
    gW1 = gradW1(ae, params)

    # RMSProp (Update We)
    We, Ve = updW1_rmsprop(ae, gW1, params)

    # pre-calculated pseudoinverse (Update Wd)
    ae['W'][1] = Wd

    return ae['W'][0]
    
   
# Calculate Pseudo-inverse
def calculate_pinv(X, We, C):
    H = We @ X
    A = (H @ H.T) + (np.identity(H.T.shape[1]) / C)
    #U, S, V = np.linalg.svd(A, full_matrices=False)
    #A_inv = V.T @ np.linalg.inv(np.diag(S)) @ U.T
    #Wd = X @ H.T @ A_inv
    Wd = X @ H.T @ np.linalg.pinv(A)
    return Wd


def ae_summary(autoencoder, force_exit=False):
    print("Summary for autoencoder")
    print("X - We -> H - Wd -> X'\n")
    print(f"X shape is: {autoencoder['A'][0].shape}")
    print(f"We shape is: {autoencoder['W'][0].shape}")
    print(f"H shape is: {autoencoder['A'][1].shape}") # this is the hidden layer
    print(f"Wd shape is: {autoencoder['W'][1].shape}")
    print(f"X' shape is: {autoencoder['A'][2].shape}")
    print("*"*25)
    print(f"A len is: {len(autoencoder['A'])}")
    print(f"Z len is: {len(autoencoder['Z'])}")
    print(f"W len is: {len(autoencoder['W'])}")
    print(f"V len is: {len(autoencoder['V'])}")
    exit() if force_exit else '' 

 
# STEP 2: Feed-Backward for AE
def gradW1(ae, params):
    #ae_summary(autoencoder, force_exit=True)
    encoder_act_func = params['encoder_act_func']

    E = (ae['A'][2] - ae['A'][0]) # (X' - X)
    
    delta = E
    
    gW1 = (ae['W'][1].T @ delta) * deriva_act(ae['Z'][0], encoder_act_func)
    gW1 = gW1 @ ae['A'][0].T

    assert gW1.shape == ae['W'][0].shape, "TEST FAILED: Dimensions of encoder weight and gradient must be equal."

    return gW1


# Update AE's weight via RMSprop
def updW1_rmsprop(ae, gW1, params):
    beta = 0.9
    epsilon = 1e-8
    
    ae['V'][0] = (beta * ae['V'][0]) + ((1 - beta) * (gW1**2))
    grms = (1 / np.sqrt(ae['V'][0] + epsilon)) * gW1
    ae['W'][0] = ae['W'][0] - (params['sae_learning_rate'] * grms)
    
    return ae['W'][0], ae['V'][0]


# Update Softmax's weight via RMSprop
def updW_sft_rmsprop(w, v, gw, params):
    beta, eps = 0.9, 1e-8
    ...
    return(w, v)


# Softmax's gradient
def gradW_softmax(x, y, a):
    ya   = y * np.log(a)
    ...
    return(gW, Cost)


# Calculate Softmax
def softmax(z):
        exp_z = np.exp(z-np.max(z))
        return(exp_z/exp_z.sum(axis=0,keepdims=True))


# save weights SAE and costo of Softmax
def save_w_dl(W, Ws, cost):
    ...
    
    
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
   