# My Utility : auxiliars functions

import numpy  as np


# Configurations of Saes and softmax as dict 
def load_config():      
    cnf_sae = np.genfromtxt('cnf_sae.csv', delimiter=',')    
    cnf_sft = np.genfromtxt('cnf_softmax.csv', delimiter=',')    
    params = {
        'p_inverse_param' : cnf_sae[0],
        'encoder_act_func' : cnf_sae[1],
        'sae_max_it' : int(cnf_sae[2]),
        'sae_minibatch_size' : cnf_sae[3],
        'sae_learning_rate' : cnf_sae[4],
        'encoder1_nodes' : int(cnf_sae[5]),
        'encoder2_nodes' : int(cnf_sae[6]),
        'sft_max_it' : cnf_sft[0],
        'sft_learning_rate' : cnf_sft[1],
        'sft_minibatch_size' : cnf_sft[2],
    }
    return params


# Initialize one-wieght
def iniW(next, prev):
    r = np.sqrt(6/(next+ prev))
    w = np.random.rand(next, prev)
    w = w*2*r-r

    return(w)
    
# STEP 1: Feed-forward of AE
def ae_forward(x,w1,w2):
    ...
    return(a)


#Activation function
def act_sigmoid(z):
    return(1/(1+np.exp(-z)))   


# Derivate of the activation function
def deriva_sigmoid(a):
    return(a*(1-a))
    
    
# Calculate Pseudo-inverse
def calculate_pinv(X, W1, C):
    ...
    ...
    return(w2)


# STEP 2: Feed-Backward for AE
def gradW1(a, w2):
    e       = a[2]-a[0]
    Cost    = np.sum(np.sum(e**2))/(2*e.shape[1])
    ...      
    return(gW1, Cost)


# Update AE's weight via RMSprop
def updW1_rmsprop(w, v, gw, mu):
    beta,eps = 0.9, 1e-8
    ...    
    return(w, v)


# Update Softmax's weight via RMSprop
def updW_sft_rmsprop(w, v, gw, mu):
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
    