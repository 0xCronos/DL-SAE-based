import numpy      as np
import utility    as ut


# Save Data : training and testing
def save_data(X, Y):
    np.savetxt("X_train.csv", X['train'], delimiter=",")
    np.savetxt("X_test.csv", X['test'], delimiter=",")
    np.savetxt("Y_train.csv", Y['train'], delimiter=",", fmt="%i")
    np.savetxt("Y_test.csv", Y['test'], delimiter=",", fmt="%i")

# Binary Label
def binary_label(Y):
    n_classes = len(np.unique(Y))
    
    binary_labels = []
    for label in Y:
        binary_class = np.zeros(n_classes)
        binary_class[label - 1] = 1
        binary_labels.append(binary_class)

    return np.array(binary_labels)


# Load data csv
def load_data_csv():
    train_data = np.genfromtxt('train.csv', delimiter=',')
    test_data = np.genfromtxt('test.csv', delimiter=',')
    
    X = {
        'train' : train_data[:, :-1],
        'test'  : test_data[:, :-1],
    }

    Y = {
        'train' : binary_label(train_data[:, -1:].flatten().astype(int)),
        'test'  : binary_label(test_data[:, -1:].flatten().astype(int)),
    }
    
    return X, Y


# Beginning ...
def main():
    X, Y  = load_data_csv()
    save_data(X, Y)
    

if __name__ == '__main__':   
    main()

