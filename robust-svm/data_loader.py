import numpy as np
from scipy.io import loadmat
from scipy import sparse
from sklearn.datasets import load_svmlight_file
from sklearn import preprocessing

def gaussian(n, d):
    x = np.zeros((n,d+1))
    y = np.zeros((n))
    for i in range(n):
        X = np.random.normal(0, 1, d)
        x[i,0:d] = X
        x[i,d] = 1
    w = np.random.normal(0,1,d)
    b = np.random.normal(0,1)

    for i in range(n):
        t = np.int32(np.sign(np.random.normal(np.dot(np.transpose(w),x[i,0:d]) + b,0.01)))
        if t == 0:
            t = -1
        y[i] = t
    return x, y, w, b

def enron():
    x = loadmat('data/enron.mat')
    x_train = x['X_train'].toarray()
    x_test = x['X_test'].toarray()

    y_train = x['y_train']
    y_test = x['y_test']

    x = np.concatenate((x_train,x_test),axis=0)
    y = np.concatenate((y_train,y_test),axis=0).flatten()

    intercept = np.ones((x.shape[0],1))
    x = np.concatenate((x, intercept),axis=1)

    perm = np.random.permutation(len(y))

    return x[perm],y[perm]

def salary():
    x_train,y_train = load_svmlight_file("data/a9a.txt")
    y_train = np.int32(y_train)
    x_train = x_train.toarray()
    x_train = x_train[:,:-1]
    x_test,y_test = load_svmlight_file("data/a9a.t")
    x_test = x_test.toarray()
    y_test = np.int32(y_test)

    x = np.concatenate((x_train,x_test),axis=0)
    y = np.concatenate((y_train,y_test),axis=0).flatten()

    intercept = np.ones((x.shape[0],1))
    x = np.concatenate((x, intercept),axis=1)

    perm = np.random.permutation(len(y))

    return x[perm],y[perm]
