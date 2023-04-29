import numpy as np

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
        if t == -1:
            t = 0
        y[i] = t
    return x, y, w, b

def enron():
    pass