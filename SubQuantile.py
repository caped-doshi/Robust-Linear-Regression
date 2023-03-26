import numpy as np
from matplotlib import pyplot as plt

def SubQ(X, y, T, p, majority=True):
    n = X.shape[0]
    theta = np.zeros(len(X[0]))
    deriv = np.zeros(len(X[0]))
    d = np.dot(X,theta) - y
    deriv = np.sum(2 * X * d[:, np.newaxis],axis=0)

    L = np.linalg.norm(np.matmul(np.transpose(X),X),2)
    alpha = 1/(2*L)

    update = -1 * alpha * deriv
    theta = theta + update

    for i in range(T):
        pred = np.matmul(X,theta)
        v = (pred - y)**2
        v_hat = sorted(v)
        v_arg_hat = np.argsort(v)

        subq_error = np.linalg.norm(v_hat[:int(n*p)],2)/int(n*p)

        X_np = X[v_arg_hat[:int(n*p)]]
        y_np = y[v_arg_hat[:int(n*p)]]
        d = np.dot(X_np,theta) - y_np
        deriv = np.sum(2 * X_np * d[:, np.newaxis],axis=0)

        L = np.linalg.norm(np.matmul(np.transpose(X_np),X_np),2)
        alpha = 1/(2*L)

        update = -1 * alpha * deriv
        theta = theta + update

    if not majority:
        X_g = X[v_arg_hat[int(n*p):]]
        y_g = y[v_arg_hat[int(n*p):]]
        return np.matmul(np.linalg.pinv(X_g),y_g)

    return theta