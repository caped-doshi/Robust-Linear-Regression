import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge

def SubQ(X, y, T, p):
    n = X.shape[0]
    X_np = X[:int(n*p)]
    y_np = y[:int(n*p)]
    theta = np.matmul(np.linalg.pinv(X),y)
    eps = 1-p
    for i in range(T):
        pred = np.matmul(X,theta)
        v = (pred - y)**2
        v_hat = sorted(v)
        v_arg_hat = np.argsort(v)

        X_np = X[v_arg_hat[:int(n*p)]]
        y_np = y[v_arg_hat[:int(n*p)]]
        d = np.dot(X_np,theta) - y_np
        deriv = np.sum(2 * X_np * d[:, np.newaxis],axis=0)

        L = np.linalg.norm(np.matmul(np.transpose(X_np),X_np),2)
        # A = np.zeros((X_np.shape[1],X_np.shape[1]))
        # for l in range(X_np.shape[0]):
        #     x_i = X_np[l,:]
        #     A += np.outer(x_i,x_i)
        #alpha = 1/(np.linalg.norm(A,2))
        alpha = 1/L

        theta = theta - alpha * deriv
    ridge = Ridge(2, fit_intercept=True, solver='cholesky')
    ridge.fit(X_np[:, :-1], y_np)
    theta = np.append(ridge.coef_, [ridge.intercept_])
    return theta