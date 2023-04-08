import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge

def SubQ(X, y, T, p):
    n = X.shape[0]
    X_np = X[:int(n*p)]
    y_np = y[:int(n*p)]
    ridge = Ridge(2, fit_intercept=True, solver='cholesky')
    ridge.fit(X[:, :-1], y)
    theta = np.append(ridge.coef_, [ridge.intercept_])
    eps = 1-p
    for i in range(T):
        pred = np.matmul(X,theta)
        v = (pred - y)**2
        partition_number = int(n*p)
        if partition_number == 0 or partition_number == X.shape[0]:
            partition_number = 1
        v_arg_hat = np.argpartition(v, partition_number)
        X_np = X[v_arg_hat[:int(n*p)]]
        y_np = y[v_arg_hat[:int(n*p)]]
        d = np.dot(X_np,theta) - y_np
        deriv = np.sum(2 * X_np * d[:, np.newaxis],axis=0) + 2 * theta

        L = np.linalg.norm(np.matmul(np.transpose(X_np),X_np),2)
        alpha = 1/L

        theta = theta - alpha * deriv
    ridge = Ridge(2, fit_intercept=True, solver='cholesky')
    ridge.fit(X_np[:, :-1], y_np)
    theta = np.append(ridge.coef_, [ridge.intercept_])
    return theta