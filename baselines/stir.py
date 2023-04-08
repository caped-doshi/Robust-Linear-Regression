import numpy as np

def stir(X,y,iters):
    theta = np.zeros(len(X[0]))
    theta_old = np.ones(len(X[0])) * 100
    M = 1
    eta = 1
    for it in range(iters):
        while np.linalg.norm(theta_old-theta, ord=2) > 2 / (eta * M):
            theta_old = theta
            residual = np.dot(X, theta) - y
            s = np.minimum(1.0/abs(residual), np.ones(len(y)) * M)
            SS = np.diag(s)
            tmp = np.dot(X.T, SS)
            theta = np.dot(np.dot(np.linalg.inv(np.dot(tmp, X) + 1e-20*np.identity(len(X[0]))), tmp),y)
            # theta = theta - (0.2 / (M * len(y))) * np.dot(np.dot(X.T, SS), residual)
        M = M * eta
    return theta