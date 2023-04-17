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
            theta = np.dot(np.dot(np.linalg.inv(np.dot(tmp, X) + 1e-10*np.identity(len(X[0]))), tmp),y)
            # theta = theta - (0.2 / (M * len(y))) * np.dot(np.dot(X.T, SS), residual)
        M = M * eta
    return theta

from data_loader import *
from RMSE import *
from noise_models.noise import *
if __name__ == "__main__":
    X,y = data_loader_drug()

    x_ = np.linspace(0.1,0.4,4)
    for eps in x_:
        print(f"epsilon:\t{eps}")
        means = []
        for j in range(10):
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8)
            y_train_noisy = addUnstructuredNoise(X_train, y_train, eps)
            theta = stir(X_train,y_train_noisy,1000)
            loss = calc_RMSE(y_test, theta, X_test)
            means.append(loss)
            print(f"Loss:\t{loss:.3f}")
        print(f"Stir:\t{np.mean(np.float32(means)):.3f}_{{({np.std(np.float32(means)):.4f})}}")