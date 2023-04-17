import numpy as np

def HT(v, k):
    # only keep k largest elements (k corrupted samples)
    not_top_k_index = (abs(v)).argsort()[:len(v)-k]
    v[not_top_k_index] = 0
    return v

def CRR(X,y):
    k = 500
    eps = 0.1
    b_new = np.ones(len(X))
    b_old = np.zeros(len(X))
    cov_ = np.dot(X.T, X) + np.identity(len(X[0])) * 0.00001
    P_x = np.dot(np.dot(X, np.linalg.inv(cov_)), X.T)
    a = np.dot((np.identity(len(X))-P_x), y)
    iter_ = 0
    while np.linalg.norm(b_new-b_old, ord=2) > eps:
        iter_ += 1
        b_old = b_new
        b_new = HT(np.dot(P_x, b_old) + a, k)
        if iter_ > 3000:  # keep increasing the number of iterations until the point where it doesn't affect things
            break

    #print(b_new)
    theta = np.dot(np.dot(np.linalg.inv(cov_), X.T), (y-b_new))

    #print(theta)
    return theta

from data_loader import *
from RMSE import *
from noise_models.noise import *
if __name__ == "__main__":
    #X,y = data_loader_drug()
    x_ = np.linspace(0.1,0.4,4)
    for eps in x_:
        print(f"epsilon:\t{eps}")
        means = []
        for j in range(30):
            X,y = gaussian(2000,200)
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8)
            y_train_noisy = addUnstructuredNoise(X_train, y_train, eps)
            theta = CRR(X_train,y_train_noisy)
            loss = calc_RMSE(y_test, theta, X_test)
            means.append(loss)
            print(f"Loss:\t{loss:.3f}")
        print(f"CRR:\t{np.mean(np.float32(means)):.3f}_{{({np.std(np.float32(means)):.4f})}}")