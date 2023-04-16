import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge
from noise_models.noise import *
from data_loader import *
from RMSE import *
import cvxpy as cp

def SubQ(X, y, T, p, j):
    n = X.shape[0]
    X_np = X[:int(n*p)]
    y_np = y[:int(n*p)]
    ridge = Ridge(2, fit_intercept=True, solver='cholesky')
    ridge.fit(X[:, :-1], y)
    theta = np.append(ridge.coef_, [ridge.intercept_])
    #theta = np.matmul(np.linalg.pinv(X),y)
    L = np.linalg.norm(np.matmul(np.transpose(X_np),X_np),2)
    t = 0
    for i in range(T):
        pred = np.matmul(X,theta)
        v = (pred - y)**2
        partition_number = int(n*p)
        #if partition_number == 0 or partition_number == X.shape[0]:
           #partition_number = 1
        if i % j == 0:
            v_arg_hat = np.argpartition(v, partition_number)
            X_np = X[v_arg_hat[:int(n*p)]]
            y_np = y[v_arg_hat[:int(n*p)]]
            L = np.linalg.norm(np.matmul(np.transpose(X_np),X_np),2)
            t = np.max(v[v_arg_hat[:int(n*p)]])
            
        d = np.dot(X_np,theta) - y_np
        deriv = np.sum(2 * X_np * d[:, np.newaxis],axis=0) + 2 * theta        
        alpha = 1/L

        theta = theta - alpha * deriv
    
    #theta = np.matmul(np.linalg.pinv(X_np),y_np)
    ridge = Ridge(2, fit_intercept=True, solver='cholesky')
    ridge.fit(X_np[:, :-1], y_np)
    theta = np.append(ridge.coef_, [ridge.intercept_])
    return theta

def SubQ2(X, y, T, p, j):
    n = X.shape[0]
    X_np = X[:int(n*p)]
    y_np = y[:int(n*p)]
    ridge = Ridge(2, fit_intercept=True, solver='cholesky')
    ridge.fit(X[:, :-1], y)
    theta = np.append(ridge.coef_, [ridge.intercept_])
    #theta = np.matmul(np.linalg.pinv(X),y)
    L = np.linalg.norm(np.matmul(np.transpose(X_np),X_np),2)
    t = 0
    for i in range(T):
        pred = np.matmul(X,theta)
        v = (pred - y)**2
        partition_number = int(n*p)
        #if partition_number == 0 or partition_number == X.shape[0]:
           #partition_number = 1
        if i % j == 0:
            v_arg_hat = np.argpartition(v, partition_number)
            X_np = X[v_arg_hat[:int(n*p)]]
            y_np = y[v_arg_hat[:int(n*p)]]
            L = np.linalg.norm(np.matmul(np.transpose(X_np),X_np),2)
            t = np.max(v[v_arg_hat[:int(n*p)]])
            P_num = np.count_nonzero(v_arg_hat[:int(n*p)] < int(n*p))
            Q_num = np.count_nonzero(v_arg_hat[:int(n*p)] > int(n*p))
            #print(f"P:\t{P_num}\tQ:\t{Q_num}")
            
        ridge = Ridge(2, fit_intercept=True, solver='cholesky')
        ridge.fit(X_np[:, :-1], y_np)
        theta = np.append(ridge.coef_, [ridge.intercept_])
    
    #theta = np.matmul(np.linalg.pinv(X_np),y_np)
    ridge = Ridge(2, fit_intercept=True, solver='cholesky')
    ridge.fit(X_np[:, :-1], y_np)
    theta = np.append(ridge.coef_, [ridge.intercept_])
    return theta

if __name__ == "__main__":
    X,y = data_loader_drug()
    
    x_ = np.linspace(0.1,0.4,4)
    for eps in x_:
        print(f"Epsilon:\t{eps}")
        means = []
        for j in range(10):
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8)
            y_train_noisy = addUnstructuredNoise(X_train, y_train, eps)
            theta = SubQ2(X_train,y_train_noisy,16,1-eps,1)
            loss = calc_RMSE(y_test, theta, X_test)
            means.append(loss)
            print(f"Loss:\t{loss:.3f}")
        print(f"SubQuantile:\t{np.mean(np.float32(means)):.3f}_{{({np.std(np.float32(means)):.4f})}}")
