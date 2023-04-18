import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge
from noise import *
from data_loader import *
import cvxpy as cp
import argparse

def SubQ(X, y, T, p, j):
    n = X.shape[0]
    X_np = X[:int(n*p)]
    y_np = y[:int(n*p)]
    ridge = Ridge(2, fit_intercept=True, solver='cholesky')
    ridge.fit(X[:, :-1], y)
    theta = np.append(ridge.coef_, [ridge.intercept_])
    L = np.linalg.norm(np.matmul(np.transpose(X_np),X_np),2)
    t = 0
    for i in range(T):
        pred = np.matmul(X,theta)
        v = (pred - y)**2
        partition_number = int(n*p)
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
    
    ridge = Ridge(2, fit_intercept=True, solver='cholesky')
    ridge.fit(X_np[:, :-1], y_np)
    theta = np.append(ridge.coef_, [ridge.intercept_])
    return theta

def SubQ2(X, y, T, p):
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
        v_arg_hat = np.argpartition(v, partition_number)
        X_np = X[v_arg_hat[:int(n*p)]]
        y_np = y[v_arg_hat[:int(n*p)]]
        L = np.linalg.norm(np.matmul(np.transpose(X_np),X_np),2)
        t = np.max(v[v_arg_hat[:int(n*p)]])
        P_num = np.count_nonzero(v_arg_hat[:int(n*p)] < int(n*p))
        Q_num = np.count_nonzero(v_arg_hat[:int(n*p)] > int(n*p))

        ridge = Ridge(2, fit_intercept=True, solver='cholesky')
        ridge.fit(X_np[:, :-1], y_np)
        theta = np.append(ridge.coef_, [ridge.intercept_])
    
    ridge = Ridge(2, fit_intercept=True, solver='cholesky')
    ridge.fit(X_np[:, :-1], y_np)
    theta = np.append(ridge.coef_, [ridge.intercept_])
    return theta

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_trials',help='run how many times',type=int,default=5)
    parser.add_argument('--num_iters',help='how many iterations of algorithm',type=int,default=32)
    parser.add_argument('--noise', help='noise ratio in range (0, 1)',type=float,default=0.1)
    parser.add_argument('--noise_type',help="oblivious, adaptive, or feature",type=str,default='oblivious')
    parser.add_argument('--quantile',help='what quantile level to minimize over', type=float,default=0.9)
    parser.add_argument('--dataset', help='dataset; drug, cal_housing, abalone, or synthetic',type=str,default='drug')
    parser.add_argument('--n', help='samples for synthetic data',type=int,default='2000')
    parser.add_argument('--d', help='dim for synthetic data',type=int,default='200')

    parsed = vars(parser.parse_args())
    num_trials = parsed['num_trials']
    num_iters = parsed['num_iters']
    noise = parsed['noise']
    noise_type = parsed['noise_type']
    p = parsed['quantile']
    dataset = parsed['dataset']

    maxLen = max([len(ii) for ii in parsed.keys()])
    fmtString = '\t%' + str(maxLen) + 's : %s'
    print('Arguments:')
    for keyPair in sorted(parsed.items()): print(fmtString % keyPair)

    if dataset == 'cal_housing':
        X, y = data_loader_cal_housing()
    elif dataset == 'abalone':
        X, y = data_loader_abalone()
    elif dataset == 'drug':
        X, y = data_loader_drug()     
    elif dataset == 'synthetic':
        n = parsed['n']
        d = parsed['d']
        X, y = gaussian(n, d) 

    if noise_type == "oblivious":
        noise_fn = addObliviousNoise
    if noise_type == "adaptive":
        noise_fn = addAdaptiveNoise
    if noise_type == "feature":
        noise_fn = addFeatureNoise

    print(f"Epsilon:\t{noise}")
    means = []
    for _ in range(num_trials):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8)

        X_train, y_train_noisy = noise_fn(X, y, noise)

        theta = SubQ2(X_train,y_train_noisy,num_iters,p)
        loss = np.sqrt(np.mean((np.dot(X_test, theta) - y_test) ** 2))
        means.append(loss)
        print(f"Loss:\t{loss:.3f}")
    print(f"SubQuantile:\t{np.mean(np.float32(means)):.3f}_{{({np.std(np.float32(means)):.4f})}}")
