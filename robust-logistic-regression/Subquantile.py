import numpy as np
from matplotlib import pyplot as plt
from noise import *
from data_loader import *
import cvxpy as cp
import argparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def SubQ(X, y, T, p, reg):
    n = X.shape[0]
    partition_number = int(n*p)
    X_np = X[np.random.permutation(n)[:int(n*p)]]
    y_np = y[np.random.permutation(n)[:int(n*p)]]
    #start with the reg-norm regularizer
    logistic = LogisticRegression(fit_intercept=True)
    logistic.fit(X[:,:-1],y)
    theta = np.append(logistic.coef_,[logistic.intercept_])
    t = 0
    iter_diff = 1
    #while iter_diff > 1e-16:
    for _ in range(8):
        #calculate the lowest error over the quantile with 
        #lowest error and compute the numerical solution at each step
        theta_prev = theta.copy()
        z = X @ theta
        h = 1 / (1 + np.exp(-z))
        v = np.log(1+np.exp(-z)) + np.multiply(1-y, z)
        #linear time function to find the quantile with the lowest error
        v_hat = sorted(v)
        #v_arg_hat = np.argpartition(v, partition_number)
        v_arg_hat = np.argsort(v)
        X_np = X[v_arg_hat[:int(n*p)]]
        y_np = y[v_arg_hat[:int(n*p)]]
        t = np.max(v[v_arg_hat[:int(n*p)]])
        P_num = np.count_nonzero(v_arg_hat[:int(n*p)] < int(n*p))
        Q_num = np.count_nonzero(v_arg_hat[:int(n*p)] > int(n*p))
        #print(f"P-num:\t{P_num}\tQ-num:\t{Q_num}")
        print(f"t:\t{t:.4f}")

        logistic = LogisticRegression(fit_intercept=True)
        logistic.fit(X_np[:,:-1],y_np)
        theta = np.append(logistic.coef_,[logistic.intercept_])

        iter_diff = np.linalg.norm(theta-theta_prev,2)
        #print(f"iter_diff:\t{iter_diff}")
    # logistic = LogisticRegression(fit_intercept=True)
    # logistic.fit(X_np[:,:-1],y_np)
    # theta = np.append(logistic.coef_,[logistic.intercept_])
    return theta

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_trials',help='run how many times',type=int,default=5)
    parser.add_argument('--num_iters',help='how many iterations of algorithm',type=int,default=64)
    parser.add_argument('--noise', help='noise ratio in range (0, 1)',type=float,default=0.1)
    parser.add_argument('--noise_type',help="oblivious, adaptive, or feature",type=str,default='adaptive')
    parser.add_argument('--dataset', help='dataset; drug, cal_housing, abalone, or synthetic',type=str,default='synthetic')
    parser.add_argument('--quantile',help='what quantile level to minimize over', type=float,default=0.9)
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
   
    if dataset == 'synthetic':
        n = parsed['n']
        d = parsed['d']
        X, y, m, b = gaussian(n, d) 

    if noise_type == "oblivious":
        noise_fn = addObliviousNoise
    if noise_type == "adaptive":
        noise_fn = addAdaptiveNoise
    if noise_type == "feature":
        noise_fn = addFeatureNoise
    #test the algorithm on the selected data num_trials times and report the result
    print(f"Epsilon:\t{noise}")
    means = []
    for _ in range(num_trials):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8)

        X_train, y_train_noisy = noise_fn(X_train, y_train, noise, m, b)

        # logistic = LogisticRegression(fit_intercept=True,solver='newton-cg')
        # logistic.fit(X_train[:,:-1],y_train_noisy)
        # erm = np.append(logistic.coef_,[logistic.intercept_])
        # pred = 1/(1 + np.exp(-1*np.dot(X_test, erm)))
        # pred[pred > 0.5] = 1
        # pred[pred <= 0.5] = -1
        # accuracy = 100*(len(y_test)-np.count_nonzero(pred - y_test)) / len(y_test)
        # print(f"ERM Acc:\t{accuracy:.2f}")

        theta = SubQ(X_train,np.int32(y_train_noisy),num_iters,p, 0)
        pred = 1/(1 + np.exp(-1*np.dot(X_test, theta)))
        pred = pred.round()
        accuracy = 100*(len(y_test)-np.count_nonzero(pred - y_test)) / len(y_test)
        print(f"SubQ Acc:\t{accuracy:.2f}\n")
        means.append(accuracy)
    print(f"SubQuantile:\t{np.mean(np.float32(means)):.3f}_{{({np.std(np.float32(means)):.4f})}}")