import numpy as np
from matplotlib import pyplot as plt
from noise import *
from data_loader import *
import cvxpy as cp
import argparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import cvxpy as cp
import tikzplotlib

def SubQ(X, y, T, p, reg):
    n = X.shape[0]
    d = X.shape[1]
    partition_number = int(n*p)

    svc = svm.LinearSVC(loss="hinge",fit_intercept=True)
    svc.fit(X[:,:-1],y)
    theta = np.append(svc.coef_,[svc.intercept_])

    t = 0
    iter_diff = 1
    iter = 0
    while iter_diff > 1e-3:
    #for _ in range(32):
        theta_prev = theta.copy()
        v = np.maximum(0, 1 - y*(X @ theta))
        v_arg_hat = np.argsort(v)
        #v_arg_hat = np.argsort(v)#[:partition_number])
        X_np = X[v_arg_hat[:partition_number]]
        y_np = y[v_arg_hat[:partition_number]]
        t = np.max(v[v_arg_hat[:partition_number]])
        P_num = np.count_nonzero(v_arg_hat[:partition_number] < partition_number)
        Q_num = np.count_nonzero(v_arg_hat[:partition_number] > partition_number)
        print(f"P-num:\t{P_num}\tQ-num:\t{Q_num}")
        #print(f"t:\t{t:.4f}")

        svc = svm.LinearSVC(loss="hinge",tol=1e-5,fit_intercept=True)
        svc.fit(X_np[:,:-1],y_np)
        theta = np.append(svc.coef_,[svc.intercept_])

        iter_diff = np.linalg.norm(theta-theta_prev,2)
        #print(f"iter_diff:\t{iter_diff}")
        if iter > T:
            break
        iter += 1

    return theta

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_trials',help='run how many times',type=int,default=5)
    parser.add_argument('--num_iters',help='how many iterations of algorithm',type=int,default=16)
    parser.add_argument('--noise', help='noise ratio in range (0, 1)',type=float,default=0.2)
    parser.add_argument('--noise_type',help="oblivious, adaptive, or feature",type=str,default='oblivious')
    parser.add_argument('--dataset', help='dataset; salary, enron, or synthetic',type=str,default='enron')
    parser.add_argument('--quantile',help='what quantile level to minimize over', type=float,default=0.8)
    parser.add_argument('--n', help='samples for synthetic data',type=int,default=1000)
    parser.add_argument('--d', help='dim for synthetic data',type=int,default=2)

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
   
    m = None
    b = None
    if dataset == 'synthetic':
        n = parsed['n']
        d = parsed['d']
        X, y, m, b = gaussian(n, d) 
    elif dataset == 'enron':
        X, y = enron()
    elif dataset == 'salary':
        X, y = salary()

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

        theta = SubQ(X_train,np.int32(y_train_noisy),num_iters,p, 0)

        pred = np.sign((X_test @ theta))
        accuracy = (len(y_test)-np.count_nonzero(pred - y_test)) / len(y_test)
        print(f"SubQ Acc:\t{accuracy:.3f}")
        means.append(accuracy)

    print(f"SubQuantile:\t{np.mean(np.float32(means)):.3f}_{{({np.std(np.float32(means)):.4f})}}")