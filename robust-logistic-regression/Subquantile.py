import numpy as np
from matplotlib import pyplot as plt
from noise import *
from data_loader import *
import cvxpy as cp
import argparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import cvxpy as cp

def SubQ(X, y, T, p, reg, X_test, y_test):
    n = X.shape[0]
    d = X.shape[1]
    partition_number = int(n*p)
    X_np = X[np.random.permutation(n)[:int(n*p)]]
    y_np = y[np.random.permutation(n)[:int(n*p)]]
    #start with the reg-norm regularizer
    theta = cp.Variable(d)
    obj = cp.Minimize(cp.sum(cp.logistic(-cp.multiply(y,X@theta))))
    try:
        prob = cp.Problem(obj).solve(solver=cp.ECOS)
    except Exception as e:
        prob = cp.Problem(obj).solve(solver=cp.ECOS)
    theta = theta.value

    t = 0
    iter_diff = 1
    #while iter_diff > 1e-16:
    for _ in range(12):
        #calculate the lowest error over the quantile with 
        #lowest error and compute the numerical solution at each step
        theta_prev = theta.copy()
        v = np.log(1 + np.exp(-y * (X @ theta)))
        #linear time function to find the quantile with the lowest error
        v_hat = sorted(v)
        #v_arg_hat = np.argpartition(v, partition_number)
        v_arg_hat = np.argsort(v)
        X_np = X[v_arg_hat[:partition_number]]
        y_np = y[v_arg_hat[:partition_number]]
        t = np.max(v[v_arg_hat[:partition_number]])
        P_num = np.count_nonzero(v_arg_hat[:partition_number] < partition_number)
        Q_num = np.count_nonzero(v_arg_hat[:partition_number] > partition_number)
        #print(f"P-num:\t{P_num}\tQ-num:\t{Q_num}")
        #print(f"t:\t{t:.4f}")

        theta = cp.Variable(d)
        obj = cp.Minimize(cp.sum(cp.logistic(-cp.multiply(y_np,X_np@theta))))
        try:
            prob = cp.Problem(obj).solve(solver=cp.ECOS)
        except Exception as e:
            prob = cp.Problem(obj).solve(solver=cp.ECOS)
        theta = theta.value

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
    parser.add_argument('--n', help='samples for synthetic data',type=int,default=200)
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
        X, y, m, b = gaussian(n, d) 
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8)

        X_train, y_train_noisy = noise_fn(X_train, y_train, noise, m, b)

        clean = np.where(y_train == y_train_noisy)[0]
        noisy = np.where(y_train != y_train_noisy)[0]
        clean_pos = np.where(y_train[clean] == 1)[0]
        clean_neg = np.where(y_train[clean] == -1)[0]
        noisy_pos = np.where(y_train[noisy] == 1)[0]
        noisy_neg = np.where(y_train[noisy] == -1)[0]

        plt.scatter(X_train[clean_pos,0],X_train[clean_pos,1], color='g')
        plt.scatter(X_train[clean_neg,0],X_train[clean_neg,1], color='r')
        plt.scatter(X_train[noisy_pos,0],X_train[noisy_pos,1], color='g', marker='x')
        plt.scatter(X_train[noisy_neg,0],X_train[noisy_neg,1], color='r', marker='x')


        theta = SubQ(X_train,np.int32(y_train_noisy),num_iters,p, 0, X_test, y_test)

        x = np.linspace(-10,10,100)
        ys = (theta[1]/theta[0])*x + theta[2]/theta[0]
        plt.plot(x,ys, 'k-', label='Decision boundary')
        plt.legend(['clean positive', 'clean negative', 'noisy positive', 'noisy negative','theta'])
        plt.xlim([-10,10])
        plt.ylim([-10,10])
        # plt.show()
        # plt.clf()
        pred = np.sign((X_test @ theta))
        #pred = 1/(1 + np.exp(-1*np.dot(X_test, theta)))
#        pred[pred == 0] = -1
        accuracy = (len(y_test)-np.count_nonzero(pred - y_test)) / len(y_test)
        print(f"SubQ Acc:\t{accuracy:.3f}\n")
        means.append(accuracy)

        logistic = LogisticRegression(fit_intercept=True)
        logistic.fit(X_train[:,:-1],y_train)
        pred = logistic.predict(X_test[:,:-1])
        accuracy = (len(y_test)-np.count_nonzero(pred - y_test)) / len(y_test)
        print(f"ERM Acc:\t{accuracy:.3f}\n")

    print(f"SubQuantile:\t{np.mean(np.float32(means)):.3f}_{{({np.std(np.float32(means)):.4f})}}")