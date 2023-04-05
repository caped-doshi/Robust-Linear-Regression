import numpy as np
import numpy.linalg as LA
import sys
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
from matplotlib import pyplot as plt
import csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import HuberRegressor,TheilSenRegressor,LinearRegression, RANSACRegressor
import tikzplotlib
from scipy.stats.distributions import chi2

def num_p_q(argsorted_arr,n,p,eps):
    m = int(n*(1-eps))
    pc = 0
    qc = 0
    for i in range(0,int(n*p)):
        if argsorted_arr[0,i] < m:
            pc += 1
        else:
            qc += 1
    return (pc,qc)

def addNoise(X, y, m, b, noise: float): 
    noisyY = np.copy(y)
    d = 50
    m_ = np.random.normal(4,4,d)
    b_ = np.random.normal(4,4,1)
    for i in range(int(y.shape[1] * (1-noise)), y.shape[1]):
        noisyY[0,i] = np.random.normal(np.dot(np.transpose(m_),X[0:d,i]) - b_,0.1)
    #print(f"a:\t{a}b:\t{b}")
    return noisyY

if __name__ == "__main__":

    meta_means = []
    meta_std = []
    x_ = np.linspace(0.1,0.40,15)
    for eps in x_:
        print(f"Initial Epsilon:\t{eps}")
        means = []
        for _ in range(10):
            
            d = 50
            pq = 1-eps
            n = 2000
            x = np.random.normal(1,1,n)
            mu = np.ones((d+1,1))
            sigma = np.eye(d+1)
            sigma[d,d] = 0

            m = np.random.normal(4,4,d)
            b = np.random.normal(4,4)
            beta_p = np.zeros((d+1,))
            beta_p[0:d] += m
            beta_p[d] = b
            beta_p = beta_p[:,np.newaxis]
            #print(f"beta_p:\t{beta_p.transpose()}")
            
            X = np.zeros((d+1,n))
            y = np.zeros((1,n))

            for i in range(n):
                x = np.random.normal(1,1,d)
                X[0:d,i] = x
                X[d,i] = 1

            for i in range(n):
                t = np.random.normal(np.dot(np.transpose(m),X[0:d,i]) + b,0.1)
                y[0,i] = t
            eps_p = 0.1
            eps_q = 0.1
            
            #print(f"X: {X.shape}")
            #print(f"y: {y.shape}")
            X_train, X_test, y_train, y_test = train_test_split(np.transpose(X),np.transpose(y), train_size = .80)
            X_train = np.transpose(X_train)
            X_test = np.transpose(X_test)
            y_train = np.transpose(y_train)
            y_test = np.transpose(y_test)

            beta_q = np.zeros((d+1,))
            #beta_q[0:d] += m
            #beta_q[d] += b + 2
            #beta_q = beta_q[:,np.newaxis]
            #print(f"beta_q:\t{beta_q.transpose()}")
            y_train_noisy = addNoise(X_train, y_train,m, b, eps)

            theta = np.matmul(np.linalg.pinv(np.transpose(X_train)),np.transpose(y_train_noisy))
            #theta = np.random.normal(0,4,d+1)
            x_test = np.linspace(-3,3,300)
            n = X_train.shape[1]
            X_prev_np_hat = np.zeros((int(n*pq)))
            pred = np.matmul(np.transpose(theta),X_train)
            v = (pred - y_train_noisy) * (pred-y_train_noisy)
            v_abs = np.abs(pred-y_train_noisy)
            v_n_abs = pred-y_train_noisy
            v_arg_hat = np.argsort(v)
            for k in range(0, 500):
                pred = np.matmul(np.transpose(theta),X_train)
                v = (pred - y_train_noisy) * (pred-y_train_noisy)
                v_abs = np.abs(pred-y_train_noisy)
                v_n_abs = pred-y_train_noisy
                v_arg_hat = np.argsort(v)
                # if k % 2 == 0:
                #     plt.hist(v_n_abs[0,:int(n*(1-eps))],bins='auto')
                #     plt.hist(v_n_abs[0,int(n*(1-eps)):],bins='auto')
                #     plt.xlabel("Residual")
                #     plt.ylabel("Frequency")
                #     tikzplotlib.save(f"test{k}.tikz")
                #     plt.clf()
                X_np_hat = v_arg_hat[0,:int(n*pq)]
                #print(f"Common Points:\t{len(np.intersect1d(X_np_hat,X_prev_np_hat))}")
                X_prev_np_hat = X_np_hat.copy()
                X_np = X_train[:,v_arg_hat[0,:int(n*pq)]]
                y_np = y_train_noisy[:,v_arg_hat[0,:int(n*pq)]]
                df = np.matmul(np.transpose(theta), X_np) - y_np
                deriv = np.zeros((d+1,1))
                df_temp = df[0]
                df_temp = np.transpose(df_temp)
                deriv = 2 * X_np.transpose() * df_temp[:,np.newaxis]
                deriv = np.sum(deriv, axis=0)
                deriv /= (n*pq)
            
                L = np.matmul(np.transpose(X_np),X_np)
                v = np.diag(L)
                L = np.sum(v)
                L = L /(n*pq)

                alpha = 1/(2*L)
                update = -1 * alpha * deriv.transpose()
                theta = theta + update[:,np.newaxis]
            
            n_p,n_q = num_p_q(v_arg_hat,n,1-eps,eps)
            #print(f"p:\t{n_p}\tq:\t{n_q}")
            #print(f"Epsilon:\t{n_q/(n_p + n_q)}")
            means.append(n_q/(n_p+n_q))
        mean = np.mean(means)
        std = np.std(means)
        print(f"mean:\t{mean}")
        print(f"std:\t{std}")
        meta_means.append(mean)
        meta_std.append(std)
    meta_means = np.float32(meta_means)
    meta_std = np.float32(meta_std)
    plt.plot(x_,meta_means, color='black')
    plt.fill_between(x_,meta_means-meta_std,meta_means+meta_std,color='black',alpha=0.5)
    plt.ylim([0, 1])
    tikzplotlib.save("final-epsilon-random-0-symmetric.tex")
    plt.show()
    pass