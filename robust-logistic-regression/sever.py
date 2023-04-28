import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import argparse

def SEVER(x_train, y_train, reg=4, p=0.01, iter=64):
    for _ in range(iter):

        logistic = LogisticRegression(fit_intercept=True)
        logistic.fit(x_train[:,:-1],y_train)
        w = np.append(logistic.coef_,[logistic.intercept_])

        #extract gradients for each point
        # losses = y_train - x_train @ w
        # grads = 2 * np.diag(losses) @ x_train + (reg * w)[None, :]
        pred = x_train @ w
        s = 1 / (1 + np.exp(-1*pred))
        losses = -y_train * np.log(s) - (1 - y_train) * np.log(s)
        grads = np.diag(s - y_train) @ x_train

        #center gradients
        grad_avg = np.mean(grads, axis=0)
        centered_grad = grads-grad_avg[None, :]

        #svd to compute top vector v
        u, s, vh = np.linalg.svd(centered_grad, full_matrices=False)
        v = vh[:, 0] #top right singular vector

        #compute outlier score
        tau = np.sum(centered_grad*v[None, :], axis=1)**2

        #in each iteration simply remove the top p fraction of outliers
        #according to the scores τi
        n_removed = int(p*len(tau))
        if n_removed == len(x_train):
            print("Warning: cannot remove p={} of values, too many iterations or p too large.".format(p))
            continue
        # idx_kept = np.argpartition(tau, -n_removed)[:-n_removed]
        idx_kept = np.argsort(tau)[:-n_removed]
        idx_kept = np.sort(idx_kept)
        x_train = x_train[idx_kept]
        y_train = y_train[idx_kept]

        # test_data = scaler.transform(x_test)
        # rmse = np.sqrt(np.mean((x_train@w - y_train)**2))
        # print(rmse)
    return w


from data_loader import *
from noise import *
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_trials',help='run how many times',type=int,default=5)
    parser.add_argument('--num_iters',help='iterations of algorithm',type=int,default=4)
    parser.add_argument('--reg',help='regularizer', type=float,default=2)
    parser.add_argument('--p',help='fraction of outlier to remove (0:1)]', type=float,default=0.3)
    parser.add_argument('--noise', help='noise ratio in range (0, 1)',type=float,default=0.1)
    parser.add_argument('--noise_type',help="oblivious, adaptive, or feature",type=str,default='adaptive')
    parser.add_argument('--dataset', help='dataset; drug, cal_housing, abalone, or synthetic',type=str,default='synthetic')
    parser.add_argument('--n', help='samples for synthetic data',type=int,default='2000')
    parser.add_argument('--d', help='dim for synthetic data',type=int,default='200')

    parsed = vars(parser.parse_args())
    num_trials = parsed['num_trials']
    num_iters = parsed['num_iters']
    reg = parsed['reg']
    p = parsed['p']
    noise = parsed['noise']
    noise_type = parsed['noise_type']
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

    print(f"Epsilon:\t{noise}")
    means = []
    for _ in range(num_trials):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8)

        X_train, y_train_noisy = noise_fn(X_train, y_train, noise, m, b)

        theta = SEVER(X_train,y_train_noisy,4,p,num_iters)
        pred = np.dot(X_test, theta)
        s = 1 / (1 + np.exp(-1*pred))
        v = -y_test*np.log(s) - (1-y_test)*np.log(s)
        loss = np.sqrt(np.mean(v) ** 2)
        means.append(loss)
        print(f"Loss:\t{loss:.3f}")
    print(f"SEVER:\t{np.mean(np.float32(means)):.3f}_{{({np.std(np.float32(means)):.4f})}}")