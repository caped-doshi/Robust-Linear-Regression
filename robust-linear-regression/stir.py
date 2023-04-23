import numpy as np
import argparse

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
from noise import *
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_trials',help='run how many times',type=int,default=5)
    parser.add_argument('--num_iters',help='how many iterations of algorithm',type=int,default=1000)
    parser.add_argument('--noise', help='noise ratio in range (0, 1)',type=float,default=0.1)
    parser.add_argument('--noise_type',help="oblivious, adaptive, or feature",type=str,default='oblivious')
    parser.add_argument('--dataset', help='dataset; drug, cal_housing, abalone, or synthetic',type=str,default='drug')
    parser.add_argument('--n', help='samples for synthetic data',type=int,default='2000')
    parser.add_argument('--d', help='dim for synthetic data',type=int,default='200')

    parsed = vars(parser.parse_args())
    num_trials = parsed['num_trials']
    num_iters = parsed['num_iters']
    noise = parsed['noise']
    noise_type = parsed['noise_type']
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

    means = []
    for _ in range(num_trials):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8)

        X_train, y_train_noisy = noise_fn(X_train, y_train, noise)

        theta = stir(X_train,y_train_noisy,num_iters)
        loss = np.sqrt(np.mean((np.dot(X_test, theta) - y_test) ** 2))
        means.append(loss)
        print(f"Loss:\t{loss:.3f}")
    print(f"CRR:\t{np.mean(np.float32(means)):.3f}_{{({np.std(np.float32(means)):.4f})}}")