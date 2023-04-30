import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import argparse
import cvxpy as cp
from sklearn import svm

def ERM(X, y):
    svc = svm.LinearSVC(loss="hinge",fit_intercept=True,max_iter=10000)
    svc.fit(X[:,:-1],y)
    theta = np.append(svc.coef_,[svc.intercept_]) 
    return theta

from data_loader import *
from noise import *
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_trials',help='run how many times',type=int,default=5)
    parser.add_argument('--num_iters',help='iterations of algorithm',type=int,default=4)
    parser.add_argument('--reg',help='regularizer', type=float,default=2)
    parser.add_argument('--noise', help='noise ratio in range (0, 1)',type=float,default=0.1)
    parser.add_argument('--noise_type',help="oblivious, adaptive, or feature",type=str,default='oblivious')
    parser.add_argument('--dataset', help='dataset; synthetic, enron, or salary',type=str,default='salary')
    parser.add_argument('--n', help='samples for synthetic data',type=int,default='2000')
    parser.add_argument('--d', help='dim for synthetic data',type=int,default='200')

    parsed = vars(parser.parse_args())
    num_trials = parsed['num_trials']
    num_iters = parsed['num_iters']
    reg = parsed['reg']
    noise = parsed['noise']
    noise_type = parsed['noise_type']
    dataset = parsed['dataset']
    n = parsed['n']
    d = parsed['d']

    maxLen = max([len(ii) for ii in parsed.keys()])
    fmtString = '\t%' + str(maxLen) + 's : %s'
    print('Arguments:')
    for keyPair in sorted(parsed.items()): print(fmtString % keyPair)

    m = None
    b = None
    if dataset == 'synthetic':
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

    print(f"Epsilon:\t{noise}")
    means = []
    for _ in range(num_trials):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8)

        X_train, y_train_noisy = noise_fn(X_train, y_train, noise, m, b)
        n = X_train.shape[0]
        partition_number = int(n*(1 - noise))

        theta = ERM(X_train,y_train)
        pred = np.sign((X_test @ theta))
        accuracy = (len(y_test)-np.count_nonzero(pred - y_test)) / len(y_test)
        means.append(accuracy)
        print(f"accuracy:\t{accuracy:.3f}")
    print(f"ERM:\t{np.mean(np.float32(means)):.3f}_{{({np.std(np.float32(means)):.4f})}}")