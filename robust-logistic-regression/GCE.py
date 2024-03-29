import numpy as np
from matplotlib import pyplot as plt
from noise import *
from data_loader import *
import cvxpy as cp
import argparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import cvxpy as cp
import tikzplotlib

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

        theta = 

        pred = np.sign((X_test @ theta))
        accuracy = (len(y_test)-np.count_nonzero(pred - y_test)) / len(y_test)
        print(f"SubQ Acc:\t{accuracy:.3f}")
        means.append(accuracy)

    print(f"GCE:\t{np.mean(np.float32(means)):.3f}_{{({np.std(np.float32(means)):.4f})}}")