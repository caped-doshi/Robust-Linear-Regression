import numpy as np
import heapq
import cvxpy as cp
import argparse

def f(y_i, y_predict):
    return (y_i - y_predict) ** 2

def objective_function(b, x, y, S, f):
    sum_exp = 0
    for i in S:
        sum_exp += -np.log(f(y[i], np.dot(b, x[i])))
    return sum_exp

def TMLE(X, y, eps, eta, R, f):
    n = len(y)
    temp_heap = []
    for i in range(n):
        heapq.heappush(temp_heap, (y[i], i))
        if len(temp_heap) > n * (1 - eps):
            heapq.heappop(temp_heap)
    S_0 = [x[1] for x in temp_heap]

    beta_prev = np.zeros(X.shape[1])
    for t in range(int(1e12)):
        temp_heap = []
        for i in S_0:
            heapq.heappush(temp_heap, (-np.log(f(y[i], np.dot(X[i], beta_prev))), i))
            if len(temp_heap) > n * (1 - 2 * eps):
                heapq.heappop(temp_heap)
        S_t = [x[1] for x in temp_heap]

        beta = cp.Variable(X.shape[1])
        obj = cp.Minimize(cp.sum_squares(cp.hstack([X[S_t] @ beta - y[S_t], beta])))
        constraints = [cp.norm(beta, 2) <= R]
        prob = cp.Problem(obj, constraints)
        prob.solve(solver = "ECOS")

        if (1 / n) * objective_function(beta.value, X, y, S_t, f) > (1 / n) * objective_function(beta_prev, X, y, S_t, f) - eta:
            return beta.value

        beta_prev = beta.value

from data_loader import *
# from RMSE import *
from noise import *
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', help='dataset; drug, cal_housing, abalone, or synthetic',type=str,default='drug')
    parser.add_argument('--num_trials',help='run how many times',type=int,default=5)
    parser.add_argument('--R_val',help='hyperparameter for SMART',type=float,default=10)
    parser.add_argument('--eta_val',help='hyperparameter for SMART',type=float,default=0.001)
    parser.add_argument('--noise_type',help="oblivious, adaptive, or feature",type=str,default='oblivious')
    parser.add_argument('--noise', help='noise ratio in range (0, 1)',type=float,default=0.1)
    parser.add_argument('--n', help='samples for synthetic data',type=int,default='2000')
    parser.add_argument('--d', help='dim for synthetic data',type=int,default='200')
    
    parsed = vars(parser.parse_args())
    
    dataset = parsed['dataset']
    num_trials = parsed['num_trials']
    eta_value = parsed['eta_val']
    R_value = parsed['R_val']
    noise_type = parsed['noise_type']
    noise = parsed['noise']
    
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
        
    maxLen = max([len(ii) for ii in parsed.keys()])
    fmtString = '\t%' + str(maxLen) + 's : %s'
    print('Arguments:')
    for keyPair in sorted(parsed.items()): print(fmtString % keyPair)

    print(f"epsilon:\t{noise}")
    means = []
    for j in range(num_trials):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8)
        y_train_noisy = noise_fn(X_train, y_train, noise)
        theta = TMLE(X_train,y_train_noisy, noise, eta_value, R_value,f)
        loss = np.sqrt(np.mean((np.dot(X_test, theta) - y_test) ** 2))
        means.append(loss)
        print(f"Loss:\t{loss:.3f}")
    print(f"SMART:\t{np.mean(np.float32(means)):.3f}_{{({np.std(np.float32(means)):.4f})}}")