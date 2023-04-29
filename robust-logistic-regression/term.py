import numpy as np
import cvxpy as cp
from sklearn.model_selection import train_test_split
import argparse

#code from https://github.com/litian96/TERM/blob/master/robust_regression/regression.py
def compute_gradients_tilting(theta, X, y, t, reg):  # our objective
    #loss = (np.dot(X, theta) - y) ** 2
    z = X @ theta
    h = 1 + np.exp(-y*z)
    loss = np.log(h)
    if t > 0:
        max_l = max(loss)
        loss = loss - max_l

    grad = np.dot(X.T, np.divide(-y, h)) / y.size
    ZZ = np.mean(np.exp(loss))
    return grad / (ZZ)

#cvxpy is mostly not supported as the expression is not convex. 
def TERM_cp(X, y, t, reg):
  theta = cp.Variable(len(X[0]))
  objective = cp.Minimize(cp.sum(cp.exp(t * ((X @ theta - y) ** 2 + reg * theta))))
  prob = cp.Problem(objective)
  result = prob.solve(verbose=True, max_iters=500)
  print(result)
  return theta.value

#this function uses the standard gradient descent to approximate the minimum
def TERM(train_X, train_y, t, alpha, num_iters,reg):
    theta = np.zeros(len(train_X[0]))
    for j in range(num_iters):
        grads_theta = compute_gradients_tilting(theta, train_X, train_y, t,reg)
        #check if objective is reached 
        if np.linalg.norm(grads_theta, ord=2) < 1e-10:
            break
        theta = theta - alpha * grads_theta
        # if j % 1000 == 0:
            # train_error = np.sqrt(np.mean((np.dot(train_X, theta) - train_y) ** 2))
            #print("training error: ", train_error)
    return theta


from data_loader import *
from noise import *
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', help='dataset; drug, cal_housing, abalone, or synthetic',type=str,default='synthetic')
    parser.add_argument('--num_trials',help='run how many times',type=int,default=5)
    parser.add_argument('--num_iters',help='how many iterations of algorithm',type=int,default=10000)
    parser.add_argument('--noise_type',help="oblivious, adaptive, or feature",type=str,default='adaptive')
    parser.add_argument('--learning_rate',help='learning rate for tilted optimization',type=float,default=0.1)
    parser.add_argument('--t', help='hyperparameter for TERM',type=float,default=-2.0)
    parser.add_argument('--reg',help="regularization parameter for ridge regression",type=float,default=0)
    parser.add_argument('--noise', help='noise ratio in range (0, 1)',type=float,default=0.1)
    parser.add_argument('--n', help='samples for synthetic data',type=int,default='2000')
    parser.add_argument('--d', help='dim for synthetic data',type=int,default='200')
    
    parsed = vars(parser.parse_args())
    
    dataset = parsed['dataset']
    num_trials = parsed['num_trials']
    num_iters = parsed['num_iters']
    noise_type = parsed['noise_type']
    alpha = parsed['learning_rate']
    t = parsed['t']
    noise = parsed['noise']
    reg = parsed['reg']
    
    m = None
    b = None
    if dataset == 'cal_housing':
        X, y = data_loader_cal_housing()
    elif dataset == 'abalone':
        X, y = data_loader_abalone()
    elif dataset == 'drug':
        X, y = data_loader_drug()     
    elif dataset == 'synthetic':
        n = parsed['n']
        d = parsed['d']
        X, y, m, b = gaussian(n, d) 
        
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

    #try the TERM method num_trials time and report the average
    print(f"epsilon:\t{noise}")
    means = []
    for j in range(num_trials):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8)
        X_train, y_train_noisy = noise_fn(X_train, y_train, noise, m, b)

        theta = TERM(X_train,y_train_noisy, t, alpha, num_iters, reg)

        pred = np.sign((X_test @ theta))
        accuracy = (len(y_test)-np.count_nonzero(pred - y_test)) / len(y_test)

        print(f"Accuracy:\t{accuracy:.3f}")
        means.append(accuracy)
    print(f"TERM:\t{np.mean(np.float32(means)):.3f}_{{({np.std(np.float32(means)):.4f})}}")