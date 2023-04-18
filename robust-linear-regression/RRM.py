import numpy as np
import cvxpy as cp
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
import argparse
from noise import *
from data_loader import *


def RRM(X, y, epsilon, tol=2e-2, max_iters=50, dist_tol=1e-3, dist_max_iters=10000):
   
    ####RRM##########
    #INPUT
    #x_aug : n x d matrix
    #y_aug : n x 1 vector
    #epsilon : scalar
    ###############

    #Output###
    #theta_new : d x 1 vector of robust estimate
    
    ##INITIALIZATION##
    n, d = X.shape
    p = np.ones(n)/n
    
    #initial theta as the solution of WLS at p_old
    theta_old = np.linalg.pinv(X.T @ np.diag(p) @ X) @ (X.T @ np.diag(p) @ y)

    for i in range(max_iters):
        #LOSS
        alpha = (y-X@theta_old)**2

        p = optimize_RRM(alpha.flatten(),epsilon,n, p, tol=dist_tol, max_iters=dist_max_iters)
        # p = optimize_scipy(alpha.flatten(),epsilon,n, p).reshape(-1, 1)
            
        #WLS SOLUTION
        theta_new  = np.linalg.pinv(X.T @ np.diag(p.flatten()) @ X) @ (X.T @ np.diag(p.flatten()) @ y)

        if np.linalg.norm(theta_old-theta_new)/np.linalg.norm(theta_old)>tol:
          # print(np.linalg.norm(theta_old-theta_new)/np.linalg.norm(theta_old))
          theta_old = theta_new
        else:
            break
    return theta_new



def optimize_RRM(alpha, epsilon, n, p_val=None, tol=1e-3, max_iters=10000):
   # Define the optimization variables
  p = cp.Variable(n)
  if p_val is None:
    p.value = np.ones(n)/n
  else:
    p.value = p_val

  # Define the objective function to be minimized
  objective = cp.Minimize(alpha.T @ p)

  # Define the constraints
  constraints = [cp.sum(cp.entr(p)) >= cp.log((1-epsilon)*n), # entropy constraint
                  cp.sum(p) == 1.0, # probability constraint
                  p >= 0] # non-negativity constraint

  # Define the problem and solve it
  prob = cp.Problem(objective, constraints)
  prob.solve(solver=cp.SCS, eps=tol, max_iters=max_iters)# feastol=1e-3, abstol=1e-3, reltol=1e-3, max_iters=100)
  return p.value

from data_loader import *
from noise import *
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_trials',help='run how many times',type=int,default=5)
    parser.add_argument('--max_iters',help='max iterations of algorithm',type=int,default=50)
    parser.add_argument('--tol',help='tolerence for algorithm convergence', type=float,default=2e-2)
    parser.add_argument('--dist_max_iters',help='max iterations of distribution optimization',type=int,default=10000)
    parser.add_argument('--dist_tol',help='tolerence for distribution optimization convergence', type=float,default=1e-3)
    parser.add_argument('--noise', help='noise ratio in range (0, 1)',type=float,default=0.1)
    parser.add_argument('--noise_type',help="oblivious, adaptive, or feature",type=str,default='oblivious')
    parser.add_argument('--dataset', help='dataset; drug, cal_housing, abalone, or synthetic',type=str,default='drug')
    parser.add_argument('--n', help='samples for synthetic data',type=int,default='2000')
    parser.add_argument('--d', help='dim for synthetic data',type=int,default='200')

    parsed = vars(parser.parse_args())
    num_trials = parsed['num_trials']
    max_iters = parsed['max_iters']
    tol = parsed['tol']
    dist_max_iters = parsed['dist_max_iters']
    dist_tol = parsed['dist_tol']
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

    print(f"Epsilon:\t{noise}")
    means = []
    for _ in range(num_trials):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8)

        X_train, y_train_noisy = noise_fn(X, y, noise)

        theta = RRM(X_train,y_train_noisy,epsilon=noise, tol=tol, max_iters=max_iters, dist_tol=dist_tol, dist_max_iters=dist_max_iters)
        loss = np.sqrt(np.mean((np.dot(X_test, theta) - y_test) ** 2))
        means.append(loss)
        print(f"Loss:\t{loss:.3f}")
    print(f"RRM:\t{np.mean(np.float32(means)):.3f}_{{({np.std(np.float32(means)):.4f})}}")

    
