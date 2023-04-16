import numpy as np
import cvxpy as cp
from scipy.io import loadmat
from sklearn import linear_model
from sklearn.linear_model import HuberRegressor
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize

def blockwise(X, y, epsilon):
    
    ##INITIALIZATION##
    n, d = X.shape
    p = np.ones([n,1])/n
    
    #initial theta as the solution of WLS at p_old
    theta_old = np.linalg.pinv(X.T @ np.diag(p.flatten()) @ X) @ (X.T @ np.diag(p.flatten()) @ y)

   
    ##################

    ####RRM##########
    #INPUT
    #x_aug : n x d matrix
    #y_aug : n x 1 vector
    #epsilon : scalar
    ###############

    #Output###
    #theta_new : d x 1 vector of robust estimate


    cnt = 1

    while (cnt==1):
        alpha = (y-X@theta_old)**2

        p = p.flatten()
        p = optimize(alpha.flatten(),epsilon,n, p).reshape(-1, 1)
        # p = optimize_scipy(alpha.flatten(),epsilon,n, p).reshape(-1, 1)
            
        theta_new  = np.linalg.pinv(X.T @ np.diag(p.flatten()) @ X) @ (X.T @ np.diag(p.flatten()) @ y)

        if np.linalg.norm(theta_old-theta_new)/np.linalg.norm(theta_old)>1e-3:
          #print(np.linalg.norm(theta_old-theta_new)/np.linalg.norm(theta_old))
          cnt = 1
          theta_old = theta_new
        else:
            cnt = 0
    return theta_new

def optimize(alpha, epsilon, n, p_val):
   # Define the optimization variables
  p = cp.Variable(n)
  p.value = p_val

  # Define the objective function to be minimized
  objective = cp.Minimize(alpha.T @ p)

  # Define the constraints
  constraints = [cp.sum(cp.entr(p)) >= cp.log((1-epsilon)*n), # entropy constraint
                  cp.sum(p) == 1.0, # probability constraint
                  p >= 0] # non-negativity constraint

  # Define the problem and solve it
  prob = cp.Problem(objective, constraints)
  prob.solve(solver=cp.ECOS, max_iters=500)# feastol=1e-3, abstol=1e-3, reltol=1e-3, max_iters=100)
  return p.value

def optimize_scipy(alpha, epsilon, n, p_val):
    def objective(p):
        return alpha.T @ p

    def constraint_entropy(p):
        return entropy(p) - np.log((1 - epsilon) * n)

    constraints = ({'type': 'eq', 'fun': lambda p: np.sum(p) - 1},  # probability constraint
                   {'type': 'ineq', 'fun': constraint_entropy},  # entropy constraint
                   {'type': 'ineq', 'fun': lambda p: -p})  # non-negativity constraint

    res = minimize(objective, p_val, method='CG', constraints=constraints, options={'maxiter': 500})

    return res.x

from data_loader import *
from RMSE import *
from noise_models.noise import *
if __name__ == "__main__":
    x_ = np.linspace(0.2,0.4,3)
    X,y = data_loader_drug()
    for eps in x_:
        print(f"epsilon:\t{eps}")
        means = []
        for j in range(4):
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8)
            y_train_noisy = addUnstructuredNoise(X_train, y_train, eps)
            theta = blockwise(X_train,y_train_noisy,eps)
            loss = calc_RMSE(y_test, theta, X_test)
            means.append(loss)
            print(f"Loss:\t{loss:.3f}")
        print(f"RRM:\t{np.mean(np.float32(means)):.3f}_{{({np.std(np.float32(means)):.4f})}}")

    
