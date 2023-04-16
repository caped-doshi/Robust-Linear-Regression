import numpy as np
import cvxpy as cp
from scipy.io import loadmat
from sklearn import linear_model
from sklearn.linear_model import HuberRegressor
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize

def RRM(X, y, epsilon):
   
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



    cnt = 1

    while (cnt==1):
        #LOSS
        alpha = (y-X@theta_old)**2

        p = optimize_RRM(alpha.flatten(),epsilon,n, p)
        # p = optimize_scipy(alpha.flatten(),epsilon,n, p).reshape(-1, 1)
            
        #WLS SOLUTION
        theta_new  = np.linalg.pinv(X.T @ np.diag(p.flatten()) @ X) @ (X.T @ np.diag(p.flatten()) @ y)

        if np.linalg.norm(theta_old-theta_new)/np.linalg.norm(theta_old)>2e-2:
          print(np.linalg.norm(theta_old-theta_new)/np.linalg.norm(theta_old))
          cnt = 1
          theta_old = theta_new
        else:
            cnt = 0
    return theta_new



def optimize_RRM(alpha, epsilon, n, p_val=None):
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
  prob.solve(solver=cp.SCS, eps=1e-3, max_iters=10000)# feastol=1e-3, abstol=1e-3, reltol=1e-3, max_iters=100)
  return p.value

from data_loader import *
from RMSE import *
from noise_models.noise import *
if __name__ == "__main__":
    x_ = np.linspace(0.1,0.4,4)
    X,y = data_loader_drug()
    for eps in x_:
        print(f"epsilon:\t{eps}")
        means = []
        for j in range(4):
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8)
            y_train_noisy = addUnstructuredNoise(X_train, y_train, eps)
            theta = RRM(X_train,y_train_noisy,eps)
            loss = np.sqrt(np.mean((np.dot(X_test, theta) - y_test) ** 2))
            means.append(loss)
            print(f"Loss:\t{loss:.3f}")
        print(f"RRM:\t{np.mean(np.float32(means)):.3f}_{{({np.std(np.float32(means)):.4f})}}")

    
