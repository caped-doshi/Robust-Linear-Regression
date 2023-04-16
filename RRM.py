import numpy as np
import cvxpy as cp
from scipy.io import loadmat
from sklearn import linear_model
from sklearn.linear_model import HuberRegressor
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize

def data_loader_drug(i=0):
    x = loadmat('C:/Users/Josh/Documents/data/qsar.mat')
    x_train = x['X_train']
    x_test = x['X_test']

    y_train = x['y_train']
    y_test = x['y_test']

    x = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0).flatten()

    intercept = np.ones((x.shape[0], 1))
    x = np.concatenate((x, intercept), axis=1)

    perm = np.random.permutation(len(y))

    return x[perm], y[perm]


def calc_RMSE(y, theta, X):
  if len(y) != len(X): 
    print("ERROR: MISMATCHING DATA/TARGET DIMENSIONS")
    return None
  if len(X) == 0: 
    print("ERROR: NO DATA")
    return None
  if len(theta) != len(X[0]): 
    print("ERROR: MISMATCHING DATA/COEF DIMENSIONS")
    return None
  return np.sqrt(np.mean((np.dot(X, theta) - y) ** 2))

def LM(X, y):
    return np.matmul(np.linalg.pinv(X),y)

def addFeatureNoise(X,y,noise:float):
  noisyY = np.copy(y)
  noisyX = np.copy(X)
  for i in range(int(len(y) * (1-noise)), len(y)):
    noisyY[i] = 10000*y[i]
    noisyX[i,:] = 100*X[i,:]
  return noisyX, noisyY

def addNoise(X, y, noise: float): 
    noisyY = np.copy(y)
    d = 100
    #m_ = np.random.normal(4,4,d)
    b_ = np.random.normal(5,5)
    for i in range(int(len(y) * (1-noise)), len(y)):
        noisyY[i] = b_
        #noisyY[i] = np.random.normal(np.dot(np.transpose(m_),X[i,0:d]) + b_,0.01)
    return noisyY



def blockwise(X, y, epsilon):

    ##INITIALIZATION##
    n, d = X.shape
    p_old = np.ones([n,1])/n
    y = y.reshape(y.shape[0], 1)
    
    #initial theta as the solution of WLS at p_old
    theta_old = np.linalg.pinv(X.T@(p_old*X))@(X.T@(p_old*y))
    ###################

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

        p_new = optimize(alpha.flatten(),epsilon,n).reshape(-1, 1)
            
        theta_new  = np.linalg.pinv(X.T@(p_new*X))@(X.T@(p_new*y))

        if np.linalg.norm(theta_old-theta_new)/np.linalg.norm(theta_old)>1e-3:
          cnt = 1
          theta_old = theta_new
        else:
            cnt = 0
        return
    return theta_new

def optimize(alpha, epsi, n):

    # Define the entropy function
  def entropy(p):
      return -np.sum(p * np.log(p))

  # Define the objective function to be minimized
  def objective(p):
      return np.dot(alpha, p)

  # Define the constraint function
  def constraint(p):
      return entropy(p) - np.log((1 - epsi) * n)

  # Define the initial guess for p
  p0 = np.ones(n) / n

  # Define the bounds for p (each element of p should be between 0 and 1)
  bounds = [(0, 1) for i in range(n)]

  # Define the constraint as a dictionary
  constraints = [{'type': 'ineq', 'fun': constraint}]

  # Use the minimize function to find the optimal p
  result = minimize(objective, p0, method='SLSQP', bounds=bounds, constraints=constraints)

  # The optimal p vector can be obtained from the 'x' attribute of the result object
  p_optimal = result.x
  print(p_optimal)
  return p_optimal
  


if __name__ == "__main__":
    n = 2000
    d = 100

    meta_means_sever = []
    meta_means_term = []
    meta_means_subq = []
    meta_means_crr = []
    meta_means_genie = []
    meta_means_ransac = []
    meta_means_huber = []
    meta_means_erm = []
    meta_means_stir = []
    meta_std_sever = []
    meta_std_term = []
    meta_std_subq = []
    meta_std_crr = []
    meta_std_ransac = []
    meta_std_genie = []
    meta_std_huber = []
    meta_std_erm = []
    meta_std_stir = []
    x_ = np.linspace(0.1,0.4,11)
    for eps in x_:
        means_sever = []
        means_term = []
        means_subq = []
        means_crr = []
        means_genie = []
        means_ransac = []
        means_huber = []
        means_erm = []
        means_stir = []
        for j in range(5):
            X,y = data_loader_drug()
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = .80)

            y_train_noisy = addNoise(X_train, y_train, eps)

            theta_blockwise = blockwise(X_train, y_train_noisy, eps)
            exit(0)
            theta_erm = LM(X_train, y_train_noisy)

            loss_erm = calc_RMSE(y_test,theta_erm,X_test)

            print(eps, loss_erm)
    

