import numpy as np
import cvxpy as cp
from scipy.io import loadmat
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import HuberRegressor,TheilSenRegressor,LinearRegression, RANSACRegressor

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

#code from https://github.com/litian96/TERM/blob/master/robust_regression/regression.py
def compute_gradients_tilting(theta, X, y, t):  # our objective
    loss = (np.dot(X, theta) - y) ** 2
    if t > 0:
        max_l = max(loss)
        loss = loss - max_l

    grad = (np.dot(np.multiply(np.exp(loss * t), (np.dot(X, theta) - y)).T, X)) / y.size
    ZZ = np.mean(np.exp(t * loss))
    return grad / (ZZ)
  


def TERM(X, y, t):
  theta = cp.Variable(len(X[0]))
  objective = cp.Minimize(cp.sum(cp.exp(t * ((X @ theta - y) ** 2))))
  prob = cp.Problem(objective)
  result = prob.solve(verbose=True, max_iters=500)
  print(result)
  return theta.value

def TERM(train_X, train_y, t, alpha, num_iters):
    theta = np.zeros(len(train_X[0]))
    for j in range(num_iters):
        grads_theta = compute_gradients_tilting(theta, train_X, train_y, t)

        if np.linalg.norm(grads_theta, ord=2) < 1e-10:
            break
        theta = theta - alpha * grads_theta
        if j % 1000 == 0:
            train_error = np.sqrt(np.mean((np.dot(train_X, theta) - train_y) ** 2))
            #print("training error: ", train_error)
    return theta

def SubQ(X, y, T, p):
    t = 0
    n = X.shape[0]
    theta = np.zeros(len(X[0]))
    deriv = np.zeros(len(X[0]))
    for i in range(T):
        pred = np.matmul(X,theta)
        v = (pred - y)**2
        v_hat = sorted(v)
        v_arg_hat = np.argsort(v)

        subq_error = np.linalg.norm(v_hat[:int(n*p)],2)/int(n*p)
        #if i % 100 == 0:
            #print(f"Iteration {i}:\tSubQ Error\t=\t{subq_error:.4f}")

        X_np = X[v_arg_hat[:int(n*p)]]
        y_np = y[v_arg_hat[:int(n*p)]]
        d = np.dot(X_np,theta) - y_np
        deriv = np.sum(2 * X_np * d[:, np.newaxis],axis=0)

        L = np.linalg.norm(np.matmul(np.transpose(X_np),X_np),2)
        alpha = 1/(2*L)

        update = -1 * alpha * deriv
        theta = theta + update
    return theta

def LM(X, y):
    return np.matmul(np.linalg.pinv(X),y)

def addNoise(X, y, noise: float): 
    noisyY = np.copy(y)
    for i in range(int(len(y) * (1-noise)), len(y)):
        noisyY[i] = np.random.normal(-1*(X[i,0]**2) + X[i,1] + 4, 0.01)
    return noisyY

if __name__ == "__main__":
    n = 10000

    for j in range(10):
        x = np.random.normal(0,1,n)
        X = np.zeros((n,3))
        y = np.zeros((n))
        for i in range(n):
            X[i,2] = 1
            X[i,1] = x[i]
            X[i,0] = x[i]**2

        eps = 0.2
        
        for i in range(n):
            t = np.random.normal(x[i]**2 - x[i] + 2,0.01,1)
            y[i] = t
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = .80)

        y_train_noisy = addNoise(X_train, y_train, eps)

        theta_term = TERM(X_train, y_train_noisy, -2, 0.01, 3000)
        theta_erm = LM(X_train, y_train_noisy)
        theta_subq = SubQ(X_train,y_train_noisy, 500, 0.6)
        theta_huber = HuberRegressor(max_iter=3000).fit(X_train,y_train_noisy).coef_
        theta_genie = LM(X_train[int(len(y) * eps):], y_train[int(len(y) * eps):])
        ransac = RANSACRegressor().fit(X_train,y_train_noisy)

        print(f"Iteration:\t{j}")
        print(f"SubQ Loss eps = {eps}:\t{calc_RMSE(y_test,theta_subq,X_test):.8f}")        
        print(f"Huber Loss eps = {eps}:\t{calc_RMSE(y_test,theta_huber,X_test):.8f}")
        print(f"ERM Loss eps = {eps}:\t{calc_RMSE(y_test,theta_erm,X_test):.8f}")
        print(f"RANSAC Loss eps = {eps}:\t{np.sqrt(np.mean((ransac.predict(X_test) - y_test) ** 2)):.8f}")
        print(f"Term Loss eps = {eps}:\t{calc_RMSE(y_test,theta_term,X_test):.8f}")
        print(f"Genie Loss eps = {eps}:\t{calc_RMSE(y_test,theta_genie,X_test):.8f}\n")


        