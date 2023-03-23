import numpy as np
import cvxpy as cp
from scipy.io import loadmat
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import HuberRegressor,TheilSenRegressor,LinearRegression, RANSACRegressor, Ridge

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

def addNoise(y, noise: float): 
  #np.random.seed(42)
  noisyY = np.copy(y)
  noisyY[:int(len(y) * noise)] = np.random.normal(5, 5, int(len(y) * noise))
  return noisyY

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

def SEVER(x_train, x_test, y_train, y_test, reg=2, p=0.01, iter=64):
    for _ in range(iter):

        ridge = Ridge(reg, fit_intercept=True, solver='cholesky')

        ridge.fit(x_train[:,:-1], y_train)

        w = np.append(ridge.coef_, [ridge.intercept_])

        #extract gradients for each point
        losses = y_train - x_train @ w
        grads = np.diag(losses) @ x_train + (reg * w)[None, :]

        #center gradients
        grad_avg = np.mean(grads, axis=0)
        centered_grad = grads-grad_avg[None, :]

        #svd to compute top vector v
        u, s, vh = np.linalg.svd(centered_grad, full_matrices=False)
        v = vh[:, 0] #top right singular vector

        #compute outlier score
        tau = np.sum(centered_grad*v[None, :], axis=1)**2

        #in each iteration simply remove the top p fraction of outliers
        #according to the scores τi
        n_removed = int(p*len(tau))
        # idx_kept = np.argpartition(tau, -n_removed)[:-n_removed]
        idx_kept = np.argsort(tau)[:-n_removed]
        idx_kept = np.sort(idx_kept)
        x_train = x_train[idx_kept]
        y_train = y_train[idx_kept]

        # test_data = scaler.transform(x_test)
        rmse = np.sqrt(np.mean((x_train@w - y_train)**2))
        # print(rmse)
    rmse = np.sqrt(np.mean((x_test@w - y_test)**2))
    return rmse

def LM(X, y):
    return np.matmul(np.linalg.pinv(X),y)
    

def data_loader_drug(i = 0):
    x = loadmat('qsar.mat')
    x_train = x['X_train']
    x_test = x['X_test']

    y_train = x['y_train']
    y_test = x['y_test']

    x = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0).flatten()

    intercept = np.ones((x.shape[0], 1))
    x = np.concatenate((x, intercept), axis=1)

    #np.random.seed(i)
    perm = np.random.permutation(len(y))

    #print(len(x), len(x[0]))

    return  x[perm], y[perm]

if __name__ == "__main__":
    for i in range(10):
      X, y = data_loader_drug()
      X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = .80)
      
      for eps in [0.4]:

        y_train_noisy = addNoise(y_train, eps)

        rmse_sever = SEVER(X_train, X_test, y_train_noisy, y_test)
        theta_term = TERM(X_train, y_train_noisy, -2, 0.01, 3000)
        theta_erm = LM(X_train, y_train_noisy)
        theta_subq = SubQ(X_train,y_train_noisy, 500, 0.6)
        theta_huber = HuberRegressor(max_iter=3000).fit(X_train,y_train_noisy).coef_
        theta_genie = LM(X_train[int(len(y) * eps):], y_train[int(len(y) * eps):])
        ransac = RANSACRegressor().fit(X_train,y_train_noisy)
        
        print(f"Iteration:\t{i}")
        print(f"SubQ Loss eps = {eps}:\t{calc_RMSE(y_test,theta_subq,X_test):.4f}")       
        print(f"SEVER Loss eps = {eps}:\t{rmse_sever:.4f}")  
        print(f"Huber Loss eps = {eps}:\t{calc_RMSE(y_test,theta_huber,X_test):.4f}")
        print(f"ERM Loss eps = {eps}:\t{calc_RMSE(y_test,theta_erm,X_test):.4f}")
        print(f"RANSAC Loss eps = {eps}:\t{np.sqrt(np.mean((ransac.predict(X_test) - y_test) ** 2))}")
        print(f"Term Loss eps = {eps}:\t{calc_RMSE(y_test,theta_term,X_test):.4f}")
        print(f"Genie Loss eps = {eps}:\t{calc_RMSE(y_test,theta_genie,X_test):.4f}\n")