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
    #theta = np.zeros(len(X[0]))
    theta = np.random.normal(0,1,3)
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

def SEVER(x_train, y_train, reg=2, p=0.01, iter=64):
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
    return w

def LM(X, y):
    return np.matmul(np.linalg.pinv(X),y)

def addNoise(X, y, noise: float): 
    noisyY = np.copy(y)
    for i in range(int(len(y) * (1-noise)), len(y)):
        noisyY[i] = np.random.normal(-1*(X[i,0]) + 4, 0.01)
    return noisyY

if __name__ == "__main__":
    n = 10000

    for eps in [0.2]:
        for j in range(1):
            x = np.random.normal(0,2,n)
            X = np.zeros((n,3))
            y = np.zeros((n))
            for i in range(n):
                X[i,0] = x[i]
                X[i,1] = 1
            
            for i in range(n):
                t = np.random.normal(2*x[i] + 3,0.01,)
                y[i] = t
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = .80)

            y_train_noisy = addNoise(X_train, y_train, eps)

            theta_sever = SEVER(X_train, y_train_noisy)
            theta_term = TERM(X_train, y_train_noisy, -2, 0.01, 3000)
            theta_erm = LM(X_train, y_train_noisy)
            theta_subq = SubQ(X_train,y_train_noisy, 1500, 1-eps)
            theta_huber = HuberRegressor(max_iter=3000).fit(X_train,y_train_noisy).coef_
            theta_genie = LM(X_train[int(y_train.shape[0] * eps):], y_train[int(y_train.shape[0] * eps):])
            ransac = RANSACRegressor().fit(X_train,y_train_noisy)
            
            print(f"Iteration:\t{j}")
            print(f"SubQ Loss\teps = {eps}:\t{calc_RMSE(y_test,theta_subq,X_test):.4f}")       
            print(f"SEVER Loss\teps = {eps}:\t{calc_RMSE(y_test,theta_sever,X_test):.8f}")  
            print(f"Huber Loss\teps = {eps}:\t{calc_RMSE(y_test,theta_huber,X_test):.4f}")
            print(f"ERM Loss\teps = {eps}:\t{calc_RMSE(y_test,theta_erm,X_test):.4f}")
            print(f"RANSAC Loss\teps = {eps}:\t{np.sqrt(np.mean((ransac.predict(X_test) - y_test) ** 2))}")
            print(f"Term Loss\teps = {eps}:\t{calc_RMSE(y_test,theta_term,X_test):.4f}")
            print(f"Genie Loss\teps = {eps}:\t{calc_RMSE(y_test,theta_genie,X_test):.4f}\n")
        


        