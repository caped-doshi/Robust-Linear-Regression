import numpy as np
import cvxpy as cp
from scipy.io import loadmat
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import HuberRegressor,TheilSenRegressor,LinearRegression, RANSACRegressor, Ridge
from matplotlib import pyplot as plt

from SubQuantile import SubQ

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
        noisyY[i] = np.random.normal(-2*(X[i,0]) +4, 1)
    return noisyY

if __name__ == "__main__":
    n = 2000

    e_subq = []
    ed_subq = []
    eu_subq = []
    e_ransac = []
    ed_ransac = []
    eu_ransac = []
    e_term = []
    ed_term = []
    eu_term = []
    e_genie = []
    ed_genie = []
    eu_genie = []
    e_sever = []
    ed_sever = []
    eu_sever = []
    for eps in np.linspace(0.1,0.9,20):
        means_sever = []
        means_term = []
        means_subq = []
        means_genie = []
        means_ransac = []
        for j in range(5):
            x = np.random.normal(0,2,n)
            X = np.zeros((n,3))
            y = np.zeros((n))
            for i in range(n):
                X[i,0] = x[i]
                X[i,1] = 1
            
            for i in range(n):
                t = np.random.normal(2*x[i] + 4,1)
                y[i] = t
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = .80)

            y_train_noisy = addNoise(X_train, y_train, eps)

            if eps > 0.5: majority = False
            else: majority = True

            theta_sever = SEVER(X_train, y_train_noisy,iter=64)
            theta_term = TERM(X_train, y_train_noisy, -2, 0.01, 3000)
            theta_erm = LM(X_train, y_train_noisy)
            theta_subq = SubQ(X_train,y_train_noisy, 1500, max(eps,1-eps),majority)
            theta_huber = HuberRegressor(max_iter=3000).fit(X_train,y_train_noisy).coef_
            theta_genie = LM(X_train[int(y_train.shape[0] * eps):], y_train[int(y_train.shape[0] * eps):])
            ransac = RANSACRegressor().fit(X_train,y_train_noisy)

            loss_subq = calc_RMSE(y_test,theta_subq,X_test)
            loss_sever = calc_RMSE(y_test,theta_sever,X_test)
            loss_huber = calc_RMSE(y_test,theta_huber,X_test)
            loss_erm = calc_RMSE(y_test,theta_erm,X_test)
            loss_ransac = np.sqrt(np.mean((ransac.predict(X_test) - y_test) ** 2))
            loss_term = calc_RMSE(y_test,theta_term,X_test)
            loss_genie = calc_RMSE(y_test,theta_genie,X_test)

            # x_test = np.linspace(np.min(x),np.max(x),300)
            # y_sever = theta_sever[0]*x_test + theta_sever[1]
            # y_huber = theta_huber[0]*x_test + theta_huber[1]
            # y_erm = theta_erm[0]*x_test + theta_erm[1]
            # y_term = theta_term[0]*x_test + theta_term[1]
            # y_subq = theta_subq[0]*x_test + theta_subq[1]
            # y_genie = theta_genie[0]*x_test + theta_genie[1]

            means_subq.append(loss_subq)
            means_ransac.append(loss_ransac)
            means_term.append(loss_term)
            means_genie.append(loss_genie)
            means_sever.append(loss_sever)
        
        mean_subq = np.mean(np.float32(means_subq))
        std_subq = np.std(np.float32(means_subq))
        e_subq.append((eps,mean_subq))
        ed_subq.append((eps,mean_subq-std_subq))
        eu_subq.append((eps,mean_subq+std_subq))

        mean_ransac = np.mean(np.float32(means_ransac))
        std_ransac = np.std(np.float32(means_ransac))
        e_ransac.append((eps,mean_ransac))
        ed_ransac.append((eps,mean_ransac-std_ransac))
        eu_ransac.append((eps,mean_ransac+std_ransac))

        mean_term = np.mean(np.float32(means_term))
        std_term = np.std(np.float32(means_term))
        e_term.append((eps,mean_term))
        ed_term.append((eps,mean_term-std_term))
        eu_term.append((eps,mean_term+std_term))

        mean_genie = np.mean(np.float32(means_genie))
        std_genie = np.std(np.float32(means_genie))
        e_genie.append((eps,mean_genie))
        ed_genie.append((eps,mean_genie-std_genie))
        eu_genie.append((eps,mean_genie+std_genie))

        mean_sever = np.mean(np.float32(means_sever))
        std_sever = np.std(np.float32(means_sever))
        e_sever.append((eps,mean_sever))
        ed_sever.append((eps,mean_sever-std_sever))
        eu_sever.append((eps,mean_sever+std_sever))

    print("SUBQUANTILE")
    for e in e_subq:
        print(f"{e}",end="")    
    print("\n")
    for e in eu_subq:
        print(f"{e}",end="")
    print("\n")
    for e in ed_subq:
        print(f"{e}",end="")   
    print("\n")   

    print("RANSAC\n")
    for e in e_ransac:
        print(f"{e}",end="")    
    print("\n")
    for e in eu_ransac:
        print(f"{e}",end="")
    print("\n")
    for e in ed_ransac:
        print(f"{e}",end="") 
    print("\n")  

    print("TERM\n")
    for e in e_term:
        print(f"{e}",end="")    
    print("\n")
    for e in eu_term:
        print(f"{e}",end="")
    print("\n")
    for e in ed_term:
        print(f"{e}",end="")   
    print("\n")

    print("GENIE\n")
    for e in e_genie:
        print(f"{e}",end="")    
    print("\n")
    for e in eu_genie:
        print(f"{e}",end="")
    print("\n")
    for e in ed_genie:
        print(f"{e}",end="")   
    print("\n")

    print("SEVER\n")
    for e in e_sever:
        print(f"{e}",end="")    
    print("\n")
    for e in eu_sever:
        print(f"{e}",end="")
    print("\n")
    for e in ed_sever:
        print(f"{e}",end="")        
        