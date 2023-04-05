import numpy as np
import cvxpy as cp
from scipy.io import loadmat
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import HuberRegressor,TheilSenRegressor,LinearRegression, RANSACRegressor, Ridge
from matplotlib import pyplot as plt
import tikzplotlib

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

def addNoise(X, y, m, b, noise: float): 
    noisyY = np.copy(y)
    d = 100
    #m_ = np.random.normal(4,4,d)
    b_ = np.random.normal(4,4)
    for i in range(int(len(y) * (1-noise)), len(y)):
        noisyY[i] = np.random.normal(b_,4)
        #noisyY[i] = np.random.normal(np.dot(np.transpose(m_),X[i,0:d]) + b_,0.01)
    return noisyY

if __name__ == "__main__":
    n = 2000
    d = 100

    meta_means_sever = []
    meta_means_term = []
    meta_means_subq = []
    meta_means_genie = []
    meta_means_ransac = []
    meta_means_huber = []
    meta_means_erm = []
    meta_std_sever = []
    meta_std_term = []
    meta_std_subq = []
    meta_std_ransac = []
    meta_std_genie = []
    meta_std_huber = []
    meta_std_erm = []
    x_ = np.linspace(0.1,0.4,20)
    for eps in x_:
        means_sever = []
        means_term = []
        means_subq = []
        means_genie = []
        means_ransac = []
        means_huber = []
        means_erm = []
        for j in range(1):
            X = np.zeros((n,d+1))
            y = np.zeros((n))
            for i in range(n):
                x = np.random.uniform(-3,3,d)
                X[i,0:d] = x
                X[i,d] = 1
            
            m = np.random.normal(4,4,d)
            b = np.random.normal(4,4)

            
            for i in range(n):
                t = np.random.normal(np.dot(np.transpose(m),X[i,0:d]) + b,0.01)
                y[i] = t
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = .80)

            y_train_noisy = addNoise(X_train, y_train,m,b, eps)

            theta_sever = SEVER(X_train, y_train_noisy,iter=64)
            theta_term = TERM(X_train, y_train_noisy, -2, 0.01, 500)
            theta_erm = LM(X_train, y_train_noisy)
            theta_subq = SubQ(X_train,y_train_noisy, 600, 1-eps)
            theta_huber = HuberRegressor(max_iter=3000).fit(X_train,y_train_noisy).coef_
            #theta_genie = LM(X_train[int(y_train.shape[0] * eps):], y_train[int(y_train.shape[0] * eps):])
            ransac = RANSACRegressor().fit(X_train,y_train_noisy)

            loss_subq = calc_RMSE(y_test,theta_subq,X_test)
            loss_sever = calc_RMSE(y_test,theta_sever,X_test)
            loss_huber = calc_RMSE(y_test,theta_huber,X_test)
            loss_erm = calc_RMSE(y_test,theta_erm,X_test)
            loss_ransac = np.sqrt(np.mean((ransac.predict(X_test) - y_test) ** 2))
            loss_term = calc_RMSE(y_test,theta_term,X_test)
            #loss_genie = calc_RMSE(y_test,theta_genie,X_test)

            print(f"\n")
            print(f"epsilon:\t{eps}")
            print(f"subq loss:\t{loss_subq}")
            print(f"term loss:\t{loss_term}")
            print(f"sever loss:\t{loss_sever}")
            print(f"huber loss:\t{loss_huber}")
            print(f"ransac loss:\t{loss_ransac}")
            print(f"erm loss:\t{loss_erm}")
            #print(f"genie loss:\t{loss_genie}")


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
            #means_genie.append(loss_genie)
            means_sever.append(loss_sever)
            means_huber.append(loss_huber)
            means_erm.append(loss_erm)
        
        a = 0.5 

        mean_subq = np.mean(np.float32(means_subq))
        std_subq = np.std(np.float32(means_subq))
        meta_means_subq.append(mean_subq)
        meta_std_subq.append(std_subq)

        mean_ransac = np.mean(np.float32(means_ransac))
        std_ransac = np.std(np.float32(means_ransac))
        meta_means_ransac.append(mean_ransac)
        meta_std_ransac.append(std_ransac)

        mean_term = np.mean(np.float32(means_term))
        std_term = np.std(np.float32(means_term))
        meta_means_term.append(mean_term)
        meta_std_term.append(std_term)

        mean_huber = np.mean(np.float32(means_huber))
        std_huber = np.std(np.float32(means_huber))
        meta_means_huber.append(mean_huber)
        meta_std_huber.append(std_huber)

        mean_erm = np.mean(np.float32(means_erm))
        std_erm = np.std(np.float32(means_erm))
        meta_means_erm.append(mean_erm)
        meta_std_erm.append(std_erm)

        mean_sever = np.mean(np.float32(means_sever))
        std_sever = np.std(np.float32(means_sever))
        meta_means_sever.append(mean_sever)
        meta_std_sever.append(mean_subq)

    meta_means_subq = np.float32(meta_means_subq)
    meta_means_sever = np.float32(meta_means_sever)
    meta_means_term = np.float32(meta_means_term)
    meta_means_ransac = np.float32(meta_means_ransac)
    meta_means_huber = np.float32(meta_means_huber)
    meta_means_erm = np.float32(meta_means_erm)
    meta_std_subq = np.float32(meta_std_subq)
    meta_std_sever = np.float32(meta_std_sever)
    meta_std_term = np.float32(meta_std_term)
    meta_std_ransac = np.float32(meta_std_ransac)
    meta_std_huber = np.float32(meta_std_huber)
    meta_std_erm = np.float32(meta_std_erm)
    plt.plot(x_,meta_means_subq, color='black')
    plt.plot(x_,meta_means_sever,color='green')
    plt.plot(x_,meta_means_term,color='red')
    plt.plot(x_,meta_means_ransac,color='orange')
    plt.plot(x_,meta_means_huber,color='purple')
    plt.plot(x_,meta_means_erm,color='cyan')
    # plt.fill_between(x_,meta_means_subq-meta_std_subq,meta_means_subq+meta_std_subq,color='black',alpha=0.5)
    # plt.fill_between(x_,meta_means_sever-meta_std_sever,meta_means_sever+meta_std_sever,color='green',alpha=0.5)
    # plt.fill_between(x_,meta_means_term-meta_std_term,meta_means_term+meta_std_term,color='red',alpha=0.5)
    # plt.fill_between(x_,meta_means_ransac-meta_std_ransac,meta_means_ransac+meta_std_ransac,color='orange',alpha=0.5)
    # plt.fill_between(x_,meta_means_huber-meta_std_huber,meta_means_huber+meta_std_huber,color='purple',alpha=0.5)
    # plt.fill_between(x_,meta_means_erm-meta_std_erm,meta_means_erm+meta_std_erm,color='cyan',alpha=0.5)
    plt.ylim([0, 1000])
    tikzplotlib.save("noisy-subq-sever.tex")
    plt.show()