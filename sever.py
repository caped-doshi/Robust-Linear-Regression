from scipy.io import loadmat
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge
import numpy as np

def SEVER(X, Y, epsilon, alpha=4, beta=3, reg=1, p=0.3, iter=8, training_num=4084):
    X = np.copy(X)
    Y = np.copy(Y).flatten()

    idxs = np.arange(len(X))
    np.random.shuffle(idxs)
    X = X[idxs]
    Y = Y[idxs]

    # n_bad = int(epsilon*len(X))

    # corrupted_idx = np.random.choice(np.arange(len(X)), n_bad)

    # X[corrupted_idx] = np.random.normal(0, 5, X[corrupted_idx].shape) + Y[corrupted_idx]*X[corrupted_idx]/(alpha*n_bad)
    # Y[corrupted_idx] = -beta

    x_test = X[training_num:]
    x_train = X[:training_num]

    y_train = Y[:training_num]
    y_train = addNoise(y_train, epsilon)
    y_test = Y[training_num:]

    for _ in range(iter):

        # scaler = RobustScaler().fit(x_train)

        # x_train = scaler.transform(x_train)

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
        rmse = np.sqrt(np.mean((x_test@w - y_test)**2))
        print(rmse)

    return rmse

def addNoise(y, noise: float): 
  #np.random.seed(42)
  noisyY = np.copy(y)
  noisyY[:int(len(y) * noise)] = np.random.normal(5, 5, int(len(y) * noise))
  return noisyY
    

    
x = loadmat('data/qsar.mat')
x_train = x['X_train']
x_test = x['X_test']

y_train = x['y_train']
y_test = x['y_test']

X = np.concatenate([x_train, x_test])
Y = np.concatenate([y_train, y_test])

X = np.concatenate([X, np.ones(len(X)).reshape(-1, 1)], axis=1)


epsilon = 0.4
alpha = 2
beta = 4


rmse = SEVER(X, Y, epsilon, alpha, beta, reg=2, p=0.05, iter=16, training_num=4084)
print(rmse)