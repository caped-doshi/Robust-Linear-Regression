from scipy.io import loadmat
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge
import numpy as np

def SEVER(X, Y, epsilon, alpha, beta):
    X = np.copy(X)
    Y = np.copy(Y)

    n_bad = int(epsilon*len(X))

    corrupted_idx = np.random.choice(np.arange(len(X)), n_bad)

    X[corrupted_idx] = np.random.normal(0, 5, X[corrupted_idx].shape) + Y[corrupted_idx]*X[corrupted_idx]/(alpha*n_bad)
    Y[corrupted_idx] = -beta

    x_train = X[:4084]
    x_test = X[4084:]

    y_train = Y[:4084]
    y_test = Y[4084:]

    scaler = RobustScaler().fit(x_train)

    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    ridge = Ridge(1, fit_intercept=False)

    ridge.fit(x_train, y_train)

    w = ridge.coef_


    #extract gradients for each point
    #center gradients
    #svd to compute top vector v
    #in each iteration simply remove the top p fraction of outliers
    #according to the scores τi
    #and instead of using a specific stopping condition, simply repeat the filter for r
    #iterations in total. This is the version of Sever that we use in our experiments in Section 3

    return w
    

    
x = loadmat('data/qsar.mat')
x_train = x['X_train']
x_test = x['X_test']

y_train = x['y_train']
y_test = x['y_test']

X = np.concatenate([x_train, x_test])
Y = np.concatenate([y_train, y_test])

X = np.concatenate([X, np.ones(len(X)).reshape(-1, 1)], axis=1)

idxs = np.arange(len(X))
np.random.shuffle(idxs)
X = X[idxs]
Y = Y[idxs]

epsilon = 0.1
alpha = 2
beta = 4


w = SEVER(X, Y, epsilon, alpha, beta)
print(w)