import numpy as np

def HT(v, k):
    # only keep k largest elements (k corrupted samples)
    not_top_k_index = (abs(v)).argsort()[:len(v)-k]
    v[not_top_k_index] = 0
    return v

def CRR(X,y):
    k = 500
    eps = 0.1
    b_new = np.ones(len(X))
    b_old = np.zeros(len(X))
    cov_ = np.dot(X.T, X) + np.identity(len(X[0])) * 0.00001
    P_x = np.dot(np.dot(X, np.linalg.inv(cov_)), X.T)
    a = np.dot((np.identity(len(X))-P_x), y)
    iter_ = 0
    while np.linalg.norm(b_new-b_old, ord=2) > eps:
        iter_ += 1
        b_old = b_new
        b_new = HT(np.dot(P_x, b_old) + a, k)
        if iter_ % 100 == 0:
            print(iter_)
        if iter_ > 3000:  # keep increasing the number of iterations until the point where it doesn't affect things
            break

    #print(b_new)
    theta = np.dot(np.dot(np.linalg.inv(cov_), X.T), (y-b_new))

    #print(theta)
    return theta