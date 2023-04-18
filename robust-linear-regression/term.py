import numpy as np
import cvxpy as cp

#code from https://github.com/litian96/TERM/blob/master/robust_regression/regression.py
def compute_gradients_tilting(theta, X, y, t):  # our objective
    loss = (np.dot(X, theta) - y) ** 2
    if t > 0:
        max_l = max(loss)
        loss = loss - max_l

    grad = (np.dot(np.multiply(np.exp(loss * t), (np.dot(X, theta) - y)).T, X) + 2*theta) / y.size
    ZZ = np.mean(np.exp(t * loss))
    return grad / (ZZ)

def TERM_cp(X, y, t):
  theta = cp.Variable(len(X[0]))
  objective = cp.Minimize(cp.sum(cp.exp(t * ((X @ theta - y) ** 2 + 2 * theta))))
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


from data_loader import *
from noise import *
if __name__ == "__main__":
    X,y = data_loader_drug()

    x_ = np.linspace(0.1,0.4,4)
    for eps in x_:
        print(f"epsilon:\t{eps}")
        means = []
        for j in range(10):
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8)
            y_train_noisy = addUnstructuredNoise(X_train, y_train, eps)
            theta = TERM(X_train,y_train_noisy, -2, 0.01, 1000)
            loss = np.sqrt(np.mean((np.dot(X_test, theta) - y_test) ** 2))
            means.append(loss)
            print(f"Loss:\t{loss:.3f}")
        print(f"TERM:\t{np.mean(np.float32(means)):.3f}_{{({np.std(np.float32(means)):.4f})}}")