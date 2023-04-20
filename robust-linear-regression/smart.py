import numpy as np
import heapq
import cvxpy as cp
import argparse

def f(y_i, y_predict):
    return (y_i - y_predict) ** 2

def objective_function(b, x, y, S, f):
    sum_exp = 0
    for i in S:
        sum_exp += -np.log(f(y[i], np.dot(b, x[i])))
    return sum_exp

def TMLE(X, y, eps, eta, R, f):
    n = len(y)
    temp_heap = []
    for i in range(n):
        heapq.heappush(temp_heap, (y[i], i))
        if len(temp_heap) > n * (1 - eps):
            heapq.heappop(temp_heap)
    S_0 = [x[1] for x in temp_heap]

    beta_prev = np.zeros(X.shape[1])
    for t in range(int(1e12)):
        temp_heap = []
        for i in S_0:
            heapq.heappush(temp_heap, (-np.log(f(y[i], np.dot(X[i], beta_prev))), i))
            if len(temp_heap) > n * (1 - 2 * eps):
                heapq.heappop(temp_heap)
        S_t = [x[1] for x in temp_heap]

        beta = cp.Variable(X.shape[1])
        obj = cp.Minimize(cp.sum_squares(cp.hstack([X[S_t] @ beta - y[S_t], beta])))
        constraints = [cp.norm(beta, 2) <= R]
        prob = cp.Problem(obj, constraints)
        prob.solve(solver = "ECOS")

        if (1 / n) * objective_function(beta.value, X, y, S_t, f) > (1 / n) * objective_function(beta_prev, X, y, S_t, f) - eta:
            return beta.value

        beta_prev = beta.value

from data_loader import *
from RMSE import *
from noise import *
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    X,y = data_loader_drug()

    x_ = np.linspace(0.1,0.4,4)
    for eps in x_:
        print(f"epsilon:\t{eps}")
        means = []
        for j in range(10):
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8)
            y_train_noisy = addUnstructuredNoise(X_train, y_train, eps)
            theta = TMLE(X_train,y_train_noisy, eps,  0.001, 1e9,f)
            loss = calc_RMSE(y_test, theta, X_test)
            means.append(loss)
            print(f"Loss:\t{loss:.3f}")
        print(f"SMART:\t{np.mean(np.float32(means)):.3f}_{{({np.std(np.float32(means)):.4f})}}")