import numpy as np
import heapq
import cvxpy as cp

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
        obj = cp.Minimize(cp.sum_squares(X[S_t] @ beta - y[S_t]))
        constraints = [cp.norm(beta, 2) <= R]
        prob = cp.Problem(obj, constraints)
        prob.solve()

        if (1 / n) * objective_function(beta.value, X, y, S_t, f) > (1 / n) * objective_function(beta_prev, X, y, S_t, f) - eta:
            return beta.value

        beta_prev = beta.value
