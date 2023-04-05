import numpy as np
import cvxpy as cp
from scipy.io import loadmat
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import HuberRegressor, TheilSenRegressor, LinearRegression, RANSACRegressor, Ridge
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

# code from https://github.com/litian96/TERM/blob/master/robust_regression/regression.py


def compute_gradients_tilting(theta, X, y, t):  # our objective
  loss = (np.dot(X, theta) - y) ** 2
  if t > 0:
    max_l = max(loss)
    loss = loss - max_l

  grad = (np.dot(np.multiply(np.exp(loss * t),(np.dot(X, theta) - y)).T, X)) / y.size
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
  return theta


def SEVER(x_train, y_train, reg=2, p=0.01, iter=64):
    for _ in range(iter):

      ridge = Ridge(reg, fit_intercept=True, solver='cholesky')

      ridge.fit(x_train[:, :-1], y_train)

      w = np.append(ridge.coef_, [ridge.intercept_])

      # extract gradients for each point
      losses = y_train - x_train @ w
      grads = np.diag(losses) @ x_train + (reg * w)[None, :]

      # center gradients
      grad_avg = np.mean(grads, axis=0)
      centered_grad = grads-grad_avg[None, :]

      # svd to compute top vector v
      u, s, vh = np.linalg.svd(centered_grad, full_matrices=True)
      v = vh[:, 0]  # top right singular vector

      # compute outlier score
      tau = np.sum(centered_grad*v[None, :], axis=1)**2

      # in each iteration simply remove the top p fraction of outliers
      # according to the scores τi
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
  optimal = np.linalg.lstsq(X, y, rcond=None)
  return optimal[0]

def addNoise(y, noise: float):
  # np.random.seed(42)
  noisyY = np.copy(y)
  noisyY[:int(len(y) * noise)] = np.random.normal(5, 5, int(len(y) * noise))
  return noisyY


def data_loader_drug(i=0):
  x = loadmat('qsar.mat')
  x_train = x['X_train']
  x_test = x['X_test']

  y_train = x['y_train']
  y_test = x['y_test']

  x = np.concatenate((x_train, x_test), axis=0)
  y = np.concatenate((y_train, y_test), axis=0).flatten()

  intercept = np.ones((x.shape[0], 1))
  x = np.concatenate((x, intercept), axis=1)

  perm = np.random.permutation(len(y))

  return x[perm], y[perm]


if __name__ == "__main__":
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
  x_ = np.linspace(0.1, 0.4, 1)
  for eps in x_:
    print(f"Epsilon:\t{eps}")
    means_sever = []
    means_term = []
    means_subq = []
    means_genie = []
    means_ransac = []
    means_huber = []
    means_erm = []
    for i in range(2):
      X, y = data_loader_drug()
      X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.80)

      y_train_noisy = addNoise(y_train, eps)

      theta_sever = SEVER(X_train, y_train_noisy)
      #theta_term = TERM(X_train, y_train_noisy, -2, 0.01, 500)
      #theta_erm = LM(X_train, y_train_noisy)
      theta_subq = SubQ(X_train, y_train_noisy,1500,1-eps)
      #theta_huber = HuberRegressor(max_iter=3000).fit(X_train, y_train_noisy).coef_
      theta_genie = LM(X_train[int(y_train.shape[0] * eps):],y_train[int(y_train.shape[0] * eps):])
      #ransac = RANSACRegressor().fit(X_train, y_train_noisy)

      print(f"Iteration:\t{i}")
      print(f"SubQ Loss eps = {eps}:\t{calc_RMSE(y_test,theta_subq,X_test):.4f}")
      print(f"SEVER Loss eps = {eps}:\t{calc_RMSE(y_test,theta_sever,X_test):.4f}")
      print(f"Huber Loss eps = {eps}:\t{calc_RMSE(y_test,theta_huber,X_test):.4f}")
      print(f"ERM Loss eps = {eps}:\t{calc_RMSE(y_test,theta_erm,X_test):.4f}")
      #print(f"RANSAC Loss eps = {eps}:\t{np.sqrt(np.mean((ransac.predict(X_test) - y_test) ** 2))}")
      print(f"Term Loss eps = {eps}:\t{calc_RMSE(y_test,theta_term,X_test):.4f}")
      print(f"Genie Loss eps = {eps}:\t{calc_RMSE(y_test,theta_genie,X_test):.4f}\n")

    mean_subq = np.mean(np.float32(means_subq))
    std_subq = np.std(np.float32(means_subq))
    meta_means_subq.append(mean_subq)
    meta_std_subq.append(std_subq)

    # mean_ransac = np.mean(np.float32(means_ransac))
    # std_ransac = np.std(np.float32(means_ransac))
    # meta_means_ransac.append(mean_ransac)
    # meta_std_ransac.append(std_ransac)

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
  #meta_means_ransac = np.float32(meta_means_ransac)
  meta_means_huber = np.float32(meta_means_huber)
  meta_means_erm = np.float32(meta_means_erm)
  meta_std_subq = np.float32(meta_std_subq)
  meta_std_sever = np.float32(meta_std_sever)
  meta_std_term = np.float32(meta_std_term)
 #meta_std_ransac = np.float32(meta_std_ransac)
  meta_std_huber = np.float32(meta_std_huber)
  meta_std_erm = np.float32(meta_std_erm)
  plt.plot(x_,meta_means_subq, color='black')
  plt.plot(x_,meta_means_sever,color='green')
  plt.plot(x_,meta_means_term,color='red')
  #plt.plot(x_,meta_means_ransac,color='orange')
  plt.plot(x_,meta_means_huber,color='purple')
  plt.plot(x_,meta_means_erm,color='cyan')

  plt.set_yscale('log')
  tikzplotlib.save("drug-discovery.tex")
  plt.show()