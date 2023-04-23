import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

def data_loader_drug(i=0):
  x = loadmat('data/qsar.mat')
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

def data_loader_abalone(i=0):
  x = []
  y = []
  with open("data/abalone.data", 'r') as f:
      lines = f.readlines()
      for sample in lines:
          sample = sample.strip().split(',')
          y.append(float(sample[-1]))
          sample[0] = 1 if sample[0] == 'M' else 0
          x.append(sample[:-1])
  x = np.array(x)
  y = np.array(y)

  # normalize the features
  mu = np.mean(x.astype(np.float32), 0)
  sigma = np.std(x.astype(np.float32), 0)
  x = (x.astype(np.float32) - mu) / (sigma + 0.000001)

  intercept = np.ones((x.shape[0], 1))
  x = np.concatenate((x, intercept), axis=1)

  # randomly shuffle data with seed i
  #np.random.seed(i)
  perm = np.random.permutation(len(y))

  return x[perm], y[perm]

def data_loader_cal_housing(i=0):
  x = []
  y = []
  with open("data/cal_housing.data", 'r') as f:
      lines = f.readlines()
      for sample in lines:
          sample = sample.strip().split(',')
          for i in range(len(sample)):
            sample[i] = float(sample[i])
          y.append(float(sample[-1]))
          x.append(sample[:-1])
  x = np.array(x)
  y = np.array(y)

  # normalize the features
  mu = np.mean(x.astype(np.float32), 0)
  sigma = np.std(x.astype(np.float32), 0)
  x = (x.astype(np.float32) - mu) / (sigma + 0.000001)

  mu = np.mean(y.astype(np.float32), 0)
  sigma = np.std(y.astype(np.float32), 0)
  y = (y.astype(np.float32) - mu) / (sigma + 0.000001)

  intercept = np.ones((x.shape[0], 1))
  x = np.concatenate((x, intercept), axis=1)

  # randomly shuffle data with seed i
  #np.random.seed(i)
  perm = np.random.permutation(len(y))

  return x[perm], y[perm]

def gaussian(n,d):
    x = np.zeros((n,d+1))
    y = np.zeros((n))
    for i in range(n):
        X = np.random.normal(0, 1, d)
        x[i,0:d] = X
        x[i,d] = 1
    m = np.random.normal(4,4,d)
    b = np.random.normal(4,4)

    for i in range(n):
        t = np.random.normal(np.dot(np.transpose(m),x[i,0:d]) + b,0.01)
        y[i] = t
    return x, y, m, b
