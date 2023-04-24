import numpy as np

def addFeatureNoise(X,y,noise:float,m=None,b=None):
  noisyY = np.copy(y)
  noisyX = np.copy(X)
  for i in range(int(len(y) * (1-noise)), len(y)):
    noisyY[i] = 10000*y[i]
    noisyX[i,:] = 100*X[i,:]
  return noisyX, noisyY

def addAdaptiveNoise(X, y, noise: float,m=None, b=None): 
    noisyY = np.copy(y)
    d = X.shape[1]
    b_ = 10
    for i in range(int(len(y) * (1-noise)), len(y)):
        noisyY[i] = noisyY[i] + b_
    return X, noisyY

def addObliviousNoise(X, y, noise: float,m=None,b=None): 
  noisyY = np.copy(y)
  d = X.shape[1]
  for i in range(int(len(y) * (1-noise)), len(y)):
      noisyY[i] = np.random.normal(5,5,1)
  return X, noisyY

''' sever outliers'''
def addOutliers(X,y,alpha:float,beta:float,noise:float):
  X_noisy = X[int(n*(1-noise)):]
  X_bad = (1/(alpha*n*noise))