import numpy as np

def addFeatureNoise(X,y,noise:float,m=None,b=None):
  noisyY = np.copy(y)
  noisyX = np.copy(X)
  for i in range(int(len(y) * (1-noise)), len(y)):
    noisyY[i] = 10000*y[i]
    noisyX[i,:] = 100*X[i,:]
  return noisyX, noisyY

def addAdaptiveNoise(X, y, noise: float,w=None, b=None): 
    noisyY = np.copy(y)
    d = X.shape[1]
    for i in range(int(len(y) * (1-noise)), len(y)):
        X[i,0:d-1] = np.random.uniform(-10,10,d-1)
        noisyY[i] = -1*np.sign(X[i,0:d-1] @ w + b)
        if noisyY[i] == 0:
          noisyY[i] = -1
    return X, noisyY

def addObliviousNoise(X, y, noise: float,m=None,b=None): 
  noisyY = np.copy(y)
  d = X.shape[1]
  for i in range(int(len(y) * (1-noise)), len(y)):
      noisyY[i] *= -1
  return X, np.int32(noisyY)