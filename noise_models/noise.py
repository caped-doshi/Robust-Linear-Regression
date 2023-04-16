import numpy as np

def addFeatureNoise(X,y,noise:float):
  noisyY = np.copy(y)
  noisyX = np.copy(X)
  for i in range(int(len(y) * (1-noise)), len(y)):
    noisyY[i] = 10000*y[i]
    noisyX[i,:] = 100*X[i,:]
  return noisyX, noisyY

def addStructuredNoise(X, y, noise: float): 
    noisyY = np.copy(y)
    d = X.shape[1]
    m_ = np.random.normal(4,0,d)
    b_ = np.random.normal(0,0)
    for i in range(int(len(y) * (1-noise)), len(y)):
        noisyY[i] = np.random.normal(np.dot(np.transpose(m_),X[i,0:d]) + b_,0.01)
    return noisyY

def addUnstructuredNoise(X, y, noise: float): 
  noisyY = np.copy(y)
  d = X.shape[1]
  for i in range(int(len(y) * (1-noise)), len(y)):
      noisyY[i] = np.random.normal(5,5,1)
  return noisyY

''' sever outliers'''
def addOutliers(X,y,alpha:float,beta:float,noise:float):
  X_noisy = X[int(n*(1-noise)):]
  X_bad = (1/(alpha*n*noise))