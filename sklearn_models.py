from data_loader import *
from noise_models.noise import *
from sklearn.linear_model import HuberRegressor, Ridge, RANSACRegressor, QuantileRegressor
import statsmodels.api as sm
import statsmodels.formula.api as smf
import argparse

def LM(X, y, noise, iters):
  ridge = Ridge(2, fit_intercept=True, solver='cholesky')
  ridge.fit(X[:, :-1], y)
  theta = np.append(ridge.coef_, [ridge.intercept_])
  return theta

def ransac(X,y, noise, iters):
  base_model = Ridge()
  r = RANSACRegressor(estimator=base_model,random_state=0,min_samples=X.shape[1]+1,max_trials=iters).fit(X,y)
  return r

def quantile(X,y,noise,iters):
  mod = sm.QuantReg(y_train_noisy,X_train)
  q = mod.fit(q=1-noise-0.1,max_iter=iters)
  return q.params

def genie(X,y, noise, iters):
  X_g = X[:int(y_train.shape[0] * (1-noise))]
  y_g = y[:int(y_train.shape[0] * (1-noise))]
  ridge = Ridge(2, fit_intercept=True, solver='cholesky')
  ridge.fit(X_g[:, :-1], y_g)
  theta = np.append(ridge.coef_, [ridge.intercept_])
  return theta

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--method',help="RANSAC, Ridge, Quantile, or Genie",type=str,default="Ridge")
    parser.add_argument('--num_trials',help='run how many times',type=int,default=5)
    parser.add_argument('--num_iters',help='how many iterations of algorithm',type=int,default=32)
    parser.add_argument('--noise', help='noise ratio in range (0, 1)',type=float,default=0.1)
    parser.add_argument('--noise_type',help="oblivious, adaptive, or feature",type=str,default='oblivious')
    parser.add_argument('--dataset', help='dataset; drug, cal_housing, or abalone',type=str,default='drug')

    parsed = vars(parser.parse_args())
    method = parsed['method']
    num_trials = parsed['num_trials']
    num_iters = parsed['num_iters']
    noise = parsed['noise']
    noise_type = parsed['noise_type']
    dataset = parsed['dataset']

    maxLen = max([len(ii) for ii in parsed.keys()])
    fmtString = '\t%' + str(maxLen) + 's : %s'
    print('Arguments:')
    for keyPair in sorted(parsed.items()): print(fmtString % keyPair)

    if dataset == 'cal_housing':
        X, y = data_loader_cal_housing()
    elif dataset == 'abalone':
        X, y = data_loader_abalone()
    elif dataset == 'drug':
        X, y = data_loader_drug()     

    if noise_type == "oblivious":
        noise_fn = addObliviousNoise
    if noise_type == "adaptive":
        noise_fn = addAdaptiveNoise
    if noise_type == "feature":
        noise_fn = addFeatureNoise

    if method == "RANSAC":
        model = ransac
    elif method == "Ridge":
        model = LM
    elif method == "Quantile":
        model = quantile
    else:
        model = genie

    print(f"epsilon:\t{noise}")
    means = []
    for j in range(num_trials):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8)
        X_train, y_train_noisy = noise_fn(X_train, y_train, noise)
        theta = model(X_train,y_train_noisy,noise,num_iters)
        if model != ransac:
          loss = np.sqrt(np.mean((np.dot(X_test, theta) - y_test) ** 2))
        else:
          loss = np.sqrt(np.mean((theta.predict(X_test) - y_test) ** 2))
        means.append(loss)
        print(f"Loss:\t{loss:.3f}")
    print(f"{method}:\t{np.mean(np.float32(means)):.3f}_{{({np.std(np.float32(means)):.4f})}}")
