from data_loader import *
from RMSE import *
from noise_models.noise import *
from sklearn.linear_model import HuberRegressor, Ridge, RANSACRegressor, QuantileRegressor

def LM(X, y):
  ridge = Ridge(2, fit_intercept=True, solver='cholesky')
  ridge.fit(X[:, :-1], y)
  theta = np.append(ridge.coef_, [ridge.intercept_])
  return theta

if __name__ == "__main__":
    X,y = data_loader_drug()

    x_ = np.linspace(0.1,0.4,4)
    for eps in x_:
        print(f"epsilon:\t{eps}")
        means = []
        for j in range(10):
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8)
            y_train_noisy = addUnstructuredNoise(X_train, y_train, eps)
            theta_erm = LM(X_train,y_train_noisy)
            loss = calc_RMSE(y_test, theta_erm, X_test)
            #theta_genie = LM(X_train[:int(y_train.shape[0] * (1-eps))], y_train[:int(y_train.shape[0] * (1-eps))])
            #loss = calc_RMSE(y_test, theta_genie, X_test)
            #base_model = Ridge()
            #ransac = RANSACRegressor(estimator=base_model,random_state=0,min_samples=X.shape[1]+1).fit(X_train,y_train_noisy)
            #loss = np.sqrt(np.mean((ransac.predict(X_test) - y_test) ** 2))
            means.append(loss)
            print(f"Loss:\t{loss:.3f}")
        print(f"ERM:\t{np.mean(np.float32(means)):.3f}_{{({np.std(np.float32(means)):.4f})}}")
