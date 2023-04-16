import numpy as np
from sklearn.linear_model import Ridge, LinearRegression

def SEVER(x_train, y_train, reg=2, p=0.01, iter=64):
    for _ in range(iter):

        ridge = Ridge(reg, fit_intercept=True, solver='cholesky')

        ridge.fit(x_train[:,:-1], y_train)

        w = np.append(ridge.coef_, [ridge.intercept_])
        #w = np.matmul(np.linalg.pinv(x_train),y_train)

        #extract gradients for each point
        losses = y_train - x_train @ w
        grads = 2 * np.diag(losses) @ x_train #+ (reg * w)[None, :]

        #center gradients
        grad_avg = np.mean(grads, axis=0)
        centered_grad = grads-grad_avg[None, :]

        #svd to compute top vector v
        u, s, vh = np.linalg.svd(centered_grad, full_matrices=False)
        v = vh[:, 0] #top right singular vector

        #compute outlier score
        tau = np.sum(centered_grad*v[None, :], axis=1)**2

        #in each iteration simply remove the top p fraction of outliers
        #according to the scores τi
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


from data_loader import *
from RMSE import *
from noise_models.noise import *
if __name__ == "__main__":
    x_ = np.linspace(0.1,0.4,4)
    X, y = gaussian(2000,200)
    for eps in x_:
        print(f"epsilon:\t{eps}")
        means = []
        for j in range(10):
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8)
            y_train_noisy = addUnstructuredNoise(X_train, y_train, eps)
            theta = SEVER(X_train,y_train_noisy)
            loss = calc_RMSE(y_test, theta, X_test)
            means.append(loss)
            print(f"Loss:\t{loss:.3f}")
        print(f"Sever:\t{np.mean(np.float32(means)):.3f}_{{({np.std(np.float32(means)):.4f})}}")