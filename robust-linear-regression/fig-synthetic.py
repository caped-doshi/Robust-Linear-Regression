import numpy as np
import cvxpy as cp
from scipy.io import loadmat
from sklearn import linear_model
from sklearn.linear_model import HuberRegressor, Ridge, RANSACRegressor, QuantileRegressor
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib import pyplot as plt
import tikzplotlib

from data_loader import *
from noise import *

from sever import SEVER
from term import TERM
from crr import CRR
from stir import stir
from SubQuantile import SubQ, SubQ2

def LM(X, y):
  ridge = Ridge(2, fit_intercept=True, solver='cholesky')
  ridge.fit(X[:, :-1], y)
  theta = np.append(ridge.coef_, [ridge.intercept_])
  return theta

if __name__ == "__main__":
    n = 10000
    d = 200

    meta_means_sever = []
    meta_means_term = []
    meta_means_subq = []
    meta_means_crr = []
    meta_means_genie = []
    meta_means_ransac = []
    meta_means_quantile = []
    meta_means_huber = []
    meta_means_erm = []
    meta_means_stir = []
    meta_std_sever = []
    meta_std_term = []
    meta_std_subq = []
    meta_std_crr = []
    meta_std_ransac = []
    meta_std_quantile = []
    meta_std_genie = []
    meta_std_huber = []
    meta_std_erm = []
    meta_std_stir = []
    x_ = np.linspace(0.1,0.4,11)
    for eps in x_:
        means_sever = []
        means_term = []
        means_subq = []
        means_crr = []
        means_genie = []
        means_ransac = []
        means_huber = []
        means_erm = []
        means_stir = []
        means_quantile = []
        for j in range(4):
            X,y,m,b = gaussian(n,d)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8)

            X_train,y_train_noisy = addAdaptiveNoise(X_train, y_train, eps, m, b)
            
            mod = sm.QuantReg(y_train_noisy,X_train)
            q = mod.fit(q=1-eps-0.1,max_iter=500)
            theta_quantile = q.params
            theta_sever = SEVER(X_train, y_train_noisy,iter=64)
            theta_term = TERM(X_train, y_train_noisy, -2, 0.1, 10000)
            theta_erm = LM(X_train, y_train_noisy)
            theta_subq = SubQ2(X_train,y_train_noisy, 64, 1-eps)
            theta_huber = HuberRegressor(max_iter=1500).fit(X_train,y_train_noisy).coef_
            theta_genie = LM(X_train[:int(y_train.shape[0] * (1-eps))], y_train[:int(y_train.shape[0] * (1-eps))])
            base_model = Ridge()
            ransac = RANSACRegressor(estimator=base_model,random_state=0,min_samples=X_train.shape[1]+1).fit(X_train,y_train_noisy)
            theta_crr = CRR(X_train,y_train_noisy,max_iters=3000)
            theta_stir = stir(X_train,y_train_noisy,500)

            loss_subq = np.sqrt(np.mean((np.dot(X_test, theta_subq) - y_test) ** 2))
            loss_sever = np.sqrt(np.mean((np.dot(X_test, theta_sever) - y_test) ** 2))
            loss_huber = np.sqrt(np.mean((np.dot(X_test, theta_huber) - y_test) ** 2))
            loss_erm = np.sqrt(np.mean((np.dot(X_test, theta_erm) - y_test) ** 2))
            loss_ransac = np.sqrt(np.mean((ransac.predict(X_test) - y_test) ** 2))
            loss_quantile = np.sqrt(np.mean((np.dot(X_test, theta_quantile) - y_test) ** 2))
            loss_term = np.sqrt(np.mean((np.dot(X_test, theta_term) - y_test) ** 2))
            loss_crr = np.sqrt(np.mean((np.dot(X_test, theta_crr) - y_test) ** 2))
            loss_stir = np.sqrt(np.mean((np.dot(X_test, theta_stir) - y_test) ** 2))
            loss_genie = np.sqrt(np.mean((np.dot(X_test, theta_genie) - y_test) ** 2))

            print(f"\n")
            print(f"epsilon:\t{eps}")
            print(f"subq loss:\t{loss_subq}")
            print(f"term loss:\t{loss_term}")
            print(f"sever loss:\t{loss_sever}")
            print(f"Quantile loss:\t{loss_quantile}")
            print(f"CRR loss:\t{loss_crr}")
            print(f"STIR loss:\t{loss_stir}")
            print(f"huber loss:\t{loss_huber}")
            print(f"ransac loss:\t{loss_ransac}")
            print(f"erm loss:\t{loss_erm}")
            print(f"genie loss:\t{loss_genie}")

            means_subq.append(loss_subq)
            means_ransac.append(loss_ransac)
            means_term.append(loss_term)
            means_genie.append(loss_genie)
            means_sever.append(loss_sever)
            means_huber.append(loss_huber)
            means_erm.append(loss_erm)
            means_crr.append(loss_crr)
            means_quantile.append(loss_quantile)
            means_stir.append(loss_stir)
        
        a = 0.5 

        mean_erm = np.mean(np.float32(means_erm))
        std_erm = np.std(np.float32(means_erm))
        meta_means_erm.append(mean_erm)
        meta_std_erm.append(std_erm)
        print(f"ERM:\t{mean_erm:.3f}_{{({std_erm:.4f})}}")

        mean_crr = np.mean(np.float32(means_crr))
        std_crr = np.std(np.float32(means_crr))
        meta_means_crr.append(mean_crr)
        meta_std_crr.append(std_crr)
        print(f"CRR:\t{mean_crr:.3f}_{{({std_crr:.4f})}}")

        mean_stir = np.mean(np.float32(means_stir))
        std_stir = np.std(np.float32(means_stir))
        meta_means_stir.append(mean_stir)
        meta_std_stir.append(std_stir)
        print(f"Stir:\t{mean_stir:.3f}_{{({std_stir:.4f})}}")

        mean_huber = np.mean(np.float32(means_huber))
        std_huber = np.std(np.float32(means_huber))
        meta_means_huber.append(mean_huber)
        meta_std_huber.append(std_huber)
        print(f"Huber:\t{mean_huber:.3f}_{{({std_huber:.4f})}}")

        mean_ransac = np.mean(np.float32(means_ransac))
        std_ransac = np.std(np.float32(means_ransac))
        meta_means_ransac.append(mean_ransac)
        meta_std_ransac.append(std_ransac)
        print(f"Ransac:\t{mean_ransac:.3f}_{{({std_ransac:.4f})}}")

        mean_quantile = np.mean(np.float32(means_quantile))
        std_quantile = np.std(np.float32(means_quantile))
        meta_means_quantile.append(mean_quantile)
        meta_std_quantile.append(std_quantile)
        print(f"Quantile:\t{mean_quantile:.3f}_{{({std_quantile:.4f})}}")

        mean_term = np.mean(np.float32(means_term))
        std_term = np.std(np.float32(means_term))
        meta_means_term.append(mean_term)
        meta_std_term.append(std_term)
        print(f"Term:\t{mean_term:.3f}_{{({std_term:.4f})}}")

        mean_sever = np.mean(np.float32(means_sever))
        std_sever = np.std(np.float32(means_sever))
        meta_means_sever.append(mean_sever)
        meta_std_sever.append(std_sever)
        print(f"Sever:\t{mean_sever:.3f}_{{({std_sever:.4f})}}")

        mean_subq = np.mean(np.float32(means_subq))
        std_subq = np.std(np.float32(means_subq))
        meta_means_subq.append(mean_subq)
        meta_std_subq.append(std_subq)
        print(f"SubQuantile:\t{mean_subq:.3f}_{{({std_subq:.4f})}}")

        mean_genie = np.mean(np.float32(means_genie))
        std_genie = np.std(np.float32(means_genie))
        meta_means_genie.append(mean_genie)
        meta_std_subq.append(std_genie)
        print(f"Genie:\t{mean_genie:.3f}_{{({std_genie:.4f})}}")

    meta_means_subq = np.float32(meta_means_subq)
    meta_means_sever = np.float32(meta_means_sever)
    meta_means_term = np.float32(meta_means_term)
    meta_means_crr = np.float32(meta_means_crr)
    meta_means_stir = np.float32(meta_means_stir)
    meta_means_quantile = np.float32(meta_means_quantile)
    meta_means_ransac = np.float32(meta_means_ransac)
    meta_means_huber = np.float32(meta_means_huber)
    meta_means_erm = np.float32(meta_means_erm)
    meta_means_genie = np.float32(meta_means_genie)

    meta_std_subq = np.float32(meta_std_subq)
    meta_std_sever = np.float32(meta_std_sever)
    meta_std_term = np.float32(meta_std_term)
    meta_std_crr = np.float32(meta_std_crr)
    meta_std_stir = np.float32(meta_std_stir)
    meta_std_quantile = np.float32(meta_std_quantile)
    meta_std_ransac = np.float32(meta_std_ransac)
    meta_std_huber = np.float32(meta_std_huber)
    meta_std_erm = np.float32(meta_std_erm)
    meta_std_genie = np.float32(meta_std_genie)

    plt.plot(x_,meta_means_subq, color='black')
    plt.plot(x_,meta_means_sever,color='green')
    plt.plot(x_,meta_means_term,color='red')
    plt.plot(x_,meta_means_ransac,color='orange')
    plt.plot(x_,meta_means_huber,color='purple')
    plt.plot(x_,meta_means_erm,color='cyan')
    plt.plot(x_,meta_means_crr,color='blue')
    plt.plot(x_,meta_means_stir,color='teal')
    plt.plot(x_,meta_means_quantile,color='gray')
    # plt.fill_between(x_,meta_means_subq-meta_std_subq,meta_means_subq+meta_std_subq,color='black',alpha=0.5)
    # plt.fill_between(x_,meta_means_sever-meta_std_sever,meta_means_sever+meta_std_sever,color='green',alpha=0.5)
    # plt.fill_between(x_,meta_means_term-meta_std_term,meta_means_term+meta_std_term,color='red',alpha=0.5)
    # plt.fill_between(x_,meta_means_ransac-meta_std_ransac,meta_means_ransac+meta_std_ransac,color='orange',alpha=0.5)
    # plt.fill_between(x_,meta_means_huber-meta_std_huber,meta_means_huber+meta_std_huber,color='purple',alpha=0.5)
    # plt.fill_between(x_,meta_means_erm-meta_std_erm,meta_means_erm+meta_std_erm,color='cyan',alpha=0.5)
    plt.ylim([0, 3])
    tikzplotlib.save("fig-synthetic.tex")
    plt.show()