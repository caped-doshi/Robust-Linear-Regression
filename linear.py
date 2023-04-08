import numpy as np
import cvxpy as cp
from scipy.io import loadmat
from sklearn import linear_model
from sklearn.linear_model import HuberRegressor
from matplotlib import pyplot as plt
import tikzplotlib

from data_loader import *

from baselines.sever import SEVER
from baselines.term import TERM
from baselines.crr import CRR
from baselines.stir import stir
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

def LM(X, y):
    return np.matmul(np.linalg.pinv(X),y)

def addFeatureNoise(X,y,noise:float):
  noisyY = np.copy(y)
  noisyX = np.copy(X)
  for i in range(int(len(y) * (1-noise)), len(y)):
    noisyY[i] = 10000*y[i]
    noisyX[i,:] = 100*X[i,:]
  return noisyX, noisyY

def addNoise(X, y, noise: float): 
    noisyY = np.copy(y)
    d = 100
    #m_ = np.random.normal(4,4,d)
    b_ = np.random.normal(5,5)
    for i in range(int(len(y) * (1-noise)), len(y)):
        noisyY[i] = b_
        #noisyY[i] = np.random.normal(np.dot(np.transpose(m_),X[i,0:d]) + b_,0.01)
    return noisyY

if __name__ == "__main__":
    n = 2000
    d = 100

    meta_means_sever = []
    meta_means_term = []
    meta_means_subq = []
    meta_means_crr = []
    meta_means_genie = []
    meta_means_ransac = []
    meta_means_huber = []
    meta_means_erm = []
    meta_means_stir = []
    meta_std_sever = []
    meta_std_term = []
    meta_std_subq = []
    meta_std_crr = []
    meta_std_ransac = []
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
        for j in range(5):
            X,y = data_loader_drug()
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = .80)

            y_train_noisy = addNoise(X_train, y_train, eps)

            theta_sever = SEVER(X_train, y_train_noisy,iter=64)
            theta_term = TERM(X_train, y_train_noisy, -2, 0.01, 500)
            theta_erm = LM(X_train, y_train_noisy)
            theta_subq = SubQ(X_train,y_train_noisy, 1500, 1-eps)
            theta_huber = HuberRegressor(max_iter=3000).fit(X_train,y_train_noisy).coef_
            #theta_genie = LM(X_train[int(y_train.shape[0] * eps):], y_train[int(y_train.shape[0] * eps):])
            #ransac = RANSACRegressor().fit(X_train,y_train_noisy)
            theta_crr = CRR(X_train,y_train_noisy)
            #theta_stir = stir(X_train,y_train_noisy,500)

            loss_subq = calc_RMSE(y_test,theta_subq,X_test)
            loss_sever = calc_RMSE(y_test,theta_sever,X_test)
            loss_huber = calc_RMSE(y_test,theta_huber,X_test)
            loss_erm = calc_RMSE(y_test,theta_erm,X_test)
            #loss_ransac = np.sqrt(np.mean((ransac.predict(X_test) - y_test) ** 2))
            loss_term = calc_RMSE(y_test,theta_term,X_test)
            loss_crr = calc_RMSE(y_test,theta_crr,X_test)
            #loss_stir = calc_RMSE(y_test,theta_stir,X_test)
            #loss_genie = calc_RMSE(y_test,theta_genie,X_test)

            print(f"\n")
            print(f"epsilon:\t{eps}")
            print(f"subq loss:\t{loss_subq}")
            print(f"term loss:\t{loss_term}")
            print(f"sever loss:\t{loss_sever}")
            print(f"CRR loss:\t{loss_crr}")
            #print(f"STIR loss:\t{loss_stir}")
            print(f"huber loss:\t{loss_huber}")
            #print(f"ransac loss:\t{loss_ransac}")
            print(f"erm loss:\t{loss_erm}")
            #print(f"genie loss:\t{loss_genie}")


            # x_test = np.linspace(np.min(x),np.max(x),300)
            # y_sever = theta_sever[0]*x_test + theta_sever[1]
            # y_huber = theta_huber[0]*x_test + theta_huber[1]
            # y_erm = theta_erm[0]*x_test + theta_erm[1]
            # y_term = theta_term[0]*x_test + theta_term[1]
            # y_subq = theta_subq[0]*x_test + theta_subq[1]
            # y_genie = theta_genie[0]*x_test + theta_genie[1]

            means_subq.append(loss_subq)
            #means_ransac.append(loss_ransac)
            means_term.append(loss_term)
            #means_genie.append(loss_genie)
            means_sever.append(loss_sever)
            means_huber.append(loss_huber)
            means_erm.append(loss_erm)
            means_crr.append(loss_crr)
            #means_stir.append(loss_stir)
        
        a = 0.5 

        mean_erm = np.mean(np.float32(means_erm))
        std_erm = np.std(np.float32(means_erm))
        meta_means_erm.append(mean_erm)
        meta_std_erm.append(std_erm)
        print(f"ERM:\t{mean_erm:.3f}_({std_erm:.4f})")

        mean_crr = np.mean(np.float32(means_crr))
        std_crr = np.std(np.float32(means_crr))
        meta_means_crr.append(mean_crr)
        meta_std_crr.append(std_crr)
        print(f"CRR:\t{mean_crr:.3f}_({std_crr:.4f})")

        # mean_stir = np.mean(np.float32(means_stir))
        # std_stir = np.std(np.float32(means_stir))
        # meta_means_stir.append(mean_stir)
        # meta_std_stir.append(std_stir)
        # print(f"Stir:\t{mean_stir:.3f}_({std_stir:.4f})")

        mean_huber = np.mean(np.float32(means_huber))
        std_huber = np.std(np.float32(means_huber))
        meta_means_huber.append(mean_huber)
        meta_std_huber.append(std_huber)
        print(f"Huber:\t{mean_huber:.3f}_({std_huber:.4f})")

        # mean_ransac = np.mean(np.float32(means_ransac))
        # std_ransac = np.std(np.float32(means_ransac))
        # meta_means_ransac.append(mean_ransac)
        # meta_std_ransac.append(std_ransac)
        # print(f"Ransac:\t{mean_ransac:.3f}_({std_ransac:.4f})")

        mean_term = np.mean(np.float32(means_term))
        std_term = np.std(np.float32(means_term))
        meta_means_term.append(mean_term)
        meta_std_term.append(std_term)
        print(f"Term:\t{mean_term:.3f}_({std_term:.4f})")

        mean_sever = np.mean(np.float32(means_sever))
        std_sever = np.std(np.float32(means_sever))
        meta_means_sever.append(mean_sever)
        meta_std_sever.append(std_sever)
        print(f"Sever:\t{mean_sever:.3f}_({std_sever:.4f})")

        mean_subq = np.mean(np.float32(means_subq))
        std_subq = np.std(np.float32(means_subq))
        meta_means_subq.append(mean_subq)
        meta_std_subq.append(std_subq)
        print(f"SubQuantile:\t{mean_subq:.3f}_({std_subq:.4f})")

    meta_means_subq = np.float32(meta_means_subq)
    meta_means_sever = np.float32(meta_means_sever)
    meta_means_term = np.float32(meta_means_term)
    meta_means_crr = np.float32(meta_means_crr)
    #meta_means_stir = np.float32(meta_means_stir)
    #meta_means_ransac = np.float32(meta_means_ransac)
    meta_means_huber = np.float32(meta_means_huber)
    meta_means_erm = np.float32(meta_means_erm)

    meta_std_subq = np.float32(meta_std_subq)
    meta_std_sever = np.float32(meta_std_sever)
    meta_std_term = np.float32(meta_std_term)
    meta_std_crr = np.float32(meta_std_crr)
    #meta_std_stir = np.float32(meta_std_stir)
    #meta_std_ransac = np.float32(meta_std_ransac)
    meta_std_huber = np.float32(meta_std_huber)
    meta_std_erm = np.float32(meta_std_erm)

    plt.plot(x_,meta_means_subq, color='black')
    plt.plot(x_,meta_means_sever,color='green')
    plt.plot(x_,meta_means_term,color='red')
    #plt.plot(x_,meta_means_ransac,color='orange')
    plt.plot(x_,meta_means_huber,color='purple')
    plt.plot(x_,meta_means_erm,color='cyan')
    plt.plot(x_,meta_means_crr,color='brown')
    #plt.plot(x_,meta_means_stir,color='pink')
    # plt.fill_between(x_,meta_means_subq-meta_std_subq,meta_means_subq+meta_std_subq,color='black',alpha=0.5)
    # plt.fill_between(x_,meta_means_sever-meta_std_sever,meta_means_sever+meta_std_sever,color='green',alpha=0.5)
    # plt.fill_between(x_,meta_means_term-meta_std_term,meta_means_term+meta_std_term,color='red',alpha=0.5)
    # plt.fill_between(x_,meta_means_ransac-meta_std_ransac,meta_means_ransac+meta_std_ransac,color='orange',alpha=0.5)
    # plt.fill_between(x_,meta_means_huber-meta_std_huber,meta_means_huber+meta_std_huber,color='purple',alpha=0.5)
    # plt.fill_between(x_,meta_means_erm-meta_std_erm,meta_means_erm+meta_std_erm,color='cyan',alpha=0.5)
    plt.ylim([0, 3])
    tikzplotlib.save("noisy-synthetic-linear-regression.tex")
    plt.show()