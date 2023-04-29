import numpy as np
from matplotlib import pyplot as plt
from Subquantile import SubQ
from term import TERM
from sever import SEVER
from noise import *
from data_loader import *
from sklearn.model_selection import train_test_split
import tikzplotlib

if __name__ == "__main__":
    n = 1000
    d = 2
    X, y, w, b = gaussian(n,d)
    noise_fn = addAdaptiveNoise
    eps = 0.2

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8)
    X_train, y_train_noisy = noise_fn(X_train, y_train, eps, w, b)

    n = len(y_train)
    partition_number = int(n*(1-eps))
    clean_pos = np.where(y_train_noisy[0:partition_number]==1)[0]
    clean_neg = np.where(y_train_noisy[0:partition_number]==-1)[0]
    noisy_pos = partition_number + np.where(y_train_noisy[partition_number:n]==1)[0]
    noisy_neg = partition_number + np.where(y_train_noisy[partition_number:n]==-1)[0]
    plt.scatter(X_train[clean_pos,0],X_train[clean_pos,1])
    plt.scatter(X_train[clean_neg,0],X_train[clean_neg,1])
    plt.scatter(X_train[noisy_pos,0],X_train[noisy_pos,1])
    plt.scatter(X_train[noisy_neg,0],X_train[noisy_neg,1])

    theta_subq = SubQ(X_train,np.int32(y_train_noisy),32,1-eps, 0)
    theta_term = TERM(X_train,np.int32(y_train_noisy),-2,0.1,10000,0)
    theta_sever = SEVER(X_train,np.int32(y_train_noisy),0,0.01,64)
    subq_acc = (len(y_test)-np.count_nonzero(np.sign((X_test@theta_subq)) - y_test)) / len(y_test)
    term_acc = (len(y_test)-np.count_nonzero(np.sign((X_test@theta_term)) - y_test)) / len(y_test)
    sever_acc = (len(y_test)-np.count_nonzero(np.sign((X_test@theta_sever)) - y_test)) / len(y_test)
    print(f"Subq Acc:\t{subq_acc:.3f}")
    print(f"TERM Acc:\t{term_acc:.3f}")
    print(f"SEVER Acc:\t{sever_acc:.3f}")
    
    x = np.linspace(-10,10,100)
    y_subq = (-theta_subq[2] - theta_subq[0]*x)/theta_subq[1]
    y_term = (-theta_term[2] - theta_term[0]*x)/theta_term[1]
    y_sever = (-theta_sever[2] - theta_sever[0]*x)/theta_sever[1]
    y_true = (-b - w[0]*x)/w[1]
    plt.plot(x,y_subq,color='black')
    plt.plot(x,y_term,color='red')
    plt.plot(x,y_sever,color='green')
    #plt.legend(['positive','negative','true','theta'])
    plt.xlim([-10,10])
    plt.ylim([-10,10])
    tikzplotlib.save("toy-synthetic.tex")
    plt.show()

    
