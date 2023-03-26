import numpy as np
import numpy.linalg as LA
import sys
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
#np.random.seed(13)
from matplotlib import pyplot as plt
import csv

from sklearn.linear_model import HuberRegressor,TheilSenRegressor,LinearRegression, RANSACRegressor

def num_p_q(argsorted_arr,n,p,eps):
    m = int(n*(1-eps))
    pc = 0
    qc = 0
    for i in range(0,int(n*p)):
        if argsorted_arr[i] < m:
            pc += 1
        else:
            qc += 1
    return (pc,qc)

def deriv_in_p_q(argsorted_arr, L, x, n,p,eps):

    p_deriv = np.zeros((1,3))
    q_deriv = np.zeros((1,3))

    for i in range(0,int(n*p)):
        index = argsorted_arr[i]
        d_i = 2*x[index]*L[index]
        if index < n*(1-eps):
            p_deriv += d_i
        else:
            q_deriv += d_i
    return p_deriv,q_deriv

def avg_loss_in_p_q(argsorted_arr,L,n,p,eps):
    p_loss = 0
    q_loss = 0

    pc = 0
    qc = 0
    for i in range(0,int(n*p)):
        index = argsorted_arr[i]
        d_i = L[index]
        if index < n*(1-eps):
            p_loss += d_i
            pc += 1
        else:
            q_loss += d_i
            qc += 1
    return p_loss/(pc+qc),q_loss/(qc+pc)

def avg_loss_total_p_q(argsorted_arr,L,n,p,eps):
    p_loss = 0
    q_loss = 0

    pc = 0
    qc = 0
    for i in range(0,n):
        index = argsorted_arr[i]
        d_i = L[index]
        if index < n*(1-eps):
            p_loss += d_i
            pc += 1
        else:
            q_loss += d_i
            qc += 1
    return p_loss/pc,q_loss/qc

def cos_array(f_deriv, g_deriv):
    cos = np.zeros((f_deriv.shape[0],))
    #print(f_deriv[0])
    for i in range(len(cos)):
        cos[i] = np.dot(f_deriv[i],np.transpose(g_deriv))/(LA.norm(f_deriv[i],2)*LA.norm(g_deriv,2))
    return cos


if __name__ == "__main__":

    for _ in range(1):

        pq = 0.8
        n = 10000
        eps = 0.8
        x = np.random.normal(0,1,n)
        p_hat = np.zeros((n,))
        m = int(n*(1-eps))
        l = int(n*eps)
        p = np.zeros((int(n*(1-eps)),))
        q = np.zeros((int(n*(eps)),))

        M = int(n*pq*0.2)

        f = open('quadratic-regression-p.csv','w',newline='')
        writer = csv.writer(f)
        header = ['Px','Py']
        writer.writerow(header)

        for i in range(int(n*(1-eps))):
            t = np.random.normal(x[i]**2 - x[i] + 2,0.01,1)
            p_hat[i] = t
            p[i] = t
            writer.writerow([x[i],t[0]])
        f.close()

        f = open('quadratic-regression-q.csv','w',newline='')
        #writer = csv.writer(f)
        #header = ['Qx','Qy']
        #writer.writerow(header)
        for i in range(int(n*(1-eps)),n):
            t = np.random.normal(-1*(x[i]**2) +x[i] + 4, 0.01, 1)
            p_hat[i] = t
            #q[int(i-m)] = t
            #writer.writerow([x[i],t[0]])
        #f.close()
        

        #plt.scatter(x[:m],p)
        #plt.scatter(x[m:],q)
        #initialize theta

        X = np.zeros((n,3))
        X_norm = np.zeros((n,))
        for i in range(n):
            X[i,0] = x[i]**2
            X[i,1] = x[i]
            X[i,2] = 1
            X_norm[i] = LA.norm(X[i],2)
        
        P = X[:m,:]
        Q = X[m:,:]

        plt.hist(X[:,0], bins=50)
        plt.gca().set(title='Frequency Histogram', ylabel='Frequency');
        #plt.show()

        # huber = HuberRegressor(max_iter=3000).fit(X, p_hat)
        # h_pred = huber.predict(P)
        # h_err = LA.norm(h_pred-p_hat[:m],2)

        # print(f"Model:\t{_}\tHuber\th-error:\t{h_err}")
        # print(f"Model:\t{_}\tHuber\th-theta:\t{huber.coef_}")
        linear = LinearRegression().fit(X, p_hat)
        l_pred = linear.predict(P)
        l_err = LA.norm(l_pred-p_hat[:m],2)

        # print(f"Model:\t{_}\tLinear\tl-error:\t{l_err}")
        # print(f"Model:\t{_}\tLinear\tl-theta:\t{linear.coef_}")

        # ransac = RANSACRegressor().fit(X, p_hat)
        # r_pred = ransac.predict(P)
        # r_err = LA.norm(r_pred-p_hat[:m],2)

        # print(f"Model:\t{_}\tRANSAC\tr-error:\t{r_err}")
        #print(f"Model:\t{_}\tRANSAC\tr-theta:\t{ransac.coef_}")

        genie = LinearRegression().fit(P,p_hat[:m])
        g_pred = genie.predict(P)
        g_err = LA.norm(g_pred-p_hat[:m],2)

        L = LA.norm(np.matmul(np.transpose(X),X),2)

        #print(f"Model:\t{_}\tGenie\tg-error:\t{g_err}")
        genie_monrose = np.matmul(np.matmul(LA.inv(np.matmul(np.transpose(P),P)), np.transpose(P)),p_hat[:m])
        print(f"Genie Predicted:\t{genie_monrose}")

        print(f"Theoretical Predicted:\t{(2/L)*(1-eps)*np.array([1,-1,1]) + (2/L)*eps*np.array([-1,1,4])}")
        
        corrupted_monrose = np.matmul(np.matmul(LA.inv(np.matmul(np.transpose(X),X)), np.transpose(X)),p_hat)
        print(f"Corrupted Predicted:\t{corrupted_monrose}\n")

        x_test = np.linspace(-3,3,300)
        t = 0
        #theta = np.random.normal(0,10,3)
        #print(theta)
        theta = np.zeros((3,))
        d = np.dot(X,theta) - p_hat
        deriv = np.sum(2 * X * d[:, np.newaxis],axis=0)
        alpha = 1/(2*L)
        update = -1 * alpha * deriv
        theta = theta + update
        
        print(f"theta0 Predicted:\t{theta}")
        x_test = np.linspace(-3,3,300)
        #y = theta[0]*x_test**2 + theta[1]*x_test + theta[2]
        for k in range(0, 200):
            pred = np.matmul(X,theta)
            v = (pred - p_hat)**2
            v_abs = np.abs(pred-p_hat)
            p_err = LA.norm(np.matmul(P,theta)-p_hat[:m],2)
            v_hat = sorted(v)
            v_arg_hat = np.argsort(v)
            deriv = np.zeros((1,3))
            L = 0
            
            X_np = X[v_arg_hat[:int(n*pq)]]
            y_np = p_hat[v_arg_hat[:int(n*pq)]]
            d = np.dot(X_np,theta) - y_np
            deriv = np.sum(2 * X_np * d[:, np.newaxis],axis=0)

            #M_set = np.random.randint(int(n*pq), size=M)
            #X_m = X_np[M_set]
            #y_m = y_np[M_set]
            #d_m = np.dot(X_m,theta) - y_m
            #deriv_m = np.sum(2 * X_m * d_m[:, np.newaxis],axis=0)
            #print(deriv)
            # L = L * 2 / (n*pq)
            
            # g = 0
            # for i in range(n):
            #     g += max(0,t-v[i])
            # g /= (n*pq)
            # g = t - g
            

            L = LA.norm(np.matmul(np.transpose(X_np),X_np),2)
            #L = LA.norm(np.matmul(np.transpose(X_m),X_m),2

            alpha = 1/(2*L)
            update = -1 * alpha * deriv
            #update = -1 * alpha * deriv_m

            theta = theta + update
            if k % 1 == 0:
                print(num_p_q(v_arg_hat,n,pq,eps))
                print(f"theta:\t{theta}")
                #print(f"Model:\t{_}\tk:{k}\tSubQ\tp-error:\t{p_err}")
                #print(f"Model:\t{_}\tk:{k}\tSubQ\ttheta:\t{theta}")
        #print(f"Model:\t{_}\tSubQ\tp-error:\t{p_err}")
        #print(f"Deriv:\t{deriv}")

        
        y = theta[2]*x_test**2 + theta[1]*x_test + theta[0]
        #plt.plot(x_test,y)

        #print(theta)

    


    #plt.show()
    pass