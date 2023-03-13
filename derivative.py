import numpy as np
import numpy.linalg as LA
import sys
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
#np.random.seed(13)
from matplotlib import pyplot as plt

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

    pq = 0.9
    n = 20
    eps = 0.1
    x = np.random.normal(0,1,n)
    p_hat = np.zeros((n,))
    m = int(n*(1-eps))
    l = int(n*eps)
    p = np.zeros((int(n*(1-eps)),))
    q = np.zeros((int(n*(eps)),))
    for i in range(int(n*(1-eps))):
        t = np.random.normal(x[i]**2 - x[i] + 2,0.01,1)
        p_hat[i] = t
        p[i] = t
    for i in range(int(n*(1-eps)),n):
        t = np.random.normal(-1*(x[i]**2) +x[i] + 4, 0.01, 1)
        p_hat[i] = t
        q[int(i-m)] = t
    

    plt.scatter(x[:m],p)
    plt.scatter(x[m:],q)
    #initialize theta

    X = np.zeros((n,3))
    X_norm = np.zeros((n,))
    for i in range(n):
        X[i,0] = 1
        X[i,1] = x[i]
        X[i,2] = x[i]**2
        X_norm[i] = LA.norm(X[i],2)
    
    P = X[:m,:]
    Q = X[m:,:]

    #print(np.matmul(np.transpose(P),P))
    #print(np.matmul(np.transpose(Q),Q))

    theta = np.random.normal(0,1,3)
    #print(theta)

    x_test = np.linspace(-3,3,300)
    y = theta[0]*x_test**2 + theta[1]*x_test + theta[2]

    #plt.plot(x_test,y)

    #plt.show()

    # gradient updates

    for k in range(0, 2500):
        pred = np.matmul(X,theta)
        v = (pred - p_hat)**2
        v_abs = np.abs(pred-p_hat)
        v_hat = sorted(v)
        v_arg_hat = np.argsort(v)
        deriv = np.zeros((1,3))
        L = 0

        for i in range(int(n*pq)):
            x_i = X[v_arg_hat[i]]
            d_i = x_i*(np.dot(theta, x_i) - p_hat[v_arg_hat[i]])
            deriv = deriv + d_i
            #L = L + np.dot(x_i,x_i)
            #print(d_i)

        L = LA.norm(np.matmul(np.transpose(X),X))

        pn = 19
        deriv_p = np.zeros((1,3))
        deriv_p = deriv_p + X[pn]*(np.dot(theta,X[pn])-p_hat[v_arg_hat[pn]])
        los = v[pn]
        #print(f"Derivative {pn}:\t{deriv_p}")
        #print(f"Loss {pn}:\t{los}")

        #f_deriv = np.matmul(np.transpose(X),v_abs)
        f_deriv = np.zeros((n,3))
        for i in range(n):
            f_deriv[i] = 2*X[i]*(pred[i] - p_hat[i])

        print(num_p_q(v_arg_hat,n,pq,eps))
        print(f"Not in p-quantile:\t{sorted(v_arg_hat[int(n*(1-eps)):])}")
        print(f"ABS:\t{v_abs[:10]}\n\t{v_abs[10:]}")
        #print(f"cos:\t{f_deriv}")
        dpq = deriv_in_p_q(v_arg_hat,pred-p_hat,X,n,pq,eps)
        #print(f"Loss:\t{v[sorted(v_arg_hat[int(n*(1-eps)):])]}")
        avg_loss_in = avg_loss_in_p_q(v_arg_hat,v,n,pq,eps)
        #print(f"P loss:\t{avg_loss_in[0]:.3f}\tQ loss:\t{avg_loss_in[1]:.3f}")
        avg_loss_total = avg_loss_total_p_q(v_arg_hat,v,n,pq,eps)
        #print(f"P tota:\t{avg_loss_total[0]:.3f}\tQ tota:\t{avg_loss_total[1]:.3f}")
        #print(f"P deriv:\t{dpq[0]/(n*pq)}\tQ deriv:\t{dpq[1]/(n*pq)}")
        #print(f"P norm:\t{LA.norm(dpq[0]/(n*pq),2)}\tQ norm:\t{LA.norm(dpq[1]/(n*pq),2)}\n")
        deriv = deriv / (n*pq)
        #print(f"Deriv:\t{deriv}")
        alpha = 1/(L)
        #print(f"L/2:\t{L/2:.3f}")
        #print(f"deriv product:\t{np.dot(deriv_p,np.transpose(deriv))}")
        cos = cos_array(f_deriv,deriv)
        print(f"cos:\t{cos[:10]}\n\t{cos[10:]}")
        prod = np.multiply(np.multiply(cos,v_abs),X_norm)
        print(f"prod:\t{prod[:10]}\n\t{prod[10:]}")
        #print(f"arcos:\t{np.degrees(np.arccos(cos))}\n")
        update = -1 * alpha * deriv
        #print(update)

        theta = theta + update
        theta = theta[0]

        #print(theta)

    y = theta[0]*x_test**2 + theta[1]*x_test + theta[2]
    plt.plot(x_test,y)


    plt.show()
    pass