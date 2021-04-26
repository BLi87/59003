import os
import sys
import csv
import statistics
import scipy
import random
import math
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import model_selection
from sklearn import linear_model
from sklearn import metrics
from mpyc.runtime import mpc
import argparse
from mpyc.runtime import mpc
from mpyc.seclists import seclist
import mpyc.gmpy as gmpy2
JAX_ENABLE_X64=1


secint = mpc.SecInt()
secfxp = mpc.SecFxp()
def read_data(filePath):
    data =  pd.read_csv(filePath, header = None)
    newdata = pd.concat([data, data[:1200]])
    # newdata = data[:1000]
    X, y= data.iloc[:,0:-1].to_numpy(), data.iloc[:,-1].to_numpy()
    return X, y

"""
solves the following problem via ADMM:

    minimize   sum( log(1 + exp(-b_i*(a_i'w + v)) ) + m*mu*norm(w,1)
"""

async def ADMM_LR(A, b, Lambda, N, rho, ABSTOL = 1e-4, RELTOL = 1e-2):
    '''
    Solves Logistic Regression via ADMM
    
    :param A: feature matrix
    :param b: response vector
    :param N: number of subsystems to use to split the examples
    :param rho: augmented Lagrangian parameter
    :alpha: over-relaxation parameter (typical values for alpha are
            % between 1.0 and 1.8)
    return: vector x = (v,w)
    '''
    m, n = len(A), len(A[0])
    max_iter = 40
    
    x = []
    for _ in range(m):
        row = list(map(float, [0] * n))
        row = list(map(secfxp, row))
        x.append(row)
    
    ztemp = [0] * n
    z = list(map(float, ztemp))
    z = list(map(secfxp, z))
    

    u = []
    for _ in range(m):
        row = list(map(float, [0] * n))
        row = list(map(secfxp, row))
        u.append(row)

#     x = np.zeros(m * n).reshape(m, n)
#     z = np.zeros(1 * n).reshape(1, n)[0]
#     u = np.zeros(m * n).reshape(m, n)

        
    for i in range(max_iter):
        print("update", i)
        try:
            # startTime = time()
            (x, z, u, r, s) = await ADMM_LR_update(A, b, Lambda, m, n, rho, x, z, u) 
            # endTime = time()
            # print("time:", endTime - startTime)
            # termination checks
            """
            r_norm = np.sum(r**2)
            s_norm = np.sum(s**2)
            eps_pri = (m * n)**(0.5) * ABSTOL + RELTOL * (max(np.sum(x**2), np.sum(m * z**2)))**0.5
            eps_dual = (n)**(0.5) * ABSTOL + RELTOL * rho * (np.sum(u**2))**(0.5)/(m)**(0.5)
            if ((r_norm <= eps_pri**2) and (s_norm <= eps_dual**2)):
                break
            """
            r_norm = secfxp(0.0)
            for i in range(len(r)):
                for j in range(len(r[i])):
                    r_norm += mpc.pow(r[i][j], 2)

            s_norm = secfxp(0.0)
            for i in range(len(s)):
                    s_norm += mpc.pow(s[i], 2)

            # eps_pri
            x_sum = secfxp(0.0)
            for i in range(len(x)):
                for j in range(len(x[i])):
                    x_sum += mpc.pow(x[i][j], 2)

            z_sum = secfxp(0.0)
            for i in range(len(z)):
                    z_sum += m * mpc.pow(z[i], 2)

            eps_pri = mpc.max(x_sum, z_sum)
            eps_pri = math.pow(await mpc.output(eps_pri), 0.5) * RELTOL + (m * n)**(0.5) * ABSTOL
            eps_pri = secfxp(eps_pri)


            # eps_dual
            u_sum = secfxp(0.0)
            for i in range(len(u)):
                for j in range(len(u[i])):
                    u_sum += mpc.pow(u[i][j], 2)

            eps_dual = RELTOL * rho * math.pow(await mpc.output(u_sum), 0.5)/(m**0.5) + (n)**(0.5) * ABSTOL
            eps_dual = secfxp(eps_dual)

            bk = mpc.lt(mpc.pow(eps_pri, 2), r_norm) + mpc.lt(mpc.pow(eps_dual, 2), s_norm)
            if not await mpc.output(bk):
                break
        except:
            return (x, z, u)

    return (x, z, u)
    
    


async def ADMM_LR_update(A, b, Lambda, m, n, rho, x, z, u):
    '''
    Update parameters
    '''
    
    # x-update
    x_new = await serial_x_update(A, b, m, x, z, u, rho)
    # z-update with relaxation
    '''
    zold = z;
    x_hat = alpha*x + (1-alpha)*zold;
    ztilde = mean(x_hat + u,2);
    ztilde(2:end) = shrinkage( ztilde(2:end), (m*N)*mu/(rho*N) );
    '''
    
    """z_new = (x_new.sum(0) + u.sum(0)) / float(m)
    z_temp = abs(z_new)- Lambda / float(m * rho)
    z_new = np.sign(z_new) * z_temp * (z_temp > 0)"""
    
    z_new = []
    for i in range(len(x[0])):
        total = secfxp(float(0))
        for j in range(len(x)):
            total += x[j][i]
    
        for k in range(len(u)):
            total += u[k][i]
        total = total / secfxp(float(m))        
        z_new.append(total)
    
    z_temp = []
    for zi in z_new:
        z_temp.append(mpc.abs(zi) - secfxp(float(Lambda) / float(m * rho)))
    
    for i in range(len(z_new)):
        # z_temp * (z_temp > 0)
        zi = mpc.lt(secfxp(0), z_temp[i])
        zi = zi * z_temp[i]
        
        # sign(z_new)
        zi = zi * (1.0 - mpc.eq(0.0, z_new[i]))
        zi = zi * (mpc.lt(0.0, z_new[i]) - 0.5) * 2.0
        
        z_new[i] = zi

    # u-update
    """s = z_new - z
    r = x_new - np.ones(m).reshape(m, 1) * z_new
    u_new = u + r"""
    s = []
    for i in range(len(z)):
        s.append(z_new[i] - z[i])
    
    r = []
    for i in range(m):
        row = []
        for j in range(len(z_new)):
            row.append(x_new[i][j] - z_new[j])
        r.append(row)
    
    u_new = []
    for i in range(m):
        row = []
        for j in range(len(u[i])):
            row.append(u[i][j] + r[i][j])
        u_new.append(row)
    
    return (x_new, z_new, u_new, r, s)

async def serial_x_update(A, b, m, x, z, u, rho):
    '''
    Perform x_i update using L-BFGS
    '''
    x_new = []
    for i in range(m): 
#         ai = list(map(float, A[i]))
#         ai = list(map(secfxp, ai))
#         bi = secfxp(float(b[i]))
        ai = await mpc.output([a_i for a_i in A[i]])
        bi = await mpc.output(b[i])

        xi = await mpc.output([x_i for x_i in x[i]])
        ui = await mpc.output([u_i for u_i in u[i]])
        zi = await mpc.output([z_i for z_i in z])
        x_temp = scipy.optimize.minimize(update_x, xi, args=(ui, zi, rho, ai, bi), method= None, options={'disp': False})
#         x_new.append(x_temp.x)
        temp_list = x_temp.x.tolist()
        x_new.append(list(map(secfxp, temp_list)))
#     x_new = np.array(x_new)
    return x_new

@mpc.coroutine   
async def update_x(x, *args):
    '''
    Update x_i
    '''
    u, z, rho, a_i, b_i = args
    res = np.log(1 + np.e**(-b_i * np.dot(a_i, x))) + (rho / 2.0) * np.sum((x - z + u)**2)
#     first = secfxp(0)
#     x = x.tolist()
#     x = list(map(secfxp, x))
#     first = -b_i * mpc.in_prod(a_i, x)
#     first = np.log(1 + np.e**(await mpc.output(first)))
    
#     second = secfxp(0)
#     for i in range(len(x)):
#         second += (x[i] - z[i] + u[i])
#     second = (rho / 2.0) * mpc.pow(second, 2)
    
#     res = first + second
    return res

def error(y_true, y_pred):
    return metrics.log_loss(y_true, y_pred, eps=1e-15, normalize=True, sample_weight=None, labels=None)


def loss_function(A, b, x, rho):
    A = np.array(A)
    b = np.array(b)
    x = np.array(x)
    m, n = A.shape
    loss = 0
    for i in range(m):
        loss += np.log(1 + np.e**(-b[i] * np.dot(A[i,:], x))) + (rho/2.0) * np.sum((x)**2)
    return(loss)


async def main():
#     if sys.argv[1:]:
#         fileName = sys.argv[1]
#     else:
#         print("No Input File.")
#         return
    fileName = "../data/banknote.csv"
    print(fileName)
    
    await mpc.start()
    # Train/test split
    X, y = read_data(fileName)

    rnd = await mpc.transfer(random.randrange(2**31), senders=0)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.7, random_state=rnd)
    
    # Normal Logistic Regression
    # clf = linear_model.LogisticRegression(penalty='l1', random_state=0, max_iter=1000, solver = 'liblinear')
    # clf.fit(X_train, y_train)
    # error_train_skit = error(y_train, clf.predict(X_train))
    # error_test_skit = error(y_test, clf.predict(X_test))
    # print(f'scikit train error: {error_train_skit}')
    # print(f'scikit test error:  {error_test_skit}')
    
    # MPC
    secint = mpc.SecInt()
    secfxp = mpc.SecFxp()
    Labmda = np.sum(np.dot(X_train.T, y_train) ** 2) ** (0.5)
    Labmda = 0.1 / 3 * Labmda
    
    X = y = []
    for row in X_train:
        row = list(map(float, row))
        newRow = list(map(secfxp, row))
        X.append(newRow)
    
    y_train = list(map(float, y_train))
    y = list(map(secfxp, y_train))
    
    params = await ADMM_LR(X, y, Labmda, N = 1, rho = 1, ABSTOL = 1e-4, RELTOL = 1e-2)
    xsec = params[0]
    xfinal = []
    for i in range(len(xsec)):
        row = []
        for j in range(len(xsec[i])):
            row.append(await mpc.output(xsec[i]))
        xfinal.append(np.array(row))
    xfinal = np.array(xfinal)
        
    zfinal = [await mpc.output(z) for z in params[1]]

    pred_train = np.dot(X_train, zfinal)
    for i in range(len(pred_train)):
        if (1 / (1 + np.e ** (-pred_train[i]))) > 0.5:
            pred_train[i] = 1
        else:
            pred_train[i] = 0     
    
    pred_test = np.dot(X_test, zfinal)
    for i in range(len(pred_test)):
        if (1 / (1 + np.e ** (-pred_test[i]))) > 0.5:
            pred_test[i] = 1
        else:
            pred_test[i] = 0 
    
    error_train_mpyc = error(y_train, pred_train)
    error_test_mpyc = error(y_test, pred_test)
    
    print(f'MPC train error: {error_train_mpyc}')
    print(f'MPC test error:  {error_test_mpyc}')
    
    print("MPC Target loss: ",loss_function(X_train, y_train, zfinal, 1.0))
    print("MPC Target loss: ",loss_function(X_test, y_test, zfinal, 1.0))
    await mpc.shutdown()

if __name__ == '__main__':
    mpc.run(main())

