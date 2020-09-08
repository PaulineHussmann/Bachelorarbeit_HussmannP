import numpy as np
import scipy.linalg
from block import block # used for convenient block matrices, requires scipy @author Brandon Amos
import Algorithmus_IPM

p = 2    # dimension of oberved parameter theta and of X_i
M = 10   # number of observations of parmeter theta
N = 100  # number of observations of Y and X
sigma = 1

# generate uniformly distributed realizations of theta (what we observed)
thetas = []
# for each theta we want one intercept term and one regressor term, so we can use the model
# M(X,theta) = theta_0 + theta_1 * X_1 and X is one dimensional
for i in range(1, 2 * M + 1):
    theta = np.round(np.random.uniform(-1, 1), decimals=5)
    thetas.append(theta)

print(thetas)


# generate normally distributed realizations of Y (the model observations)
Ys = []
for j in range(1, N + 1):
    y = np.round(np.random.normal(0, sigma), decimals=5)
    Ys.append(y)

print(Ys)

# generate the input X:
Xs = []
for l in range(1, N + 1):
    x = np.round(np.random.uniform(-1, 1), decimals=5)
    Xs.append(x)

print(Xs)

# model: Y_i = X_i * theta_j for each theta_j
# sum_{j=1}^M sum_{i=1}^N (Y_i - X_i * theta_j)^2 / 2sigma

# we use the implemented algorithm and therefore calculate:
pre = np.reshape(np.concatenate((np.ones(N), np.asarray(Xs))), (2,N))
# G_2 = 1/(sigma*M) * np.matmul(pre, pre.transpose())
G_2 = np.array([[10.0, 0.26272100000000004], [0.26272100000000004, 3.626079655489999]])
print(G_2)
c_2 = 1/(sigma*M) * np.matmul(pre,Ys)
print(c_2)
G = scipy.linalg.block_diag(G_2,G_2,G_2,G_2,G_2,G_2,G_2,G_2,G_2,G_2,np.zeros((M*p,M*p)))
c = np.concatenate((np.tile(c_2,M),np.zeros(M*p)))
pre2 = -1/M * np.ones(2*M)
A = np.vstack((np.concatenate((np.zeros(p*M),-1/M*np.ones(p*M))),np.hstack((np.identity(p*M),np.identity(p*M))),np.hstack((-np.identity(p*M),np.identity(p*M)))))
epsilon = [10**-6]
b = np.concatenate(( -1*np.asarray(epsilon), np.asarray(thetas),-1*np.asarray(thetas)))
# we start with an arbitrary point x_0,y_0,lambda_0 and use the implemented algorithm to find a starting point
# and solve with the implemented predictor-corrector algorithm
x_0 = np.ones(2*M*p)
y_0 = np.ones(2*M*p+1)
lam_0 = np.ones(2*M*p+1)
print(Algorithmus_IPM.start(G,A,b,c,x_0,y_0,lam_0))