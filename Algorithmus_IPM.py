# algorithm 1 from thesis (\ref{alg})

import numpy as np
import random

primal_res = []
dual_res = []
steplength = []
obj_func = []
# IMPORTANT: dimensions are never checked in the functions!
# Ensure that G is pos def symmetric (n x n), A is (m x n),
# y, b and lambda have length m, x and c have length n

# We use the heuristic from Nocedal (p.484) to find a starting point, given an arbitrary point (x_bar,y_bar,lam_bar):
def start(G, A, b, c, x_bar, y_bar, lam_bar):
    n = len(x_bar)
    m = len(y_bar)
    affines = linSolver1(G, A, b, c, x_bar, y_bar, lam_bar, 0, 0)
    #print(affines)
    y_aff = affines[n:n + m]
    lam_aff = affines[n + m:]
    y_0 = np.maximum(np.ones(m), np.abs(y_bar + y_aff))  # component wise
    lam_0 = np.maximum(np.ones(m), np.abs(lam_bar + lam_aff))
    return predictorCorrector(G, A, b, c, x_bar, y_0, lam_0)


# main algorithm:
def predictorCorrector(G, A, b, c, x, y, lam):
    n = len(x)
    m = len(y)
    iteration = np.concatenate((x, y, lam), axis=None)
    for i in range(1, 15):  # 15 iterations usually are enough for the iterations to not change anymore
        # (even 10 would be sufficient)
        # I've also tried 100 and 1000 iterations without any change
        objective = np.matmul(np.matmul(iteration[:n],G),iteration[:n]) + np.matmul(c,iteration[:n])
        obj_func.append(objective)
        affines = np.round(linSolver1(G, A, b, c, iteration[:n], iteration[n:n+m], iteration[n+m:], 0,
                             0),decimals=5)  # as sigma is 0 anyways, we do not have to compute mu before this step
        y_aff = affines[n:n + m]  # left border included, right border excluded!
        lam_aff = affines[n + m:]
        mu = np.matmul(iteration[n:n+m], iteration[n+m:]) / m
        alpha_aff = affineAlphaSolver(iteration[n:n+m], iteration[n+m:], y_aff, lam_aff)
        if alpha_aff <= 0 or alpha_aff > 1:
            print("failed")
            break
        mu_aff = np.matmul((iteration[n:n+m] + alpha_aff * y_aff).transpose(), iteration[n+m:] + alpha_aff * lam_aff) / m
        sigma = pow(mu_aff / mu, 3)
        deltas = np.round(np.nan_to_num(linSolver2(G, A, b, c, iteration[:n], iteration[n:n + m], iteration[n + m:],
                                          np.round(y_aff, decimals=7), np.round(lam_aff, decimals=7), sigma, mu)),decimals=5)
        delta_y = deltas[n:n + m]
        delta_lam = deltas[n + m:]
        tau = 1-(1/(2**i)) # converges to 1 to accelerate convergence as proposed in Nocedal
        alpha_hat = min(primalDualAlpha(iteration[n:n+m], delta_y, tau), primalDualAlpha(iteration[n+m:], delta_lam, tau))
        if alpha_hat <= 0 or alpha_hat > 1:
            print("failed")
            break
        steplength.append(alpha_hat)
        iteration = iteration + alpha_hat * deltas
        print(iteration[:n])
    print(primal_res)
    print(dual_res)
    print(steplength)
    print(obj_func)
    return iteration


# the following function solves problem 16.58 from Nocedal for delta_x, delta_y and delta_lambda:
def linSolver1(G, A, b, c, x, y, lam, sig, mu):  # Warning: dimensions are not checked!
    m = len(y)
    biglam = np.diag(lam)
    bigy = np.diag(y)
    rd = np.matmul(G, x) - np.matmul(A.transpose(), lam) + c
    rp = np.matmul(A, x) - y - b
    # normal equations form matrix and right hand side:
    N = G + np.matmul(np.matmul(A.transpose(), np.linalg.inv(bigy)), np.matmul(biglam, A))
    d = -rd + np.matmul(np.matmul(A.transpose(), np.linalg.inv(bigy)),
                        np.matmul(biglam, -rp - y + sig*mu*np.matmul(np.linalg.inv(biglam), np.ones(m))))
    # we want to solve N*delta_x = d with Cholesky
    L = np.linalg.cholesky(N)
    helper_vector = np.linalg.solve(L,d)
    delta_x = np.linalg.solve(L.transpose(),helper_vector)
    delta_y = np.matmul(A, delta_x) + rp
    delta_lam = np.matmul(np.linalg.inv(bigy), -np.matmul(biglam,delta_y) - np.matmul(np.matmul(biglam, bigy), np.ones(m))
                          + sig*mu*np.ones(m))
    deltas = np.concatenate((delta_x,delta_y,delta_lam))
    return deltas
# efficiency compared to normal (Gauss) solver is not estimated!

# the following function solves problem 16.67 from Nocedal for delta_x, delta_y and delta_lambda:
def linSolver2(G, A, b, c, x, y, lam, y_aff, lam_aff, sig, mu):
    m = len(y)
    biglam = np.diag(lam)
    bigy = np.diag(y)
    biglamaff = np.diag(lam_aff)
    bigyaff = np.diag(y_aff)
    rd = np.matmul(G, x) - np.matmul(A.transpose(), lam) + c
    rp = np.matmul(A, x) - y - b
    primal_res.append(np.linalg.norm(rp))
    dual_res.append(np.linalg.norm(rd))
    # normal equations form matrix and right hand side:
    N = G + np.matmul(np.matmul(A.transpose(), np.linalg.inv(bigy)), np.matmul(biglam, A))
    d = -rd + np.matmul(np.matmul(A.transpose(), np.linalg.inv(bigy)),
                        np.matmul(biglam, -rp - y
                                  - np.matmul(np.matmul(np.linalg.inv(biglam), biglamaff),
                                              np.matmul(bigyaff, np.ones(m))) + sig * mu *
                                  np.matmul(np.linalg.inv(biglam), np.ones(m))))
    # we want to solve N*delta_x = d with Cholesky
    L = np.linalg.cholesky(N)
    helper_vector = np.linalg.solve(L, d)
    delta_x = np.linalg.solve(L.transpose(), helper_vector)
    delta_y = np.matmul(A, delta_x) + rp
    delta_lam = np.matmul(np.linalg.inv(bigy),
                          -np.matmul(biglam, delta_y) - np.matmul(np.matmul(biglam, bigy), np.ones(m))
                          + sig * mu * np.ones(m) - np.matmul(np.matmul(biglamaff, bigyaff), np.ones(m)))
    deltas = np.concatenate((delta_x, delta_y, delta_lam))
    return deltas


# the following function returns the maximum alpha in (0,1], for which (y,lambda) + alpha*(y_aff,lambda_aff) >=0:
def affineAlphaSolver(y, lam, y_aff, lam_aff):
    point = np.concatenate((y, lam), axis=None)
    direction = np.concatenate((y_aff, lam_aff), axis=None)
    alpha = []
    for i in range(0, len(point)):
        if direction[i] >= 0:
            alpha.append(1)
        else:
            alpha.append(min(1,-point[i] / direction[i]))
    return min(alpha)


# the following function can be used to compute either of the values alpha_primal or alpha_dual, using y or lambda
# respectively (as in Nocedal, 16.66):
def primalDualAlpha(var, delta_var, tau):
    alpha = []
    for i in range(0, len(var)):
        if delta_var[i] >= 0:
            alpha.append(1)
        else:
            alpha.append((-1 * tau * var[i]) / delta_var[i])
    return min(min(alpha), 1)


# testing area
#G = np.array([[2, 0], [0, 2]])
#A = np.array([[1, 0], [-1, 0], [0,1],[0,-1]])
#y = np.array([1, 2, 3,4])
#x = np.array([1, 1])
#c = np.array([1, 1])
#b = np.array([1,2, 3, 4])
#lam = np.array([1, 1, 1,1])

#print(predictorCorrector(G, A, b, c, np.array([-9 / 7, -43 / 31]), np.array([9 / 7, 12 / 7, 105 / 31, 112 / 31]),
                         #np.array([16 / 7, 13 / 7, 66 / 31, 59 / 31])))
# with this data I calculated all steps by hand and 'proved' correctness of the algorithm that way
