import numpy as np
import scipy.linalg
import Algorithmus_IPM
from matplotlib import pyplot
import pandas as pd

p = 1  # dimension of oberved parameter theta and of X_i
M = 1000  # number of observations of parameter theta
# N = 4110  # number of observations of Y
sigma = 1

# generate uniformly distributed realizations of theta (what we observed)
thetas = np.random.uniform(-1, 1, M)
# print(thetas)
bins = np.linspace(-1, 1, 20)
pyplot.hist(thetas, bins)
pyplot.xlabel("Realisierung von theta ~ U(-1,1)")
pyplot.ylabel("Häufigkeit")
pyplot.show()

# M(X,theta) = theta

# dataset Y (height of female students at FU Berlin: https://userpage.fu-berlin.de/soga/200/2010_data_sets/students.csv):
data = pd.read_csv('students.csv')
data.query('gender == "Female"', inplace=True)
Y = data['height']

bins_2 = np.linspace(140, 200, 100)
pyplot.hist(Y, bins_2)
pyplot.title("Körpergröße von weiblichen Studierenden")
pyplot.xlabel("Körpergröße in cm")
pyplot.ylabel("Häufigkeit")
pyplot.show()

# model: y_i = theta_j + eta (normally distribuded noise) for each theta_j
# \frac{1}{NM} \sum_{i=1}^N \sum_{j=1}^M \frac{-1}{2} |y_i - \theta_j|^2

# we use the implemented algorithm and therefore calculate:

G_2 = 1 / (sigma * M)
c_2 = 1 / (sigma * M) * sum(Y)
# print(c_2)
G_array = np.concatenate((np.tile(G_2, M), np.zeros(M)))
# print(G_array)
G = np.diag(G_array)
c = np.concatenate((np.tile(c_2, M), np.zeros(M)))
pre = -1 / M * np.ones(2 * M)
A = np.vstack((np.concatenate((np.zeros(M), -1 / M * np.ones(M))), np.hstack((np.identity(M), np.identity(M))),
               np.hstack((-np.identity(M), np.identity(M)))))
epsilon = [10**-2] # we try epsilon = 1e-02 to 1e10 to view differences
b = np.concatenate((-1 * np.asarray(epsilon), np.asarray(thetas), -1 * np.asarray(thetas)))
# we start with an arbitrary point x_0,y_0,lambda_0 and use the implemented algorithm to find a starting point
# and solve with the implemented predictor-corrector algorithm
x_0 = np.concatenate((1 / M * np.ones(M * p), np.zeros(M * p)))
y_0 = np.ones(2 * M * p + 1)
lam_0 = np.ones(2 * M * p + 1)
solution = Algorithmus_IPM.start(G, A, b, c, x_0, y_0, lam_0)
# print(solution[:M])

bins_3 = np.linspace(min(solution[:M]), max(solution[:M]), 20)
pyplot.hist(solution[:M], bins_3)
pyplot.xlabel("Optimales theta^*")
pyplot.ylabel("Häufigkeit")
pyplot.show()
