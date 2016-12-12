# -*- coding: utf-8 -*-
import numpy as np
from k_means import *
import math
from scipy.linalg import *
import matplotlib.pyplot as plt


def import_data(file_name):
    X = np.load(file_name)
    return X


def clustering(file_name, k, sigma, c):
    X = import_data('new_array.npy')
    W = np.zeros((200, 200))
    D = np.zeros((200, 200))
    U = np.zeros((200, c))
    T = np.zeros((200, c))
    temp = np.zeros((200, 200))
    for i in range(200):
        distance = np.zeros(200)
        for j in range(200):
            distance[j] = cal_distance(X[i],X[j])
        distance[i] = 1000
        k_near = np.argsort(distance)
        for l in range(k):
            W[i, k_near[l]] = math.exp(-np.power(distance[k_near[l]], 2) / (2 * np.power(sigma, 2)))
    W = (W.T + W) / 2
    for i in range(200):
        D[i, i] = np.sum(W[i])
        temp[i, i] = np.power(D[i, i], -1/2.0)
    L = D - W
    L_sym = np.dot(np.dot(temp, L), temp)
    eigvalues, eigvectors = eig(L_sym)
    arg_val = np.argsort(eigvalues)
    for i in range(c):
        U[:, i] = eigvectors[:, arg_val[i]]
    for i in range(200):
        T[i] = U[i] / np.linalg.norm(U[i])
    np.save('T_array.npy', T)
    cluster = Kmeans('T_array.npy', 2, 200)
    accuracy = cluster.main()
    return accuracy


if '__main__' == __name__:
    accuracy1 = []
    accuracy2 = []

    for i in range(3, 12):
        temp = clustering('new_array.npy', i, 1, 2)
        accuracy1.append(temp)
    plt.plot(range(3, 12), accuracy1)
    plt.show()

    for j in np.arange(0.5, 5, 0.5):
        temp = clustering('new_array.npy', 5, j, 2)
        accuracy2.append(temp)
    plt.plot(np.arange(0.5, 5, 0.5), accuracy2)
    plt.show()
