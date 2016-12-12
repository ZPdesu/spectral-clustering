# -*- coding: utf-8 -*-
import numpy as np


def import_data(file_name):
    X = np.load(file_name)
    return X


def cal_distance(array_A, array_B):
    return np.sqrt(sum(np.power(array_A - array_B, 2)))


def cal_squared_error(means, A):
    temp = np.sum(np.power(means - A, 2), axis=1)
    print temp


# 仅仅谱聚类时候用到
def cal_accuracy(label):
    num = 0
    for i in range(100):
        if label[i] == 0:
            num += 1
    for j in range(100, 200):
        if label[j] == 1:
            num += 1
    if num < 100:
        num = 200 - num
    print '正确率为'
    accuracy = num / 200.0
    print accuracy
    return accuracy


class Kmeans:

    def __init__(self, file_name, k, iteration):
        self.X = import_data(file_name)
        self.k = k
        self.iteration = iteration

    def initial_means(self, n, init_means):
        min_x = []
        max_x = []
        for i in range(n):
            min_x.append(min(self.X[:, i]))
            max_x.append(max(self.X[:, i]))
        for i in range(self.k):
            for j in range(n):
                init_means[i, j] = (max_x[j] - min_x[j]) * np.random.rand() + min_x[j]

    def calculate_means(self, n, means, label, distance):
        dist_arr = np.zeros(self.k)
        for i in range(self.X.shape[0]):
            for j in range(self.k):
                dist_arr[j] = cal_distance(self.X[i], means[j])
            label[i] = dist_arr.argsort()[0]
            distance[i] = np.sort(dist_arr)[0]

        for i in range(self.k):
            mean_list = []
            for j in range(self.X.shape[0]):
                if label[j] == i:
                    mean_list.append(self.X[j])
            mean_arr = np.array(mean_list)
            means[i] = np.mean(mean_arr, axis=0)
        return means, label, distance

    def main(self):
        m, n = np.shape(self.X)
        label = np.zeros(m, int)
        distance = np.zeros(m)
        means = np.zeros((self.k, n))
        self.initial_means(n, means)
        for i in range(self.iteration):
            pre_means = means.copy()
            self.calculate_means(n, means, label, distance)
            if (pre_means == means).all():
                break
        #print '均值是:'
        #print means, '\n'
        #print '每一类的数量:'
        #print np.bincount(label), '\n'
        # 仅仅谱聚类时候用到
        return cal_accuracy(label)






if '__main__' == __name__:
    A = np.array([[1, -1], [5.5, -4.5], [1,4], [6, 4.5], [9, 0.0]])
    k = Kmeans('X_array.npy', 5, 100)
    k.main()
