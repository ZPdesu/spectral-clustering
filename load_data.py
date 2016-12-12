from numpy import *


def load_data():
    X = []

    with open('pattern.txt') as fr:
        for i in range(1000):
            row = fr.readline().split()
            row[0] = float(row[0])
            row[1] = float(row[1])
            X.append(row)
    X = array(X)
    save('X_array.npy', X)
    return X


def load_data2():
    X = []

    with open('pattern2.txt') as fr:
        for i in range(200):
            row = fr.readline().split()
            row[0] = float(row[0])
            row[1] = float(row[1])
            X.append(row)
    X = array(X)
    save('new_array.npy', X)

if '__main__' == __name__:
    #load_data()
    load_data2()

