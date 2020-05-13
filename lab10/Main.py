import time
import numpy as np
from random import randrange
import matplotlib.pyplot as plt
from math import sqrt
import scipy as sp
import scipy.linalg

from scipy._lib.six import xrange

crout_list = []
doolitl_list = []
cholesky_list = []
scipy_list = []
matrix_size_list = []


def crout(M):
    n = len(M)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for k in range(0, n):
        U[k, k] = 1
        for j in range(k, n):
            sum0 = sum([L[j, s] * U[s, k] for s in range(0, j)])
            L[j, k] = M[j, k] - sum0

        for j in range(k + 1, n):
            sum1 = sum([L[k, s] * U[s, j] for s in range(0, k)])
            U[k, j] = (M[k, j] - sum1) / L[k, k]
    return L, U


def doolitl(M):
    n = len(M)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        for k in range(i, n):
            sum = 0;
            for j in range(i):
                sum += (L[i][j] * U[j][k]);
            U[i][k] = M[i][k] - sum;
        for k in range(i, n):
            if (i == k):
                L[i][i] = 1;
            else:
                sum = 0;
                for j in range(i):
                    sum += (L[k][j] * U[j][i]);
                L[k][i] = ((M[k][i] - sum) / U[i][i]);
    return L, U


def cholesky(M):
    n = len(M)
    L = np.zeros((n, n))

    for i in xrange(n):
        for k in xrange(i + 1):
            tmp_sum = sum(L[i][j] * L[k][j] for j in xrange(k))
            if i == k:
                L[i][k] = sqrt(M[i][i] - tmp_sum)
            else:
                L[i][k] = (1.0 / L[k][k] * (M[i][k] - tmp_sum))
    return L, L.T


def test():
    n = randrange(10, 100)
    A = np.random.rand(n, n)
    B = sp.array(np.dot(A, A.transpose()).tolist())

    s1_time = time.time()
    crout(B)
    e1_time = time.time()

    s2_time = time.time()
    doolitl(B)
    e2_time = time.time()

    s3_time = time.time()
    cholesky(B)
    e3_time = time.time()

    s4_time = time.time()
    sp.linalg.lu(A)
    e4_time = time.time()

    return e1_time - s1_time, e2_time - s2_time, e3_time - s3_time, e4_time - s4_time, n


def write_diagram(values, sizes, title):
    plt.plot(sizes, values, 'bo', label=title)
    plt.xlabel("Matrix size")
    plt.ylabel("Time")
    plt.title("Pomiary czasu metody rozkładu LU")
    plt.savefig(title)


def write_to_file():
    fd = open("test_results.txt", "w+")
    fd.write("Matrix size \t Crout Method \t Doolitl Method \t Cholesky Method \t Scipy Method\n")
    for i in range(500):
        fd.write(f"{matrix_size_list[i]} \t {crout_list[i]} \t {doolitl_list[i]} \t {cholesky_list[i]} \t {scipy_list[i]}\n")
    fd.close()


if __name__ == '__main__':
    for i in range(500):
        crout_time, doolitl_time, cholesky_time, scipy_time, n = test()
        crout_list.append(crout_time)
        doolitl_list.append(doolitl_time)
        cholesky_list.append(cholesky_time)
        scipy_list.append(scipy_time)
        matrix_size_list.append(n)

    write_to_file()
    write_diagram(crout_list, matrix_size_list, "Crout method")
    write_diagram(doolitl_list, matrix_size_list, "Doolitl method")
    write_diagram(cholesky_list, matrix_size_list, "Cholesky method")
    write_diagram(scipy_list, matrix_size_list, "Scipy method")
