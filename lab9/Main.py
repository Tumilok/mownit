import time
import copy
import numpy as np
from random import randrange
from tabulate import tabulate
import matplotlib.pyplot as plt

without_pivoting_list = []
with_pivoting_list = []
matrix_size_list = []

def gauss(A, B):
    n = len(A)
    for i in range(0, n - 1):
        for j in range(i + 1, n):
            to_mult = float(A[j][i]) / float(A[i][i])
            for k in range(i, n):
                A[j][k] -= to_mult * A[i][k]
            B[j] -= to_mult * B[i]

    res = [0] * n
    res[n - 1] = float(B[n - 1]) / float(A[n - 1][n - 1])
    for i in range(n - 2, -1, -1):
        sum = B[i]
        for j in range(i + 1, n):
            sum -= A[i][j] * res[j]
        res[i] = float(sum) / float(A[i][i])
    return res


def connect_matrices(A, B):
    i = 0
    for row in A:
        row.append(B[i])
        i += 1


def gauss_pivoting(A, B):
    connect_matrices(A, B)
    n = len(A)
    for i in range(n):
        for j in range(i, n):
            if abs(A[i][i]) < abs(A[j][i]):
                A[i], A[j] = A[j], A[i]

        for j in range(i + 1, n):
            to_mult = float(A[j][i]) / float(A[i][i])
            for k in range(i, n + 1):
                A[j][k] -= to_mult * A[i][k]

    res = [0] * n
    res[n - 1] = float(A[n - 1][n]) / float(A[n - 1][n - 1])
    for i in range(n - 1, -1, -1):
        temp = 0
        for j in range(i + 1, n):
            temp += A[i][j] * res[j]
        res[i] = float((A[i][n] - temp)) / float(A[i][i])
    return res


def get_random_matrix(n):
    return np.random.rand(n, n)


def get_random_vector(n):
    return np.random.rand(n)


def test():
    n = randrange(10, 300)
    A = get_random_matrix(n).tolist()
    B = get_random_vector(n).tolist()

    s1_time = time.time()
    gauss(copy.deepcopy(A), copy.deepcopy(B))
    e1_time = time.time()

    s2_time = time.time()
    gauss_pivoting(copy.deepcopy(A), copy.deepcopy(B))
    e2_time = time.time()

    return e1_time - s1_time, e2_time - s2_time, n


def write_diagram(values, sizes, title):
    plt.plot(sizes, values, 'bo', label="Gauss " + title)
    plt.xlabel("Matrix size")
    plt.ylabel("Time")
    plt.title("Pomiary czasu metody eliminacji Gaussa")
    plt.legend()
    plt.savefig(title)

def write_to_file():
    fd = open("test_results.txt", "w+")
    fd.write("With pivoting \t Without pivoting \t Matrix size\n")
    for i in range(500):
        fd.write(f"{matrix_size_list[i]} \t\t {with_pivoting_list[i]} \t {without_pivoting_list[i]}\n")
    fd.close()

if __name__ == '__main__':
    for i in range(500):
        without_pivot, with_pivot, n = test()
        without_pivoting_list.append(with_pivot)
        with_pivoting_list.append(with_pivot)
        matrix_size_list.append(n)

    write_to_file()
    write_diagram(without_pivoting_list, matrix_size_list, "without pivoting")
    write_diagram(with_pivoting_list, matrix_size_list, "with pivoting")
