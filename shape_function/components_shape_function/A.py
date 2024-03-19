from shape_function.shape_function import p
import numpy as np


# Для удобства
def functional_matrix(point):
    return np.dot(p(point), p(point).T)


# Вычисление производной обратной матрицы
def inverse_dif(dA, A):
    return -np.dot(np.linalg.inv(A), np.dot(dA, np.linalg.inv(A)))


def A(point, w, all_points):
    size = len(p(point))
    A_local = np.zeros((size, size))
    for i in range(len(w)):
        A_local += w[i] * functional_matrix(all_points[i])

    return np.array(A_local)


def dA_dx(point, dw, all_points):
    size = len(p(point))
    A_local = np.zeros((size, size))
    for i in range(len(dw)):
        A_local += dw[i] * functional_matrix(all_points[i])

    return np.array(A_local)


def dA_dy(point, dw, all_points):
    size = len(p(point))
    A_local = np.zeros((size, size))
    for i in range(len(dw)):
        A_local += dw[i] * functional_matrix(all_points[i])

    return np.array(A_local)



