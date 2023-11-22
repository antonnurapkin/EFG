from components_shape_function.weight_function_first import weight_func, d_weight_func, d2_weight_func
from components_shape_function.p import p
import numpy as np


def functional_matrix(point):
    return np.dot(p(point), p(point).T)


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


def d2A_dx2(point, d2w, all_points):
    size = len(p(point))
    A_local = np.zeros((size, size))
    for i in range(len(d2w)):
        A_local += d2w[i] * functional_matrix(all_points[i])

    return np.array(A_local)


def d2A_dy2(point, d2w, all_points):
    size = len(p(point))
    A_local = np.zeros((size, size))
    for i in range(len(d2w)):
        A_local += d2w[i] * functional_matrix(all_points[i])

    return np.array(A_local)


def d2A_dydx(point, d2w, all_points):
    size = len(p(point))
    A_local = np.zeros((size, size))
    for i in range(len(d2w)):
        A_local += d2w[i] * functional_matrix(all_points[i])

    return np.array(A_local)



