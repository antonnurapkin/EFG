from components_shape_function.weight_function_first import weight_func, d_weight_func, d2_weight_func
from components_shape_function.p import p
import numpy as np


def functional_matrix(point):
    return np.dot(p(point), p(point).T)


def inverse_dif(dA, A):
    return -np.dot(np.linalg.inv(A), np.dot(dA, np.linalg.inv(A)))


def A(point, all_points):
    size = len(p(point))
    A_local = np.zeros((size, size))
    for point_i in all_points:
        A_local += weight_func(x=point.x, x_i=point_i.x) * weight_func(x=point.y, x_i=point_i.y) \
                   * functional_matrix(point_i)

    return np.array(A_local)


def dA_dx(point, all_points):
    size = len(p(point))
    A_local = np.zeros((size, size))
    for point_i in all_points:
        A_local += d_weight_func(x=point.x, x_i=point_i.x) * weight_func(x=point.y, x_i=point_i.y) \
                   * functional_matrix(point_i)

    return np.array(A_local)


def dA_dy(point, all_points):
    size = len(p(point))
    A_local = np.zeros((size, size))
    for point_i in all_points:
        A_local += weight_func(x=point.x, x_i=point_i.x) * d_weight_func(x=point.y, x_i=point_i.y) \
                   * functional_matrix(point_i)

    return np.array(A_local)


def d2A_dx2(point, all_points):
    size = len(p(point))
    A_local = np.zeros((size, size))
    for point_i in all_points:
        A_local += d2_weight_func(x=point.x, x_i=point_i.x) * weight_func(x=point.y, x_i=point_i.y) \
                   * functional_matrix(point_i)

    return np.array(A_local)


def d2A_dy2(point, all_points):
    size = len(p(point))
    A_local = np.zeros((size, size))
    for point_i in all_points:
        A_local += weight_func(x=point.x, x_i=point_i.x) * d2_weight_func(x=point.y, x_i=point_i.y) \
                   * functional_matrix(point_i)

    return np.array(A_local)


def d2A_dydx(point, all_points):
    size = len(p(point))
    A_local = np.zeros((size, size))
    for point_i in all_points:
        A_local += d_weight_func(x=point.x, x_i=point_i.x) * d_weight_func(x=point.y, x_i=point_i.y) \
                   * functional_matrix(point_i)

    return np.array(A_local)



