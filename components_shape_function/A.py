from EFG.components_shape_function.weight_function import weight_func, d_weight_func, d2_weight_func
from EFG.params import m
import numpy as np


def A(any_point, all_points):
    A_local = np.zeros((m + 2, m + 2))
    for point_i in all_points:
        A_local += weight_func(x=any_point.x, x_i=point_i.x) * weight_func(x=any_point.y, x_i=point_i.y) \
                   * np.array([[1,            point_i.x,              point_i.y],
                        [point_i.x,    point_i.x ** 2,         point_i.x * point_i.y],
                        [point_i.y,    point_i.x * point_i.y,  point_i.y ** 2]])

    return np.array(A_local)


def dA_dx(point, all_points):
    A_local = np.zeros((m + 2, m + 2))
    for point_i in all_points:
        A_local += d_weight_func(x=point.x, x_i=point_i.x) * weight_func(x=point.y, x_i=point_i.y) \
                   * np.array([[1,            point_i.x,              point_i.y],
                        [point_i.x,    point_i.x ** 2,         point_i.x * point_i.y],
                        [point_i.y,    point_i.x * point_i.y,  point_i.y ** 2]])
    return np.array(A_local)


def dA_dy(point, all_points):
    A_local = np.zeros((m + 2, m + 2))
    for point_i in all_points:
        A_local += weight_func(x=point.x, x_i=point_i.x) * d_weight_func(x=point.y, x_i=point_i.y) \
                   * np.array([[1,            point_i.x,              point_i.y],
                        [point_i.x,    point_i.x ** 2,         point_i.x * point_i.y],
                        [point_i.y,    point_i.x * point_i.y,  point_i.y ** 2]])
    return np.array(A_local)


def d2A_dx2(point, all_points):
    A_local = np.zeros((m + 2, m + 2))
    for point_i in all_points:
        A_local += d2_weight_func(x=point.x, x_i=point_i.x) * weight_func(x=point.y, x_i=point_i.y) \
                   * np.array([[1,            point_i.x,              point_i.y],
                        [point_i.x,    point_i.x ** 2,         point_i.x * point_i.y],
                        [point_i.y,    point_i.x * point_i.y,  point_i.y ** 2]])
    return np.array(A_local)


def d2A_dy2(point, all_points):
    A_local = np.zeros((m + 2, m + 2))
    for point_i in all_points:
        A_local += weight_func(x=point.x, x_i=point_i.x) * d2_weight_func(x=point.y, x_i=point_i.y) \
                   * np.array([[1,            point_i.x,              point_i.y],
                        [point_i.x,    point_i.x ** 2,         point_i.x * point_i.y],
                        [point_i.y,    point_i.x * point_i.y,  point_i.y ** 2]])
    return np.array(A_local)


def d2A_dydx(point, all_points):
    A_local = np.zeros((m + 2, m + 2))
    for point_i in all_points:
        A_local += d_weight_func(x=point.x, x_i=point_i.x) * d_weight_func(x=point.y, x_i=point_i.y) \
                   * np.array([[1,            point_i.x,              point_i.y],
                        [point_i.x,    point_i.x ** 2,         point_i.x * point_i.y],
                        [point_i.y,    point_i.x * point_i.y,  point_i.y ** 2]])
    return np.array(A_local)
