from EFG.components_shape_function.weight_function import d_weight_func, weight_func, d2_weight_func
from components_shape_function.p import p
from EFG.params import m
import numpy as np

def dB_dx(point, all_points):
    B_local = np.zeros((m + 2, len(all_points)))
    k = 0
    for point_i in all_points:
        # B_local = np.insert(B_local, k, p(point_i).T \
        #                     * d_weight_func(x=point.x, x_i=point_i.x) * weight_func(x=point.y, x_i=point_i.y))
        B_local[0, k] = 1 * d_weight_func(x=point.x, x_i=point_i.x) * weight_func(x=point.y, x_i=point_i.y)
        B_local[1, k] = point_i.x * d_weight_func(x=point.x, x_i=point_i.x) * weight_func(x=point.y, x_i=point_i.y)
        B_local[2, k] = point_i.y * d_weight_func(x=point.x, x_i=point_i.x) * weight_func(x=point.y, x_i=point_i.y)
        k += 1

    return np.array(B_local)


def dB_dy(point, all_points):
    B_local = np.zeros((m + 2, len(all_points)))
    k = 0
    for point_i in all_points:
        # B_local = np.insert(B_local, k, p(point_i).T \
        #                     * weight_func(x=point.x, x_i=point_i.x) * d_weight_func(x=point.y, x_i=point_i.y))
        B_local[0, k] = 1 * weight_func(x=point.x, x_i=point_i.x) * d_weight_func(x=point.y, x_i=point_i.y)
        B_local[1, k] = point_i.x * weight_func(x=point.x, x_i=point_i.x) * d_weight_func(x=point.y, x_i=point_i.y)
        B_local[2, k] = point_i.y * weight_func(x=point.x, x_i=point_i.x) * d_weight_func(x=point.y, x_i=point_i.y)
        k += 1

    return np.array(B_local)


def d2B_dx2(point, all_points):
    B_local = np.zeros((m + 2, len(all_points)))
    k = 0
    for point_i in all_points:
        # B_local = np.insert(B_local, k, p(point_i).T \
        #                     * d2_weight_func(x=point.x, x_i=point_i.x) * weight_func(x=point.y, x_i=point_i.y))
        B_local[0, k] = 1 * d2_weight_func(x=point.x, x_i=point_i.x) * weight_func(x=point.y, x_i=point_i.y)
        B_local[1, k] = point_i.x * d2_weight_func(x=point.x, x_i=point_i.x) * weight_func(x=point.y, x_i=point_i.y)
        B_local[2, k] = point_i.y * d2_weight_func(x=point.x, x_i=point_i.x) * weight_func(x=point.y, x_i=point_i.y)
        k += 1

    return np.array(B_local)


def d2B_dy2(point, all_points):
    B_local = np.zeros((m + 2, len(all_points)))
    k = 0
    for point_i in all_points:
        # B_local = np.insert(B_local, k, p(point_i).T \
        #                     * weight_func(x=point.x, x_i=point_i.x) * d2_weight_func(x=point.y, x_i=point_i.y))
        B_local[0, k] = 1 * weight_func(x=point.x, x_i=point_i.x) * d2_weight_func(x=point.y, x_i=point_i.y)
        B_local[1, k] = point_i.x * weight_func(x=point.x, x_i=point_i.x) * d2_weight_func(x=point.y, x_i=point_i.y)
        B_local[2, k] = point_i.y * weight_func(x=point.x, x_i=point_i.x) * d2_weight_func(x=point.y, x_i=point_i.y)
        k += 1

    return np.array(B_local)


def d2B_dydx(point, all_points):
    B_local = np.zeros((m + 2, len(all_points)))
    k = 0
    for point_i in all_points:
        # B_local = np.insert(B_local, k, p(point_i).T \
        #                     * d_weight_func(x=point.x, x_i=point_i.x) * d_weight_func(x=point.y, x_i=point_i.y))
        B_local[0, k] = 1 * d_weight_func(x=point.x, x_i=point_i.x) * d_weight_func(x=point.y, x_i=point_i.y)
        B_local[1, k] = point_i.x * d_weight_func(x=point.x, x_i=point_i.x) * d_weight_func(x=point.y, x_i=point_i.y)
        B_local[2, k] = point_i.y * d_weight_func(x=point.x, x_i=point_i.x) * d_weight_func(x=point.y, x_i=point_i.y)
        k += 1

    return np.array(B_local)


def B(point, all_points):
    B_local = np.zeros((m + 2, len(all_points)))
    k = 0
    for point_i in all_points:
        # B_local[:, k] = p(point_i).T * weight_func(x=point.x, x_i=point_i.x) * weight_func(x=point.y, x_i=point_i.y)

        B_local[0, k] = 1 * weight_func(x=point.x, x_i=point_i.x) * weight_func(x=point.y, x_i=point_i.y)
        B_local[1, k] = point_i.x * weight_func(x=point.x, x_i=point_i.x) * weight_func(x=point.y, x_i=point_i.y)
        B_local[2, k] = point_i.y * weight_func(x=point.x, x_i=point_i.x) * weight_func(x=point.y, x_i=point_i.y)

        k += 1

    return np.array(B_local)
