from components_shape_function.weight_function_first import d_weight_func, weight_func, d2_weight_func
from components_shape_function.p import p
import numpy as np


def dB_dx(point, dw, all_points):
    B_local = np.zeros((len(p(point)), len(all_points)))
    k = 0
    for i in range(len(dw)):
        B_local[:, k] = p(all_points[i]).T * dw[i]

        k += 1

    return np.array(B_local)


def dB_dy(point, dw, all_points):
    B_local = np.zeros((len(p(point)), len(all_points)))
    k = 0
    for i in range(len(dw)):
        B_local[:, k] = p(all_points[i]).T * dw[i]

        k += 1

    return np.array(B_local)


def d2B_dx2(point, d2w, all_points):
    B_local = np.zeros((len(p(point)), len(all_points)))
    k = 0
    for i in range(len(d2w)):
        B_local[:, k] = p(all_points[i]).T * d2w[i]

        k += 1

    return np.array(B_local)


def d2B_dy2(point, d2w, all_points):
    B_local = np.zeros((len(p(point)), len(all_points)))
    k = 0
    for i in range(len(d2w)):
        B_local[:, k] = p(all_points[i]).T * d2w[i]

        k += 1

    return np.array(B_local)


def d2B_dydx(point, d2w, all_points):
    B_local = np.zeros((len(p(point)), len(all_points)))
    k = 0
    for i in range(len(d2w)):
        B_local[:, k] = p(all_points[i]).T * d2w[i]

        k += 1

    return np.array(B_local)


def B(point, w, all_points):
    B_local = np.zeros((len(p(point)), len(all_points)))
    k = 0
    for i in range(len(w)):
        B_local[:, k] = p(all_points[i]).T * w[i]

        k += 1

    return np.array(B_local)
