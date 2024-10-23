from src.shape_function.components_shape_function.p import p
import numpy as np


def B(point, w, all_points):
    B_local = np.zeros((len(p(point)), len(all_points)))

    for i in range(len(w)):
        B_local[:, i] = p(all_points[i]).T * w[i]

    return np.array(B_local)


def dB_dx(point, dw, all_points):
    B_local = np.zeros((len(p(point)), len(all_points)))

    for i in range(len(dw)):
        B_local[:, i] = p(all_points[i]).T * dw[i]

    return np.array(B_local)


def dB_dy(point, dw, all_points):
    B_local = np.zeros((len(p(point)), len(all_points)))

    for i in range(len(dw)):
        B_local[:, i] = p(all_points[i]).T * dw[i]

    return np.array(B_local)
