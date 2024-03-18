import numpy as np

from params import WEIGHT_FUNCTION_TYPE


# Весовая функция и её производные
def weight_func(r):
    if WEIGHT_FUNCTION_TYPE == "quadratic":
        if 0 <= r <= 1:
            return 1 - 6 * r ** 2 + 8 * r ** 3 - 3 * r ** 4
        else:
            return 0
    elif WEIGHT_FUNCTION_TYPE == "cubic":
        if 0 <= r <= 0.5:
            return 2 / 3 - 4 * r ** 2 + 4 * r ** 3
        elif 0.5 < r <= 1:
            return 4 / 3 - 4 * r + 4 * r ** 2 - 4 / 3 * r ** 3
        elif r > 1:
            return 0


def d_weight_func(r, dr):
    if WEIGHT_FUNCTION_TYPE == "quadratic":
        if 0 <= r <= 1:
            return (-12 * r + 24 * r ** 2 - 12 * r ** 3) * dr
        else:
            return 0
    elif WEIGHT_FUNCTION_TYPE == "cubic":
        if 0 <= r <= 0.5:
            return (-8 * r + 12 * r ** 2) * dr
        elif 0.5 < r <= 1:
            return (-4 + 8 * r - 4 * r ** 2) * dr
        elif r > 1:
            return 0


def weight_func_array(r, drdx, drdy, weight_function=False):
    w = np.zeros(r.shape)
    dwdx = np.zeros(drdx.shape)
    dwdy = np.zeros(drdy.shape)

    for i in range(len(r)):
        w[i] = weight_func(r[i])
        dwdx[i] = d_weight_func(r[i], drdx[i])
        dwdy[i] = d_weight_func(r[i], drdy[i])

    w = w[w != 0]
    dwdx = dwdx[dwdx != 0]
    dwdy = dwdy[dwdy != 0]

    if weight_function:
        return w
    else:
        return w, dwdx, dwdy
