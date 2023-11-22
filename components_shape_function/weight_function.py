import numpy as np

from params import WEIGHT_FUNCTION_TYPE


# Весовая функция и её производные
def weight_func(r):
    if WEIGHT_FUNCTION_TYPE == "quadratic":
        if r <= 1:
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
        if r <= 1:
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


def d2_weight_func(r, d2r):
    if WEIGHT_FUNCTION_TYPE == "quadratic":
        if r <= 1:
            return (-12 + 48 * r - 36 * r ** 2) * d2r
        else:
            return 0
    elif WEIGHT_FUNCTION_TYPE == "cubic":
        if 0 <= r <= 0.5:
            return (-8 + 24 * r) * d2r
        elif 0.5 < r <= 1:
            return (8 - 8 * r) * d2r
        elif r > 1:
            return 0


def weight_func_array(r, drdx, drdy, d2rdx2, d2rdy2, d2rdxdy):
    w = np.zeros(r.shape)
    dwdx = np.zeros(drdx.shape)
    dwdy = np.zeros(drdy.shape)
    d2wdx2 = np.zeros(d2rdx2.shape)
    d2wdy2 = np.zeros(d2rdy2.shape)
    d2wdxdy = np.zeros(d2rdx2.shape)

    for i in range(len(r)):
        w[i] = weight_func(r[i])
        dwdx[i] = d_weight_func(r[i], drdx[i])
        dwdy[i] = d_weight_func(r[i], drdy[i])
        d2wdx2[i] = d2_weight_func(r[i], d2rdx2[i])
        d2wdy2[i] = d2_weight_func(r[i], d2rdy2[i])
        d2wdxdy[i] = d2_weight_func(r[i], d2rdxdy[i])

    w = w[w != 0]
    dwdx = dwdx[dwdx != 0]
    dwdy = dwdy[dwdy != 0]
    d2wdx2 = d2wdx2[d2wdx2 != 0]
    d2wdy2 = d2wdy2[d2wdy2 != 0]
    d2wdxdy = d2wdxdy[d2wdxdy != 0]

    return w, dwdx, dwdy, d2wdx2, d2wdy2, d2wdxdy
