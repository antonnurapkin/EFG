from params import step_x, step_y, alpha_x
import numpy as np


def Dirac_delta(x):
    if x == 0:
        return 1
    else:
        return 0


# Весовая функция и её производные
def weight_func(x, x_i):
    d_c = min([step_x, step_y])

    d_s = alpha_x * d_c
    r = abs(x - x_i) / d_s

    # if r <= 1:
    #     return 1 - 6 * r ** 2 + 8 * r ** 3 - 3 * r ** 4
    # else:
    #     return 0

    if 0 <= r <= 0.5:
        return 2 / 3 - 4 * r ** 2 + 4 * r ** 3
    elif 0.5 < r <= 1:
        return 4 / 3 - 4 * r + 4 * r ** 2 - 4 / 3 * r ** 3
    elif r > 1:
        return 0


def d_weight_func(x, x_i):

    d_c = min([step_x, step_y])

    d_s = alpha_x * d_c
    r = abs(x - x_i) / d_s

    # if r <= 1:
    #     return -12 * r + 24 * r ** 2 - 12 * r ** 3 * np.sign(x - x_i) / d_s
    # else:
    #     return 0

    if 0 <= r <= 0.5:
        return (-8 * r + 12 * r ** 2) * np.sign(x - x_i) / d_s
    elif 0.5 < r <= 1:
        return (-4 + 8 * r - 4 * r ** 2) * np.sign(x - x_i) / d_s
    elif r > 1:
        return 0


def d2_weight_func(x, x_i):

    d_c = min([step_x, step_y])

    d_s = alpha_x * d_c
    r = abs(x - x_i) / d_s

    # if r <= 1:
    #     return (-12 + 48 * r - 36 * r ** 2) * np.sign(x - x_i) / (d_s ** 2)
    # else:
    #     return 0

    if 0 <= r <= 0.5:
        return (-8 + 24 * r) * np.sign(x - x_i) / (d_s ** 2) + (-8 * r + 12 * r ** 2) * 2 * Dirac_delta(x - x_i) / d_s
    elif 0.5 < r <= 1:
        return (8 - 8 * r) * np.sign(x - x_i) / (d_s ** 2) + (-4 + 8 * r - 4 * r ** 2) * 2 * Dirac_delta(x - x_i) / d_s
    elif r > 1:
        return 0
