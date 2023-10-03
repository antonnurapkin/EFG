from EFG.params import step_x, step_y
import numpy as np


# Весовая функция и её производные
def weight_func(x, x_i, alpha_s = 3):

    d_c = (step_x ** 2 + step_y ** 2 )**0.5

    d_s = alpha_s * d_c
    r = abs(x - x_i) / d_s

    if 0 <= r <= 0.5:
        return 2 / 3 - 4 * r ** 2 + 4 * r ** 3
    elif 0.5 < r <= 1:
        return 4 / 3 - 4 * r + 4 * r ** 2 - 4 / 3 * r ** 3
    elif r > 1:
        return 0


def d_weight_func(x, x_i, alpha_s = 2):

    d_c = (step_x ** 2 + step_y ** 2 )**0.5

    d_s = alpha_s * d_c
    r = abs(x - x_i) / d_s

    if 0 <= r <= 0.5:
        return (-8 * r + 12 * r ** 2) * np.sign(x - x_i)
    elif 0.5 < r <= 1:
        return (-4 + 8 * r - 8 * r ** 2) * np.sign(x - x_i)
    elif r > 1:
        return 0


def d2_weight_func(x, x_i, alpha_s=2):
    d_c = (step_x ** 2 + step_y ** 2) ** 0.5

    d_s = alpha_s * d_c
    r = abs(x - x_i) / d_s

    if 0 <= r <= 0.5:
        return (-8 + 24 * r) * np.sign(x - x_i)
    elif 0.5 < r <= 1:
        return (8 - 8 * r) * np.sign(x - x_i)
    elif r > 1:
        return 0