from params import b, l_x, l_y, THICKNESS, D_init_const
import numpy as np


def w(x, y):
    coef = 16 * b[2] / ((np.pi ** 6) * D_init_const)

    series = 0

    M = 20
    N = 20

    for m in range(1, M + 1):
        for n in range(1, N + 1):
            divider = m * n * (m ** 2 / l_x ** 2 + n ** 2 / l_y ** 2) ** 2
            series += np.sin(m * np.pi * x / l_x) * np.sin(n * np.pi * y / l_y) / divider

    return coef * series

print(w(x=l_x / 2, y=l_y / 2))