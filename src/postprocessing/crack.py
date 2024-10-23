import numpy as np

from src.params import CRACK_LENGTH, CRACK_HALF_WIDTH, NODES_ON_CRACK, B


def create_upper_crack_bound():
    x0 = 0
    y0 = B / 2
    step = CRACK_LENGTH / (NODES_ON_CRACK - 1)

    y_arr = []
    x_arr = []

    for i in range(NODES_ON_CRACK):
        x = step * i
        y_high_bound = CRACK_HALF_WIDTH * np.sqrt(1 - ((x - x0) / CRACK_LENGTH) ** 2) + y0

        y_arr.append(y_high_bound)
        x_arr.append(x)

    return y_arr, x_arr


def create_lower_crack_bound():
    x0 = 0
    y0 = B / 2

    step = CRACK_LENGTH / (NODES_ON_CRACK - 1)

    y_arr = []
    x_arr = []

    for i in range(NODES_ON_CRACK):
        x = step * i
        y_low_bound = -1 * CRACK_HALF_WIDTH * np.sqrt(1 - ((x - x0) / CRACK_LENGTH) ** 2) + y0

        y_arr.append(y_low_bound)
        x_arr.append(x)

    return y_arr, x_arr
