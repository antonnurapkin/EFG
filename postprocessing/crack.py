import numpy as np

from params import CRACK_LENGTH, CRACK_HALF_WIDTH, NODES_ON_CRACK, B


def create_upper_crack_bound():
    rad = (CRACK_LENGTH ** 2 + CRACK_HALF_WIDTH ** 2) / (2 * CRACK_HALF_WIDTH)
    x0 = 0
    y0_high_bound = (B / 2 + CRACK_HALF_WIDTH) - rad

    step = CRACK_LENGTH / (NODES_ON_CRACK - 1)

    y_arr = []
    x_arr = []

    for i in range(NODES_ON_CRACK):
        x = step * i
        y_high_bound = np.sqrt(rad ** 2 - (x - x0) ** 2) + y0_high_bound

        y_arr.append(y_high_bound)
        x_arr.append(x)

    return y_arr, x_arr


def create_lower_crack_bound():
    rad = (CRACK_LENGTH ** 2 + CRACK_HALF_WIDTH ** 2) / (2 * CRACK_HALF_WIDTH)
    x0 = 0
    y0_low_bound = (B / 2 - CRACK_HALF_WIDTH) + rad

    step = CRACK_LENGTH / (NODES_ON_CRACK - 1)

    y_arr = []
    x_arr = []

    for i in range(NODES_ON_CRACK):
        x = step * i
        y_low_bound = -np.sqrt(rad ** 2 - (x - x0) ** 2) + y0_low_bound

        y_arr.append(y_low_bound)
        x_arr.append(x)

    return y_arr, x_arr
