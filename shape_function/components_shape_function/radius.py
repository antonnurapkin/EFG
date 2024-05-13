import numpy as np
from params import DS


def calculate_r(coords , q_point):
    q_point_array = [[q_point.x], [q_point.y]] * np.ones(coords.shape)

    r_array = np.sqrt(np.square(coords[0, :] - q_point_array[0, :]) + np.square(coords[1, :] - q_point_array[1, :])) / DS

    return r_array


def drdx(r_array, coords, q_point):
    q_point_array = [[q_point.x], [q_point.y]] * np.ones(coords.shape)
    drdx_array = (-coords[0, :] + q_point_array[0, :]) / (r_array * DS * DS)

    return drdx_array


def drdy(r_array, coords, q_point):
    q_point_array = [[q_point.x], [q_point.y]] * np.ones(coords.shape)
    drdx_array = (-coords[1, :] + q_point_array[1, :]) / (r_array * DS * DS)

    return drdx_array


def r_derivatives(r_array, coords, q_point):
    drdy_array = drdy(r_array, coords, q_point)
    drdx_array = drdx(r_array, coords, q_point)

    return drdx_array, drdy_array








