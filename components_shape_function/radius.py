import numpy as np

from params import WEIGHT_FUNCTION_TYPE, ds_x, ds_y


def calculate_r(coords , q_point):
    q_point_array = [[q_point.x], [q_point.y]] * np.ones(coords.shape)

    r_array = np.sqrt(np.square(coords[0, :] - q_point_array[0, :]) + np.square(coords[1, :] - q_point_array[1, :])) / ds_x

    return r_array


def drdx(r_array, coords, q_point):
    q_point_array = [[q_point.x], [q_point.y]] * np.ones(coords.shape)
    drdx_array = (coords[0, :] - q_point_array[0, :]) / r_array

    return drdx_array


def drdy(r_array, coords, q_point):
    q_point_array = [[q_point.x], [q_point.y]] * np.ones(coords.shape)
    drdx_array = (coords[1, :] - q_point_array[1, :]) / r_array

    return drdx_array


def d2rdx2(r_array, coords, q_point):
    q_point_array = [[q_point.x], [q_point.y]] * np.ones(coords.shape)
    d2rdx2_array = np.square(coords[0, :] - q_point_array[0, :]) / np.power(r_array, 3) + np.power(r_array, -1.0)

    return d2rdx2_array


def d2rdy2(r_array, coords, q_point):
    q_point_array = [[q_point.x], [q_point.y]] * np.ones(coords.shape)
    d2rdy2_array = np.square(coords[1, :] - q_point_array[1, :]) / np.power(r_array, 3) + np.power(r_array, -1.0)

    return d2rdy2_array


def d2rdxdy(r_array, coords, q_point):
    q_point_array = [[q_point.x], [q_point.y]] * np.ones(coords.shape)
    d2rdxdy_array = (coords[1, :] - q_point_array[1, :]) * (coords[0, :] - q_point_array[0, :]) / np.power(r_array, 3)

    return d2rdxdy_array

def r_derivatives(r_array, coords, q_point):
    drdy_array = drdy(r_array, coords, q_point)
    drdx_array = drdx(r_array, coords, q_point)

    d2rdx2_array = d2rdx2(r_array, coords, q_point)
    d2rdy2_array = d2rdy2(r_array, coords, q_point)
    d2rdxdy_array = d2rdxdy(r_array, coords, q_point)

    return  drdx_array, drdy_array, d2rdx2_array, d2rdy2_array, d2rdxdy_array








