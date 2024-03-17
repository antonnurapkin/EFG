import numpy as np

from shape_function import dFdx, dFdy, F
from components_shape_function.weight_function import weight_func_array
from components_shape_function.radius import r_derivatives


def dF_array(q_point, nodes_in_domain, r_array, coords):

    drdx, drdy = r_derivatives(r_array, coords, q_point)
    w, dwdx, dwdy = weight_func_array(r_array, drdx, drdy)

    F = np.vstack([
        dFdx(q_point, nodes_in_domain, w, dwdy),
        dFdy(q_point, nodes_in_domain, w, dwdx)
    ])

    return F


def B_matrix(F_i):
    B = np.array([[F_i[0], 0],
                  [0, F_i[1]],
                  [F_i[0], F_i[1]]])

    return B


def search_nodes_in_domain(r_array):

    global_indexes = np.array([])

    for i in range(len(r_array)):
        if r_array[i] <= 1:
            global_indexes = np.append(global_indexes, i)

    return global_indexes


def get_x_coord(r, fi):
    return r * np.cos(fi)

def get_y_coord(r, fi):
    return r * np.sin(fi)


def N(g_pos, left_bound, right_bound):
    return (right_bound - g_pos) / (right_bound - left_bound)


def find_nearest_nodes(point, nodes_coords, y_value=False, x_value=False, y_bound=False, x_bound=False):
    if x_bound:
        value = x_value
        target_coord = 0
        other_coord = 1
        point_coord = point.y
    elif y_bound:
        value = y_value
        target_coord = 1
        other_coord = 0
        point_coord = point.x

    bound_nodes_coords = np.sort(nodes_coords[:, np.where(nodes_coords[target_coord] == value)[0]], axis=1)

    idx = (np.abs(bound_nodes_coords[other_coord] - point_coord)).argmin()
    if bound_nodes_coords[other_coord][idx] > point_coord:
        return bound_nodes_coords[other_coord][idx], bound_nodes_coords[other_coord][idx - 1]
    elif bound_nodes_coords[other_coord][idx] < point_coord:
        return bound_nodes_coords[other_coord][idx + 1], bound_nodes_coords[other_coord][idx]
