import numpy as np

from shape_function.shape_function import dFdx, dFdy, F
from shape_function.components_shape_function.weight_function import weight_func_array
from shape_function.components_shape_function.radius import r_derivatives


def dF_array(q_point, nodes_in_domain, r_array, coords, drdx_add, drdy_add):

    drdx, drdy = r_derivatives(r_array=r_array, coords=coords, q_point=q_point)

    if drdy_add is not None and drdx_add is not None :
        drdx = add_crack_nodes(dr=drdx, dr_add=drdx_add)
        drdy = add_crack_nodes(dr=drdy, dr_add=drdy_add)

    w, dwdx, dwdy = weight_func_array(r=r_array, drdx=drdx, drdy=drdy)

    dF = np.vstack([
        dFdx(point=q_point, all_points=nodes_in_domain, w=w, dwdx=dwdx),
        dFdy(point=q_point, all_points=nodes_in_domain, w=w, dwdy=dwdy)
    ])

    return dF


def add_crack_nodes(dr, dr_add):
    ids = np.where(dr_add != 0)[0]

    for index in ids:
        dr[index] = dr_add[index]

    return dr


def B_matrix(F_i):
    B = np.array([[F_i[0], 0],
                  [0, F_i[1]],
                  [F_i[1], F_i[0]]])

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

def get_tetta(x, y):
    return np.arctan(y / x)

def get_rad(tetta, x):
    return x / np.cos(tetta)


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


def distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
