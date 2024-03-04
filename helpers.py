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


def delete_duplicates(arr):
    to_delete = [i for i in range(len(arr) - 1) if arr[i] == arr[i + 1]]

    return np.delete(arr, to_delete)[:-1]


def indexes_for_bound_nodes(n_x, n_y):
    left = np.array([i for i in range(n_y)])
    bottom = np.array([n_y * (i + 1) - 1 for i in range(n_x)])
    right = np.flip(np.array([(n_x - 1) * n_y + i for i in range(n_y)]))
    top = np.flip(np.array([n_y * i for i in range(n_x)]))

    global_bound_indexes = delete_duplicates(np.concatenate([left, bottom, right, top]))
    local_bound_indexes = np.arange(0, len(global_bound_indexes), 1)

    result = np.transpose(np.vstack((global_bound_indexes, local_bound_indexes)))

    return result


def get_bound_elems(cells):
    left_bound = cells[:, 0]
    bottom_bound = cells[-1, :][1:]
    right_bound = np.flip(cells[:, -1][:-1])
    top_bound = np.flip(cells[0, :][1:-1])

    return np.concatenate((left_bound, bottom_bound, right_bound, top_bound))


def create_s_matrix(cells, indexes):
    s = np.zeros((2, len(indexes)))

    for i in range(len(cells)):
        for j in range(len(cells[i])):
            for node in cells[i][j].nodes:
                if node.global_index in indexes[:, 0]:
                    insert_index = np.where(indexes[:, 0] == node.global_index)[0][0]
                    s[0][insert_index] = node.get_x_coord
                    s[1][insert_index] = node.get_y_coord

    return s


def get_x_coord(r, fi):
    return r * np.cos(fi)

def get_y_coord(r, fi):
    return r * np.sin(fi)
