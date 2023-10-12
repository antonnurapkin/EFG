from params import D, n_x, n_y, elem_x, elem_y
from meshing import cells
import numpy as np


def indexes_for_nodes(elem_x, elem_y, n_x, n_y):
    left = np.array([i for i in range(n_y)])
    bottom = np.array([n_y * (i + 1) - 1 for i in range(n_x)])
    right = np.flip(np.array([(n_x - 1) * n_y + i for i in range(n_y)]))
    top = np.flip(np.array([n_y * i for i in range(n_x)]))

    global_bound_indexes = delete_duplicates(np.concatenate([left, bottom, right, top]))
    local_bound_indexes = np.arange(0, len(global_bound_indexes), 1)

    result = np.transpose(np.vstack((global_bound_indexes, local_bound_indexes)))

    return result


def delete_duplicates(arr):

    to_delete = []

    for i in range(len(arr) - 1):
        if arr[i] == arr[i + 1]:
            to_delete.append(i)

    return np.delete(arr, to_delete)[:-1]


def N(g_pos, left_bound, right_bound):
    return (right_bound - g_pos) / (right_bound - left_bound)


def create_s_matrix(cells, indexes):
    s = np.zeros((2,len(indexes)))

    for i in range(len(cells)):
        for j in range(len(cells[i])):
            for node in cells[i][j].nodes:
                if node.global_index in indexes[:,0]:
                    insert_index = np.where(indexes[:,0] == node.global_index)[0][0]
                    s[0][insert_index] = node.x
                    s[1][insert_index] = node.y

    return s

indexes = indexes_for_nodes(elem_x, elem_y, n_x, n_y)
S = create_s_matrix(cells, indexes)
print(S)




