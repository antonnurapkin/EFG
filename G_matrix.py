from params import D, n_x, n_y, l_x, l_y
from meshing import cells
import numpy as np


def N(g_pos, left_bound, right_bound):
    return (right_bound - g_pos) / (right_bound - left_bound)


def create_G_nodal_vector(point, cell, nodes_in_domain, F, weight, jacobian, global_indexes):
    size = len(nodes_in_domain)
    G_local_vector_n_indexes = np.empty((0, 3))

    for i in range(size):
        if cell.x == 0 or cell.x == l_x - cell.w:
            N_i = N(point.y, cell.y)
        for j in range(size):
            F[i]
            K_local_vector_n_indexes = np.append(
                K_local_vector_n_indexes,
                [
                    [
                        jacobian * weight * np.dot(np.transpose(B_i), np.dot(D, B_j)),
                        global_indexes[i],
                        global_indexes[j]
                    ]
                ],
                axis=0
            )

    return K_local_vector_n_indexes
