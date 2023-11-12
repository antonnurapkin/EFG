from helpers import B_matrix, search_nodes_in_domain
from params import D
import numpy as np


def create_K_nodal_vector(B, D, nodes_in_domain, weight, jacobian, global_indexes):
    size = len(nodes_in_domain)
    K_local_vector_n_indexes = np.empty((0, 3))

    for i in range(size):
        B_i = B[:, i]
        for j in range(size):
            B_j = B[:, j]
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


def K_global(cells, n_x, n_y, nodes, coords):
    K_global = np.zeros((n_x * n_y, n_x * n_y))

    for i in range(len(cells)):
        for j in range(len(cells[i])):
            for point in cells[i][j].gauss_points:
                global_indexes = search_nodes_in_domain(q_point=point, coords=coords)

                nodes_in_domain = nodes[global_indexes.astype(int)]

                B = B_matrix(q_point=point, nodes_in_domain=nodes_in_domain)

                # Создание матрицы K
                K_local_vector_n_indexes = create_K_nodal_vector(
                    B,
                    D,
                    nodes_in_domain,
                    point.weight,
                    cells[i][j].jacobian,
                    global_indexes
                )

                index_1 = K_local_vector_n_indexes[:, 1]
                index_2 = K_local_vector_n_indexes[:, 2]
                K_global[index_1.astype(int), index_2.astype(int)] += K_local_vector_n_indexes[:, 0]

    print("Матрица жесткости сформирована")

    return K_global