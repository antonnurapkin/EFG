from EFG.helpers import B_matrix, search_nodes_in_domain
from params import D, n_x, n_y
from meshing import cells
import numpy as np
import pandas as pd
from time import time

start = time()

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


def K_global_assemble(K_global, K_local_vector_n_indexes):
    for elem in K_local_vector_n_indexes:
        K_global[int(elem[1]), int(elem[2])] += elem[0]

    return K_global


def create_K_global(cells, n_x, n_y):
    K_global = np.zeros((n_x * n_y, n_x * n_y))

    for i in range(len(cells)):
        for j in range(len(cells[i])):
            for point in cells[i][j].gauss_points:
                nodes_in_domain, global_indexes = search_nodes_in_domain(q_point=point, current_cell=cells[i][j])
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

    return K_global


# K, f = create_K_global(cells, n_x, n_y)
#
# global_indexes = []
# all_nodes = []
# for row in cells:
#     for cell in row:
#         for node in cell.nodes:
#             if len(global_indexes) == 0:
#                 global_indexes.append(node.global_index)
#                 all_nodes.append(node)
#             elif len(global_indexes) > 0 and node.global_index not in global_indexes:
#                 global_indexes.append(node.global_index)
#                 all_nodes.append(node)
#
# # Создание вектора f
# B =
# f_local_vector_n_indexes = create_f_vector(
#     all_nodes,
#     B,
#     b,
#     1,
#     cells[0][0].jacobian,
#     global_indexes
# )
#
# f_global = f_global_assemble(f_global, f_local_vector_n_indexes)

# x_gp = []
# y_gp = []
#
# x_n = []
# y_n = []
#
#
# for row in cells:
#     for cell in row:
#         for point in cell.gauss_points:
#             x_gp.append(point.x)
#             y_gp.append(point.y)
#
#         for node in cell.nodes:
#
#
# z = np.zeros((n_y, n_x))
#
# k = 0
# for i in range(n_x):
#     for j in range(n_y):
#         z[j, i] = f[k]
#         k += 1
#
# fig = go.Figure(data=
#     go.Contour(
#         z=z,
#         x=[0.0, 0.25, 0.5, 0.75, 0.5],
#         y=[0.0, 0.25, 0.5, 0.75, 0.5]
#     ))
# fig.show()
