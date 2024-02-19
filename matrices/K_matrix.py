from helpers import F_array, B_matrix, search_nodes_in_domain
from components_shape_function.radius import calculate_r
from params import D
import numpy as np


# TODO: Проверить алгоритм работы над индексами, в целом индексацию, их использование при создании матрицы

def K_global(integration_points, nodes, nodes_coords):
    K_global = np.zeros((2 * len(nodes), 2 * len(nodes)))


    for point in integration_points:
        r_array = calculate_r(q_point=point, coords=nodes_coords)
        global_indexes = search_nodes_in_domain(r_array=r_array)

        nodes_in_domain = nodes[global_indexes.astype(int)]

        F = F_array(q_point=point, nodes_in_domain=nodes_in_domain, r_array=r_array, coords=nodes_coords)

        # Создание матрицы узловой матрица жёсткости
        for i in range(len(nodes_in_domain)):
            for j in range(len(nodes_in_domain)):
                B_i = B_matrix(F[:, i])
                B_j = B_matrix(F[:, j])

                K_local = point.jacobian * point.weight * np.dot(np.transpose(B_i), np.dot(D, B_j))
                k = int(global_indexes[i])
                l = int(global_indexes[j])

                K_global[k, k] += K_local[0, 0]
                K_global[k, l] += K_local[0, 1]
                K_global[l, l] += K_local[1, 1]
                K_global[l, k] += K_local[1, 0]

    print("Матрица жесткости сформирована...")

    return K_global
