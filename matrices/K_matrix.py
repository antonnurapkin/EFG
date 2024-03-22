from helpers import dF_array, B_matrix, search_nodes_in_domain
from shape_function.components_shape_function.radius import calculate_r
from params import D
import numpy as np


def K_global(integration_points, nodes, nodes_coords):
    K = np.zeros((2 * len(nodes), 2 * len(nodes)), dtype=np.float64)
    for point in integration_points:
        # Вычисляется некое характеристическое расстояние от точки интегрирования до всех узлов
        r_array = calculate_r(q_point=point, coords=nodes_coords)

        # Вычисляются индексы тех, кто в области поддержки
        global_indexes = search_nodes_in_domain(r_array=r_array)

        nodes_in_domain = nodes[global_indexes.astype(int)]

        # Производная функции формы
        dF = dF_array(q_point=point, nodes_in_domain=nodes_in_domain, r_array=r_array, coords=nodes_coords)

        # Создание матрицы узловой матрица жёсткости
        for i in range(len(nodes_in_domain)):
            for j in range(len(nodes_in_domain)):
                # Матрица связи деформаций и перемещений
                B_i = B_matrix(dF[:, i])
                B_j = B_matrix(dF[:, j])

                K_local = point.jacobian * point.weight * np.dot(np.transpose(B_i), np.dot(D, B_j))

                k = int(global_indexes[i])
                m = int(global_indexes[j])

                K[2 * k: 2 * k + 2, 2 * m: 2 * m + 2] += K_local

    print("Матрица жесткости сформирована...")

    return K
