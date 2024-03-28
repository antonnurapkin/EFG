import numpy as np

from helpers import search_nodes_in_domain, dF_array, B_matrix
from params import D
from shape_function.components_shape_function.radius import calculate_r


def calculate_stress(u, integration_points, nodes, nodes_coords):
    stress = np.zeros((3, len(integration_points)))

    i = 0
    for point in integration_points:

        # Вычисляется некое характеристическое расстояние от точки интегрирования до всех узлов
        r_array = calculate_r(q_point=point, coords=nodes_coords)

        # Вычисляются индексы тех, кто в области поддержки
        global_indexes = search_nodes_in_domain(r_array=r_array)

        nodes_in_domain = nodes[global_indexes.astype(int)]

        # Производная функции формы
        dF = dF_array(q_point=point, nodes_in_domain=nodes_in_domain, r_array=r_array, coords=nodes_coords)

        # Создание матрицы узловой матрица жёсткости
        for j in range(len(nodes_in_domain)):
            B_i = B_matrix(dF[:, j])

            k = int(global_indexes[j])

            u_nodal = np.array([[nodes[k].u_real], [nodes[k].v_real]])
            stress_temp = np.dot(D, np.dot(B_i, u_nodal))

            stress[0, i] = stress_temp[0]
            stress[1, i] = stress_temp[1]
            stress[2, i] = stress_temp[2]

        i += 1

    return stress
