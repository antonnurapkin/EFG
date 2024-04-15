import numpy as np

from helpers import search_nodes_in_domain, dF_array, B_matrix
from integration_points import Point
from params import D, A, R0
from shape_function.components_shape_function.radius import calculate_r, r_derivatives


def calculate_stress(nodes, nodes_coords):
    x = np.linspace(R0 + 0.001, A, 100)
    integration_points = []

    for i in range(len(x)):
        integration_points.append(Point(x=x[i], y=0, weight=None, jacobian=None))

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

        stress_temp = np.zeros((3,1))
        # Создание матрицы узловой матрица жёсткости
        for j in range(len(nodes_in_domain)):
            B_j = B_matrix(dF[:, j])
            stress_temp += np.dot(D, np.dot(B_j, np.array([[nodes_in_domain[j].u_real], [nodes_in_domain[j].v_real]])))

        stress[0, i] = stress_temp[0]
        stress[1, i] = stress_temp[1]
        stress[2, i] = stress_temp[2]

        i += 1

    return stress, integration_points