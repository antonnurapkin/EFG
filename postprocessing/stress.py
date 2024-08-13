from copy import copy

import numpy as np

from crack.utils import check_crack_interaction
from helpers import search_nodes_in_domain, dF_array, B_matrix
from integration_points import Point
from params import D, A
from shape_function.components_shape_function.radius import calculate_r, r_derivatives


def calculate_stress(nodes, nodes_coords, integration_points, crack_top_ind):
    print("Расчёт напряжений...")

    stress = np.zeros((3, len(integration_points)))

    i = 0
    for point in integration_points:

        # Вычисляется некое характеристическое расстояние от точки интегрирования до всех узлов
        r_array = calculate_r(q_point=point, coords=nodes_coords)

        # Вычисляются индексы тех, кто в области поддержки
        global_indexes = search_nodes_in_domain(r_array=r_array)

        nodes_in_domain = nodes[global_indexes.astype(int)]

        # Отсев узлов, в случае нахождения рядом с трещиной
        r_array, drdx_add, drdy_add, nodes_in_domain = check_crack_interaction(
            point=point,
            r_array=r_array,
            nodes_in_domain=nodes_in_domain,
            crack_top_ind=crack_top_ind,
            global_indexes=global_indexes,
            nodes=nodes
        )

        dF = dF_array(
            q_point=point,
            nodes_in_domain=nodes_in_domain,
            r_array=r_array,
            coords=nodes_coords,
            drdx_add=drdx_add,
            drdy_add=drdy_add
        )

        stress_temp = np.zeros((3, 1))
        # Создание матрицы узловой матрица жёсткости
        for j in range(len(nodes_in_domain)):
            B_j = B_matrix(dF[:, j])
            stress_temp += np.dot(D, np.dot(B_j, np.array(
                [[nodes_in_domain[j].u_solution], [nodes_in_domain[j].v_solution]])))

        stress[0, i] = stress_temp[0]
        stress[1, i] = stress_temp[1]
        stress[2, i] = stress_temp[2]

        i += 1

    return stress
