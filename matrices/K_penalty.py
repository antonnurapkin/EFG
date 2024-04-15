from helpers import search_nodes_in_domain
from integration_points import create_integration_points_bound
from shape_function.components_shape_function.radius import calculate_r, r_derivatives
from params import PENALTY
import numpy as np

from shape_function.components_shape_function.weight_function import weight_func_array
from shape_function.shape_function import F


def K_penalty(nodes, nodes_coords, y_value=False, x_value=False, y_bound=False, x_bound=False):
    K = np.zeros((2 * len(nodes), 2 * len(nodes)), dtype=np.float64)

    # Вычисления координат точек Гаусса на границах
    # Нижняя горизонтальная граница, v = 0
    integration_points_bottom = np.flip(create_integration_points_bound(nodes_coords, y_value=y_value, y_bound=y_bound))

    # Правая вертикальная граница, u = 0
    integration_points_left = create_integration_points_bound(nodes_coords, x_value=x_value, x_bound=x_bound)

    # Нижняя горизонтальная граница, v = 0
    for points_between_nodes in integration_points_bottom:  # Общий цикл
        for point in points_between_nodes:  # Цикл для точек внутри "ячейки"

            # Вычисляется некое характеристическое расстояние от точки интегрирования до всех узлов
            r_array = calculate_r(q_point=point, coords=nodes_coords)

            # Вычисляются индексы тех, кто в области поддержки
            global_indexes = search_nodes_in_domain(r_array=r_array)

            nodes_in_domain = nodes[global_indexes.astype(int)]

            drdx, drdy = r_derivatives(r_array, nodes_coords, point)
            w, dwdx, dwdy = weight_func_array(r_array, drdx, drdy)
            F_array = F(point, nodes_in_domain, w)

            S = np.array([[1, 0], [0, 1]])
            # Создание матрицы узловой матрица жёсткости
            for i in range(len(nodes_in_domain)):
                F_i = F_array[i]
                for j in range(len(nodes_in_domain)):
                    # Матрица связи деформаций и перемещений
                    F_j = F_array[j]

                    K_local = point.jacobian * point.weight * np.dot(np.transpose(S * F_i), np.dot(S * PENALTY, S * F_j))

                    k = int(global_indexes[i])
                    m = int(global_indexes[j])

                    K[2 * k: 2 * k + 2, 2 * m: 2 * m + 2] += K_local

    # Нижняя горизонтальная граница, v = 0
    for points_between_nodes in integration_points_left:  # Общий цикл
        for point in points_between_nodes:  # Цикл для точек внутри "ячейки"

            # Вычисляется некое характеристическое расстояние от точки интегрирования до всех узлов
            r_array = calculate_r(q_point=point, coords=nodes_coords)

            # Вычисляются индексы тех, кто в области поддержки
            global_indexes = search_nodes_in_domain(r_array=r_array)

            nodes_in_domain = nodes[global_indexes.astype(int)]

            drdx, drdy = r_derivatives(r_array, nodes_coords, point)
            w, dwdx, dwdy = weight_func_array(r_array, drdx, drdy)
            F_array = F(point, nodes_in_domain, w)

            S = np.array([[1, 0], [0, 1]])
            # Создание матрицы узловой матрица жёсткости
            for i in range(len(nodes_in_domain)):
                F_i = F_array[i]
                for j in range(len(nodes_in_domain)):
                    # Матрица связи деформаций и перемещений
                    F_j = F_array[j]

                    K_local = point.jacobian * point.weight * np.dot(np.transpose(S * F_i),
                                                                         np.dot(S * PENALTY, S * F_j))

                    k = int(global_indexes[i])
                    m = int(global_indexes[j])

                    K[2 * k: 2 * k + 2, 2 * m: 2 * m + 2] += K_local

    print(f"Штрафная матрица жесткости {K.shape} сформирована...")

    return K
