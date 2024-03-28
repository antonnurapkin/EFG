from shape_function.components_shape_function.radius import calculate_r, r_derivatives
from shape_function.components_shape_function.weight_function import weight_func_array
from helpers import search_nodes_in_domain, N, find_nearest_nodes
from params import NODES_NUMBER_RADIAL
from integration_points import create_integration_points_bound
from shape_function.shape_function import F
import numpy as np


def G_global(nodes, nodes_coords, y_value=False, x_value=False, y_bound=False, x_bound=False):

    # Вычисления координат точек Гаусса на границах
    # Нижняя горизонтальная граница, v = 0
    integration_points_bottom = np.flip(create_integration_points_bound(nodes_coords, y_value=y_value, y_bound=y_bound))

    # Правая вертикальная граница, u = 0
    integration_points_left = create_integration_points_bound(nodes_coords, x_value=x_value, x_bound=x_bound)

    rows = 2 * len(nodes)
    cols = NODES_NUMBER_RADIAL * 2 * 2
    G = np.zeros((rows, cols))

    bound_index = 0  # Индексирования узлов на границе

    # Нижняя горизонтальная граница, v = 0
    for points_between_nodes in integration_points_bottom:  # Общий цикл
        for point in points_between_nodes:  # Цикл для точек внутри "ячейки"

            r_array = calculate_r(q_point=point, coords=nodes_coords)
            global_indexes = search_nodes_in_domain(r_array=r_array)

            nodes_in_domain = nodes[global_indexes.astype(int)]

            drdx, drdy = r_derivatives(r_array, nodes_coords, point)
            w, dwdx, dwdy = weight_func_array(r_array, drdx, drdy)
            F_array = F(point, nodes_in_domain, w)

            # Поиск соседних узлов
            right, left = find_nearest_nodes(point=point, nodes_coords=nodes_coords, y_value=y_value, y_bound=y_bound)

            N1 = N(g_pos=point.x, left_bound=left, right_bound=right)
            N2 = 1 - N1
            S = np.array([[1, 0], [0, 1]])  # Для удобства

            for i in range(len(nodes_in_domain)):
                G1 = -point.jacobian * point.weight * F_array[i] * N1 * S
                G2 = -point.jacobian * point.weight * F_array[i] * N2 * S

                k = int(global_indexes[i])
                m = bound_index
                l = m + 1

                G[2 * k: 2 * k + 2, 2 * m: 2 * m + 2] += G1
                G[2 * k: 2 * k + 2, 2 * l: 2 * l + 2] += G2

        bound_index += 1

    bound_index += 1

    # Левая вертикальная граница, u = 0
    for points_between_nodes in integration_points_left:
        for point in points_between_nodes:
            r_array = calculate_r(q_point=point, coords=nodes_coords)
            global_indexes = search_nodes_in_domain(r_array=r_array)

            nodes_in_domain = nodes[global_indexes.astype(int)]

            drdx, drdy = r_derivatives(r_array, nodes_coords, point)
            w, dwdx, dwdy = weight_func_array(r_array, drdx, drdy)
            F_array = F(point, nodes_in_domain, w)

            right, left = find_nearest_nodes(point=point, nodes_coords=nodes_coords, x_value=x_value, x_bound=x_bound)

            N1 = N(g_pos=point.y, left_bound=left, right_bound=right)
            N2 = 1 - N1
            S = np.array([[1, 0], [0, 1]])

            for i in range(len(nodes_in_domain)):
                G1 = -point.jacobian * point.weight * F_array[i] * N1 * S
                G2 = -point.jacobian * point.weight * F_array[i] * N2 * S

                k = int(global_indexes[i])
                m = bound_index
                l = m + 1

                G[2 * k: 2 * k + 2, 2 * m: 2 * m + 2] += G1
                G[2 * k: 2 * k + 2, 2 * l: 2 * l + 2] += G2

        bound_index += 1

    print(f"Матрица G {G.shape} создана...")

    return G




