import numpy as np
from helpers import search_nodes_in_domain
from shape_function.components_shape_function.radius import calculate_r, r_derivatives
from integration_points import create_integration_points_bound
from shape_function.shape_function import F
from shape_function.components_shape_function.weight_function import weight_func_array
from params import PENALTY


def f_penalty(nodes, nodes_coords, x_bound=False, y_bound=False, x_value=False, y_value=False):
    f_global = np.zeros((2 * len(nodes), 1), dtype=np.float64)

    # Нижняя горизонтальная граница, v = 0
    integration_points_bottom = np.flip(create_integration_points_bound(nodes_coords, y_value=y_value, y_bound=y_bound))

    # Правая вертикальная граница, u = 0
    integration_points_left = create_integration_points_bound(nodes_coords, x_value=x_value, x_bound=x_bound)

    # Нижняя горизонтальная граница, v = 0
    for points_between_nodes in integration_points_bottom:
        for point in points_between_nodes:

            r_array = calculate_r(q_point=point, coords=nodes_coords)
            global_indexes = search_nodes_in_domain(r_array=r_array)

            nodes_in_domain = nodes[global_indexes.astype(int)]

            drdx, drdy = r_derivatives(r_array, nodes_coords, point)
            w, dwdx, dwdy = weight_func_array(r_array, drdx, drdy)

            F_array = F(point, nodes_in_domain, w)

            u_exact = np.array([[u_radial(r=point.x, tetta=np.pi / 2)], [0]])
            S = np.array([[1, 0], [0, 1]])

            for i in range(len(nodes_in_domain)):
                F_i = F_array[i]
                f_local = point.jacobian * point.weight * np.dot(np.transpose(S * F_i), np.dot(S * PENALTY, u_exact))

                k = int(global_indexes[i])

                f_global[2 * k] += f_local[0]
                f_global[2 * k + 1] += f_local[1]

    for points_between_nodes in integration_points_left:
        for point in points_between_nodes:

            r_array = calculate_r(q_point=point, coords=nodes_coords)
            global_indexes = search_nodes_in_domain(r_array=r_array)

            nodes_in_domain = nodes[global_indexes.astype(int)]

            drdx, drdy = r_derivatives(r_array, nodes_coords, point)
            w, dwdx, dwdy = weight_func_array(r_array, drdx, drdy)

            F_array = F(point, nodes_in_domain, w)

            u_exact = np.array([[0], [u_radial(r=point.y, tetta=0)]])
            S = np.array([[1, 0], [0, 1]])

            for i in range(len(nodes_in_domain)):
                F_i = F_array[i]
                f_local = point.jacobian * point.weight * np.dot(np.transpose(S * F_i), np.dot(S * PENALTY, u_exact))

                k = int(global_indexes[i])

                f_global[2 * k] += f_local[0]
                f_global[2 * k + 1] += f_local[1]

    print(f"Штрафной вектор сил {f_global.shape} сформирован...")

    return f_global
