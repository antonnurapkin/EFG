from components_shape_function.radius import calculate_r, r_derivatives
from components_shape_function.weight_function import weight_func_array
from helpers import search_nodes_in_domain
from integration_points import create_integration_points_bound
from shape_function import F
import numpy as np


# TODO: Количество точек интегрирования сделать константой

def N(g_pos, left_bound, right_bound):
    return (right_bound - g_pos) / (right_bound - left_bound)


def find_nearest_nodes(point, nodes_coords, y_value=False, x_value=False, y_bound=False, x_bound=False):
    if x_bound:
        value = x_value
        target_coord = 0
        other_coord = 1
        point_coord = point.y
    elif y_bound:
        value = y_value
        target_coord = 1
        other_coord = 0
        point_coord = point.x

    bound_nodes_coords = np.sort(nodes_coords[:, np.where(nodes_coords[target_coord] == value)[0]], axis=1)

    idx = (np.abs(bound_nodes_coords[other_coord] - point_coord)).argmin()
    if bound_nodes_coords[other_coord][idx] > point_coord:
        return bound_nodes_coords[other_coord][idx], bound_nodes_coords[other_coord][idx - 1]
    elif bound_nodes_coords[other_coord][idx] < point_coord:
        return bound_nodes_coords[other_coord][idx + 1], bound_nodes_coords[other_coord][idx]


def G_global(nodes, nodes_coords, nodes_number,y_value=False, x_value=False, y_bound=False, x_bound=False):

    # Нижняя горизонтальная граница, v = 0
    integration_points_y = create_integration_points_bound(nodes_coords, y_value=y_value, y_bound=y_bound)

    # Правая вертикальная граница, u = 0
    integration_points_x = np.flip(create_integration_points_bound(nodes_coords, x_value=x_value, x_bound=x_bound))

    rows = 2 * len(nodes)
    cols = nodes_number * 2 * 2
    G = np.zeros((rows, cols))

    bound_index = 0

    for points_between_nodes in integration_points_y:
        print(bound_index)
        for point in points_between_nodes:

            r_array = calculate_r(q_point=point, coords=nodes_coords)
            global_indexes = search_nodes_in_domain(r_array=r_array)

            nodes_in_domain = nodes[global_indexes.astype(int)]

            drdx, drdy = r_derivatives(r_array, nodes_coords, point)
            w, dwdx, dwdy = weight_func_array(r_array, drdx, drdy)
            F_array = F(point, nodes_in_domain, w)

            right, left = find_nearest_nodes(point=point, nodes_coords=nodes_coords, y_value=y_value, y_bound=y_bound)

            N1 = N(point.x, right, left)
            N2 = 1 - N1
            S = np.array([[0, 0], [0, 1]])

            for i in range(len(nodes_in_domain)):
                F_i = F_array[i] * np.eye(2)

                G1 = -point.jacobian * point.weight * np.dot(np.transpose(F_i), N1 * S)
                G2 = -point.jacobian * point.weight * np.dot(np.transpose(F_i), N2 * S)

                k = int(global_indexes[i])
                m = bound_index

                G[2 * k: 2 * k + 2, 2 * m: 2 * m + 2] += G1
                G[2 * k: 2 * k + 2, 2 * m: 2 * m + 2] += G2


    for points_between_nodes in integration_points_x:
        print(bound_index)
        for point in points_between_nodes:
            r_array = calculate_r(q_point=point, coords=nodes_coords)
            global_indexes = search_nodes_in_domain(r_array=r_array)

            nodes_in_domain = nodes[global_indexes.astype(int)]

            drdx, drdy = r_derivatives(r_array, nodes_coords, point)
            w, dwdx, dwdy = weight_func_array(r_array, drdx, drdy)
            F_array = F(point, nodes_in_domain, w)

            right, left = find_nearest_nodes(point=point, nodes_coords=nodes_coords, x_value=x_value, x_bound=x_bound)

            N1 = N(point.y, right, left)
            N2 = 1 - N1
            S = np.array([[1, 0], [0, 0]])

            for i in range(len(nodes_in_domain)):
                F_i = F_array[i] * np.eye(2)

                G1 = -point.jacobian * point.weight * np.dot(np.transpose(F_i), N1 * S)
                G2 = -point.jacobian * point.weight * np.dot(np.transpose(F_i), N2 * S)

                k = int(global_indexes[i])
                m = bound_index

                G[2 * k: 2 * k + 2, 2 * m: 2 * m + 2] += G1
                G[2 * k: 2 * k + 2, 2 * m: 2 * m + 2] += G2

        bound_index += 1


    print("Матрица G создана...")

    return G




