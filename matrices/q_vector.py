import numpy as np

from helpers import search_nodes_in_domain, find_nearest_nodes, N
from integration_points import create_integration_points_bound
from exact_solution import u_radial


def q_global(nodes, nodes_coords, rows, y_value=False, x_value=False, y_bound=False, x_bound=False):
    # Нижняя горизонтальная граница, v = 0
    integration_points_y = create_integration_points_bound(nodes_coords, y_value=y_value, y_bound=y_bound)

    # Правая вертикальная граница, u = 0
    integration_points_x = np.flip(create_integration_points_bound(nodes_coords, x_value=x_value, x_bound=x_bound))

    q = np.zeros((rows, 1))

    bound_index = 0

    for points_between_nodes in integration_points_y:
        for point in points_between_nodes:

            right, left = find_nearest_nodes(point=point, nodes_coords=nodes_coords, y_value=y_value, y_bound=y_bound)

            N1 = N(point.x, right, left)
            N2 = 1 - N1
            S = np.array([[1, 0], [0, 1]])

            # Вычисления точного решениия(см. учебник Демидова) в конкретной точке Гаусса
            u_exact = np.array([[0], [u_radial(r=point.x, tetta=0)]])

            q1 = -point.jacobian * point.weight * np.dot(N1 * S, u_exact)
            q2 = -point.jacobian * point.weight * np.dot(N2 * S, u_exact)

            k = bound_index
            l = bound_index + 1

            q[2 * k] += q1[0]
            q[2 * k + 1] += q1[1]

            q[2 * l] += q2[0]
            q[2 * l + 1] += q2[1]

        bound_index += 1

    bound_index += 1

    # Левая вертикальная граница, u = 0
    for points_between_nodes in integration_points_x:
        for point in points_between_nodes:
            right, left = find_nearest_nodes(point=point, nodes_coords=nodes_coords, x_value=x_value, x_bound=x_bound)

            N1 = N(point.y, right, left)
            N2 = 1 - N1
            S = np.array([[1, 0], [0, 1]])

            u_exact = np.array([u_radial(r=point.y, tetta=np.pi / 2), 0])

            q1 = -point.jacobian * point.weight * np.dot(N1 * S, u_exact)
            q2 = -point.jacobian * point.weight * np.dot(N2 * S, u_exact)

            k = bound_index
            l = bound_index + 1

            q[2 * k] += q1[0]
            q[2 * k + 1] += q1[1]

            q[2 * l] += q2[0]
            q[2 * l + 1] += q2[1]

        bound_index += 1

    print("Вектор q создан...")

    return q
