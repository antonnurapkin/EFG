import numpy as np

from src.helpers import find_nearest_nodes, N
from src.integration_points import create_integration_points_bound


def q_global(nodes_coords, rows, y_value=False, x_value=False, y_bound=False, x_bound=False):
    # Нижняя горизонтальная граница, v = 0
    integration_points_bottom = np.flip(create_integration_points_bound(nodes_coords, y_value=y_value, y_bound=y_bound))

    q = np.zeros((rows, 1))

    bound_index = 0

    # Нижняя горизонтальная граница, v = 0
    for points_between_nodes in integration_points_bottom:
        for point in points_between_nodes:

            right, left = find_nearest_nodes(point=point, nodes_coords=nodes_coords, y_value=y_value, y_bound=y_bound)

            N1 = N(point.x, left, right)
            N2 = 1 - N1
            S = np.array([[1, 0], [0, 1]])

            # Вычисления точного решениия(см. учебник Демидова) в конкретной точке Гаусса
            u_exact = np.array([[0], [0]])

            q1 = -point.jacobian * point.weight * np.dot(N1 * S, u_exact)
            q2 = -point.jacobian * point.weight * np.dot(N2 * S, u_exact)

            k = bound_index
            l = bound_index + 1

            q[2 * k] += q1[0]
            q[2 * k + 1] += q1[1]

            q[2 * l] += q2[0]
            q[2 * l + 1] += q2[1]

        bound_index += 1

    print(f"Вектор q {q.shape} создан...")

    return q
