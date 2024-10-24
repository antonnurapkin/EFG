import numpy as np
from src.helpers import search_nodes_in_domain
from src.shape_function.components_shape_function.radius import calculate_r, r_derivatives
from src.integration_points import create_integration_points_bound
from src.shape_function.shape_function import F
from src.shape_function.components_shape_function.weight_function import weight_func_array
from src.params import P


def f_global(nodes, nodes_coords, x_bound=False, y_bound=False, x_value=False, y_value=False):

    f_global = np.zeros((2 * len(nodes), 1), dtype=np.float64)

    integration_points = create_integration_points_bound(nodes_coords, y_bound=y_bound, x_bound=x_bound, x_value=x_value, y_value=y_value)

    for points_between_nodes in integration_points:
        for point in points_between_nodes:

            r_array = calculate_r(q_point=point, coords=nodes_coords)
            global_indexes = search_nodes_in_domain(r_array=r_array)

            nodes_in_domain = nodes[global_indexes.astype(int)]

            drdx, drdy = r_derivatives(r_array, nodes_coords, point)
            w, dwdx, dwdy = weight_func_array(r_array, drdx, drdy)

            F_array = F(point, nodes_in_domain, w)

            for i in range(len(nodes_in_domain)):
                t = np.array([[0], [P]])
                f_local = point.jacobian * point.weight * F_array[i] * t

                k = int(global_indexes[i])

                f_global[2 * k] += f_local[0]
                f_global[2 * k + 1] += f_local[1]

    print(f"Вектор сил {f_global.shape} сформирован...")

    return f_global
