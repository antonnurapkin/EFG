import numpy as np
from helpers import search_nodes_in_domain
from components_shape_function.radius import calculate_r, r_derivatives
from components_shape_function.weight_function import weight_func_array
from shape_function import F
from integration_points import create_integration_points_bound


def f_global(nodes, nodes_coords, nodes_number, x_bound=False, y_bound=False):

    f_global = np.zeros((2 * len(nodes), 1))

    integration_points = create_integration_points_bound(nodes_coords, x_bound=x_bound, y_bound=y_bound)

    for point in Gauss_points:
        r_array = calculate_r(q_point=point, coords=coords)
        global_indexes = search_nodes_in_domain(r_array=r_array)

        nodes_in_domain = nodes[global_indexes.astype(int)]

        drdx, drdy, d2rdx2, d2rdy2, d2rdxdy = r_derivatives(r_array, coords, point)
        w = weight_func_array(r_array, drdx, drdy, d2rdx2, d2rdy2, d2rdxdy)[0]

        F_array = F(point, nodes_in_domain, w)

        f_local_vector_n_indexes = create_f_vector(
            F_array,
            b,
            nodes_in_domain,
            point.weight,
            point.jacobian,
            global_indexes
        )

        indexes = f_local_vector_n_indexes[:, 1]
        f_global[indexes.astype(int), 0] += f_local_vector_n_indexes[:, 0]

    print("Вектор сил сформирован")

    return f_global
