import numpy as np
from helpers import search_nodes_in_domain
from components_shape_function.radius import calculate_r, r_derivatives
from components_shape_function.weight_function import weight_func_array
from shape_function import F
from params import b, THICKNESS, l_x, l_y, elem_x, elem_y
from plotly import graph_objects as go


def create_f_vector(F, b, nodes_in_domain, weight, jacobian, global_indexes):
    size = len(nodes_in_domain)
    f_local_vector_n_index = np.empty((0, 2))

    for i in range(size):
        f_local_vector_n_index = np.append(
            f_local_vector_n_index,
            [
                [
                    THICKNESS * jacobian * weight * F[i] * b[2, 0],
                    global_indexes[i]
                ]
            ],
            axis=0
        )

    return f_local_vector_n_index

def f_global(cells, n_x, n_y, nodes, coords):

    f_global = np.zeros((n_x * n_y, 1))

    for i in range(len(cells)):
        for j in range(len(cells[i])):
            for point in cells[i][j].gauss_points:
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
                    cells[i][j].jacobian,
                    global_indexes
                )

                indexes = f_local_vector_n_indexes[:, 1]
                f_global[indexes.astype(int), 0] += f_local_vector_n_indexes[:, 0]

    print("Вектор сил сформирован")

    return f_global
