from src.helpers import dF_array, B_matrix, search_nodes_in_domain
from src.shape_function.components_shape_function.radius import calculate_r, r_derivatives
from src.shape_function.components_shape_function.weight_function import weight_func_array, weight_func
from src.shape_function.shape_function import F
from src.params import D, DS
import numpy as np
from src.crack.utils import check_crack_interaction
from plotly import graph_objects as go


def K_global(nodes, integration_points, nodes_coords, crack_top_ind):
    K = np.zeros((2 * len(nodes), 2 * len(nodes)), dtype=np.float64)

    for point in integration_points:
        # Первичный расчёт
        r_array = calculate_r(q_point=point, coords=nodes_coords)
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

        global_indexes = search_nodes_in_domain(r_array=r_array)
        # Создание матрицы узловой матрица жёсткости
        for i in range(len(nodes_in_domain)):
            for j in range(len(nodes_in_domain)):
                # Матрица связи деформаций и перемещений
                B_i = B_matrix(dF[:, i])
                B_j = B_matrix(dF[:, j])

                K_local = point.jacobian * point.weight * np.dot(np.transpose(B_i), np.dot(D, B_j))

                k = int(global_indexes[i])
                m = int(global_indexes[j])

                K[2 * k: 2 * k + 2, 2 * m: 2 * m + 2] += K_local

    print(f"Матрица жесткости {K.shape} сформирована...")

    return K
