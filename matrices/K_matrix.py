from helpers import dF_array, B_matrix, search_nodes_in_domain
from shape_function.components_shape_function.radius import calculate_r, r_derivatives
from shape_function.components_shape_function.weight_function import weight_func_array, weight_func
from shape_function.shape_function import F
from params import D, DS
import numpy as np
from crack.utils import check_crack_interaction
from plotly import graph_objects as go


def K_global(nodes, integration_points, nodes_coords, crack_top_ind):
    K = np.zeros((2 * len(nodes), 2 * len(nodes)), dtype=np.float64)

    l = 0
    temp = 195
    for point in integration_points:

        if 0.48 <= point.x <= 0.52 and 0.48 <= point.y <= 0.52:
            print("Debug")

        # Вычисляется некое характеристическое расстояние от точки интегрирования до всех узлов
        r_array = calculate_r(q_point=point, coords=nodes_coords)

        # Вычисляются индексы тех, кто в области поддержки
        global_indexes = search_nodes_in_domain(r_array=r_array)

        nodes_in_domain = nodes[global_indexes.astype(int)]

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

        if False:
            w = np.zeros(r_array.shape)

            for i in range(len(r_array)):
                w[i] = weight_func(r_array[i])

            fig = go.Figure(go.Scatter3d(x=nodes_coords[0], y=nodes_coords[1], z=w, mode="lines+markers"))
            fig.add_trace(go.Scatter3d(x=[point.x], y=[point.y], z=[0]))
            fig.update_xaxes(range=[0, 1])
            fig.update_yaxes(range=[0, 1])

            # fig.add_shape(
            #     type="circle",
            #     x0=point.x - DS,
            #     y0=point.y - DS,
            #     x1=point.x + DS,
            #     y1=point.y + DS
            # )
            #
            # fig.add_shape(
            #     type="line",
            #     x0=0,
            #     y0=0.5,
            #     x1=0.5,
            #     y1=0.5,
            # )

            fig.show()
            exit()

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

        l += 1

    print(f"Матрица жесткости {K.shape} сформирована...")

    return K
