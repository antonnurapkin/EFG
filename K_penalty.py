from helpers import search_nodes_in_domain, get_bound_elems, indexes_for_bound_nodes
from shape_function import F
from params import D, penalty_alpha
import numpy as np


def create_K_penalty_nodal_vector(F, D, nodes_in_domain, weight, jacobian, global_indexes):
    size = len(nodes_in_domain)
    K_local_vector_n_indexes = np.empty((0, 3))

    for i in range(size):
        for j in range(size):
            K_local_vector_n_indexes = np.append(
                K_local_vector_n_indexes,
                [
                    [
                        jacobian * weight * F[i] * penalty_alpha * F[j],
                        global_indexes[i],
                        global_indexes[j]
                    ]
                ],
                axis=0
            )

    return K_local_vector_n_indexes


def create_K_penalty_global(bound_cells, indexes, n_x, n_y, nodes, coords):
    K_penalty = np.zeros((n_x * n_y, n_x * n_y))

    index_node = 0
    index_cell = 0

    counter = 0

    i = 1
    j = 0

    jIncrease = True
    isCorner = False
    isLast = False

    while counter < len(indexes):

        #  Костыль для последнего узла
        if isLast:
            index_cell = 0

        for point in bound_cells[index_cell].boundary_Gauss_points[:4]:
            global_indexes = search_nodes_in_domain(q_point=point, coords=coords)

            nodes_in_domain = nodes[global_indexes.astype(int)]

            F_array = F(point, nodes_in_domain)

            # Создание матрицы K
            K_local_vector_n_indexes = create_K_penalty_nodal_vector(
                F_array,
                D,
                nodes_in_domain,
                point.weight,
                bound_cells[i].jacobian,
                global_indexes
            )

            index_1 = K_local_vector_n_indexes[:, 1]
            index_2 = K_local_vector_n_indexes[:, 2]
            K_penalty[index_1.astype(int), index_2.astype(int)] += K_local_vector_n_indexes[:, 0]

        if len(bound_cells[index_cell].boundary_Gauss_points[:4]) == 8:
            bound_cells[index_cell].boundary_Gauss_points = bound_cells[index_cell].boundary_Gauss_points[4:]

        # Смена индексов и координат
        index_node += 1
        counter += 1
        if isCorner is False:
            index_cell += 1

        if index_node == (n_y - 1) * i + (n_x - 1) * j:
            if jIncrease:
                j += 1
                jIncrease = False
            else:
                i += 1
                jIncrease = True
            isCorner = True

            if isCorner:
                index_cell -= 1
                isCorner = False
        elif index_node == len(indexes) - 1:
            isLast = True
            index_cell = 0

    return K_penalty


def K_penalty_global(cells, n_x, n_y, nodes, coords):
    indexes = indexes_for_bound_nodes(n_x=n_x, n_y=n_y)
    bound_cells = get_bound_elems(cells=cells)

    return create_K_penalty_global(
        bound_cells=bound_cells,
        indexes=indexes,
        n_x=n_x,
        n_y=n_y,
        nodes=nodes,
        coords=coords
    )
