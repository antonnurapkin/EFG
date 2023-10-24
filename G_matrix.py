from helpers import search_nodes_in_domain, indexes_for_bound_nodes, get_bound_elems, create_s_matrix
from shape_function import F
import numpy as np


def N(g_pos, left_bound, right_bound):
    return (right_bound - g_pos) / (right_bound - left_bound)


def create_G_nodal_vector(F, Ni, weight, jacobian, global_indexes, bound_local_index):
    G_local_vector_n_indexes = np.empty((0, 3))

    for i in range(len(F)):
        G_local_vector_n_indexes = np.append(
            G_local_vector_n_indexes,
            [
                [
                    -jacobian * weight * Ni * F[i],
                    global_indexes[i],
                    bound_local_index
                ]
            ],
            axis=0
        )

    return G_local_vector_n_indexes


def create_G_matrix(bound_cells, coords, indexes, n_x, n_y):
    index_node = 0
    index_cell = 0

    current_coord = 1

    counter = 0

    i = 1
    j = 0

    jIncrease = True
    isCorner = False
    isLast = False

    G_global = np.zeros((n_x * n_y, len(indexes)))

    while counter < len(coords[0]):

        #  Костыли для последнего узла
        if isLast:
            # index_node -= 1
            second = 0
            index_cell = 0
            current_coord = 0
        else:
            second = index_node + 1

        first = index_node

        length_bpoints = len(bound_cells[index_cell].boundary_Gauss_points)
        for point in bound_cells[index_cell].boundary_Gauss_points[:4]:
            nodes_in_domain, global_indexes = search_nodes_in_domain(q_point=point, current_cell=bound_cells[index_cell])

            if current_coord == 1:
                s_coord = point.y
            else:
                s_coord = point.x

            N_interpolant = N(s_coord, coords[current_coord][first], coords[current_coord][second])

            F_array = F(point, nodes_in_domain)

            G_local_vector_n_indexes = create_G_nodal_vector(
                F_array,
                N_interpolant,
                point.weight,
                bound_cells[index_cell].jacobian,
                global_indexes,
                index_node
            )

            index_1 = G_local_vector_n_indexes[:, 1]
            index_2 = G_local_vector_n_indexes[:, 2]

            G_global[index_1.astype(int), index_2.astype(int)] += G_local_vector_n_indexes[:, 0]

        if length_bpoints == 8:
            bound_cells[index_cell].boundary_Gauss_points = bound_cells[index_cell].boundary_Gauss_points[4:]

        # Смена индексов и координат
        index_node += 1
        counter += 1
        if isCorner is False:
            index_cell += 1

        if index_node == (n_y - 1) * i + (n_x - 1) * j:
            if jIncrease:
                j += 1
                current_coord = 0
                jIncrease = False
            else:
                i += 1
                current_coord = 1
                jIncrease = True
            isCorner = True

            if isCorner:
                index_cell -= 1
                isCorner = False
        elif index_node == len(coords[0]) - 1:
            isLast = True
            index_cell = 0

    return G_global

def G_global(cells, n_x, n_y):
    indexes = indexes_for_bound_nodes(n_x, n_y)
    S = create_s_matrix(cells, indexes)
    bound_cells = get_bound_elems(cells)

    # for cell in bound_cells:
    #     print("Координаты клетки", cell.x, cell.y)
    #     print("Координаты точек:", end=" ")
    #     for point in cell.boundary_Gauss_points:
    #         print(round(point.x,5), round(point.y,5), end=", ")
    #     print()

    return create_G_matrix(bound_cells, S, indexes, n_x, n_y)




