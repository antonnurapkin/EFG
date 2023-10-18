from params import D, n_x, n_y, elem_x, elem_y
from meshing import cells, Cell
from helpers import search_nodes_in_domain
from shape_function import F
import numpy as np
import pandas as pd


def indexes_for_nodes(n_x, n_y):
    left = np.array([i for i in range(n_y)])
    bottom = np.array([n_y * (i + 1) - 1 for i in range(n_x)])
    right = np.flip(np.array([(n_x - 1) * n_y + i for i in range(n_y)]))
    top = np.flip(np.array([n_y * i for i in range(n_x)]))

    global_bound_indexes = delete_duplicates(np.concatenate([left, bottom, right, top]))
    local_bound_indexes = np.arange(0, len(global_bound_indexes), 1)

    result = np.transpose(np.vstack((global_bound_indexes, local_bound_indexes)))

    return result


def delete_duplicates(arr):
    to_delete = [i for i in range(len(arr) - 1) if arr[i] == arr[i + 1]]

    return np.delete(arr, to_delete)[:-1]


def N(g_pos, left_bound, right_bound):
    return (right_bound - g_pos) / (right_bound - left_bound)


def create_s_matrix(cells, indexes):
    s = np.zeros((2, len(indexes)))

    for i in range(len(cells)):
        for j in range(len(cells[i])):
            for node in cells[i][j].nodes:
                if node.global_index in indexes[:, 0]:
                    insert_index = np.where(indexes[:, 0] == node.global_index)[0][0]
                    s[0][insert_index] = node.x
                    s[1][insert_index] = node.y

    return s

def get_bound_elems(cells):
    left_bound = cells[:, 0]
    bottom_bound = cells[-1, :][1:]
    right_bound = np.flip(cells[:, -1][:-1])
    top_bound = np.flip(cells[0, :][1:-1])

    return np.concatenate((left_bound, bottom_bound, right_bound, top_bound))


def create_G_nodal_vector(F, Ni, weight, jacobian, global_indexes, bound_local_index):
    G_local_vector_n_indexes = np.empty((0, 3))

    for i in range(len(F)):
        G_local_vector_n_indexes = np.append(
            G_local_vector_n_indexes,
            [
                [
                    jacobian * weight * Ni * F[i],
                    global_indexes[i],
                    bound_local_index
                ]
            ],
            axis=0
        )

    return G_local_vector_n_indexes


def create_G_matrix(bound_cells: list[Cell], coords, indexes, n_x, n_y):
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

        first = index_node

        if isLast:
            index_node -= 1
            second = 0
            index_cell = 0
            first = index_node
        else:
            second = index_node + 1


        for point in bound_cells[index_cell].boundary_Gauss_points[:2]:
            nodes_in_domain, global_indexes = search_nodes_in_domain(q_point=point, current_cell=bound_cells[index_cell])

            if current_coord:
                s_coord = point.y
            else:
                s_coord = point.x

            print("Координаты ячейки:", bound_cells[index_cell].x, bound_cells[index_cell].y)
            print(coords[current_coord][first], (point.x, point.y), coords[current_coord][second])
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

        if len(bound_cells[index_cell].boundary_Gauss_points) == 4:
            bound_cells[index_cell].boundary_Gauss_points = bound_cells[index_cell].boundary_Gauss_points[:2]

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


indexes = indexes_for_nodes(n_x, n_y)
S = create_s_matrix(cells, indexes)
bound_cells = get_bound_elems(cells)

# for cell in bound_cells:
#     print("Координаты ячейки:",cell.x, cell.y)
#     for point in cell.boundary_Gauss_points:
#         print(point.x,",",point.y, end=" | ")
#
#     print()
G = create_G_matrix(bound_cells, S, indexes, n_x, n_y)
#
# df = pd.DataFrame(G)
# df.to_excel("G.xlsx")



