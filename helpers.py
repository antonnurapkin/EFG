import numpy as np

from meshing import Cell, Point
from params import ds_x, ds_y

from shape_function import d2Fdx2, d2Fdy2, d2Fdydx, F


def calculate_r(point, node, ds_x, ds_y):
    r_x = abs(point.x - node.x) / ds_x
    r_y = abs(point.y - node.y) / ds_y

    return r_x, r_y


def B_matrix(q_point, nodes_in_domain):

    B_vector = np.vstack(
        (
            -d2Fdx2(q_point, nodes_in_domain),
            -d2Fdy2(q_point, nodes_in_domain),
            -2 * d2Fdydx(q_point, nodes_in_domain)
        )
    )

    return np.resize(B_vector, (3, len(nodes_in_domain)))

def F_vector(q_point, nodes_in_domain):
    return F(q_point, nodes_in_domain)


def search_nodes_in_domain(current_cell: Cell, q_point: Point):
    nodes_in_domain = []
    all_interested_cells = current_cell.neighbors
    all_interested_cells.append(current_cell)

    global_indexes = np.array([])

    # Просмотр соседей
    for neighbor in all_interested_cells:
        # nei_x = neighbor.x
        # nei_y = neighbor.y
        for node in neighbor.nodes:
            r_x, r_y = calculate_r(point=q_point, node=node, ds_x=ds_x, ds_y=ds_y)
            if r_x <= 1 and r_y <= 1:
                if len(global_indexes) == 0:
                    global_indexes = np.append(global_indexes, node.global_index)
                    nodes_in_domain.append(node)
                elif len(global_indexes) > 0 and node.global_index not in global_indexes:
                    global_indexes = np.append(global_indexes, node.global_index)
                    nodes_in_domain.append(node)

    return nodes_in_domain, global_indexes


def delete_duplicates(arr):
    to_delete = [i for i in range(len(arr) - 1) if arr[i] == arr[i + 1]]

    return np.delete(arr, to_delete)[:-1]


def indexes_for_bound_nodes(n_x, n_y):
    left = np.array([i for i in range(n_y)])
    bottom = np.array([n_y * (i + 1) - 1 for i in range(n_x)])
    right = np.flip(np.array([(n_x - 1) * n_y + i for i in range(n_y)]))
    top = np.flip(np.array([n_y * i for i in range(n_x)]))

    global_bound_indexes = delete_duplicates(np.concatenate([left, bottom, right, top]))
    local_bound_indexes = np.arange(0, len(global_bound_indexes), 1)

    result = np.transpose(np.vstack((global_bound_indexes, local_bound_indexes)))

    return result


def get_bound_elems(cells):
    left_bound = cells[:, 0]
    bottom_bound = cells[-1, :][1:]
    right_bound = np.flip(cells[:, -1][:-1])
    top_bound = np.flip(cells[0, :][1:-1])

    return np.concatenate((left_bound, bottom_bound, right_bound, top_bound))


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
