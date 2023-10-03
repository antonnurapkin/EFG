import numpy as np

from EFG.meshing import Cell, Point
from EFG.params import ds_x, ds_y

from EFG.shape_function import d2Fdx2, d2Fdy2, d2Fdydx


def calculate_r(point, node, ds_x, ds_y):
    r_y = abs(point.x - node.x) / ds_y
    r_x = abs(point.y - node.y) / ds_x

    return r_x, r_y


def B_matrix(q_point, nodes_in_domain):

    F_vector = np.vstack(
        (
            -d2Fdx2(q_point, nodes_in_domain),
            -d2Fdy2(q_point, nodes_in_domain),
            -2 * d2Fdydx(q_point, nodes_in_domain)
        )
    )

    return np.resize(F_vector, (3, len(nodes_in_domain)))


def search_nodes_in_domain(current_cell: Cell, q_point: Point):
    nodes_in_domain = []
    all_interested_cells = current_cell.neighbors
    all_interested_cells.append(current_cell)

    global_indexes = np.array([])

    # Просмотр соседей
    for neighbor in all_interested_cells:
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
