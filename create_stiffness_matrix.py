from meshing import cells
from meshing import Cell
from meshing import Node
from meshing import Point
from params import ds_x, ds_y
from shape_function import d2Fdx2, d2Fdydx, d2Fdy2
from meshing import cells
from shape_function import A


def search_nodes_in_domain(current_cell: Cell, point: Point):
    nodes_in_domain = []
    global_indexes = []

    all_interested_cells = current_cell.neighbors
    all_interested_cells.append(current_cell)

    # Просмотр соседей
    for neighbor in all_interested_cells:
        for node in neighbor.nodes:
            r_x, r_y = calculate_r(point_x=point.x, node_x=node.x, ds_x=ds_x, ds_y=ds_y)
            if r_x <= 1 and r_y <= 1:
                if len(global_indexes) == 0:
                    global_indexes.append(node.global_index)
                    nodes_in_domain.append(node)
                elif len(global_indexes) > 0 and node.global_index not in global_indexes:
                    global_indexes.append(node.global_index)
                    nodes_in_domain.append(node)

    return nodes_in_domain


def calculate_r(point_x, node_x, ds_x, ds_y):
    r_x = abs(point_x - node_x) / ds_x
    r_y = abs(point_x - node_x) / ds_y

    return r_x, r_y


def shape_functions_array(nodes_in_domain):
    F = [[],[],[]]
    
    for node in nodes_in_domain:
        F[0].append(-d2Fdx2(node, nodes_in_domain))
        F[1].append(-d2Fdy2(node, nodes_in_domain))
        F[0].append(-2 * d2Fdydx(node, nodes_in_domain))
                    
    return F

for i in range(len(cells)):
    for j in range(len(cells[i])):
        for point in cells[i][j].gauss_points:
            nodes_in_domain = search_nodes_in_domain(current_cell=cells[i][j], point=point)
            F = shape_functions_array(nodes_in_domain=nodes_in_domain)
            break
        





        



