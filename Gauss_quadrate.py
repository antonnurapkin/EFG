from meshing import cells
from meshing import Cell
from meshing import Node
from meshing import Point
from params import ds_x, ds_y
from shape_function import d2Fdx2, d2Fdydx, d2Fdy2
from meshing import cells
from shape_function import A


def search_nodes_in_domain(cell: Cell, point: Point):
    nodes_in_domain = []

    for neighbor in cell.neighbors:
        for node in neighbor.nodes:
            
            r_x = abs(node.x - neighbor.x) / ds_x
            r_y = abs(node.x - neighbor.x) / ds_y

            if r_x <= 1 and r_y <= 1:
                nodes_in_domain.append(node)

    return nodes_in_domain


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
            nodes_in_domain = search_nodes_in_domain(cell=cells[i][j], point=point)
            F = shape_functions_array(nodes_in_domain=nodes_in_domain)
            break
        
print(F)




        



