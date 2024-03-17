from helpers import search_nodes_in_domain
from components_shape_function.radius import calculate_r, r_derivatives
from components_shape_function.weight_function import weight_func_array
from shape_function import F
import numpy as np


def write_solution(nodes, u):
    for i in range(len(nodes)):
        if nodes[i].u_solution is None and nodes[i].v_solution is None:
            nodes[i].u_solution = u[2 * i][0]
            nodes[i].v_solution = u[2 * i + 1][0]

    return nodes


def calculate_nodal_displacement(F, nodes):
    u, v = 0, 0
    for i in range(len(F)):
        u += F[i] * nodes[i].u_solution
        v += F[i] * nodes[i].v_solution

    return u, v


def calculate_real_displacements(nodes, coords):
    for node in nodes:
        r_array = calculate_r(q_point=node, coords=coords)
        global_indexes = search_nodes_in_domain(r_array=r_array)

        nodes_in_domain = nodes[global_indexes.astype(int)]

        drdx, drdy = r_derivatives(r_array, coords, node)
        w = weight_func_array(r_array, drdx, drdy, weight_function=True)

        F_array = F(node, nodes_in_domain, w)
        node.u_real, node.v_real = calculate_nodal_displacement(F_array, nodes_in_domain)

    return nodes


def get_real_displacements(nodes, u, coords):

    nodes = write_solution(nodes=nodes, u=u)
    nodes = calculate_real_displacements(nodes=nodes, coords=coords)
    return nodes



