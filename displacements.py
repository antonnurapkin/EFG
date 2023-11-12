from helpers import search_nodes_in_domain
from shape_function import F
import numpy as np


def write_solve_results(nodes, u):
    for i in range(len(nodes)):
        if nodes[i].z_solve is None:
            nodes[i].z_solve = u[i]

    return nodes


def calculate_nodal_displacement(F, nodes):
    u = 0
    for i in range(len(F)):
        u += F[i] * nodes[i].z_solve

    return u


def calculate_real_displacements(nodes, coords):
    for node in nodes:
        global_indexes = search_nodes_in_domain(q_point=node, coords=coords)

        nodes_in_domain = nodes[global_indexes.astype(int)]
        F_array = F(node, nodes_in_domain)
        node.z = calculate_nodal_displacement(F_array, nodes_in_domain)

    return nodes


def create_U_array(nodes, n_x, n_y):

    U = []

    for node in nodes:
        U.append(node.z)

    return np.transpose(np.reshape(np.array(U), (n_y, n_x)))


def get_displaments_array(nodes, u, n_x, n_y, coords):

    nodes = write_solve_results(nodes=nodes, u=u)
    nodes = calculate_real_displacements(nodes=nodes, coords=coords)
    U = create_U_array(nodes=nodes, n_x=n_x, n_y=n_y)
    return U



