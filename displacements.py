from helpers import search_nodes_in_domain
from shape_function import F
import numpy as np


def set_solve_results(cells, u):
    for row in cells:
        for cell in row:
            for node in cell.nodes:
                if node.z_solve is None:
                    node.z_solve = u[node.global_index]

    return cells


def calculate_nodal_displacement(F, nodes):
    u = 0
    for i in range(len(F)):
        u += F[i] * nodes[i].z_solve

    return u


def calculate_real_displacements(cells):
    for row in cells:
        for cell in row:
            for node in cell.nodes:
                if node.calculated_displ is False:
                    nodes_in_domain, global_indexes = search_nodes_in_domain(cell, node)
                    F_array = F(node, nodes_in_domain)

                    node.z = calculate_nodal_displacement(F_array, nodes_in_domain)

    return cells


def create_U_array(cells, n_x, n_y):

    added_indexes = []

    U = np.zeros((n_y * n_x, 1))
    for i in range(len(cells[0])):
        for j in range(len(cells)):
            for node in cells[j][i].nodes:
                if node.global_index not in added_indexes:
                    U[node.global_index] = node.z
                    added_indexes.append(node.global_index)

    return np.transpose(np.reshape(U, (n_y, n_x)))


def get_displaments_array(cells, u, n_x, n_y):

    cells = set_solve_results(cells=cells, u=u)
    cells = calculate_real_displacements(cells=cells)
    U = create_U_array(cells, n_x, n_y)
    return U



