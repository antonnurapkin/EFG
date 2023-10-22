import numpy as np
from helpers import search_nodes_in_domain, F_vector
from params import b, THICKNESS

def create_f_vector(F, b, nodes_in_domain, weight, jacobian, global_indexes):
    size = len(nodes_in_domain)
    f_local_vector_n_index = np.empty((0, 2))

    for i in range(size):
        f_local_vector_n_index = np.append(
            f_local_vector_n_index,
            [
                [
                    THICKNESS * jacobian * weight * F[i] * b[2, 0],
                    global_indexes[i]
                ]
            ],
            axis=0
        )

    return f_local_vector_n_index


def f_global_assemble(f_global, f_local_vector_n_indexes):
    for elem in f_local_vector_n_indexes:
        f_global[int(elem[1])] += elem[0]

    return f_global


def f_global(cells, n_x, n_y):

    f_global = np.zeros((n_x * n_y, 1))

    for i in range(len(cells)):
        for j in range(len(cells[i])):
            for point in cells[i][j].gauss_points:

                nodes_in_domain, global_indexes = search_nodes_in_domain(q_point=point, current_cell=cells[i][j])

                F = F_vector(q_point=point, nodes_in_domain=nodes_in_domain)

                f_local_vector_n_indexes = create_f_vector(
                    F,
                    b,
                    nodes_in_domain,
                    point.weight,
                    cells[i][j].jacobian,
                    global_indexes
                )
                #TODO: Оптимизировать с помощью индексации в numpy
                f_global = f_global_assemble(f_global, f_local_vector_n_indexes)

    return f_global
