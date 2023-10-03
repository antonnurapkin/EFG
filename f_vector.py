import numpy as np


def create_f_vector(nodes_in_domain, B, b, weight, jacobian, global_indexes):
    size = len(nodes_in_domain)
    f_local_vector_n_index = np.empty((0,2))

    for i in range(size):
        B_i = B[:, i]
        f_local_vector_n_index = np.append(
            f_local_vector_n_index,
            [
                [
                    jacobian * weight * np.dot(np.transpose(B_i), b),
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
