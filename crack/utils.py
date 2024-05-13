import numpy as np
from params import K, CONST, DS, x1, x2, y1, y2
from crack.radius import r_diffraction_method
from crack.radius import drdx, drdy


def check_crack_interaction(point, r_array, nodes_in_domain, crack_top_ind, nodes, global_indexes):
    drdx_array, drdy_array = None, None

    if is_interaction(point):
        if crack_top_ind in global_indexes:
            r_array, drdx_array, drdy_array, nodes_in_domain = diffraction_method(
                r_array=r_array,
                crack_top_ind=crack_top_ind,
                point=point,
                nodes_in_domain=nodes_in_domain,
                nodes=nodes
            )
        else:
            drdx_array, drdy_array = None, None
            r_array, nodes_in_domain = remove_other_coast(r_array=r_array, point=point, nodes_in_domain=nodes_in_domain)

    return r_array, drdx_array, drdy_array, nodes_in_domain


def is_interaction(point):
    a = 1 + K ** 2
    b = (2 * K * (CONST - point.y) - 2 * point.x)
    c = (point.x ** 2 + (CONST - point.y) ** 2 - DS ** 2)

    roots = np.roots([a, b, c])

    # Если область пересекает трещину
    for root in roots:
        if np.isreal(root) and x1 <= root <= x2 and y1 <= crack_line(root) <= y2:
            return True

    # Если вся трещина внутри области поддержки
    if point.x - DS <= x1 <= point.x + DS and \
            point.x - DS <= x2 <= point.x + DS and \
            point.y - DS <= y1 <= point.y + DS and \
            point.y - DS <= y2 <= point.y + DS:
        return True

    return False


def crack_line(x):
    return K * x + CONST


def diffraction_method(r_array, crack_top_ind, point, nodes_in_domain, nodes):
    nodes_in_domain_new = list(nodes_in_domain)  # TODO: Сделать удаление по значению с ndarray

    drdx_array = np.zeros(r_array.shape)
    drdy_array = np.zeros(r_array.shape)

    for node in nodes_in_domain:
        if (point.y < crack_line(point.x) and node.y > crack_line(node.x)) or \
                (point.y > crack_line(point.x) and node.y < crack_line(node.x)):
            r = r_diffraction_method(point=point, crack_top=nodes[crack_top_ind], node=node)
            r_array[node.global_index] = r

            if r > 1:
                nodes_in_domain_new.remove(node)
            else:
                drdx_array[node.global_index] = drdx(point=point, crack_top=nodes[crack_top_ind], node=node)
                drdy_array[node.global_index] = drdy(point=point, crack_top=nodes[crack_top_ind], node=node)

    return r_array, drdx_array, drdy_array, np.array(nodes_in_domain_new)


def remove_other_coast(r_array, point, nodes_in_domain):
    nodes_in_domain_new = list(nodes_in_domain)

    for node in nodes_in_domain:
        if (point.y < crack_line(point.x) and node.y > crack_line(node.x)) or \
                (point.y > crack_line(point.x) and node.y < crack_line(node.x)):
            r_array[node.global_index] = 1.5
            nodes_in_domain_new.remove(node)

    return r_array, np.array(nodes_in_domain_new)
