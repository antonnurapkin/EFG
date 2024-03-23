import numpy as np
from helpers import get_x_coord, get_y_coord


class Node:
    def __init__(self, x, y, global_index):
        self.x = x
        self.y = y
        self.global_index = global_index

        self.u_solution = None
        self.v_solution = None

        self.u_real = None
        self.u_real = None


def create_nodes(nodes_number_radial, nodes_number_tetta, a, b, fi_delta, r0):

    nodes = np.array([])
    global_index = 0

    # Вертикальная граница
    for i in range(nodes_number_tetta):
        x_right_bound = a
        y_right_bound = i * (b / (nodes_number_tetta - 1))
        fi_current = fi_delta * i

        distance = np.sqrt((x_right_bound - get_x_coord(r0, fi_current)) ** 2 + (y_right_bound - get_y_coord(r0, fi_current)) ** 2)
        r_delta = distance / (nodes_number_radial - 1)

        for j in range(nodes_number_radial):

            if j == nodes_number_radial - 1:
                x_coord = x_right_bound
            else:
                x_coord = get_x_coord(r_delta * j + r0, fi_current)

            y_coord = get_y_coord(r_delta * j + r0, fi_current)

            nodes = np.append(nodes, [Node(x_coord, y_coord, global_index)])
            global_index += 1

    fi_middle = fi_current

    # Горизонтальная граница
    for i in range(nodes_number_tetta - 1):
        x_top_bound = a - (i + 1) * (a / (nodes_number_tetta - 1))
        y_top_bound = b
        fi_current = fi_delta * (i + 1) + fi_middle

        distance = np.sqrt((x_top_bound - get_x_coord(r0, fi_current)) ** 2 + (y_top_bound - get_y_coord(r0, fi_current)) ** 2)
        r_delta = distance / (nodes_number_radial - 1)

        for j in range(nodes_number_radial):

            if j == nodes_number_radial - 1:
                y_coord = y_top_bound
            else:
                y_coord = get_y_coord(r_delta * j + r0, fi_current)

            x_coord = get_x_coord(r_delta * j + r0, fi_current)

            nodes = np.append(nodes, [Node(x_coord, y_coord, global_index)])
            global_index += 1

    print("Узлы созданы...")

    return nodes


def get_nodes_coords(nodes):
    coords = np.empty((2, 0), dtype='f')

    for node in nodes:
        coords = np.append(arr=coords, values=[[node.x], [node.y]], axis=1)

    print("Получены координаты узлов...")

    return np.around(coords, decimals=10)


