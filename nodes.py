import numpy as np
from helpers import get_x_coord, get_y_coord
from params import A, B, NODES_NUMBER_RADIAL_NEAR_BOUNDS, NODES_NUMBER_RADIAL_NEAR_HOLE, NODES_NUMBER_TETTA, R0, FI_DELTA, MULTIPLY_COEFF


class Node:
    def __init__(self, x, y, global_index):
        self.x = x
        self.y = y
        self.global_index = global_index

        self.u_solution = None
        self.v_solution = None

        self.u_real = None
        self.u_real = None


def create_nodes():

    nodes = np.array([])
    global_index = 0

    # Вертикальная граница
    for i in range(NODES_NUMBER_TETTA):
        x_right_bound = A
        y_right_bound = i * (B / (NODES_NUMBER_TETTA - 1))
        fi_current = FI_DELTA * i

        distance = np.sqrt((x_right_bound - get_x_coord(R0, fi_current)) ** 2 + (y_right_bound - get_y_coord(R0, fi_current)) ** 2)
        r_temp = R0

        for k in range(NODES_NUMBER_RADIAL_NEAR_HOLE):
            x_coord = get_x_coord(r_temp, fi_current)
            y_coord = get_y_coord(r_temp, fi_current)

            nodes = np.append(nodes, [Node(x_coord, y_coord, global_index)])
            global_index += 1

            r_temp += MULTIPLY_COEFF * r_temp * FI_DELTA

        r_delta = (distance - (r_temp - R0)) / (NODES_NUMBER_RADIAL_NEAR_BOUNDS - 1)

        for j in range(NODES_NUMBER_RADIAL_NEAR_BOUNDS):

            if j == NODES_NUMBER_RADIAL_NEAR_BOUNDS - 1:
                x_coord = x_right_bound
            else:
                x_coord = get_x_coord(r_delta * j + r_temp, fi_current)

            y_coord = get_y_coord(r_delta * j + r_temp, fi_current)

            nodes = np.append(nodes, [Node(x_coord, y_coord, global_index)])
            global_index += 1

    fi_middle = fi_current

    # Горизонтальная граница
    for i in range(NODES_NUMBER_TETTA - 1):
        x_top_bound = A - (i + 1) * (A / (NODES_NUMBER_TETTA - 1))
        y_top_bound = B
        fi_current = FI_DELTA * (i + 1) + fi_middle

        distance = np.sqrt((x_top_bound - get_x_coord(R0, fi_current)) ** 2 + (y_top_bound - get_y_coord(R0, fi_current)) ** 2)
        r_temp = R0

        for k in range(NODES_NUMBER_RADIAL_NEAR_HOLE):
            x_coord = get_x_coord(r_temp, fi_current)
            y_coord = get_y_coord(r_temp, fi_current)

            nodes = np.append(nodes, [Node(x_coord, y_coord, global_index)])
            global_index += 1

            r_temp += MULTIPLY_COEFF * r_temp * FI_DELTA

        r_delta = (distance - (r_temp - R0)) / (NODES_NUMBER_RADIAL_NEAR_BOUNDS - 1)

        for j in range(NODES_NUMBER_RADIAL_NEAR_BOUNDS):

            if j == NODES_NUMBER_RADIAL_NEAR_BOUNDS - 1:
                y_coord = y_top_bound
            else:
                y_coord = get_y_coord(r_delta * j + r_temp, fi_current)

            x_coord = get_x_coord(r_delta * j + r_temp, fi_current)

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


