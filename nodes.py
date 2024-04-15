import numpy as np
from params import A, B, NODES_NUMBER_ON_BOUND_X, NODES_NUMBER_ON_BOUND_Y


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

    step_x = A / (NODES_NUMBER_ON_BOUND_X - 1)
    step_y = B / (NODES_NUMBER_ON_BOUND_Y - 1)

    for i in range(NODES_NUMBER_ON_BOUND_X):
        for j in range(NODES_NUMBER_ON_BOUND_Y):

            nodes = np.append(nodes, [Node(step_x * i, step_y * j, global_index)])
            global_index += 1

    return nodes


def get_nodes_coords(nodes):
    coords = np.empty((2, 0), dtype='f')

    for node in nodes:
        coords = np.append(arr=coords, values=[[node.x], [node.y]], axis=1)

    print("Получены координаты узлов...")

    return np.around(coords, decimals=10)


