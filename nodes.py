import numpy as np


class Node:
    def __init__(self, x, y, global_index):
        self.x = x
        self.y = y
        self.z = 0

        self.z_solve = None
        self.global_index = global_index


def create_nodes(n_x, n_y, step_x, step_y):
    nodes = []
    global_index = 0
    for i in range(n_x):
        for j in range(n_y):
            nodes.append(Node(x=step_x * i, y=step_y * j, global_index=global_index))
            global_index += 1

    print("Узлы созданы")

    return np.array(nodes, dtype="object")


def get_coords_array(nodes):
    coords = np.empty((2, 0))

    for node in nodes:
        coords = np.append(arr=coords, values=[[node.x], [node.y]], axis=1)

    return coords


