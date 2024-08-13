import numpy as np
from params import A, B, NODES_NUMBER_ON_BOUND_X, NODES_NUMBER_ON_BOUND_Y, CRACK_LENGTH, CRACK_HALF_WIDTH, NODES_ON_CRACK


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
    step_y = A / (NODES_NUMBER_ON_BOUND_Y - 1)

    for i in range(NODES_NUMBER_ON_BOUND_X):
        for j in range(NODES_NUMBER_ON_BOUND_Y):

            nodes = np.append(nodes, [Node(x=step_x * i, y=step_y * j, global_index=global_index)])
            global_index += 1

    nodes, crack_top_ind = create_crack(nodes=nodes, global_index=global_index)

    nodes = seal_around_top(nodes=nodes, crack_top_ind=crack_top_ind)

    print("Узлы созданы...")

    return nodes, crack_top_ind


def create_crack(nodes, global_index):
    x0 = 0
    y0 = B / 2

    step = CRACK_LENGTH / (NODES_ON_CRACK - 1)

    crack_top_ind = None

    for i in range(NODES_ON_CRACK):
        x = step * i
        if i != NODES_ON_CRACK - 1:
            y_high_bound = CRACK_HALF_WIDTH * np.sqrt(1 - ((x - x0) / CRACK_LENGTH) ** 2) + y0
            y_low_bound = -1 * CRACK_HALF_WIDTH * np.sqrt(1 - ((x - x0) / CRACK_LENGTH) ** 2) + y0

            nodes = np.append(nodes, [Node(x=x, y=y_high_bound, global_index=global_index)])
            global_index += 1

            nodes = np.append(nodes, [Node(x=x, y=y_low_bound, global_index=global_index)])
            global_index += 1
        else:
            y_high_bound = CRACK_HALF_WIDTH * np.sqrt(1 - ((x - x0) / CRACK_LENGTH) ** 2) + y0
            nodes = np.append(nodes, [Node(x=x, y=y_high_bound, global_index=global_index)])
            crack_top_ind = global_index
            global_index += 1

    return nodes, crack_top_ind


def seal_around_top(nodes, crack_top_ind):

    global_index = nodes[-1].global_index + 1

    # Параметры уплотнения
    rad_step = CRACK_LENGTH / 5

    fi_0 = np.pi - np.pi / 9
    fi_current = fi_0
    fi_step = np.pi / 12
    fi_end = -fi_0

    # Координаты вершины трещины
    crack_top = nodes[crack_top_ind]
    x0, y0 = crack_top.x, crack_top.y

    while fi_current >= fi_end:
        for i in range(1, 5):
            nodes = np.append(
                nodes,
                [Node(
                    x=i * rad_step * np.cos(fi_current) + x0,
                    y=i * rad_step * np.sin(fi_current) + y0,
                    global_index=global_index
                )]
            )

            global_index += 1

        fi_current -= fi_step

    return nodes


def get_nodes_coords(nodes):
    coords = np.empty((2, 0), dtype='f')

    for node in nodes:
        coords = np.append(arr=coords, values=[[node.x], [node.y]], axis=1)

    print("Получены координаты узлов...")

    return np.around(coords, decimals=10)


