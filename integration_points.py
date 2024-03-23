import numpy as np
from params import NODES_NUMBER_RADIAL


# Точка Гаусса
# Для удобства, в объекте находится и вес точки и якобиан ячейки, в которой она находится
class Point:
    def __init__(self, x, y, weight, jacobian=None):
        self.x = x
        self.y = y
        self.weight = weight

        # Это характеристика клетки, в которой находится данная точка
        self.jacobian = jacobian


def get_global_coord(ksi, nu, coords):
    N = np.array([
        1 / 4 * (1 + ksi) * (1 - nu),
        1 / 4 * (1 + ksi) * (1 + nu),
        1 / 4 * (1 - ksi) * (1 + nu),
        1 / 4 * (1 - ksi) * (1 - nu),
    ])

    return np.dot(np.transpose(N), coords)


def calculate_jacobain(x_coords, y_coords, ksi, nu):
    coeffs_ksi = np.array([
        0.25 * (1 - nu),
        0.25 * (1 + nu),
        -0.25 * (1 + nu),
        -0.25 * (1 - nu)
    ])
    coeffs_nu = np.array([
        -0.25 * (1 + ksi),
        0.25 * (1 + ksi),
        0.25 * (1 - ksi),
        -0.25 * (1 - ksi)
    ])

    J11 = np.dot(np.transpose(coeffs_ksi), x_coords)
    J12 = np.dot(np.transpose(coeffs_ksi), y_coords)
    J21 = np.dot(np.transpose(coeffs_nu), x_coords)
    J22 = np.dot(np.transpose(coeffs_nu), y_coords)

    return np.linalg.det(np.array([[J11, J12], [J21, J22]]))


# Функция для вычисления координат точек Гаусса внутри "ячеек"
def create_integration_points(nodes_coords):

    integration_points = np.array([])

    point_local_coords = [
        [-1 / np.sqrt(3), 1 / np.sqrt(3)],
        [1 / np.sqrt(3), 1 / np.sqrt(3)],
        [1 / np.sqrt(3), -1 / np.sqrt(3)],
        [-1 / np.sqrt(3), -1 / np.sqrt(3)]
    ]

    weight = 1

    for i in range(len(nodes_coords[0]) // NODES_NUMBER_RADIAL - 1):
        for j in range(NODES_NUMBER_RADIAL - 1):
            x1, y1 = nodes_coords[:, i * NODES_NUMBER_RADIAL + j]
            x2, y2 = nodes_coords[:, i * NODES_NUMBER_RADIAL + j + 1]
            x3, y3 = nodes_coords[:, (i + 1) * NODES_NUMBER_RADIAL + j + 1]
            x4, y4 = nodes_coords[:, (i + 1) * NODES_NUMBER_RADIAL + j]

            nodes_x_coords = np.array([x1, x2, x3, x4])
            nodes_y_coords = np.array([y1, y2, y3, y4])

            for p in point_local_coords:
                point_x_coord = get_global_coord(ksi=p[0], nu=p[1], coords=nodes_x_coords)
                point_y_coord = get_global_coord(ksi=p[0], nu=p[1], coords=nodes_y_coords)

                jacobian = calculate_jacobain(x_coords=nodes_x_coords, y_coords=nodes_y_coords, ksi=p[0], nu=p[1])

                integration_points = np.append(
                    integration_points,
                    [
                        Point(x=point_x_coord, y=point_y_coord, weight=weight, jacobian=jacobian)
                    ]
                )

    print("Точки интегрирования созданы...")

    return integration_points


# Функция для вычисления координат точек Гаусса на границах
def create_integration_points_bound(nodes_coords, y_value=False, x_value=False, y_bound=False, x_bound=False):
    integration_points = []

    # local_coords = [-1 / np.sqrt(3), 1 / np.sqrt(3)]
    # weights = [1, 1]

    local_coords = [-0.861136, -0.339981, 0.339981, 0.861136]
    weights = [0.347854, 0.652145, 0.652145, 0.347854]

    # Сделано для удобства, чтобы эту функцию можно использовать для любой границы
    if y_bound:
        current_coord = 1
        other_coord = 0
        value = y_value
    elif x_bound:
        current_coord = 0
        other_coord = 1
        value = x_value

    # Вычисления координат узлов на текущей границе
    bound_nodes_coords = np.sort(nodes_coords[:, np.where(nodes_coords[current_coord] == value)[0]], axis=1)

    for i in range(len(bound_nodes_coords[other_coord]) - 1):
        # Вычисления ширины участка интегрирования и центра
        width = (bound_nodes_coords[other_coord][i + 1] - bound_nodes_coords[other_coord][i]) / 2 # Половина ширины
        centre_coord = bound_nodes_coords[other_coord][i] + width

        temp = np.array([])

        for j in range(len(local_coords)):

            # Сделано для удобства, чтобы эту функцию можно использовать для любой границы
            if y_bound:
                point_x_coord = centre_coord + width * local_coords[j]
                point_y_coord = value
            elif x_bound:
                point_x_coord = value
                point_y_coord = centre_coord + width * local_coords[j]

            jacobian = width

            temp = np.append(
                temp,
                [
                    Point(x=point_x_coord, y=point_y_coord, weight=weights[j], jacobian=jacobian)
                ]
            )

        integration_points.append(temp)

    print("Точки интегрирования созданы...")

    return np.array(integration_points)











