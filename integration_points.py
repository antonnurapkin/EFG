import numpy as np
from params import CELLS_NUMBER, A, B


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
def create_integration_points():

    integration_points = np.array([])

    point_local_coords = [-0.861136, -0.339981, 0.339981, 0.861136]
    weights = [0.347854, 0.652145]

    step_x = A / CELLS_NUMBER
    step_y = B / CELLS_NUMBER

    for i in range(CELLS_NUMBER):
        for j in range(CELLS_NUMBER):
            x1, y1 = step_x * i, step_y * j
            x2, y2 = step_x * (i + 1), step_y * j
            x3, y3 = step_x * (i + 1), step_y * (j + 1)
            x4, y4 = step_x * i, step_y * (j + 1)

            nodes_x_coords = np.array([x1, x2, x3, x4])
            nodes_y_coords = np.array([y1, y2, y3, y4])

            for k in range(len(point_local_coords)):
                for l in range(len(point_local_coords)):

                    if np.abs(point_local_coords[k]) == 0.861136 or np.abs(point_local_coords[l]) == 0.861136:
                        weight = weights[0]
                    else:
                        weight = weights[1]

                    point_x_coord = get_global_coord(ksi=point_local_coords[k], nu=point_local_coords[l], coords=nodes_x_coords)
                    point_y_coord = get_global_coord(ksi=point_local_coords[k], nu=point_local_coords[l], coords=nodes_y_coords)

                    jacobian = calculate_jacobain(
                        x_coords=nodes_x_coords,
                        y_coords=nodes_y_coords,
                        ksi=point_local_coords[k],
                        nu=point_local_coords[l]
                    )

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











