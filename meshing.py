import numpy as np
from plotly import graph_objects as go
import plotly.express as px
from params import n_x, n_y, elem_x, elem_y, step_x, step_y, l_x, l_y


# Класс для клетки
class Cell:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.w = step_x
        self.h = step_y
        self.nodes = []
        self.gauss_points = []
        self.neighbors = []
        self.jacobian = self.w * self.h / 4
        self.boundary_Gauss_points = None

        # Создаются в процессе при необходимости
        # self.left_bound_gauss_points = None
        # self.right_bound_gauss_points = None
        # self.top_bound_gauss_points = None
        # self.bottom_bound_gauss_points = None


# Класс для узла
class Node:
    def __init__(self, x, y, local_index):
        self.x = x
        self.y = y
        self.z = 0

        self.z_solve = None
        self.local_index = local_index
        self.global_index = None
        self.calculated_displ = False


# Класс для точки Гаусса
class Point:
    def __init__(self, x, y, weight):
        self.x = x
        self.y = y
        self.weight = weight


# Создание списков соседей
def append_neighbors(all_cells):
    for i in range(len(all_cells)):
        for j in range(len(all_cells[i])):
            # Верхня граница
            if i == 0:
                if j == 0:
                    all_cells[i][j].neighbors.extend(
                        [
                            all_cells[i + 1][j],
                            all_cells[i + 1][j + 1],
                            all_cells[i][j + 1]
                        ]
                    )
                elif j == len(all_cells[i]) - 1:
                    all_cells[i][j].neighbors.extend(
                        [
                            all_cells[i + 1][j],
                            all_cells[i + 1][j - 1],
                            all_cells[i][j - 1]
                        ]
                    )
                elif 0 < j < len(all_cells[i]) - 1:
                    all_cells[i][j].neighbors.extend(
                        [
                            all_cells[i][j - 1],
                            all_cells[i + 1][j - 1],
                            all_cells[i + 1][j],
                            all_cells[i + 1][j + 1],
                            all_cells[i][j + 1],
                        ]
                    )
                
            elif 0 < i < len(all_cells) - 1: 
                if j == 0:
                    all_cells[i][j].neighbors.extend(
                        [
                            all_cells[i - 1][j],
                            all_cells[i - 1][j + 1],
                            all_cells[i + 1][j],
                            all_cells[i + 1][j + 1],
                            all_cells[i][j + 1],
                        ]
                    )
                
                elif 0 < j < len(all_cells[i]) - 1:
                    all_cells[i][j].neighbors.extend(
                        [
                            all_cells[i - 1][j - 1],
                            all_cells[i - 1][j],
                            all_cells[i - 1][j + 1],
                            all_cells[i][j - 1],
                            all_cells[i][j + 1],
                            all_cells[i + 1][j - 1],
                            all_cells[i + 1][j],
                            all_cells[i + 1][j + 1],
                        ]
                    )
                elif j == len(all_cells[i]) - 1:
                    all_cells[i][j].neighbors.extend(
                        [
                            all_cells[i + 1][j],
                            all_cells[i + 1][j - 1],
                            all_cells[i][j - 1],
                            all_cells[i - 1][j - 1],
                            all_cells[i - 1][j],
                        ]
                )
            elif i == len(all_cells) - 1:
                if j == 0:
                    all_cells[i][j].neighbors.extend(
                        [
                            all_cells[i - 1][j],
                            all_cells[i - 1][j + 1],
                            all_cells[i][j + 1]
                        ]
                    )
                    
                elif j == len(all_cells[i]) - 1:
                    all_cells[i][j].neighbors.extend(
                        [
                            all_cells[i - 1][j],
                            all_cells[i - 1][j - 1],
                            all_cells[i][j - 1]
                        ]
                    )
                elif 0 < j < len(all_cells[i]) - 1:
                    all_cells[i][j].neighbors.extend(
                        [
                            all_cells[i][j - 1],
                            all_cells[i - 1][j - 1],
                            all_cells[i - 1][j],
                            all_cells[i - 1][j + 1],
                            all_cells[i][j + 1],
                        ]
                    )  
        
    return all_cells


# Создание узлов
def append_nodes(all_cells):
    for row in all_cells:
        for cell in row:

            cell.nodes.extend(
                [
                    Node(x=cell.x, y=cell.y, local_index=1),
                    Node(x=cell.x + cell.w, y=cell.y, local_index=2),
                    Node(x=cell.x + cell.w, y=cell.y + cell.h, local_index=3),
                    Node(x=cell.x, y=cell.y + cell.h, local_index=4)
                ]
            )

    return all_cells


# Глобальная индексация узлов
def global_indexes_for_nodes(all_cells):
    for j in range(elem_y):
        for i in range(elem_x):
            all_cells[j][i].nodes[0].global_index = i * n_y + j
            all_cells[j][i].nodes[1].global_index = (i + 1) * n_y + j 
            all_cells[j][i].nodes[2].global_index = (i + 1) * n_y + j + 1
            all_cells[j][i].nodes[3].global_index = i * n_y + j + 1

    return all_cells


# Функции для определения точек Гаусса внутри клетки
# TODO: Исправить эти функции, чтобы были параметры по умолчанию и можно было использовать для граничных гаусс. точек
def x(ksi, nu, x0, x1):
    return (x1 - x0) * (ksi + 1) / 2 + x0


def y(ksi, nu, y0, y1):
    return (y1 - y0) * (nu + 1) / 2 + y0


def create_Gauss_points_for_bound(cell):

    coords_gpoints_on_bound = [
        -0.8611363,
        - 0.3399810,
        0.3399810,
        0.8611363,
    ]

    weight_gpoints_on_bound = [
        0.3478548,
        0.6521451,
        0.6521451,
        0.3478548,
    ]

    # Гауссовы точки для границы
    if cell.x == 0 and (cell.y != 0 or round(cell.y, 5) != round(l_y - cell.h, 5)):
        cell.boundary_Gauss_points = []
        for i in range(len(coords_gpoints_on_bound)):
            cell.boundary_Gauss_points.append(
                Point(
                    x=0,
                    y=y(None, coords_gpoints_on_bound[i], cell.y, cell.y + cell.h),
                    weight=weight_gpoints_on_bound[i]
                )
            )

    elif round(cell.x, 5) == round(l_x - cell.w, 5) and (cell.y != 0 or round(cell.y, 5) != round(l_y - cell.h, 5)):
        cell.boundary_Gauss_points = []
        for i in range(len(coords_gpoints_on_bound)):
            cell.boundary_Gauss_points.append(
                Point(
                    x=l_x,
                    y=y(None, coords_gpoints_on_bound[i], cell.y, cell.y + cell.h),
                    weight=weight_gpoints_on_bound[i]
                )
            )

    if cell.y == 0 and (cell.x != 0 or round(cell.x, 5) != round(l_x - cell.w, 5)):
        cell.boundary_Gauss_points = []
        for i in range(len(coords_gpoints_on_bound)):
            cell.boundary_Gauss_points.append(
                Point(
                    x=x(coords_gpoints_on_bound[i], None, cell.x, cell.x + cell.w),
                    y=0,
                    weight=weight_gpoints_on_bound[i]
                )
            )

    elif round(cell.y, 5) == round(l_y - cell.h, 5) and (cell.x != 0 or round(cell.x, 5) != round(l_x - cell.w, 5)):
        cell.boundary_Gauss_points = []
        for i in range(len(coords_gpoints_on_bound)):
            cell.boundary_Gauss_points.append(
                Point(
                    x=x(coords_gpoints_on_bound[i], None, cell.x, cell.x + cell.w),
                    y=l_y,
                    weight=weight_gpoints_on_bound[i]
                )
            )

    # Углы
    if cell.x == 0 and cell.y == 0:
        cell.boundary_Gauss_points = []
        for i in range(len(coords_gpoints_on_bound)):
            cell.boundary_Gauss_points.append(
                Point(
                    x=0,
                    y=y(None, coords_gpoints_on_bound[i], cell.y, cell.y + cell.h),
                    weight=weight_gpoints_on_bound[i]
                )
            )
        for i in range(len(coords_gpoints_on_bound)):
            cell.boundary_Gauss_points.append(
                    Point(
                        x=x(coords_gpoints_on_bound[i], None, cell.x, cell.x + cell.w),
                        y=0,
                        weight=weight_gpoints_on_bound[i]
                    )
                )

    elif cell.x == 0 and round(cell.y, 5) == round(l_y - cell.h, 5):
        cell.boundary_Gauss_points = []
        for i in range(len(coords_gpoints_on_bound)):
            cell.boundary_Gauss_points.append(
                Point(
                    x=0,
                    y=y(None, coords_gpoints_on_bound[i], cell.y, cell.y + cell.h),
                    weight=weight_gpoints_on_bound[i]
                )
            )
        for i in range(len(coords_gpoints_on_bound)):
            cell.boundary_Gauss_points.append(
                Point(
                    x=x(coords_gpoints_on_bound[i], None, cell.x, cell.x + cell.w),
                    y=l_y,
                    weight=weight_gpoints_on_bound[i]
                )
            )

    elif round(cell.x, 5) == round(l_x - cell.w, 5) and round(cell.y, 5) == round(l_y - cell.h, 5):
        cell.boundary_Gauss_points = []
        for i in range(len(coords_gpoints_on_bound)):
            cell.boundary_Gauss_points.append(
                Point(
                    x=x(coords_gpoints_on_bound[i], None, cell.x, cell.x + cell.w),
                    y=l_y,
                    weight=weight_gpoints_on_bound[i]
                )
            )
        for i in range(len(coords_gpoints_on_bound)):
            cell.boundary_Gauss_points.append(
                Point(
                    x=l_x,
                    y=y(None, coords_gpoints_on_bound[i], cell.y, cell.y + cell.h),
                    weight=weight_gpoints_on_bound[i]
                )
            )

    elif round(cell.x, 5) == round(l_x - cell.w, 5) and cell.y == 0:
        cell.boundary_Gauss_points = []
        for i in range(len(coords_gpoints_on_bound)):
            cell.boundary_Gauss_points.append(
                Point(
                    x=l_x,
                    y=y(None, coords_gpoints_on_bound[i], cell.y, cell.y + cell.h),
                    weight=weight_gpoints_on_bound[i]
                )
            )
        for i in range(len(coords_gpoints_on_bound)):
            cell.boundary_Gauss_points.append(
                Point(
                    x=x(coords_gpoints_on_bound[i], None, cell.x, cell.x + cell.w),
                    y=0,
                    weight=weight_gpoints_on_bound[i]
                )
            )


    return cell

# Cоздание точек Гаусса
def append_Gauss_points(all_cells):
    # По часовой стрелке
    # coords_gpoints_in_elem = [
    #     [ -1 / np.sqrt(3), 1 / np.sqrt(3)],
    #     [ 1 / np.sqrt(3), 1 / np.sqrt(3)],
    #     [ 1 / np.sqrt(3), -1 / np.sqrt(3)],
    #     [ -1 / np.sqrt(3), -1 / np.sqrt(3)]
    # ]

    coords_gpoints_in_elem = [[0, 0]]

    weight = 2

    for i in range(len(all_cells)):
        for j in range(len(all_cells[i])):
            cell = all_cells[i][j]
            for coord in coords_gpoints_in_elem:
                cell.gauss_points.append(
                    Point(
                        x=x(coord[0], coord[1], cell.x, cell.x + cell.w),
                        y=y(coord[0], coord[1], cell.y, cell.y + cell.h),
                        weight=weight
                    )
                )

            all_cells[i][j] = create_Gauss_points_for_bound(cell)
                
    return all_cells
            

# Создание ячеек
def create_cells(n_x, n_y, l_x, l_y):

    all_cells = []

    # TODO: Заменить на генератор
    for i in range(0, n_y - 1):
        row = []
        for j in range(0, n_x - 1):
            row.append(Cell(x=j * step_x, y=i * step_y))
        all_cells.append(row)

    all_cells = append_neighbors(all_cells=all_cells)
    all_cells = append_nodes(all_cells=all_cells)
    all_cells = append_Gauss_points(all_cells=all_cells)
    all_cells = global_indexes_for_nodes(all_cells=all_cells)

    return np.array(all_cells, dtype="object")







    










        

