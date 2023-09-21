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


# Класс для узла
class Node:
    def __init__(self, x, y, local_index):
        self.x = x
        self.y = y
        self.z = 0

        self.local_index = local_index
        self.global_index = None


# Класс для точки Гаусса
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


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
def x(ksi, nu, x0, x1):
    return (x1 - x0) * (ksi + 1) / 2 + x0


def y(ksi, nu, y0, y1):
    return (y1 - y0) * (nu + 1) / 2 + y0


# Cоздание точек Гаусса
def append_Gauss_points(all_cells):
    # По часовой стрелке
    coords = [
        [ -1 / np.sqrt(3), 1 / np.sqrt(3)],
        [ 1 / np.sqrt(3), 1 / np.sqrt(3)],
        [ 1 / np.sqrt(3), -1 / np.sqrt(3)],
        [ -1 / np.sqrt(3), -1 / np.sqrt(3)]
    ]

    for row in all_cells:
        for cell in row:
            for coord in coords:
                cell.gauss_points.append(
                    Point(
                        x=x(coord[0], coord[1], cell.x, cell.x + cell.w),
                        y=y(coord[0], coord[1], cell.y, cell.y + cell.h)
                    )
                )
                
    return all_cells
            

# Создание ячеек
def create_cells(n_x, n_y, l_x, l_y):

    all_cells = []

    for i in range(0, n_y - 1):
        row = []
        for j in range(0, n_x - 1):
            row.append(Cell(x=j * step_x, y=i * step_y))
        all_cells.append(row)

    all_cells = append_neighbors(all_cells=all_cells)
    all_cells = append_nodes(all_cells=all_cells)
    all_cells = append_Gauss_points(all_cells=all_cells)
    all_cells = global_indexes_for_nodes(all_cells=all_cells)

    return all_cells


cells = create_cells(n_x, n_y, l_x, l_y)


# x_gp = []
# y_gp = []

# x_n = []
# y_n = []


# for row in cells:
#     for cell in row:
#         for point in cell.gauss_points:
#             x_gp.append(point.x)
#             y_gp.append(point.y)

#         for node in cell.nodes:
#             x_n.append(node.x)
#             y_n.append(node.y)



# fig = go.Figure(go.Scatter(x=x_gp, y=y_gp, mode='markers'))
# fig.add_trace(go.Scatter(x=x_n, y=y_n, mode='markers'))

# fig.show()











    










        

